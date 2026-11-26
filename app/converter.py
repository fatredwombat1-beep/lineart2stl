"""Convert a line-art (or photo) image into a single-piece, printable STL outline.

Pipeline: image -> black/white mask -> vector polygons (mm) -> thicken lines ->
optional solid frame -> drop tiny specks -> bridge floating islands to the main
body (minimum spanning tree) -> extrude -> STL + preview PNG.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import manifold3d
import trimesh
from shapely import affinity, make_valid
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import nearest_points, unary_union

from portrait import photo_to_mask


@dataclass
class Settings:
    mode: str = "lineart"        # "lineart" or "photo"
    width_mm: float = 200.0      # overall output width incl. frame
    thickness_mm: float = 2.4
    thicken_mm: float = 0.4      # extra line width (total, both sides)
    min_speck_mm2: float = 1.0   # pieces smaller than this are removed
    bridge_mm: float = 1.0       # bridge width
    frame_mm: float = 4.0        # solid frame width, 0 = no frame
    invert: bool = False         # set if image is light lines on dark
    crop: bool = True            # crop to drawing before adding frame
    absorb_border: bool = True   # merge a border already drawn in the image into the frame
    detail: int = 5              # photo mode only: detail on people 1..10
    background: int = 3          # photo mode only: background lines 0 (none)..10

    def clamp(self) -> "Settings":
        self.mode = "photo" if self.mode == "photo" else "lineart"
        self.width_mm = min(max(self.width_mm, 20), 1000)
        self.thickness_mm = min(max(self.thickness_mm, 0.4), 50)
        self.thicken_mm = min(max(self.thicken_mm, 0), 5)
        self.min_speck_mm2 = min(max(self.min_speck_mm2, 0), 500)
        self.bridge_mm = min(max(self.bridge_mm, 0.4), 10)
        self.frame_mm = min(max(self.frame_mm, 0), 50)
        self.detail = int(min(max(self.detail, 1), 10))
        self.background = int(min(max(self.background, 0), 10))
        return self


class ConversionError(ValueError):
    pass


MAX_PX = 2400          # working resolution (long side)
MAX_ISLANDS = 1500     # safety valve for very noisy images


# ---------------------------------------------------------------- image -> mask
def _mask_from_image(data: bytes, s: Settings) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ConversionError("Could not read that file as an image.")
    h, w = img.shape[:2]
    scale = MAX_PX / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    if s.invert:
        img = 255 - img

    if s.mode == "lineart":
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        mask, info = photo_to_mask(img, s.detail, s.background)
        s._photo_info = info
    # remove single-pixel noise, close pinholes
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

    # Lines must be the minority; if not, the image is probably inverted.
    if mask.mean() > 127:
        mask = 255 - mask
    if mask.max() == 0:
        raise ConversionError("No lines found in the image.")

    if s.crop:
        ys, xs = np.nonzero(mask)
        pad = 2
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
        mask = mask[y0:y1, x0:x1]
    return mask


# ------------------------------------------------------------- mask -> polygons
def _polys(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _polys(g)]
    return []


def _mask_to_geometry(mask: np.ndarray, px_mm: float):
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        raise ConversionError("No lines found in the image.")
    hier = hier[0]
    polys = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1 or len(c) < 3:      # holes are added with their parent
            continue
        holes, ch = [], hier[i][2]
        while ch != -1:
            if len(contours[ch]) >= 3:
                holes.append(contours[ch][:, 0, :])
            ch = hier[ch][0]
        polys += _polys(make_valid(Polygon(c[:, 0, :], holes)))
    geom = unary_union(polys)
    H = mask.shape[0]
    # pixels -> mm, flip Y so the print isn't mirrored
    geom = affinity.affine_transform(geom, [px_mm, 0, 0, -px_mm, 0, H * px_mm])
    # smooth pixel staircase
    r = px_mm * 0.75
    geom = geom.buffer(r, join_style=1).buffer(-r, join_style=1).simplify(px_mm * 0.4)
    return geom


def _border_insets(mask: np.ndarray, band=0.08, cover=0.8):
    """Pixels from each edge (left, top, right, bottom) to the inner edge of a
    border line already drawn in the image; 0 where there is none."""
    m = mask > 0
    H, W = m.shape

    def side(lines, n):
        lim = int(n * band)
        cov = lines[:lim].mean(axis=1)
        hits = np.nonzero(cov > cover)[0]
        return int(hits.max()) + 1 if len(hits) else 0

    return (side(m.T, W), side(m, H), side(m.T[::-1], W), side(m[::-1], H))


# ------------------------------------------------------------- bridging (MST)
def _bridge(a_geom, b_geom, width):
    a, b = nearest_points(a_geom, b_geom)
    d = a.distance(b)
    if d > 1e-9:
        ext = 0.3
        dx, dy = (b.x - a.x) / d * ext, (b.y - a.y) / d * ext
        line = LineString([(a.x - dx, a.y - dy), (b.x + dx, b.y + dy)])
    else:
        line = LineString([(a.x - 0.2, a.y), (a.x + 0.2, a.y)])
    return line.buffer(width / 2, cap_style=2), d


def _connect(parts, width):
    """Prim's MST from the largest part; returns bridge polygons."""
    n = len(parts)
    if n <= 1:
        return []
    simp = [p.simplify(0.1) for p in parts]          # cheaper distance queries
    bounds = np.array([p.bounds for p in parts])
    in_tree = np.zeros(n, bool)
    in_tree[0] = True
    best = np.full(n, np.inf)
    link = np.zeros(n, int)

    def relax(j):
        bj = bounds[j]
        for i in np.nonzero(~in_tree)[0]:
            bi = bounds[i]
            gap = max(bj[0] - bi[2], bi[0] - bj[2], 0) ** 2 + max(bj[1] - bi[3], bi[1] - bj[3], 0) ** 2
            if gap ** 0.5 >= best[i]:
                continue
            d = simp[i].distance(simp[j])
            if d < best[i]:
                best[i], link[i] = d, j

    relax(0)
    bridges = []
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        i = int(np.argmin(cand))
        in_tree[i] = True
        br, _ = _bridge(parts[i], parts[link[i]], width)
        bridges.append(br)
        relax(i)
    return bridges


# ---------------------------------------------------------------- main entry
def convert(data: bytes, s: Settings) -> dict:
    s.clamp()
    mask = _mask_from_image(data, s)
    content_w = s.width_mm - 2 * s.frame_mm
    if content_w <= 5:
        raise ConversionError("Frame is too wide for that overall width.")
    px_mm = content_w / mask.shape[1]
    geom = _mask_to_geometry(mask, px_mm)
    cw, ch = mask.shape[1] * px_mm, mask.shape[0] * px_mm

    if s.thicken_mm > 0:
        geom = geom.buffer(s.thicken_mm / 2, join_style=1)
    # keep inside the drawing area
    geom = geom.intersection(box(0, 0, cw, ch))

    if s.frame_mm > 0:
        f, ov = s.frame_mm, 0.3
        il, it, ir, ib = (_border_insets(mask) if s.absorb_border else (0, 0, 0, 0))
        il, it, ir, ib = (v * px_mm + ov for v in (il, it, ir, ib))
        frame = box(-f, -f, cw + f, ch + f).difference(box(il, ib, cw - ir, ch - it))
        geom = unary_union([geom, frame])

    parts = sorted(_polys(make_valid(geom)), key=lambda p: -p.area)
    if not parts:
        raise ConversionError("Nothing printable was left after processing.")
    main, rest = parts[0], parts[1:]
    kept = [p for p in rest if p.area >= s.min_speck_mm2]
    removed = len(rest) - len(kept)
    if len(kept) > MAX_ISLANDS:
        raise ConversionError(
            f"{len(kept)} separate pieces found - the image is too noisy. "
            "Raise 'Remove specks smaller than' or use a cleaner image.")

    parts = [main] + kept
    bridges = _connect(parts, s.bridge_mm)
    body = unary_union(parts + bridges)
    # tidy any hairline leftovers from floating-point unions
    for _ in range(3):
        ps = sorted(_polys(make_valid(body)), key=lambda p: -p.area)
        if len(ps) == 1:
            break
        extra = _connect(ps, s.bridge_mm)
        bridges += extra
        body = unary_union(ps + extra)
    ps = sorted(_polys(make_valid(body)), key=lambda p: -p.area)
    body = ps[0]
    # reduce vertex count (smaller STL) and make sure the outline is clean
    body = sorted(_polys(make_valid(body.simplify(0.03).buffer(0))), key=lambda p: -p.area)[0]

    # move to positive coordinates
    minx, miny, _, _ = body.bounds
    body = affinity.translate(body, -minx, -miny)
    bridges = [affinity.translate(b, -minx, -miny) for b in bridges]

    mesh = _extrude(body, s.thickness_mm)
    stl = mesh.export(file_type="stl")

    # weakest-link check
    thin = "ok"
    if len(_polys(body.buffer(-0.3))) > 1:
        thin = "some joins are under 0.6 mm - consider more line thickening"

    bx = body.bounds
    stats = {
        "size_mm": [round(bx[2] - bx[0], 1), round(bx[3] - bx[1], 1), s.thickness_mm],
        "specks_removed": removed,
        "islands_joined": len(kept),
        "bridges": len(bridges),
        "single_piece": len(ps) == 1,
        "watertight": bool(mesh.is_watertight),
        "thin_check": thin,
        "settings": asdict(s),
        "photo": getattr(s, "_photo_info", None),
    }
    return {"stl": stl, "preview": _preview(body, bridges), "stats": stats}


def _extrude(poly: Polygon, h: float) -> trimesh.Trimesh:
    """Robust extrusion via Manifold (always yields a closed, manifold mesh)."""
    rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
    cs = manifold3d.CrossSection(rings, manifold3d.FillRule.EvenOdd)
    m = cs.extrude(h).to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(m.vert_properties)[:, :3], faces=np.asarray(m.tri_verts), process=True)


def _preview(body: Polygon, bridges, long_side=1000) -> bytes:
    minx, miny, maxx, maxy = body.bounds
    sc = long_side / max(maxx - minx, maxy - miny)
    W, H = int((maxx - minx) * sc) + 20, int((maxy - miny) * sc) + 20
    img = np.full((H, W, 3), 255, np.uint8)

    def ring(r):
        a = np.asarray(r.coords)
        return np.column_stack([(a[:, 0] - minx) * sc + 10, H - ((a[:, 1] - miny) * sc + 10)]).round().astype(np.int32)

    def fill(p, col):
        cv2.fillPoly(img, [ring(p.exterior)] + [ring(i) for i in p.interiors], col, lineType=cv2.LINE_AA)

    fill(body, (20, 20, 20))
    for b in bridges:
        for p in _polys(b.intersection(body)):
            fill(p, (40, 40, 220))     # BGR red
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()
