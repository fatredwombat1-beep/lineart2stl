"""Photo -> clean line drawing.

People get bold outlines (segmentation), faces get clean drawn features from
MediaPipe face landmarks, clothing/hands get moderate detail, and the
background is reduced to a few long lines that hint at the scene.
Returns a uint8 mask (lines = 255) at the input resolution.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects

try:
    import mediapipe as mp
    _mp_ok = True
except Exception:          # pragma: no cover - fall back to edges only
    _mp_ok = False

if _mp_ok:
    _FM = mp.solutions.face_mesh
    _FEATURES = {
        "oval": _FM.FACEMESH_FACE_OVAL,
        "l_brow": _FM.FACEMESH_LEFT_EYEBROW,
        "r_brow": _FM.FACEMESH_RIGHT_EYEBROW,
        "l_eye": _FM.FACEMESH_LEFT_EYE,
        "r_eye": _FM.FACEMESH_RIGHT_EYE,
        "lips": _FM.FACEMESH_LIPS,
    }
NOSE_BOTTOM = [64, 98, 97, 2, 326, 327, 294]
NOSE_SMALL = [98, 97, 2, 326, 327]
R_EYE_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
L_EYE_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263]
R_BROW = [70, 63, 105, 66, 107]
L_BROW = [336, 296, 334, 293, 300]
NOSE_BRIDGE = [168, 6, 197, 195, 5]
MOUTH_LINE = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]


def _chains(conns):
    """Turn an unordered set of landmark index pairs into ordered polylines."""
    adj = {}
    for a, b in conns:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, chains = set(), []
    starts = [n for n, v in adj.items() if len(v) == 1] + list(adj)
    for s in starts:
        if s in seen:
            continue
        chain, cur, prev = [s], s, None
        seen.add(s)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                if len(chain) > 2 and chain[0] in adj[cur]:
                    chain.append(chain[0])          # closed loop
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
            seen.add(cur)
        chains.append(chain)
    return chains


if _mp_ok:
    _CHAINS = {k: _chains(v) for k, v in _FEATURES.items()}


# --------------------------------------------------------------- detectors
def _person_mask(rgb):
    if not _mp_ok:
        return None
    h, w = rgb.shape[:2]
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as seg:
        m = seg.process(rgb).segmentation_mask
    if m is None:
        return None
    m = (m > 0.5).astype(np.uint8) * 255
    k = max(3, (w + h) // 300) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
    # drop small blobs
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 0.01 * h * w:
            keep[lab == i] = 255
    return keep if keep.any() else None


def _faces(rgb):
    """Return a list of (N,2) float landmark arrays in image pixels."""
    if not _mp_ok:
        return []
    h, w = rgb.shape[:2]
    boxes = []
    for sel in (1, 0):
        with mp.solutions.face_detection.FaceDetection(model_selection=sel, min_detection_confidence=0.5) as fd:
            res = fd.process(rgb)
        for d in res.detections or []:
            b = d.location_data.relative_bounding_box
            boxes.append((b.xmin * w, b.ymin * h, b.width * w, b.height * h))
    # de-duplicate overlapping boxes
    uniq = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        if all(abs(cx - (u[0] + u[2] / 2)) > u[2] / 2 or abs(cy - (u[1] + u[3] / 2)) > u[3] / 2 for u in uniq):
            uniq.append(b)

    faces = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.4) as fm:
        for x, y, bw, bh in uniq:
            s = max(bw, bh) * 2.0
            cx, cy = x + bw / 2, y + bh / 2
            x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
            x1, y1 = int(min(cx + s / 2, w)), int(min(cy + s / 2, h))
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            sc = 512 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            r = fm.process(crop)
            if not r.multi_face_landmarks:
                continue
            ch, cw = crop.shape[:2]
            pts = np.array([(p.x * cw / sc + x0, p.y * ch / sc + y0) for p in r.multi_face_landmarks[0].landmark])
            faces.append(pts)
    return faces


# ------------------------------------------------------------------ edges
def _dog_lines(gray, sigma, region, keep_pct):
    """Difference-of-Gaussians dark-edge response, thresholded per region by percentile."""
    g = gray.astype(np.float32) / 255
    g1 = cv2.GaussianBlur(g, (0, 0), sigma)
    g2 = cv2.GaussianBlur(g, (0, 0), sigma * 1.6)
    resp = g2 - g1                      # >0 on the dark side of an edge
    vals = resp[region]
    if vals.size == 0 or keep_pct <= 0:
        return np.zeros(gray.shape, bool)
    t = max(np.percentile(vals, 100 - keep_pct), 0.004)
    return (resp > t) & region


def _clean(lines, min_len_px):
    """Keep strokes whose skeleton is at least min_len_px long."""
    sk = skeletonize(lines)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    good = np.zeros(n, bool)
    good[1:] = st[1:, cv2.CC_STAT_AREA] >= min_len_px
    return good[lab]


def _lower_oval(pts, chain, cut_y):
    """Split the face-oval loop into runs that lie below cut_y (jaw and cheeks)."""
    runs, cur = [], []
    for i in chain:
        if pts[i, 1] >= cut_y:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    # the loop may start mid-run; join first and last if they wrap around
    if len(runs) > 1 and chain[0] == chain[-1] and runs[0][0] == chain[0] and runs[-1][-1] == chain[-1]:
        runs[0] = runs.pop() + runs[0]
    return [r for r in runs if len(r) > 2]


def photo_to_mask(bgr: np.ndarray, detail: int = 5, background: int = 3, px_mm: float = 0.1):
    """detail: 1..10 detail on people; background: 0 (none) .. 10 (lots).
    px_mm: printed size of one pixel, so line widths can be set in millimetres."""
    h, w = bgr.shape[:2]
    L = max(h, w)
    mm = lambda v: max(2, int(round(v / px_mm)))   # mm -> px
    # widths before the converter's global thickening (+0.4 mm by default)
    stroke = mm(0.5)          # general lines
    outline = mm(0.8)         # person silhouette
    feat = mm(0.4)            # facial features

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 40, 9)
    gray = cv2.bilateralFilter(gray, 9, 40, 9)

    person = _person_mask(rgb)
    faces = _faces(rgb)
    if person is None:
        person = np.zeros((h, w), np.uint8)

    # Lower face (below the eyebrows) is drawn from landmarks, so photo edges are
    # removed there. Forehead and hair keep their photo lines.
    lower_face = np.zeros((h, w), np.uint8)
    for pts in faces:
        brow_y = pts[R_BROW + L_BROW, 1].mean()
        for ch in _CHAINS["oval"]:
            poly = pts[ch].copy()
            poly[:, 1] = np.maximum(poly[:, 1], brow_y)
            cv2.fillPoly(lower_face, [poly.round().astype(np.int32)], 255)
    person_r = (person > 0) & (lower_face == 0)
    bg_r = (person == 0) & (lower_face == 0)
    if not person.any():
        person_r, bg_r = lower_face == 0, np.zeros((h, w), bool)

    sigma = L / 900
    in_lines = _dog_lines(gray, sigma, person_r, 2.0 + 1.3 * detail)
    in_lines = _clean(in_lines, L / 70)
    bg_lines = _dog_lines(gray, sigma * 1.5, bg_r, 0.6 * background)
    bg_lines = _clean(bg_lines, L / 12)
    erase = cv2.erode(lower_face, np.ones((3, 3), np.uint8))
    lines = (in_lines | bg_lines) & (erase == 0)

    out = np.zeros((h, w), np.uint8)
    out[skeletonize(lines)] = 255
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke, stroke)))

    if person.any():
        cs, _ = cv2.findContours(person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = [cv2.approxPolyDP(c, L / 800, True) for c in cs]
        cv2.polylines(out, cs, True, 255, outline, cv2.LINE_AA)

    # clear any outline/edge pixels that crossed a lower face, then draw
    # features with thin, print-sized lines
    out[erase > 0] = 0
    for pts in faces:
        face_mm = np.ptp(pts[:, 0]) * px_mm
        small = face_mm < 30

        def line(idx, th=feat, closed=False):
            cv2.polylines(out, [pts[idx].round().astype(np.int32)], closed, 255, th, cv2.LINE_AA)

        eye_y = pts[R_EYE_UPPER + L_EYE_UPPER, 1].mean()
        for ch in _CHAINS["oval"]:
            for run in _lower_oval(pts, ch, eye_y):
                line(run, th=stroke)
        line(R_BROW); line(L_BROW)
        if small:
            # simplified face: upper eyelids, nose tip, single mouth line
            line(R_EYE_UPPER); line(L_EYE_UPPER)
            line(NOSE_SMALL)
            line(MOUTH_LINE)
        else:
            for k in ("l_eye", "r_eye"):
                for ch in _CHAINS[k]:
                    line(ch)
            if len(pts) > 470:
                for c in (468, 473):
                    r = max(2, int(np.linalg.norm(pts[c + 1] - pts[c + 3]) / 2.6))
                    cv2.circle(out, tuple(pts[c].round().astype(int)), r, 255, -1, cv2.LINE_AA)
            line(NOSE_BOTTOM)
            for ch in _CHAINS["lips"]:
                line(ch)

    out = (out > 127).astype(np.uint8) * 255
    return out, {"faces": len(faces), "person_found": bool(person.any())}
