"""Photo -> line art with the "Informative Drawings" generator (ONNX).

The network does the drawing (faces, hair, clothing). This module then:
- optionally thins out background lines using MediaPipe person segmentation,
- guarantees a printable minimum line width (in mm).
Returns a uint8 mask (lines = 255) at the input resolution.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np
from skimage.morphology import skeletonize

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
_sessions: dict = {}
_lock = threading.Lock()
# Added to the Otsu threshold. Measured on the real weights, Otsu sits near 0.41
# (fine) / 0.43 (bold) at every detail level; fine needs a lower cut to keep
# glasses, eyes and mouths unbroken.
THRESH_OFFSET = {"fine": -0.06, "bold": 0.0}


def available(style: str = "fine") -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, f"lineart_{style}.onnx"))


def _session(style):
    with _lock:
        if style not in _sessions:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.environ.get("MODEL_THREADS", "0")) or os.cpu_count() or 4
            _sessions[style] = ort.InferenceSession(
                os.path.join(MODEL_DIR, f"lineart_{style}.onnx"), opts, providers=["CPUExecutionProvider"])
        return _sessions[style]


def _infer(bgr, style, long_side):
    h, w = bgr.shape[:2]
    sc = long_side / max(h, w)
    img = cv2.resize(bgr, (max(8, round(w * sc)), max(8, round(h * sc))), interpolation=cv2.INTER_AREA)
    ih, iw = img.shape[:2]
    ph, pw = (-ih) % 8, (-iw) % 8                        # network needs multiples of 4; use 8
    img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    y = _session(style).run(None, {"image": x})[0][0, 0][:ih, :iw]
    strength = 1.0 - np.clip(y, 0, 1)                     # network draws dark lines on white
    return cv2.resize(strength, (w, h), interpolation=cv2.INTER_CUBIC)


def photo_to_mask(bgr: np.ndarray, detail: int = 5, background: int = 5,
                  px_mm: float = 0.1, style: str = "fine"):
    """detail 1..10 -> network resolution (more detail = finer, more lines).
    background 0..10 -> how many lines to keep outside the people (10 = all)."""
    h, w = bgr.shape[:2]
    L = max(h, w)
    long_side = int(512 + (detail - 1) * 112)            # 512 .. 1520 px
    strength = _infer(bgr, style, long_side)

    # threshold: Otsu on the line response, clamped to a sensible range
    s8 = (strength * 255).astype(np.uint8)
    t, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(np.clip(t / 255 + THRESH_OFFSET.get(style, 0.0), 0.25, 0.6))
    lines = strength > t

    info = {"engine": "lineart-" + style, "threshold": round(t, 2)}

    # background thinning
    if background < 10:
        person = None
        try:
            from portrait import _person_mask
            person = _person_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            person = None
        info["person_found"] = person is not None
        if person is not None:
            near = cv2.dilate(person, np.ones((L // 100 | 1, L // 100 | 1), np.uint8)) > 0
            # long straight verticals off the people (door edges, downpipes) read as
            # noise; brick courses and the ground are mostly horizontal and stay
            off = skeletonize(lines & ~near).astype(np.uint8) * 255
            segs = cv2.HoughLinesP(off, 2, np.pi / 180, threshold=L // 40,
                                   minLineLength=L // 14, maxLineGap=L // 50)
            vert = np.zeros((h, w), np.uint8)
            for x1, y1, x2, y2 in (segs[:, 0] if segs is not None else []):
                if abs(x2 - x1) <= abs(y2 - y1) * np.tan(np.radians(8)):
                    cv2.line(vert, (int(x1), int(y1)), (int(x2), int(y2)), 255, max(3, L // 250))
            lines = lines & ~((vert > 0) & ~near)
            n, lab, st, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
            # fraction of each stroke lying on a person
            on = np.bincount(lab[near].ravel(), minlength=n)
            frac = on / np.maximum(st[:, cv2.CC_STAT_AREA], 1)
            sk = skeletonize(lines)
            length = np.bincount(lab[sk].ravel(), minlength=n)
            min_len = (L / 6) * (1 - background / 10) ** 2     # bg 0: remove all off-person strokes
            keep = (frac > 0.5) | (length >= min_len) if background > 0 else (frac > 0.5)
            keep[0] = False
            lines = keep[lab]

    # printable minimum width: keep the network's bold strokes, but make sure
    # every stroke is at least ~0.5 mm (before the converter's thickening)
    min_w = max(2, int(round(0.5 / px_mm)))
    sk = skeletonize(lines).astype(np.uint8) * 255
    out = cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_w, min_w)))
    out = np.maximum(out, lines.astype(np.uint8) * 255)
    return out, info
