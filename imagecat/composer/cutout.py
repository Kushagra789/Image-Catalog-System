"""Extracting an object from its catalog image as an RGBA cutout.

Mask methods:
  alpha    - the source PNG already has transparency
  polygon  - the outline stored with the object (YOLO-seg mask or OpenCV contour)
  rembg    - learned salient-object matting (rembg, U2Net by default), limited to
             the object's box and, if known, a slightly enlarged outline
  key      - background keying: pixels similar to the crop border and connected
             to it are removed (works well for flat backgrounds/cartoons)
  grabcut  - OpenCV GrabCut initialised with the bounding box (photos)
  none     - the plain rectangular crop

``auto`` picks by evidence: alpha if present; an exact OpenCV contour (pixel
outline on flat art) as-is; otherwise rembg when its model is already
downloaded (soft, accurate edges for photos and coarse YOLO-seg outlines);
then polygon, key, GrabCut, rectangle. ``auto`` never triggers a download.

Limitations: rembg finds the *salient* object in the crop, so with heavily
overlapping objects the outline constraint matters; very fine structures and
soft shadows can still be imperfect.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import config  # noqa: F401  (points U2NET_HOME at models/rembg inside the project)
from ..imaging import load_rgba
from ..models import ObjectRecord

MASK_METHODS = ("auto", "alpha", "polygon", "rembg", "key", "grabcut", "none")
REMBG_MODEL = os.environ.get("IMAGECAT_REMBG_MODEL", "u2net")
_EXACT_POLYGON_SOURCES = ("opencv_shape", "opencv_color_region", "opencv_foreground", "composite")
_sessions: dict[str, object] = {}
_session_lock = threading.Lock()


def rembg_status(model: str = REMBG_MODEL) -> tuple[bool, str]:
    """(usable without downloading, explanation)."""
    try:
        from rembg.sessions import sessions_class
    except Exception as exc:
        return False, f"rembg not installed ({exc})"
    cls = next((c for c in sessions_class if c.name() == model), None)
    if cls is None:
        return False, f"unknown rembg model '{model}'"
    # Ask rembg where it keeps the file (its layout changed between versions).
    path = cls.resolve_existing(f"{model}.onnx")
    if path is None:
        return False, (f"model '{model}' not downloaded yet (goes to {cls.model_dir()}); "
                       "choose the 'rembg' method once to download it")
    return True, f"rembg model '{model}' ready ({path})"


def _rembg_session(model: str):
    with _session_lock:
        if model not in _sessions:
            from rembg import new_session
            _sessions[model] = new_session(model)
        return _sessions[model]


def _polygon_source(obj: ObjectRecord) -> str:
    return obj.attributes.get("polygon_source") or obj.source


class CutoutError(Exception):
    pass


def _polygon_mask(obj: ObjectRecord, size: tuple[int, int]) -> np.ndarray | None:
    poly = obj.attributes.get("polygon")
    if not poly or len(poly) < 3:
        return None
    w, h = size
    full = np.zeros((h, w), np.uint8)
    cv2.fillPoly(full, [np.asarray(poly, dtype=np.int32)], 255)
    total = int((full > 0).sum())
    x, y, bw, bh = obj.x, obj.y, obj.width, obj.height
    crop = full[y:y + bh, x:x + bw]
    inside = int((crop > 0).sum())
    # After a manual bbox edit the stored outline may no longer describe the box.
    if total == 0 or inside < 0.5 * total or inside < 0.1 * bw * bh:
        return None
    return crop


def _rembg_mask(full_rgb: np.ndarray, obj: ObjectRecord, model: str = REMBG_MODEL) -> np.ndarray | None:
    """Salient-object mask from rembg for the object's box (with context padding)."""
    try:
        from rembg import remove
    except Exception:
        return None
    h, w = full_rgb.shape[:2]
    x, y, bw, bh = obj.x, obj.y, obj.width, obj.height
    pad_x, pad_y = max(8, bw // 10), max(8, bh // 10)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    window = Image.fromarray(np.ascontiguousarray(full_rgb[y0:y1, x0:x1]))
    try:
        out = remove(window, session=_rembg_session(model), only_mask=True, post_process_mask=True)
    except Exception:
        return None
    mask = np.asarray(out.convert("L"))[y - y0:y - y0 + bh, x - x0:x - x0 + bw].copy()
    poly = _polygon_mask(obj, (w, h))
    if poly is not None:
        # Keep rembg's soft edges, but only near the known outline, so another
        # salient object inside the crop is not included.
        margin = max(3, int(0.03 * max(bw, bh)))
        allowed = cv2.dilate(poly, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
        mask = np.minimum(mask, allowed)
    if (mask > 128).mean() < 0.05:
        return None
    return mask


def _alpha_mask(rgba: np.ndarray) -> np.ndarray | None:
    alpha = rgba[:, :, 3]
    if (alpha < 250).mean() < 0.02:
        return None
    return alpha.copy()


def _key_mask(rgb: np.ndarray, tolerance: float = 25.0) -> np.ndarray | None:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    # Only meaningful when the border is (mostly) one colour.
    if np.mean(np.linalg.norm(border - bg, axis=1) < tolerance) < 0.7:
        return None
    bg_like = (np.linalg.norm(lab - bg, axis=2) < tolerance).astype(np.uint8)
    n, comps = cv2.connectedComponents(bg_like, connectivity=4)
    edge_labels = set(np.unique(np.concatenate([comps[0], comps[-1], comps[:, 0], comps[:, -1]])))
    background = np.isin(comps, [l for l in edge_labels if l != 0]) & (bg_like > 0)
    mask = (~background).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if (mask > 0).mean() < 0.05:
        return None
    return mask


def _grabcut_mask(full_rgb: np.ndarray, obj: ObjectRecord) -> np.ndarray | None:
    h, w = full_rgb.shape[:2]
    x, y, bw, bh = obj.x, obj.y, obj.width, obj.height
    pad_x, pad_y = max(10, bw // 5), max(10, bh // 5)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    window = np.ascontiguousarray(full_rgb[y0:y1, x0:x1, ::-1])  # BGR
    rect = (x - x0, y - y0, bw, bh)
    if rect[2] >= window.shape[1] - 1 and rect[3] >= window.shape[0] - 1:
        return None  # no background around the box to learn from
    mask = np.zeros(window.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(window, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    crop = fg[y - y0:y - y0 + bh, x - x0:x - x0 + bw]
    if (crop > 0).mean() < 0.05:
        return None
    return crop


def _flatten(rgba: np.ndarray) -> np.ndarray:
    """RGBA -> RGB on white. Opaque images (e.g. JPEG photos) are returned as a
    view without copying; otherwise integer arithmetic keeps memory low for
    large photos (no float64 copies of the whole image)."""
    alpha = rgba[:, :, 3]
    if alpha.min() == 255:
        return rgba[:, :, :3]
    a = alpha[:, :, None].astype(np.uint16)
    return ((rgba[:, :, :3].astype(np.uint16) * a + 255 * (255 - a)) // 255).astype(np.uint8)


def extract_cutout(image_path: Path | str, obj: ObjectRecord, method: str = "auto") -> tuple[Image.Image, str, str]:
    """Return (RGBA cutout trimmed to its visible pixels, method used, note)."""
    if method not in MASK_METHODS:
        raise CutoutError(f"Unknown mask method '{method}'.")
    rgba_full = np.asarray(load_rgba(image_path))
    h, w = rgba_full.shape[:2]
    x, y, bw, bh = obj.x, obj.y, obj.width, obj.height
    if bw < 1 or bh < 1 or x < 0 or y < 0 or x + bw > w or y + bh > h:
        raise CutoutError(f"Object #{obj.id} bounding box lies outside its image.")
    crop = rgba_full[y:y + bh, x:x + bw].copy()
    crop_rgb = crop[:, :, :3]
    # Flatten transparent pixels to white for the colour-based methods.
    flat = _flatten(crop)

    if method == "auto":
        exact = obj.attributes.get("polygon") and _polygon_source(obj).startswith(_EXACT_POLYGON_SOURCES)
        rembg_ready = rembg_status()[0]
        if exact or not rembg_ready:
            order = ["alpha", "polygon", "key", "grabcut", "none"]
        else:
            order = ["alpha", "rembg", "polygon", "key", "grabcut", "none"]
    else:
        order = [method]
    mask, used, note = None, "none", ""
    full_flat = None
    for m in order:
        if m == "alpha":
            mask = _alpha_mask(crop)
        elif m == "polygon":
            mask = _polygon_mask(obj, (w, h))
        elif m == "key":
            mask = _key_mask(flat)
        elif m in ("grabcut", "rembg"):
            if full_flat is None:
                full_flat = _flatten(rgba_full)
            mask = _grabcut_mask(full_flat, obj) if m == "grabcut" else _rembg_mask(full_flat, obj)
        else:
            mask = np.full((bh, bw), 255, np.uint8)
        if mask is not None:
            used = m
            break
        note += f"{m} not applicable; "
    if mask is None:  # an explicitly requested method failed
        mask = np.full((bh, bw), 255, np.uint8)
        used = "none"
        note += "fell back to rectangular crop."

    if used not in ("none", "rembg"):  # rembg masks already have soft edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)  # soften jagged edges slightly
    alpha = (mask.astype(np.float32) * (crop[:, :, 3].astype(np.float32) / 255.0)).astype(np.uint8)
    out = np.dstack([crop_rgb, alpha])
    ys, xs = np.nonzero(alpha > 10)
    if len(xs) == 0:
        raise CutoutError(f"Object #{obj.id}: mask '{used}' removed every pixel.")
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(out, "RGBA"), used, note.strip()
