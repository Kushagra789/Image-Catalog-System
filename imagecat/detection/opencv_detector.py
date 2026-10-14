"""Classical (non-learned) analysis with OpenCV: shapes, colour regions, foreground.

How it works
------------
1. The image is classified as ``blank``, ``flat`` (few distinct colours, e.g.
   cartoons, clip-art, diagrams) or ``photo``.
2. For flat images a small colour palette is extracted, each pixel is assigned
   to its nearest palette colour (in CIELAB) and connected regions of each colour
   are analysed:
     * clean geometric outlines  -> ``opencv_shape`` (circle, square, triangle...)
     * irregular outlines        -> ``opencv_color_region`` ("red region")
   A shape partly hidden by another colour is recognised by fitting complete
   shape hypotheses (ellipse / rectangle / triangle) to the visible outline,
   and flagged ``occluded``.
3. Connected foreground blobs made of several colours (e.g. a cartoon character,
   an apple with a leaf) are reported as ``opencv_foreground`` objects with a
   polygon mask, so they can be relabelled and used in composites.

Limitations: no semantic understanding (it never says "apple" or "boy"),
photographs are skipped because gradients/texture produce meaningless regions,
and shading or outlines in cartoons fragment regions. Confidence is always None
because these heuristics have no calibrated probability.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..colors import name_color_bgr
from ..models import BBox, SOURCE_COLOR_REGION, SOURCE_FOREGROUND, SOURCE_SHAPE
from .base import Detection, polygon_from_contour

KIND_BLANK = "blank"
KIND_FLAT = "flat"
KIND_PHOTO = "photo"

_WORK_MAX_SIDE = 1024
# Thresholds chosen from measurements (top-16 colour coverage on the foreground):
# real photos 0.35-0.60 (incl. a product photo on white), real flat cartoons
# 0.745-0.90, synthetic fixtures >= 0.94.
FLAT_COVERAGE = 0.80
FLAT_FG_COVERAGE = 0.68


def classify_image(bgr: np.ndarray) -> tuple[str, dict]:
    """Return (kind, metrics) where kind is blank / flat / photo."""
    small = cv2.resize(bgr, (min(256, bgr.shape[1]), min(256, bgr.shape[0])), interpolation=cv2.INTER_AREA)
    gray_std = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std())
    channel_range = float((small.reshape(-1, 3).max(axis=0).astype(int) - small.reshape(-1, 3).min(axis=0)).max())
    if gray_std < 3.0 and channel_range < 25:
        return KIND_BLANK, {"gray_std": round(gray_std, 2), "channel_range": channel_range}

    q = (small >> 4).reshape(-1, 3).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    counts = np.sort(np.bincount(keys))[::-1]
    coverage = float(counts[:16].sum() / counts.sum())

    # The same measure on the foreground only (pixels that differ from the
    # median border colour). Without this, a product photo on a plain white
    # background looks "flat" because the background dominates the count.
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    fg = (np.linalg.norm(lab - np.median(border, axis=0), axis=2) > 20).reshape(-1)
    fg_keys = keys[fg]
    if len(fg_keys):
        fg_counts = np.sort(np.bincount(fg_keys))[::-1]
        fg_coverage = float(fg_counts[:16].sum() / len(fg_keys))
    else:
        fg_coverage = 1.0
    kind = KIND_FLAT if coverage >= FLAT_COVERAGE and fg_coverage >= FLAT_FG_COVERAGE else KIND_PHOTO
    return kind, {"gray_std": round(gray_std, 2), "top16_color_coverage": round(coverage, 3),
                  "top16_foreground_coverage": round(fg_coverage, 3)}


class OpenCVDetector:
    name = "opencv"

    def __init__(self, min_area_frac: float = 0.003, foreground_min_area_frac: float = 0.01,
                 max_objects: int = 25):
        self.min_area_frac = min_area_frac
        self.foreground_min_area_frac = foreground_min_area_frac
        self.max_objects = max_objects

    # ------------------------------------------------------------------ public
    def detect(self, bgr: np.ndarray) -> tuple[list[Detection], str, str]:
        """Returns (detections, image_kind, message)."""
        kind, metrics = classify_image(bgr)
        if kind == KIND_BLANK:
            return [], kind, f"Image is blank/uniform (gray std {metrics['gray_std']}); nothing to analyse."
        if kind == KIND_PHOTO:
            return [], kind, (
                "Image looks photographic (top-16 colour coverage "
                f"{metrics['top16_color_coverage']:.0%} overall, "
                f"{metrics['top16_foreground_coverage']:.0%} on the foreground; flat images need "
                f">= {FLAT_COVERAGE:.0%} and >= {FLAT_FG_COVERAGE:.0%}); shape/colour-region analysis skipped "
                "because gradients and texture would produce meaningless regions."
            )

        scale = 1.0
        work = bgr
        if max(bgr.shape[:2]) > _WORK_MAX_SIDE:
            scale = _WORK_MAX_SIDE / max(bgr.shape[:2])
            work = cv2.resize(bgr, (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
        detections = self._analyse_flat(work)
        if scale != 1.0:
            detections = [self._rescale(d, 1.0 / scale, bgr.shape) for d in detections]
        detections.sort(key=lambda d: d.bbox.area, reverse=True)
        detections = detections[: self.max_objects]
        return detections, kind, f"Flat/illustration image: {len(detections)} shape/region/foreground object(s)."

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _rescale(det: Detection, factor: float, shape) -> Detection:
        h_img, w_img = shape[:2]
        b = det.bbox
        x, y = int(round(b.x * factor)), int(round(b.y * factor))
        w = max(1, min(w_img - x, int(round(b.width * factor))))
        h = max(1, min(h_img - y, int(round(b.height * factor))))
        det.bbox = BBox(x, y, w, h)
        if "polygon" in det.attributes:
            det.attributes["polygon"] = [[int(round(px * factor)), int(round(py * factor))]
                                         for px, py in det.attributes["polygon"]]
        return det

    @staticmethod
    def _palette(lab: np.ndarray, bgr: np.ndarray) -> list[dict]:
        """Extract up to 12 dominant colours, merging near-identical shades."""
        flat_bgr = bgr.reshape(-1, 3)
        flat_lab = lab.reshape(-1, 3).astype(np.float32)
        q = (flat_bgr >> 4).astype(np.int32)
        keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        counts = np.bincount(keys, minlength=4096)
        min_count = max(20, int(0.002 * len(keys)))
        candidates = [k for k in np.argsort(-counts) if counts[k] >= min_count][:40]

        palette: list[dict] = []
        for k in candidates:
            sel = keys == k
            mean_lab = flat_lab[sel].mean(axis=0)
            mean_bgr = flat_bgr[sel].mean(axis=0)
            for entry in palette:
                if np.linalg.norm(entry["lab"] - mean_lab) < 15:
                    total = entry["count"] + counts[k]
                    entry["lab"] = (entry["lab"] * entry["count"] + mean_lab * counts[k]) / total
                    entry["bgr"] = (entry["bgr"] * entry["count"] + mean_bgr * counts[k]) / total
                    entry["count"] = total
                    break
            else:
                palette.append({"lab": mean_lab, "bgr": mean_bgr, "count": int(counts[k])})
            if len(palette) >= 12:
                break
        return palette

    @staticmethod
    def _background_index(bgr: np.ndarray, lab: np.ndarray, palette: list[dict]) -> int | None:
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]).astype(np.float32)
        dists = np.stack([np.linalg.norm(border - p["lab"], axis=1) for p in palette], axis=1)
        nearest = dists.argmin(axis=1)
        near_enough = dists.min(axis=1) < 20
        counts = np.bincount(nearest[near_enough], minlength=len(palette))
        idx = int(counts.argmax())
        return idx if counts[idx] >= 0.5 * len(border) else None

    @staticmethod
    def _shape_hypotheses(points: np.ndarray) -> list[tuple[str, object]]:
        """Candidate complete shapes fitted to boundary points."""
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out: list[tuple[str, object]] = []
        if len(pts) >= 5:
            out.append(("ellipse", cv2.fitEllipse(pts)))
        if len(pts) >= 3:
            out.append(("rect", cv2.minAreaRect(pts)))
            _, tri = cv2.minEnclosingTriangle(pts)
            if tri is not None:
                out.append(("triangle", tri.reshape(-1, 2)))
        return out

    @staticmethod
    def _hypothesis_mask(kind: str, params, shape) -> np.ndarray:
        mask = np.zeros(shape, np.uint8)
        if kind == "ellipse":
            (cx, cy), (ma, mi), ang = params
            if ma > 0 and mi > 0:
                cv2.ellipse(mask, ((cx, cy), (ma, mi), ang), 255, -1)
        elif kind == "rect":
            cv2.fillPoly(mask, [np.round(cv2.boxPoints(params)).astype(np.int32)], 255)
        else:
            cv2.fillPoly(mask, [np.round(params).astype(np.int32)], 255)
        return mask > 0

    def _classify_shape(self, contour: np.ndarray, region: np.ndarray, bg_mask: np.ndarray,
                        has_background: bool) -> dict:
        """Name the geometric shape of a single-colour region, or shape=None if irregular.

        Each hypothesis (ellipse / rectangle / triangle) is fitted only to the
        part of the outline that borders the background, so a shape partly
        covered by another colour can still be recognised (flagged ``occluded``).
        A hypothesis is accepted when it contains the region and everything
        inside it is either the region or another (occluding) foreground colour.
        """
        area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        hull_peri = cv2.arcLength(hull, True) or 1.0
        info: dict = {
            "solidity": round(solidity, 3),
            "circularity": round(4 * math.pi * hull_area / hull_peri ** 2, 3),
            "occluded": False,
        }

        hull_mask = np.zeros(region.shape, np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        gap = (hull_mask > 0) & ~region
        gap_px = int(gap.sum())
        gap_bg_frac = float((gap & bg_mask).sum() / gap_px) if gap_px else 0.0

        # A star has deep, background-filled concavities and ~10 vertices.
        approx_raw = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if 0.35 < solidity < 0.75 and 8 <= len(approx_raw) <= 12 and gap_bg_frac > 0.8:
            info.update(shape="star", vertices=len(approx_raw))
            return info
        if solidity < 0.9 and gap_bg_frac > 0.5:
            info.update(shape=None, vertices=len(approx_raw))  # irregular outline
            return info

        # Pass 1 fits the whole outline. Pass 2 (only if needed) fits just the
        # outline points that touch the background/image border, which lets a
        # shape whose silhouette is cut by an occluder be completed.
        h, w = region.shape
        pts = contour.reshape(-1, 2)
        point_sets = [pts]
        if has_background:
            near_bg = cv2.dilate(bg_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            on_border = (pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] >= h - 2)
            visible = pts[near_bg[pts[:, 1], pts[:, 0]] | on_border]
            if 5 <= len(visible) < len(pts) and len(visible) >= 0.3 * len(pts):
                point_sets.append(visible)

        region_px = float(region.sum()) or 1.0
        best = None
        for fit_pts in point_sets:
            fit_pts = cv2.convexHull(fit_pts.reshape(-1, 1, 2).astype(np.int32)).reshape(-1, 2)
            for kind, params in self._shape_hypotheses(fit_pts):
                cand = self._hypothesis_mask(kind, params, region.shape)
                cand_px = float(cand.sum())
                if cand_px == 0:
                    continue
                inside = float((cand & region).sum())
                containment = inside / region_px
                own = inside / cand_px
                covered = float((cand & ~bg_mask).sum()) / cand_px if has_background else own
                # Occluders may fill at most 40% of the completed shape.
                score = min(containment, covered) if own >= 0.6 else 0.0
                if best is None or score > best[0] + 1e-6:
                    occ = float((cand & ~region & ~bg_mask).sum()) / cand_px if has_background else 0.0
                    best = (score, kind, params, occ)
            if best is not None and best[0] >= 0.95:
                break

        if best is not None and best[0] >= 0.93:
            score, kind, params, occ = best
            info["fit_score"] = round(score, 3)
            info["occluded_fraction"] = round(occ, 3)
            info["occluded"] = bool(occ > 0.015)
            if kind == "ellipse":
                (_, _), (ma, mi), _ = params
                ratio = min(ma, mi) / max(ma, mi)
                info.update(shape="circle" if ratio > 0.9 else "ellipse", vertices=0)
            elif kind == "rect":
                (_, _), (rw, rh), _ = params
                aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) else 0
                info.update(shape="square" if aspect > 0.9 else "rectangle", vertices=4)
            else:
                info.update(shape="triangle", vertices=3)
            return info

        # Unoccluded regular polygons with more sides.
        approx = cv2.approxPolyDP(hull, 0.03 * hull_peri, True)
        n = len(approx)
        info["vertices"] = n
        if solidity >= 0.95 and n == 5:
            info["shape"] = "pentagon"
        elif solidity >= 0.95 and n == 6:
            info["shape"] = "hexagon"
        else:
            info["shape"] = None  # irregular -> colour region
        return info

    def _analyse_flat(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        img_area = h * w
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        palette = self._palette(lab, bgr)
        if not palette:
            return []
        bg_idx = self._background_index(bgr, lab, palette)

        flat_lab = lab.reshape(-1, 3).astype(np.float32)
        dists = np.stack([np.linalg.norm(flat_lab - p["lab"], axis=1) for p in palette], axis=1)
        labels = dists.argmin(axis=1)
        labels[dists.min(axis=1) > 25] = -1  # anti-aliasing / outline pixels
        labels = labels.reshape(h, w)
        bg_mask = labels == bg_idx if bg_idx is not None else np.zeros((h, w), bool)

        kernel = np.ones((3, 3), np.uint8)
        min_area = self.min_area_frac * img_area
        detections: list[Detection] = []

        for idx, entry in enumerate(palette):
            if idx == bg_idx:
                continue
            mask = (labels == idx).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            color_name = name_color_bgr(entry["bgr"].round().astype(np.uint8))
            b, g, r = (int(round(c)) for c in entry["bgr"])
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                # A huge region touching all borders is a background, not an object.
                if bg_idx is None and area > 0.4 * img_area and x == 0 and y == 0 \
                        and x + bw == w and y + bh == h:
                    continue
                single = np.zeros((h, w), np.uint8)
                cv2.drawContours(single, [contour], -1, 255, -1)
                region = (single > 0) & (mask > 0)
                shape = self._classify_shape(contour, region, bg_mask, bg_idx is not None)
                attrs = {
                    "color_hex": color_hex,
                    "area_px": int(area),
                    "polygon": polygon_from_contour(contour),
                    **shape,
                }
                if shape.get("shape"):
                    detections.append(Detection(
                        label=shape["shape"], bbox=BBox(x, y, bw, bh), source=SOURCE_SHAPE,
                        confidence=None, color=color_name, attributes=attrs))
                else:
                    detections.append(Detection(
                        label=f"{color_name} region", bbox=BBox(x, y, bw, bh), source=SOURCE_COLOR_REGION,
                        confidence=None, color=color_name, attributes=attrs))

        detections.extend(self._foreground_objects(bgr, labels, bg_idx, palette))
        return detections

    def _foreground_objects(self, bgr, labels, bg_idx, palette) -> list[Detection]:
        """Multi-colour connected blobs (characters, fruit with leaves, ...)."""
        if bg_idx is None:
            return []
        h, w = labels.shape
        # Unassigned (anti-aliasing) pixels are excluded so that faint fringes
        # cannot chain unrelated objects together along the image border.
        fg = ((labels >= 0) & (labels != bg_idx)).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.foreground_min_area_frac * h * w:
                continue
            comp = np.zeros((h, w), np.uint8)
            cv2.drawContours(comp, [contour], -1, 255, -1)
            comp_labels = labels[(comp > 0) & (labels >= 0) & (labels != bg_idx)]
            if comp_labels.size == 0:
                continue
            counts = np.bincount(comp_labels, minlength=len(palette))
            significant = [i for i, c in enumerate(counts) if c >= 0.05 * comp_labels.size]
            if len(significant) < 2:
                continue  # single colour: already reported as a shape/region
            main = int(counts.argmax())
            color_name = name_color_bgr(palette[main]["bgr"].round().astype(np.uint8))
            x, y, bw, bh = cv2.boundingRect(contour)
            attrs = {
                "area_px": int(area),
                "polygon": polygon_from_contour(contour),
                "colors": sorted({name_color_bgr(palette[i]["bgr"].round().astype(np.uint8))
                                  for i in significant}),
            }
            face = face_evidence(bgr[y:y + bh, x:x + bw], comp[y:y + bh, x:x + bw] > 0)
            if face:
                label = "cartoon character"
                attrs["character_evidence"] = face
                attrs["note"] = ("Heuristic: face-like pair of eyes found. Boy/girl is not inferred; set it in the "
                                 "Manual Editor if you know it.")
            else:
                label = "foreground object"
                attrs["note"] = "Multi-colour blob; relabel it (e.g. 'cartoon boy') in the Manual Editor."
            out.append(Detection(label=label, bbox=BBox(x, y, bw, bh), source=SOURCE_FOREGROUND,
                                 confidence=None, color=color_name, attributes=attrs))
        return out


def face_evidence(bgr: np.ndarray, mask: np.ndarray) -> dict | None:
    """Look for two eye-like blobs: small, compact, dark, side by side, in the
    upper half of a figure. Deliberately colour-agnostic about skin.

    Returns evidence (eye centres in crop coordinates) or None.
    """
    h, w = mask.shape
    fig_area = float(mask.sum())
    if fig_area == 0 or h < 20 or w < 20:
        return None
    value = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = ((value < 90) & mask).astype(np.uint8)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        # Filled outer contour: a drawn eye is often an outline ring around the
        # pupil, which only looks compact once its inside is filled.
        a = float(cv2.contourArea(c))
        bx, by, bw_, bh_ = cv2.boundingRect(c)
        if not (0.0003 * fig_area <= a <= 0.06 * fig_area):
            continue
        cx, cy = bx + bw_ / 2, by + bh_ / 2
        if cy > 0.5 * h or not (0.4 <= bw_ / max(1, bh_) <= 2.5) or a / max(1, bw_ * bh_) < 0.5:
            continue  # eyes are compact; outlines and hair strands are not
        blobs.append((float(cx), float(cy), a))
    best = None
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            (x1, y1, a1), (x2, y2, a2) = blobs[i], blobs[j]
            dx, dy = abs(x1 - x2), abs(y1 - y2)
            if dy > 0.06 * h or not (0.06 * w <= dx <= 0.6 * w) or max(a1, a2) > 3 * min(a1, a2):
                continue
            score = dy / h + abs(a1 - a2) / max(a1, a2)
            if best is None or score < best[0]:
                best = (score, [[round(x1), round(y1)], [round(x2), round(y2)]])
    if best is None:
        return None
    return {"method": "two compact dark blobs side by side in the upper half of a multi-colour figure",
            "eyes": best[1]}
