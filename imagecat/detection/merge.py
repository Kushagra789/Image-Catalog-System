"""De-duplication of detections coming from several engines.

Every merge keeps one object and records the others in
``attributes["merged_from"]`` and ``attributes["alternative_labels"]``, so no
detector output is silently thrown away.

1. Model vs model (YOLO / YOLOE)
   * compatible labels (same, synonym, broad/specific like fruit/apple, or both
     person-like) and IoU >= SAME_OBJECT_IOU -> same object; the most specific
     label wins.
   * compatible labels and the smaller box lies inside the larger one
     (>= PART_CONTAINMENT), e.g. a head detected as "cartoon character" inside
     a full-body "doll" -> part of the same object; the larger box wins.
   * incompatible labels but nearly identical boxes (IoU >= SAME_REGION_IOU),
     e.g. "kite" and "fruit" on one clip-art apple -> one object with competing
     labels. The model suited to the image wins (YOLOE on illustrations, COCO
     YOLO on photos).
2. OpenCV vs model
   * A multi-colour foreground blob overlapping a model detection
     (IoU >= SAME_OBJECT_IOU) is the same object: the model names it, OpenCV
     contributes its outline. The result is a *whole* object.
   * A geometric shape / colour region covering the same area as a model
     detection (IoU >= SAME_REGION_IOU) is merged too. The model's label only
     wins if its confidence is >= CONFIDENT; otherwise the geometric label is
     kept (models are often wrong on drawings) and the guess is an alternative.
     Shapes are never merged into a *whole* object (an apple's round body stays
     a separate "circle" part).
3. Person-like labels on illustrations are unified to "cartoon character"; the
   model's original label is kept in ``label_mapping``.

Confidence always belongs to the detection whose label is kept; it is never
averaged or invented.
"""
from __future__ import annotations

from ..models import BBox, SOURCE_FOREGROUND, SOURCE_YOLO, SOURCE_YOLOE
from .base import Detection
from .labels import CHARACTER_LABEL, compatible, is_person_like, specificity

SAME_OBJECT_IOU = 0.5
SAME_REGION_IOU = 0.8
PART_CONTAINMENT = 0.85
CONFIDENT = 0.5


def containment(small: BBox, big: BBox) -> float:
    """Fraction of ``small`` that lies inside ``big``."""
    x1, y1 = max(small.x, big.x), max(small.y, big.y)
    x2 = min(small.x + small.width, big.x + big.width)
    y2 = min(small.y + small.height, big.y + big.height)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / small.area if small.area else 0.0


def _summary(d: Detection) -> dict:
    return {"label": d.label, "source": d.source, "confidence": d.confidence, "bbox": list(d.bbox.as_tuple())}


def _absorb(keep: Detection, other: Detection, reason: str) -> None:
    """Fold ``other`` (and everything it had already absorbed) into ``keep``."""
    history = keep.attributes.setdefault("merged_from", [])
    history.append({**_summary(other), "reason": reason})
    history.extend(other.attributes.get("merged_from", []))
    alts = keep.attributes.setdefault("alternative_labels", [])
    for cand in [{"label": other.label, "source": other.source, "confidence": other.confidence},
                 *other.attributes.get("alternative_labels", [])]:
        if cand["label"] != keep.label and all(a["label"] != cand["label"] for a in alts):
            alts.append(cand)
    for key in ("polygon", "character_evidence", "shape", "label_mapping"):
        if key not in keep.attributes and key in other.attributes:
            keep.attributes[key] = other.attributes[key]
            if key == "polygon":
                keep.attributes["polygon_source"] = other.attributes.get("polygon_source", other.source)
    if other.attributes.get("whole_object"):
        keep.attributes["whole_object"] = True


def unify_character_labels(dets: list[Detection], illustration: bool) -> None:
    if not illustration:
        return
    for d in dets:
        if is_person_like(d.label) and d.label != CHARACTER_LABEL:
            d.attributes["label_mapping"] = {
                "model_label": d.label,
                "reason": "person-like label on an illustration; gender/age is not inferred automatically",
            }
            d.label = CHARACTER_LABEL


def merge_model_detections(dets: list[Detection], illustration: bool = False) -> list[Detection]:
    preferred = SOURCE_YOLOE if illustration else SOURCE_YOLO
    # Preferred model first, then by confidence: earlier entries win label conflicts.
    ordered = sorted(dets, key=lambda d: (d.source != preferred, -(d.confidence or 0.0), -d.bbox.area))
    kept: list[Detection] = []
    for d in ordered:
        target, reason = None, ""
        for k in kept:
            iou = d.bbox.iou(k.bbox)
            if compatible(d.label, k.label):
                if iou >= SAME_OBJECT_IOU:
                    target, reason = k, "same object (overlap)"
                    break
                small, big = (d, k) if d.bbox.area <= k.bbox.area else (k, d)
                if containment(small.bbox, big.bbox) >= PART_CONTAINMENT:
                    target, reason = k, "part of a larger detection"
                    break
            elif iou >= SAME_REGION_IOU:
                target, reason = k, "same region, competing label"
                break
        if target is None:
            kept.append(d)
            continue
        if reason == "same object (overlap)":
            better = max((target, d), key=lambda x: (specificity(x.label), x.confidence or 0.0))
        elif reason == "part of a larger detection":
            better = max((target, d), key=lambda x: x.bbox.area)
        else:
            better = target  # already ranked first by domain preference and confidence
        worse = d if better is target else target
        _absorb(better, worse, reason)
        if better is d:
            kept[kept.index(target)] = d
    return kept


def merge_with_opencv(model_dets: list[Detection], cv_dets: list[Detection]) -> list[Detection]:
    result = list(model_dets)
    model_ids = {id(d) for d in model_dets}
    # Foreground blobs first, so model detections become "whole objects" before
    # shapes are considered.
    for cv in sorted(cv_dets, key=lambda d: d.source != SOURCE_FOREGROUND):
        models = [m for m in result if id(m) in model_ids]
        if cv.source == SOURCE_FOREGROUND:
            match = max(models, key=lambda m: m.bbox.iou(cv.bbox), default=None)
            if match is not None and match.bbox.iou(cv.bbox) >= SAME_OBJECT_IOU:
                _absorb(match, cv, "same object as an OpenCV foreground blob")
                match.attributes["whole_object"] = True
                if containment(match.bbox, cv.bbox) >= 0.9 and cv.bbox.area > match.bbox.area:
                    # The blob is the measured pixel extent of the whole object
                    # (e.g. apple body + stem + leaf); keep box and outline consistent.
                    match.attributes["model_bbox"] = list(match.bbox.as_tuple())
                    match.bbox = cv.bbox
                    match.attributes["polygon"] = cv.attributes.get("polygon", match.attributes.get("polygon"))
                    match.attributes["polygon_source"] = cv.source
                continue
        else:
            parts_of = [m for m in models if m.attributes.get("whole_object")]
            candidates = [m for m in models if not m.attributes.get("whole_object")]
            match = max(candidates, key=lambda m: m.bbox.iou(cv.bbox), default=None)
            if match is not None and match.bbox.iou(cv.bbox) >= SAME_REGION_IOU:
                if (match.confidence or 0.0) >= CONFIDENT:
                    _absorb(match, cv, "same region as an OpenCV shape/region")
                else:
                    # Low-confidence model label on the same region: the geometric
                    # description is more trustworthy; keep the guess as an alternative.
                    idx = result.index(match)
                    _absorb(cv, match, "low-confidence model label for the same region")
                    result[idx] = cv
                    model_ids.discard(id(match))
                continue
            whole = max(parts_of, key=lambda m: containment(cv.bbox, m.bbox), default=None)
            if whole is not None and containment(cv.bbox, whole.bbox) >= PART_CONTAINMENT:
                cv.attributes["part_of"] = {"label": whole.label, "bbox": list(whole.bbox.as_tuple())}
        overlapping = max((m for m in result if id(m) in model_ids), key=lambda m: m.bbox.iou(cv.bbox),
                          default=None)
        if overlapping is not None and overlapping.bbox.iou(cv.bbox) > 0:
            cv.attributes["overlaps_model"] = {"label": overlapping.label,
                                               "iou": round(overlapping.bbox.iou(cv.bbox), 3)}
        result.append(cv)
    return result
