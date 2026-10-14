"""High-level application service: upload + detect, rerun policy, manual corrections.

Correction policy
-----------------
* Manually added objects, edited detections, and deleted detections (kept as
  soft-deleted "tombstones") are *corrections*. Composite layer objects are also
  protected because they are known exactly from the composition.
* Detection only runs automatically on upload. Re-running is always explicit:
    - ``keep_manual``  replaces only untouched automatic objects; new detections
      that overlap a protected object (including a deleted one) are skipped, so
      corrections stay authoritative. No confirmation needed.
    - ``replace_all``  discards everything, corrections included. Raises
      ``ConfirmationRequired`` unless ``confirm_replace=True``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .catalog import ImageCatalog
from .config import Settings
from .db import Database, utcnow
from .detection import DetectionEngine, DetectionResult, IllustrationDetector, OpenCVDetector, YoloDetector
from .models import (
    BBox,
    ImageRecord,
    ObjectRecord,
    ORIGIN_MANUAL,
    SOURCE_COMPOSITE,
    SOURCE_MANUAL,
    STATUS_FAILED,
)

RERUN_KEEP_MANUAL = "keep_manual"
RERUN_REPLACE_ALL = "replace_all"
OVERLAP_SKIP_IOU = 0.5

_UNSET: Any = object()


class ConfirmationRequired(Exception):
    def __init__(self, n_corrections: int):
        self.n_corrections = n_corrections
        super().__init__(
            f"This image has {n_corrections} manual correction(s). Replacing them requires "
            "explicit confirmation."
        )


class ValidationError(ValueError):
    pass


@dataclass
class UploadOutcome:
    record: ImageRecord
    detection: DetectionResult | None


def normalize_label(label: str) -> str:
    label = re.sub(r"\s+", " ", (label or "").strip().lower())
    if not label:
        raise ValidationError("Label cannot be empty.")
    if len(label) > 100:
        raise ValidationError("Label is too long (max 100 characters).")
    return label


def normalize_color(color: str | None) -> str | None:
    if color is None:
        return None
    color = re.sub(r"\s+", " ", color.strip().lower())
    return color or None


def validate_confidence(confidence: float | None) -> float | None:
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Confidence must be a number between 0 and 1, or empty.") from exc
    if not 0.0 <= value <= 1.0:
        raise ValidationError("Confidence must be between 0 and 1.")
    return value


def validate_bbox(bbox, width: int, height: int) -> BBox:
    try:
        x, y, w, h = (int(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Bounding box must be four integers (x, y, width, height).") from exc
    if w < 1 or h < 1:
        raise ValidationError("Bounding box width and height must be at least 1 pixel.")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValidationError(
            f"Bounding box ({x}, {y}, {w}, {h}) falls outside the image ({width}x{height})."
        )
    return BBox(x, y, w, h)


def is_protected(obj: ObjectRecord) -> bool:
    return obj.is_correction or obj.source == SOURCE_COMPOSITE


class CatalogService:
    def __init__(self, settings: Settings | None = None, yolo: YoloDetector | None = _UNSET,
                 illustration: IllustrationDetector | None = _UNSET):
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.db = Database(self.settings.db_path)
        self.catalog = ImageCatalog(self.db, self.settings.images_dir)
        if yolo is _UNSET:
            yolo = (YoloDetector(self.settings.yolo_model, self.settings.models_dir,
                                 self.settings.yolo_confidence)
                    if self.settings.enable_yolo else None)
        if illustration is _UNSET:
            illustration = (IllustrationDetector(self.settings.illustration_model, self.settings.models_dir,
                                                 self.settings.illustration_confidence)
                            if self.settings.enable_illustration_model else None)
        self.engine = DetectionEngine(yolo, OpenCVDetector(), illustration)

    # ------------------------------------------------------------- uploading
    def upload(self, data: bytes, filename: str, display_name: str | None = None,
               auto_detect: bool = True, allow_duplicates: bool = False,
               **detect_kwargs) -> UploadOutcome:
        record = self.catalog.add_image_bytes(data, filename, display_name, allow_duplicates)
        result = None
        if auto_detect:
            result = self.run_detection(record.id, mode="initial", **detect_kwargs)
            record = self.db.get_image(record.id)  # type: ignore[assignment]
        return UploadOutcome(record, result)

    # ------------------------------------------------------------- detection
    def corrections(self, image_id: int) -> list[ObjectRecord]:
        return [o for o in self.db.list_objects(image_id, include_deleted=True) if o.is_correction]

    def run_detection(self, image_id: int, mode: str = RERUN_KEEP_MANUAL, confirm_replace: bool = False,
                      use_yolo: bool = True, use_opencv: bool = True,
                      yolo_confidence: float | None = None,
                      use_illustration_model: bool = True) -> DetectionResult:
        record = self.db.get_image(image_id)
        if record is None:
            raise ValidationError(f"Image #{image_id} does not exist.")
        existing = self.db.list_objects(image_id, include_deleted=True)
        protected = [o for o in existing if is_protected(o)]

        if mode == RERUN_REPLACE_ALL and protected and not confirm_replace:
            raise ConfirmationRequired(len(protected))
        if mode not in ("initial", RERUN_KEEP_MANUAL, RERUN_REPLACE_ALL):
            raise ValidationError(f"Unknown detection mode '{mode}'.")

        result = self.engine.detect_path(self.catalog.path_for(record), use_yolo=use_yolo,
                                         use_opencv=use_opencv, yolo_confidence=yolo_confidence,
                                         use_illustration_model=use_illustration_model)
        details = {
            "engines": [r.to_dict() for r in result.reports],
            "image_kind": result.image_kind,
            "yolo_confidence_threshold": yolo_confidence,
        }
        message = result.message

        with self.db.connect() as conn:
            # Take the write lock, then re-read the objects: detection takes
            # seconds, and a correction made meanwhile (e.g. in another browser
            # tab) must be protected too, never deleted from a stale list.
            conn.execute("BEGIN IMMEDIATE")
            existing = self.db.list_objects(image_id, include_deleted=True, conn=conn)
            protected = [o for o in existing if is_protected(o)]
            if mode == RERUN_REPLACE_ALL and protected and not confirm_replace:
                raise ConfirmationRequired(len(protected))
            if result.status == STATUS_FAILED:
                # Never wipe existing objects because a detector broke.
                if existing:
                    message += " Existing objects were left unchanged."
            else:
                if mode == RERUN_REPLACE_ALL:
                    to_remove, keep = existing, []
                else:
                    to_remove = [o for o in existing if not is_protected(o)]
                    keep = protected
                self.db.hard_delete_objects([o.id for o in to_remove], conn)
                skipped = 0
                blocked = [k.bbox for k in keep] + [
                    BBox(*k.attributes["original"]["bbox"]) for k in keep
                    if isinstance(k.attributes.get("original"), dict) and k.attributes["original"].get("bbox")]
                for det in result.detections:
                    if any(det.bbox.iou(b) >= OVERLAP_SKIP_IOU for b in blocked):
                        skipped += 1
                        continue
                    self.db.insert_object(
                        image_id=image_id, label=det.label, color=det.color,
                        confidence=det.confidence, bbox=det.bbox.as_tuple(), source=det.source,
                        attributes=det.attributes, conn=conn,
                    )
                if keep:
                    message += f" {len(keep)} protected correction(s) preserved"
                    message += f"; {skipped} overlapping new detection(s) skipped." if skipped else "."
                details["skipped_due_to_corrections"] = skipped
                details["removed_auto_objects"] = len(to_remove)
            n_active = conn.execute(
                "SELECT COUNT(*) FROM objects WHERE image_id = ? AND is_deleted = 0", (image_id,)
            ).fetchone()[0]
            self.db.log_detection_run(image_id, mode, result.status, message, n_active, details, conn=conn)
            self.db.update_image(
                image_id,
                conn=conn,
                detection_status=result.status,
                detection_message=message,
                detected_at=utcnow(),
                dominant_colors=result.dominant_colors,
            )
        result.message = message
        return result

    # ------------------------------------------------------ manual corrections
    def _image_or_raise(self, image_id: int) -> ImageRecord:
        record = self.db.get_image(image_id)
        if record is None:
            raise ValidationError(f"Image #{image_id} does not exist.")
        return record

    def _object_or_raise(self, object_id: int) -> ObjectRecord:
        obj = self.db.get_object(object_id)
        if obj is None:
            raise ValidationError(f"Object #{object_id} does not exist.")
        return obj

    def add_object(self, image_id: int, label: str, bbox, color: str | None = None,
                   confidence: float | None = None, notes: str | None = None) -> ObjectRecord:
        record = self._image_or_raise(image_id)
        box = validate_bbox(bbox, record.width, record.height)
        attrs = {"notes": notes} if notes else {}
        object_id = self.db.insert_object(
            image_id=image_id, label=normalize_label(label), color=normalize_color(color),
            confidence=validate_confidence(confidence), bbox=box.as_tuple(),
            source=SOURCE_MANUAL, origin=ORIGIN_MANUAL, attributes=attrs,
        )
        return self.db.get_object(object_id)  # type: ignore[return-value]

    def update_object(self, object_id: int, label: str | None = None, color: str | None = _UNSET,
                      confidence: float | None = _UNSET, bbox=None) -> ObjectRecord:
        """Edit an object. Unspecified fields are unchanged.

        If an automatic detection is relabelled and no confidence is given, the
        old confidence is cleared: it described the *old* label, so keeping it
        would misrepresent the model's certainty.
        """
        obj = self._object_or_raise(object_id)
        record = self._image_or_raise(obj.image_id)
        changes: dict[str, Any] = {}
        if label is not None:
            new_label = normalize_label(label)
            if new_label != obj.label:
                changes["label"] = new_label
        if color is not _UNSET:
            new_color = normalize_color(color)
            if new_color != obj.color:
                changes["color"] = new_color
        if confidence is not _UNSET:
            new_conf = validate_confidence(confidence)
            if new_conf != obj.confidence:
                changes["confidence"] = new_conf
        elif "label" in changes and obj.origin != ORIGIN_MANUAL and obj.confidence is not None:
            changes["confidence"] = None
        if bbox is not None:
            box = validate_bbox(bbox, record.width, record.height)
            if box.as_tuple() != obj.bbox.as_tuple():
                changes.update(x=box.x, y=box.y, width=box.width, height=box.height)
        if not changes:
            return obj

        attrs = dict(obj.attributes)
        if obj.origin != ORIGIN_MANUAL and "original" not in attrs:
            attrs["original"] = {
                "label": obj.label, "color": obj.color, "confidence": obj.confidence,
                "bbox": list(obj.bbox.as_tuple()), "source": obj.source,
            }
        if "confidence" in changes and confidence is _UNSET:
            attrs["confidence_note"] = "Model confidence cleared because the label was changed manually."
        attrs.setdefault("edit_history", []).append(
            {"at": utcnow(), "changes": {k: v for k, v in changes.items()}})
        self.db.update_object(object_id, **changes, is_edited=True, attributes=attrs)
        return self.db.get_object(object_id)  # type: ignore[return-value]

    def delete_object(self, object_id: int) -> None:
        """Manual objects are removed; automatic ones are soft-deleted so that
        re-running detection in ``keep_manual`` mode does not bring them back."""
        obj = self._object_or_raise(object_id)
        if obj.origin == ORIGIN_MANUAL:
            with self.db.connect() as conn:
                self.db.hard_delete_objects([object_id], conn)
        else:
            self.db.update_object(object_id, is_deleted=True)

    def restore_object(self, object_id: int) -> ObjectRecord:
        obj = self._object_or_raise(object_id)
        self.db.update_object(object_id, is_deleted=False)
        return self.db.get_object(obj.id)  # type: ignore[return-value]

    # ------------------------------------------------------------- reporting
    def health_report(self) -> dict[str, Any]:
        """Read-only consistency check of the database and the image folder.

        Nothing is repaired automatically: problems are reported so the user can
        decide what to do (data is never deleted behind their back).
        """
        report: dict[str, Any] = {"problems": [], "warnings": []}
        with self.db.connect() as conn:
            report["integrity_check"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
            fk = conn.execute("PRAGMA foreign_key_check").fetchall()
            report["foreign_key_violations"] = len(fk)
            bad_boxes = conn.execute(
                """SELECT o.id, o.image_id FROM objects o JOIN images i ON i.id = o.image_id
                   WHERE o.x < 0 OR o.y < 0 OR o.width < 1 OR o.height < 1
                      OR o.x + o.width > i.width OR o.y + o.height > i.height""").fetchall()
            bad_conf = conn.execute(
                "SELECT COUNT(*) FROM objects WHERE confidence IS NOT NULL AND (confidence < 0 OR confidence > 1)"
            ).fetchone()[0]
        if report["integrity_check"] != "ok":
            report["problems"].append(f"SQLite integrity check: {report['integrity_check']}")
        if fk:
            report["problems"].append(f"{len(fk)} foreign-key violation(s)")
        if bad_boxes:
            report["problems"].append(
                f"{len(bad_boxes)} object(s) with a box outside their image: "
                + ", ".join(f"#{o} (image #{i})" for o, i in bad_boxes[:10]))
        if bad_conf:
            report["problems"].append(f"{bad_conf} object(s) with confidence outside 0..1")

        records = self.db.list_images()
        known = {r.filename for r in records}
        missing = [r for r in records if not self.catalog.path_for(r).exists()]
        on_disk = {p.name for p in self.catalog.images_dir.iterdir() if p.is_file()}
        orphans = sorted(on_disk - known)
        for r in missing:
            report["problems"].append(f"Image #{r.id} ('{r.display_name}'): file {r.filename} is missing")
        if orphans:
            report["warnings"].append(f"{len(orphans)} file(s) in the images folder have no catalog record: "
                                      + ", ".join(orphans[:10]))
        ids = {r.id for r in records}
        for r in records:
            if r.is_composite:
                gone = [i for i in (r.source_asset_ids or {}).get("image_ids", []) if i not in ids]
                if gone:
                    report["warnings"].append(
                        f"Composite #{r.id} lists deleted source image(s) {gone}; its stored provenance "
                        "snapshot still describes them.")
        report["images"] = len(records)
        report["missing_files"] = len(missing)
        report["orphan_files"] = len(orphans)
        report["ok"] = not report["problems"]
        return report

    def image_metadata(self, image_id: int) -> dict[str, Any]:
        record = self._image_or_raise(image_id)
        return {
            "image": record.to_dict(),
            "file_path": str(self.catalog.path_for(record)),
            "objects": [o.to_dict() for o in self.db.list_objects(image_id, include_deleted=True)],
            "detection_runs": self.db.list_detection_runs(image_id),
        }
