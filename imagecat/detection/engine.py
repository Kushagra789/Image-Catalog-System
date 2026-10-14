"""Runs all detectors on an image and merges their results."""
from __future__ import annotations

from pathlib import Path

from ..colors import dominant_palette
from ..imaging import load_rgb, pil_to_bgr
from ..models import STATUS_EMPTY, STATUS_FAILED, STATUS_SUCCESS
from .base import Detection, DetectionResult, EngineReport
from .labels import CHARACTER_LABEL
from .merge import merge_model_detections, merge_with_opencv, unify_character_labels
from .opencv_detector import KIND_BLANK, KIND_FLAT, OpenCVDetector, classify_image
from .yolo_detector import YoloDetector
from .yoloe_detector import IllustrationDetector


class DetectionEngine:
    def __init__(self, yolo: YoloDetector | None, opencv: OpenCVDetector | None = None,
                 illustration: IllustrationDetector | None = None):
        self.yolo = yolo
        self.opencv = opencv or OpenCVDetector()
        self.illustration = illustration

    def engine_status(self) -> dict[str, str]:
        status = {"opencv": "available"}
        for name, det in (("yolo", self.yolo), ("yoloe", self.illustration)):
            if det is None:
                status[name] = "disabled"
            else:
                status[name] = "available" if det.available else f"unavailable: {det.reason}"
        return status

    def detect_path(self, path: Path | str, use_yolo: bool = True, use_opencv: bool = True,
                    yolo_confidence: float | None = None, use_illustration_model: bool = True) -> DetectionResult:
        try:
            bgr = pil_to_bgr(load_rgb(path))
        except Exception as exc:
            return DetectionResult(
                status=STATUS_FAILED,
                message=f"Detection failed: image could not be read ({exc}).",
                detections=[],
                reports=[EngineReport("loader", ran=True, ok=False, message=str(exc))],
            )
        return self.detect(bgr, use_yolo=use_yolo, use_opencv=use_opencv, yolo_confidence=yolo_confidence,
                           use_illustration_model=use_illustration_model)

    def detect(self, bgr, use_yolo: bool = True, use_opencv: bool = True,
               yolo_confidence: float | None = None, use_illustration_model: bool = True) -> DetectionResult:
        reports: list[EngineReport] = []
        model_dets: list[Detection] = []
        cv_dets: list[Detection] = []
        kind = None

        # OpenCV first: it also decides whether the image is an illustration.
        if use_opencv:
            try:
                cv_dets, kind, msg = self.opencv.detect(bgr)
                reports.append(EngineReport("opencv", True, True, msg, len(cv_dets)))
            except Exception as exc:
                reports.append(EngineReport("opencv", True, False, f"OpenCV error: {exc}"))
        else:
            reports.append(EngineReport("opencv", False, True, "OpenCV analysis disabled for this run."))
            kind = classify_image(bgr)[0]
        illustration = kind == KIND_FLAT

        if use_yolo and self.yolo is not None:
            if self.yolo.available:
                try:
                    found = self.yolo.detect(bgr, confidence=yolo_confidence)
                    msg = f"{len(found)} object(s) above confidence threshold."
                    if found and illustration:
                        msg += " Image is a flat illustration: verify these labels."
                        for d in found:
                            d.attributes["domain_warning"] = (
                                "YOLO is trained on photographs; labels on illustrations are often wrong.")
                    model_dets += found
                    reports.append(EngineReport("yolo", True, True, msg, len(found)))
                except Exception as exc:
                    reports.append(EngineReport("yolo", True, False, f"YOLO error: {exc}"))
            else:
                reports.append(EngineReport("yolo", False, False, self.yolo.reason))
        else:
            reports.append(EngineReport("yolo", False, True, "YOLO disabled for this run."))

        gender_hints = []
        if self.illustration is None or not use_illustration_model:
            reports.append(EngineReport("yoloe", False, True, "Illustration model disabled for this run."))
        elif not illustration:
            reports.append(EngineReport("yoloe", False, True,
                                        "Illustration model not used: image is not a flat illustration."))
        elif not self.illustration.available:
            reports.append(EngineReport("yoloe", False, False, self.illustration.reason))
        else:
            try:
                found = self.illustration.detect(bgr)
                for d in found:
                    d.attributes["domain_warning"] = "Open-vocabulary model on an illustration: verify the label."
                model_dets += found
                gender_hints = self.illustration.gender_suggestions(bgr)
                reports.append(EngineReport("yoloe", True, True,
                                            f"{len(found)} object(s) from the illustration vocabulary.", len(found)))
            except Exception as exc:
                reports.append(EngineReport("yoloe", True, False, f"Illustration model error: {exc}"))

        unify_character_labels(model_dets, illustration)
        n_raw = len(model_dets) + len(cv_dets)
        detections = merge_with_opencv(merge_model_detections(model_dets, illustration), cv_dets)
        attach_gender_suggestions(detections, gender_hints)
        palette = dominant_palette(bgr) if kind != KIND_BLANK else None

        ran_ok = [r for r in reports if r.ran and r.ok]
        if detections:
            status = STATUS_SUCCESS
            message = f"{len(detections)} object(s) detected."
            if n_raw > len(detections):
                message += f" {n_raw - len(detections)} duplicate detection(s) merged."
        elif not ran_ok:
            status = STATUS_FAILED
            message = "Detection failed: no detector could run. " + " | ".join(
                f"{r.engine}: {r.message}" for r in reports)
        else:
            status = STATUS_EMPTY
            message = "No objects detected." + (" The image appears blank." if kind == KIND_BLANK else "")
        problems = [f"{r.engine}: {r.message}" for r in reports if not r.ok]
        if problems and status != STATUS_FAILED:
            message += " Warnings: " + " | ".join(problems)
        return DetectionResult(status=status, message=message, detections=detections, reports=reports,
                               image_kind=kind, dominant_colors=palette)


GENDER_WARNING = ("Appearance-based guess from the illustration model. It is unreliable (in evaluation it "
                  "called a cartoon boy a girl) and is never applied automatically.")


def attach_gender_suggestions(dets: list[Detection], hints) -> None:
    """Attach boy/girl hints to characters as *suggestions only*."""
    for d in dets:
        if d.label != CHARACTER_LABEL:
            continue
        overlapping = [(label, score) for label, score, box in hints if box.iou(d.bbox) >= 0.5]
        if overlapping:
            label, score = max(overlapping, key=lambda t: t[1])
            d.attributes["suggested_labels"] = [{
                "label": f"cartoon {label}", "score": score, "source": "yoloe", "warning": GENDER_WARNING}]
