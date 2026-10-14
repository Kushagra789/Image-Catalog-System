"""Ultralytics YOLO wrapper (COCO classes: person, apple, orange, ...).

The detector is optional. If ``ultralytics`` is not installed or the weights
cannot be loaded/downloaded, ``available`` is False and ``reason`` explains why;
the rest of the application keeps working with OpenCV analysis only.

Confidence values are the model's own class scores, stored unmodified.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from ..colors import dominant_color
from ..models import BBox, SOURCE_YOLO
from .base import Detection

log = logging.getLogger(__name__)


class YoloDetector:
    name = "yolo"

    def __init__(self, model_name: str, models_dir: Path, confidence: float = 0.25):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.confidence = confidence
        self._model = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def weights_path(self) -> Path:
        return self.models_dir / self.model_name

    def load(self) -> bool:
        """Load the model once. Returns True when usable."""
        if self._model is not None:
            return True
        if self._error is not None:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from ultralytics import YOLO  # heavy import, done lazily
            except Exception as exc:  # ImportError, or torch DLL problems on Windows
                self._error = f"ultralytics/torch not importable: {exc}"
                return False
            try:
                self.models_dir.mkdir(parents=True, exist_ok=True)
                # Passing a path inside models/ makes Ultralytics download the
                # official weights there on first use (needs internet once).
                self._model = YOLO(str(self.weights_path))
            except Exception as exc:
                self._error = (f"could not load YOLO weights '{self.model_name}' "
                               f"(first run needs internet to download them): {exc}")
                return False
        return True

    @property
    def available(self) -> bool:
        return self.load()

    @property
    def reason(self) -> str:
        if self._model is not None:
            return f"YOLO model '{self.model_name}' loaded."
        return self._error or "YOLO not loaded yet."

    def detect(self, bgr: np.ndarray, confidence: float | None = None) -> list[Detection]:
        if not self.load():
            raise RuntimeError(self.reason)
        conf = self.confidence if confidence is None else confidence
        results = self._model.predict(bgr, conf=conf, verbose=False)
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        polygons = None
        if getattr(result, "masks", None) is not None:
            polygons = result.masks.xy  # list of (N, 2) arrays in image coordinates

        h_img, w_img = bgr.shape[:2]
        detections = []
        for i, ((x1, y1, x2, y2), score, cls) in enumerate(zip(xyxy, scores, classes)):
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(w_img, int(round(x2))), min(h_img, int(round(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            attrs: dict = {"class_id": int(cls), "model": self.model_name}
            mask = None
            if polygons is not None and i < len(polygons) and len(polygons[i]) >= 3:
                poly = np.round(polygons[i]).astype(np.int32)
                attrs["polygon"] = poly.tolist()
                import cv2
                mask = np.zeros((h_img, w_img), np.uint8)
                cv2.fillPoly(mask, [poly], 255)
            color, color_hex = self._object_color(bgr, (x1, y1, x2, y2), mask)
            if color_hex:
                attrs["color_hex"] = color_hex
            detections.append(Detection(
                label=str(names[int(cls)]),
                bbox=BBox(x1, y1, x2 - x1, y2 - y1),
                source=SOURCE_YOLO,
                confidence=round(float(score), 4),
                color=color,
                attributes=attrs,
            ))
        return detections

    @staticmethod
    def _object_color(bgr, box, mask) -> tuple[str | None, str | None]:
        x1, y1, x2, y2 = box
        if mask is not None and mask[y1:y2, x1:x2].any():
            name, hex_, _ = dominant_color(bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2])
            return name, hex_
        # No mask: use the central 60% of the box to reduce background influence.
        w, h = x2 - x1, y2 - y1
        cx1, cy1 = x1 + int(0.2 * w), y1 + int(0.2 * h)
        cx2, cy2 = max(cx1 + 1, x2 - int(0.2 * w)), max(cy1 + 1, y2 - int(0.2 * h))
        name, hex_, _ = dominant_color(bgr[cy1:cy2, cx1:cx2])
        return name, hex_
