"""Open-vocabulary detector for illustrations (Ultralytics YOLOE, prompt-free).

The prompt-free YOLOE model ships with a ~4,500-class vocabulary. Used raw it is
unreliable on drawings (in evaluation: a cartoon girl's head -> "joker" 0.95, an
apple photo -> "fig" 0.92). Its classification layer, however, scores every
class independently, so we keep only the rows for a curated vocabulary
(``labels.ILLUSTRATION_VOCAB``). Scores of the kept classes are unchanged; we
only remove irrelevant competitors.

A second copy restricted to {"boy", "girl"} produces *suggestions* for
characters. These are appearance-based guesses and are never applied as a label
automatically: in evaluation the model called both a cartoon boy and a cartoon
girl "girl".
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..colors import dominant_color
from ..models import BBox, SOURCE_YOLOE
from .base import Detection
from .labels import GENDER_VOCAB, ILLUSTRATION_VOCAB


def restrict_vocabulary(yolo_model, words: list[str]) -> list[str]:
    """Shrink a prompt-free YOLOE model's vocabulary in place to ``words``."""
    import torch

    model = yolo_model.model
    names = model.names
    inverse = {v: k for k, v in names.items()}
    keep = [w for w in words if w in inverse]
    if not keep:
        raise ValueError("None of the requested classes exist in the model vocabulary.")
    idx = torch.tensor([inverse[w] for w in keep])
    head = model.model[-1]
    found = False
    for attr in ("lrpc", "one2one_lrpc"):
        for lrpc in getattr(head, attr, None) or []:
            old = lrpc.vocab
            if isinstance(old, torch.nn.Linear):
                new = torch.nn.Linear(old.in_features, len(keep), bias=old.bias is not None)
            elif isinstance(old, torch.nn.Conv2d):
                new = torch.nn.Conv2d(old.in_channels, len(keep), 1, bias=old.bias is not None)
            else:
                raise TypeError(f"Unexpected vocabulary layer {type(old).__name__}")
            new.weight.data = old.weight.data[idx].clone()
            if old.bias is not None:
                new.bias.data = old.bias.data[idx].clone()
            lrpc.vocab = new.to(old.weight.device)
            found = True
    if not found:
        raise TypeError("Model is not a prompt-free YOLOE model (no LRPC vocabulary head).")
    head.nc = len(keep)
    if hasattr(head, "no") and hasattr(head, "reg_max"):
        head.no = head.nc + head.reg_max * 4
    model.names = dict(enumerate(keep))
    yolo_model.predictor = None  # force a rebuild with the new class count
    return keep


class IllustrationDetector:
    name = "yoloe"

    def __init__(self, model_name: str, models_dir: Path, confidence: float = 0.2,
                 vocabulary: list[str] | None = None):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.confidence = confidence
        self.vocabulary = list(vocabulary or ILLUSTRATION_VOCAB)
        self._model = None
        self._gender_model = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def weights_path(self) -> Path:
        return self.models_dir / self.model_name

    def load(self) -> bool:
        if self._model is not None:
            return True
        if self._error is not None:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from ultralytics import YOLO
            except Exception as exc:
                self._error = f"ultralytics/torch not importable: {exc}"
                return False
            try:
                main = YOLO(str(self.weights_path))
                restrict_vocabulary(main, self.vocabulary)
                gender = YOLO(str(self.weights_path))
                restrict_vocabulary(gender, GENDER_VOCAB)
            except Exception as exc:
                self._error = (f"could not load illustration model '{self.model_name}' "
                               f"(first run downloads ~40 MB): {exc}")
                return False
            self._model, self._gender_model = main, gender
        return True

    @property
    def available(self) -> bool:
        return self.load()

    @property
    def reason(self) -> str:
        if self._model is not None:
            return f"Illustration model '{self.model_name}' loaded ({len(self.vocabulary)} classes)."
        return self._error or "Illustration model not loaded yet."

    def detect(self, bgr: np.ndarray, confidence: float | None = None) -> list[Detection]:
        if not self.load():
            raise RuntimeError(self.reason)
        conf = self.confidence if confidence is None else confidence
        result = self._model.predict(bgr, conf=conf, verbose=False)[0]
        return self._to_detections(result, bgr)

    def gender_suggestions(self, bgr: np.ndarray) -> list[tuple[str, float, BBox]]:
        """Raw boy/girl scores. Callers must treat them as unverified hints."""
        if not self.load():
            return []
        result = self._gender_model.predict(bgr, conf=0.05, verbose=False)[0]
        out = []
        if result.boxes is None:
            return out
        for (x1, y1, x2, y2), score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                                 result.boxes.cls.cpu().numpy().astype(int)):
            box = BBox(int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)))
            out.append((result.names[int(cls)], round(float(score), 4), box))
        return out

    def _to_detections(self, result, bgr) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        h_img, w_img = bgr.shape[:2]
        polygons = result.masks.xy if getattr(result, "masks", None) is not None else None
        out = []
        for i, ((x1, y1, x2, y2), score, cls) in enumerate(zip(
                boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int))):
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(w_img, int(round(x2))), min(h_img, int(round(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            attrs: dict = {"model": self.model_name, "vocabulary": "curated illustration subset"}
            mask = None
            if polygons is not None and i < len(polygons) and len(polygons[i]) >= 3:
                import cv2
                poly = np.round(polygons[i]).astype(np.int32)
                attrs["polygon"] = poly.tolist()
                mask = np.zeros((h_img, w_img), np.uint8)
                cv2.fillPoly(mask, [poly], 255)
            sub_mask = mask[y1:y2, x1:x2] if mask is not None and mask[y1:y2, x1:x2].any() else None
            color, color_hex, _ = dominant_color(bgr[y1:y2, x1:x2], sub_mask)
            if color_hex:
                attrs["color_hex"] = color_hex
            out.append(Detection(
                label=str(result.names[int(cls)]), bbox=BBox(x1, y1, x2 - x1, y2 - y1),
                source=SOURCE_YOLOE, confidence=round(float(score), 4), color=color, attributes=attrs,
            ))
        return out
