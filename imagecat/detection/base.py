"""Shared detection result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import BBox


@dataclass
class Detection:
    label: str
    bbox: BBox
    source: str
    confidence: float | None = None  # only set when the model actually produced one
    color: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineReport:
    """What one detector did on one image."""
    engine: str
    ran: bool
    ok: bool
    message: str
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "ran": self.ran, "ok": self.ok,
                "message": self.message, "count": self.count}


@dataclass
class DetectionResult:
    status: str
    message: str
    detections: list[Detection]
    reports: list[EngineReport]
    image_kind: str | None = None
    dominant_colors: list[dict] | None = None


def polygon_from_contour(contour, max_points: int = 400) -> list[list[int]]:
    """Convert an OpenCV contour to a compact JSON-friendly polygon."""
    import cv2

    pts = contour
    eps = 0.5
    while len(pts) > max_points:
        pts = cv2.approxPolyDP(contour, eps, True)
        eps *= 1.6
    return [[int(p[0][0]), int(p[0][1])] for p in pts]
