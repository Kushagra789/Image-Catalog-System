"""Plain data classes shared across the application."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Detection status values stored on each image.
STATUS_PENDING = "pending"      # uploaded, detection not run yet
STATUS_SUCCESS = "success"      # at least one object found
STATUS_EMPTY = "empty"          # detectors ran fine, nothing found
STATUS_FAILED = "failed"        # no detector could run / image unreadable
STATUS_SKIPPED = "skipped"      # e.g. composites: objects come from composition, not detection
DETECTION_STATUSES = (STATUS_PENDING, STATUS_SUCCESS, STATUS_EMPTY, STATUS_FAILED, STATUS_SKIPPED)

# Object sources.
SOURCE_YOLO = "yolo"
SOURCE_YOLOE = "yoloe"  # open-vocabulary illustration model
SOURCE_SHAPE = "opencv_shape"
SOURCE_COLOR_REGION = "opencv_color_region"
SOURCE_FOREGROUND = "opencv_foreground"
SOURCE_MANUAL = "manual"
SOURCE_COMPOSITE = "composite"

ORIGIN_AUTO = "auto"
ORIGIN_MANUAL = "manual"


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def iou(self, other: "BBox") -> float:
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class ImageRecord:
    id: int
    name: str
    display_name: str
    filename: str
    width: int
    height: int
    created_at: str
    detection_status: str
    detection_message: str | None = None
    detected_at: str | None = None
    file_hash: str | None = None
    is_composite: bool = False
    source_asset_ids: dict[str, Any] | None = None
    composition_metadata: dict[str, Any] | None = None
    dominant_colors: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectRecord:
    id: int
    image_id: int
    label: str
    color: str | None
    confidence: float | None  # None = no calibrated confidence exists (never invented)
    x: int
    y: int
    width: int
    height: int
    source: str
    origin: str
    is_edited: bool
    is_deleted: bool
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def bbox(self) -> BBox:
        return BBox(self.x, self.y, self.width, self.height)

    @property
    def is_correction(self) -> bool:
        """True when this row reflects a human decision."""
        return self.origin == ORIGIN_MANUAL or self.is_edited or self.is_deleted

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
