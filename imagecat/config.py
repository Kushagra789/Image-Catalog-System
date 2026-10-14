"""Application paths and tunable settings.

Everything lives inside the project folder by default. Set the
``IMAGECAT_DATA_DIR`` environment variable to use a different data folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep Ultralytics' settings file inside the project instead of %APPDATA%.
# The directory must exist before ultralytics is imported, or it falls back.
_yolo_cfg = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics")))
_yolo_cfg.mkdir(parents=True, exist_ok=True)
# rembg downloads its matting model here instead of ~/.u2net.
os.environ.setdefault("U2NET_HOME", str(PROJECT_ROOT / "models" / "rembg"))


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("IMAGECAT_DATA_DIR", PROJECT_ROOT / "data"))
    )
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    # Segmentation model: gives boxes, real confidences AND object masks
    # (masks make background removal in the composer much better).
    yolo_model: str = field(default_factory=lambda: os.environ.get("IMAGECAT_YOLO_MODEL", "yolo11n-seg.pt"))
    yolo_confidence: float = 0.25
    enable_yolo: bool = field(default_factory=lambda: os.environ.get("IMAGECAT_DISABLE_YOLO", "0") != "1")
    # Open-vocabulary model used on illustrations (cartoons, clip-art) only.
    illustration_model: str = field(
        default_factory=lambda: os.environ.get("IMAGECAT_ILLUSTRATION_MODEL", "yoloe-26s-seg-pf.pt"))
    illustration_confidence: float = 0.2
    enable_illustration_model: bool = field(
        default_factory=lambda: os.environ.get("IMAGECAT_DISABLE_ILLUSTRATION_MODEL", "0") != "1")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "catalog.db"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
