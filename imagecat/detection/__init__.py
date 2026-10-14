from .base import Detection, DetectionResult, EngineReport
from .engine import DetectionEngine
from .opencv_detector import OpenCVDetector, classify_image
from .yolo_detector import YoloDetector
from .yoloe_detector import IllustrationDetector

__all__ = [
    "Detection", "DetectionResult", "EngineReport", "DetectionEngine",
    "OpenCVDetector", "YoloDetector", "IllustrationDetector", "classify_image",
]
