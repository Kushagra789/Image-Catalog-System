"""Detection on synthetic fixtures (OpenCV path; YOLO is tested in test_yolo.py)."""
import numpy as np
import pytest

from imagecat import fixtures
from imagecat.colors import name_color_rgb
from imagecat.detection import OpenCVDetector, classify_image
from imagecat.imaging import pil_to_bgr
from imagecat.models import (
    BBox,
    SOURCE_COLOR_REGION,
    SOURCE_FOREGROUND,
    SOURCE_SHAPE,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_SUCCESS,
)


def _detect(name):
    img, truth = fixtures.generate(name)
    dets, kind, msg = OpenCVDetector().detect(pil_to_bgr(img))
    return dets, truth, kind


def _best_match(gt, dets):
    gt_box = BBox(*gt.bbox)
    return max(dets, key=lambda d: d.bbox.iou(gt_box), default=None)


@pytest.mark.parametrize("rgb,expected", [
    ((220, 20, 30), "red"), ((255, 140, 0), "orange"), ((40, 160, 40), "green"),
    ((30, 70, 220), "blue"), ((245, 215, 0), "yellow"), ((130, 50, 180), "purple"),
    ((120, 60, 20), "brown"), ((250, 205, 165), "beige"), ((240, 110, 170), "pink"),
    ((255, 255, 255), "white"), ((10, 10, 10), "black"), ((128, 128, 128), "gray"),
])
def test_color_naming(rgb, expected):
    assert name_color_rgb(rgb) == expected


def test_image_kind_classification():
    assert classify_image(pil_to_bgr(fixtures.generate("blank")[0]))[0] == "blank"
    assert classify_image(pil_to_bgr(fixtures.generate("shapes")[0]))[0] == "flat"
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    assert classify_image(noise)[0] == "photo"


def test_shapes_detected_with_labels_colors_and_boxes():
    dets, truth, kind = _detect("shapes")
    assert kind == "flat"
    for gt in truth:
        d = _best_match(gt, dets)
        assert d is not None
        assert d.bbox.iou(BBox(*gt.bbox)) > 0.9, gt
        assert d.label == gt.label, (gt, d.label)
        assert d.color == gt.color, (gt, d.color)
        assert d.source == SOURCE_SHAPE
        assert d.confidence is None  # heuristics never invent a confidence


def test_color_regions_detected():
    dets, truth, _ = _detect("colors")
    for gt in truth:
        d = _best_match(gt, dets)
        assert d.bbox.iou(BBox(*gt.bbox)) > 0.9
        assert d.source == SOURCE_COLOR_REGION
        assert d.label == gt.label and d.color == gt.color


def test_overlapping_shapes_detected_and_occlusion_flagged():
    dets, truth, _ = _detect("overlapping_shapes")
    found = {}
    for gt in truth:
        d = _best_match(gt, dets)
        assert d.bbox.iou(BBox(*gt.bbox)) > 0.9
        assert (d.label, d.color) == (gt.label, gt.color)
        found[gt.label] = d
    # Boxes really overlap in this fixture.
    assert found["square"].bbox.iou(found["circle"].bbox) > 0
    assert found["circle"].bbox.iou(found["triangle"].bbox) > 0
    # Square (bitten by the circle) and circle (bitten by the triangle) are
    # occluded; the triangle lies on top.
    assert found["square"].attributes["occluded"] is True
    assert found["circle"].attributes["occluded"] is True
    assert found["triangle"].attributes["occluded"] is False


@pytest.mark.parametrize("name", ["apple", "apple_orange", "cartoon_boy", "cartoon_girl"])
def test_multicolor_objects_found_as_foreground_with_polygon(name):
    dets, truth, _ = _detect(name)
    fgs = [d for d in dets if d.source == SOURCE_FOREGROUND]
    for gt in truth:
        if gt.label == "orange":  # single dominant colour -> reported as a circle instead
            continue
        d = _best_match(gt, fgs)
        assert d is not None and d.bbox.iou(BBox(*gt.bbox)) > 0.9, gt
        assert len(d.attributes["polygon"]) >= 3


def test_blank_image_reports_empty(service, upload_fixture):
    out = upload_fixture("blank")
    assert out.detection.status == STATUS_EMPTY
    assert "blank" in out.detection.message.lower()
    assert out.record.detection_status == STATUS_EMPTY
    assert service.db.list_objects(out.record.id) == []


def test_successful_detection_persisted(service, upload_fixture):
    out = upload_fixture("shapes")
    assert out.record.detection_status == STATUS_SUCCESS
    objs = service.db.list_objects(out.record.id)
    assert len(objs) == 6
    assert all(o.confidence is None and o.origin == "auto" for o in objs)
    runs = service.db.list_detection_runs(out.record.id)
    assert runs[0]["status"] == STATUS_SUCCESS and runs[0]["num_objects"] == 6


def test_detection_failure_reported_when_no_engine_can_run(service, upload_fixture):
    rec = upload_fixture("shapes", auto_detect=False).record
    result = service.run_detection(rec.id, use_yolo=False, use_opencv=False)
    assert result.status == STATUS_FAILED
    assert "no detector could run" in result.message.lower()
    assert service.db.get_image(rec.id).detection_status == STATUS_FAILED


def test_detection_failure_on_corrupted_file_keeps_objects(service, upload_fixture):
    rec = upload_fixture("shapes").record
    before = len(service.db.list_objects(rec.id))
    service.catalog.path_for(rec).write_bytes(b"corrupted")
    result = service.run_detection(rec.id)
    assert result.status == STATUS_FAILED
    assert "could not be read" in result.message
    assert len(service.db.list_objects(rec.id)) == before  # nothing wiped


def test_unavailable_yolo_reported_not_hidden(settings):
    from imagecat.detection import YoloDetector
    from imagecat.service import CatalogService

    broken = YoloDetector("missing.pt", settings.models_dir)
    broken._error = "simulated: ultralytics not installed"  # what load() records on ImportError
    svc = CatalogService(settings, yolo=broken, illustration=None)
    from tests.conftest import fixture_png
    out = svc.upload(fixture_png("shapes"), "shapes.png")
    assert out.detection.status == STATUS_SUCCESS  # OpenCV still worked
    yolo_report = next(r for r in out.detection.reports if r.engine == "yolo")
    assert yolo_report.ok is False
    assert "Warnings" in out.detection.message


def _product_photo(seed=0):
    """Photo-like test image: an orange sphere with shading, low-frequency
    colour texture (peel, lighting) and sensor noise on a plain white
    background, which dominates the image and fooled the old flat/photo rule.
    Its metrics resemble a real product photo measured in the project
    (overall top-16 coverage ~0.88, foreground ~0.60)."""
    import cv2

    rng = np.random.default_rng(seed)
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - 320) ** 2 + (yy - 240) ** 2) / 150.0
    inside = d < 1
    shade = np.clip(1.0 - 0.55 * d ** 2 + 0.25 * np.exp(-((xx - 280) ** 2 + (yy - 190) ** 2) / 900.0), 0, 1)
    img = np.full((h, w, 3), 255.0)
    for c, base in enumerate((10, 120, 245)):  # BGR orange
        field = cv2.GaussianBlur(rng.normal(0, 1, (h, w)), (0, 0), 6)
        texture = field / field.std() * 22  # blotchy colour variation, per channel
        img[..., c] = np.where(inside, base * (0.35 + 0.65 * shade) + texture, 255)
    img += rng.normal(0, 8, img.shape) * inside[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def test_product_photo_on_white_background_is_not_flat():
    from imagecat.detection.opencv_detector import FLAT_FG_COVERAGE

    kind, metrics = classify_image(_product_photo())
    assert metrics["top16_color_coverage"] >= 0.8  # the old rule alone would have said "flat"
    assert metrics["top16_foreground_coverage"] < FLAT_FG_COVERAGE
    assert kind == "photo"
    dets, _, msg = OpenCVDetector().detect(_product_photo())
    assert dets == [] and "foreground" in msg
