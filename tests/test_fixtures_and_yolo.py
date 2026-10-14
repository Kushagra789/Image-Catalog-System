"""Fixture reproducibility, and YOLO on real photographs (skipped if unavailable)."""
import hashlib

import pytest

from imagecat import fixtures
from imagecat.config import Settings
from imagecat.detection import YoloDetector
from imagecat.imaging import load_rgb, pil_to_bgr
from imagecat.models import STATUS_SUCCESS


def test_all_required_fixtures_exist():
    required = {"apple", "orange", "cartoon_boy", "cartoon_girl", "shapes", "colors",
                "apple_orange", "overlapping_shapes"}
    assert required <= set(fixtures.FIXTURES)


@pytest.mark.parametrize("name", list(fixtures.FIXTURES))
def test_fixtures_are_deterministic(name):
    a, ta = fixtures.generate(name)
    b, tb = fixtures.generate(name)
    assert hashlib.sha256(a.tobytes()).digest() == hashlib.sha256(b.tobytes()).digest()
    assert ta == tb


def test_write_all(tmp_path):
    paths = fixtures.write_all(tmp_path)
    assert {p.stem for p in paths} == set(fixtures.FIXTURES)


# ------------------------------------------------------------------- YOLO
@pytest.fixture(scope="module")
def yolo():
    s = Settings()
    if not (s.models_dir / s.yolo_model).exists():
        pytest.skip("YOLO weights not downloaded (run the app once with internet).")
    det = YoloDetector(s.yolo_model, s.models_dir)
    if not det.available:
        pytest.skip(f"YOLO unavailable: {det.reason}")
    return det


@pytest.fixture(scope="module")
def sample_photo():
    try:
        from ultralytics.utils import ASSETS
    except Exception:
        pytest.skip("ultralytics not installed")
    return ASSETS / "bus.jpg"  # real photograph shipped with ultralytics


def test_yolo_detects_real_photo_with_real_confidences(yolo, sample_photo):
    dets = yolo.detect(pil_to_bgr(load_rgb(sample_photo)))
    labels = [d.label for d in dets]
    assert "bus" in labels and "person" in labels
    for d in dets:
        assert d.source == "yolo"
        assert 0.0 < d.confidence <= 1.0
        assert d.confidence >= yolo.confidence  # threshold respected
        assert d.bbox.width > 0 and d.bbox.height > 0
        assert "polygon" in d.attributes  # segmentation mask outline


def test_yolo_pipeline_end_to_end(settings, yolo, sample_photo):
    from imagecat.service import CatalogService

    svc = CatalogService(settings, yolo=yolo, illustration=None)
    out = svc.upload(sample_photo.read_bytes(), "bus.jpg")
    assert out.record.detection_status == STATUS_SUCCESS
    objs = svc.db.list_objects(out.record.id)
    assert any(o.source == "yolo" and o.label == "bus" and o.confidence for o in objs)
    # Photos are not analysed for shapes (would produce meaningless regions).
    assert not any(o.source.startswith("opencv") for o in objs)


def test_yolo_blank_image_is_empty(settings, yolo):
    from imagecat.service import CatalogService
    from tests.conftest import fixture_png

    svc = CatalogService(settings, yolo=yolo, illustration=None)
    out = svc.upload(fixture_png("blank"), "blank.png")
    assert out.detection.status == "empty"
