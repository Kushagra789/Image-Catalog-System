"""Regression tests for the defect found in the real-browser (Playwright) test run.

With the learned models enabled, a model label with confidence >= 0.5 replaced a
correct OpenCV shape / colour region on flat illustrations: the orange blob in
`colors` became YOLO "kite" (0.57) and the orange ellipse in `shapes` became
YOLOE "orange" (0.59). The scores below are the ones measured in that run.
"""
import pytest

from imagecat import fixtures
from imagecat.config import Settings
from imagecat.detection import Detection, DetectionEngine
from imagecat.detection.merge import CONFIDENT, merge_with_opencv
from imagecat.imaging import pil_to_bgr
from imagecat.models import BBox


def det(label, box, source, conf, **attrs):
    return Detection(label=label, bbox=BBox(*box), source=source, confidence=conf, attributes=dict(attrs))


def test_threshold_value():
    assert CONFIDENT == 0.65


def test_kite_does_not_replace_orange_colour_region():
    out = merge_with_opencv([det("kite", (240, 299, 149, 131), "yolo", 0.5728)],
                            [det("orange region", (240, 300, 149, 129), "opencv_color_region", None)])
    assert [d.label for d in out] == ["orange region"]
    assert out[0].confidence is None
    assert out[0].attributes["alternative_labels"][0]["label"] == "kite"  # still one click away


def test_low_confidence_fruit_does_not_replace_ellipse():
    out = merge_with_opencv([det("orange", (510, 300, 120, 100), "yoloe", 0.5864)],
                            [det("ellipse", (510, 300, 121, 101), "opencv_shape", None, shape="ellipse")])
    assert [d.label for d in out] == ["ellipse"]


def test_confident_label_still_names_the_orange_fixture():
    out = merge_with_opencv([det("orange", (234, 142, 172, 181), "yolo", 0.7121)],
                            [det("circle", (235, 151, 171, 170), "opencv_shape", None, shape="circle")])
    assert [d.label for d in out] == ["orange"] and out[0].confidence == 0.7121


# ------------------------------------------- end to end with the real models
@pytest.fixture(scope="module")
def full_engine():
    from imagecat.detection import IllustrationDetector, YoloDetector

    s = Settings()
    yolo = YoloDetector(s.yolo_model, s.models_dir)
    ill = IllustrationDetector(s.illustration_model, s.models_dir)
    if not (s.models_dir / s.yolo_model).exists() or not (s.models_dir / s.illustration_model).exists():
        pytest.skip("model weights not downloaded")
    if not (yolo.available and ill.available):
        pytest.skip("learned models unavailable")
    return DetectionEngine(yolo, illustration=ill)


def _labels(engine, name):
    return {(d.label, d.color) for d in engine.detect(pil_to_bgr(fixtures.generate(name)[0])).detections}


def test_full_engine_keeps_all_six_shapes(full_engine):
    labels = _labels(full_engine, "shapes")
    for want in [("circle", "red"), ("square", "blue"), ("triangle", "green"), ("rectangle", "yellow"),
                 ("pentagon", "purple"), ("ellipse", "orange")]:
        assert want in labels, (want, labels)


def test_full_engine_keeps_all_six_colour_swatches(full_engine):
    regions = {label for label, _ in _labels(full_engine, "colors") if label.endswith(" region")}
    assert regions == {f"{c} region" for c in ("red", "green", "blue", "yellow", "orange", "purple")}


def test_full_engine_still_names_the_orange(full_engine):
    assert any(label == "orange" for label, _ in _labels(full_engine, "orange"))
