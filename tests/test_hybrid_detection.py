"""Hybrid cartoon detection and duplicate merging."""
import pytest

from imagecat import fixtures
from imagecat.config import Settings
from imagecat.detection import Detection, DetectionEngine, OpenCVDetector
from imagecat.detection.engine import attach_gender_suggestions
from imagecat.detection.labels import compatible, group_of, specificity
from imagecat.detection.merge import (
    merge_model_detections,
    merge_with_opencv,
    unify_character_labels,
)
from imagecat.imaging import pil_to_bgr
from imagecat.models import BBox


def det(label, box, source="yolo", conf=0.9, **attrs):
    return Detection(label=label, bbox=BBox(*box), source=source, confidence=conf, attributes=dict(attrs))


# ------------------------------------------------------------------ labels
@pytest.mark.parametrize("a,b,expected", [
    ("apple", "apple", True), ("fruit", "apple", True), ("apple", "orange", False),
    ("person", "doll", True), ("cartoon character", "child", True), ("kite", "fruit", False),
    ("motorbike", "motorcycle", True), ("dog", "animal", True), ("dog", "cat", False),
])
def test_label_compatibility(a, b, expected):
    assert compatible(a, b) is expected


def test_specificity_prefers_specific_labels():
    assert specificity("apple") > specificity("fruit")
    assert group_of("girl") == "person"


# --------------------------------------------------------- model vs model
def test_same_object_keeps_most_specific_label_and_its_own_confidence():
    out = merge_model_detections([det("fruit", (10, 10, 100, 100), "yoloe", 0.8),
                                  det("apple", (12, 12, 100, 100), "yolo", 0.4)])
    assert len(out) == 1
    assert out[0].label == "apple"
    assert out[0].confidence == 0.4  # confidence of the kept label, never averaged
    assert out[0].attributes["alternative_labels"][0]["label"] == "fruit"


def test_part_inside_whole_is_merged_into_larger_box():
    body = det("doll", (100, 100, 200, 500), "yoloe", 0.7)
    head = det("cartoon character", (120, 100, 150, 150), "yoloe", 0.3)
    out = merge_model_detections([body, head], illustration=True)
    assert len(out) == 1 and out[0].bbox.as_tuple() == (100, 100, 200, 500)
    assert out[0].attributes["merged_from"][0]["reason"] == "part of a larger detection"


def test_competing_labels_same_region_prefer_domain_model():
    a = det("kite", (100, 100, 400, 400), "yolo", 0.62)
    b = det("fruit", (102, 98, 400, 402), "yoloe", 0.59)
    on_drawing = merge_model_detections([a, b], illustration=True)
    assert [d.label for d in on_drawing] == ["fruit"]
    assert on_drawing[0].attributes["alternative_labels"][0]["label"] == "kite"
    a2 = det("kite", (100, 100, 400, 400), "yolo", 0.62)
    b2 = det("fruit", (102, 98, 400, 402), "yoloe", 0.59)
    assert [d.label for d in merge_model_detections([a2, b2], illustration=False)] == ["kite"]


def test_distinct_objects_are_not_merged():
    out = merge_model_detections([det("apple", (0, 0, 100, 100)), det("orange", (300, 0, 100, 100)),
                                  det("apple", (40, 40, 100, 100))])  # IoU < 0.5 with the first apple
    assert len(out) == 3


# --------------------------------------------------------- opencv vs model
def test_foreground_blob_gives_outline_and_whole_extent():
    model = det("apple", (81, 161, 169, 159), "yoloe", 0.21)
    blob = det("foreground object", (80, 135, 171, 186), "opencv_foreground", None, polygon=[[80, 135], [250, 135], [250, 320]])
    body = det("circle", (80, 160, 171, 161), "opencv_shape", None)
    out = merge_with_opencv([model], [body, blob])
    labels = sorted(d.label for d in out)
    assert labels == ["apple", "circle"]  # the round body stays a separate part
    apple = next(d for d in out if d.label == "apple")
    assert apple.bbox.as_tuple() == (80, 135, 171, 186)  # stem + leaf included
    assert apple.attributes["model_bbox"] == [81, 161, 169, 159]
    assert apple.attributes["polygon_source"] == "opencv_foreground"
    circle = next(d for d in out if d.label == "circle")
    assert circle.attributes["part_of"]["label"] == "apple"


def test_low_confidence_model_label_does_not_override_shape():
    out = merge_with_opencv([det("frisbee", (50, 60, 142, 142), "yolo", 0.46)],
                            [det("circle", (50, 60, 141, 141), "opencv_shape", None)])
    assert len(out) == 1 and out[0].label == "circle" and out[0].confidence is None
    assert out[0].attributes["alternative_labels"][0] == {"label": "frisbee", "source": "yolo", "confidence": 0.46}


def test_confident_model_label_absorbs_shape():
    out = merge_with_opencv([det("orange", (234, 142, 172, 181), "yolo", 0.71)],
                            [det("circle", (235, 151, 171, 170), "opencv_shape", None, shape="circle")])
    assert len(out) == 1 and out[0].label == "orange" and out[0].confidence == 0.71
    assert out[0].attributes["shape"] == "circle"


# ------------------------------------------------------- character labels
def test_person_like_labels_unified_only_on_illustrations():
    dets = [det("doll", (0, 0, 10, 10), "yoloe"), det("person", (20, 0, 10, 10))]
    unify_character_labels(dets, illustration=True)
    assert [d.label for d in dets] == ["cartoon character"] * 2
    assert dets[0].attributes["label_mapping"]["model_label"] == "doll"
    photo = [det("person", (0, 0, 10, 10))]
    unify_character_labels(photo, illustration=False)
    assert photo[0].label == "person"


def test_gender_is_only_ever_a_suggestion():
    character = det("cartoon character", (0, 0, 100, 200))
    attach_gender_suggestions([character], [("girl", 0.2, BBox(0, 0, 100, 200))])
    assert character.label == "cartoon character"  # never changed automatically
    sug = character.attributes["suggested_labels"][0]
    assert sug["label"] == "cartoon girl" and "unreliable" in sug["warning"]


@pytest.mark.parametrize("name", ["cartoon_boy", "cartoon_girl"])
def test_face_heuristic_finds_synthetic_characters(name):
    dets, _, _ = OpenCVDetector().detect(pil_to_bgr(fixtures.generate(name)[0]))
    chars = [d for d in dets if d.label == "cartoon character"]
    assert len(chars) == 1
    assert chars[0].confidence is None  # heuristic: no invented confidence
    assert len(chars[0].attributes["character_evidence"]["eyes"]) == 2


@pytest.mark.parametrize("name", ["apple", "apple_orange", "shapes", "overlapping_shapes", "colors"])
def test_face_heuristic_has_no_false_positives_on_non_characters(name):
    dets, _, _ = OpenCVDetector().detect(pil_to_bgr(fixtures.generate(name)[0]))
    assert not any(d.label == "cartoon character" for d in dets)


def test_engine_reports_merges_in_message():
    eng = DetectionEngine(yolo=None, illustration=None)
    result = eng.detect(pil_to_bgr(fixtures.generate("cartoon_boy")[0]))
    assert any(d.label == "cartoon character" for d in result.detections)
    assert any(r.engine == "yoloe" and not r.ran for r in result.reports)


# ------------------------------------------------- real illustration model
@pytest.fixture(scope="module")
def illustration():
    from imagecat.detection import IllustrationDetector

    s = Settings()
    if not (s.models_dir / s.illustration_model).exists():
        pytest.skip("Illustration model weights not downloaded.")
    detector = IllustrationDetector(s.illustration_model, s.models_dir)
    if not detector.available:
        pytest.skip(f"Illustration model unavailable: {detector.reason}")
    return detector


def test_vocabulary_is_restricted(illustration):
    names = set(illustration._model.model.names.values())
    assert names == set(illustration.vocabulary) and "heart" not in names and "joker" not in names
    assert set(illustration._gender_model.model.names.values()) == {"boy", "girl"}


def test_illustration_model_names_cartoon_orange(illustration):
    found = illustration.detect(pil_to_bgr(fixtures.generate("orange")[0]))
    best = max(found, key=lambda d: d.confidence)
    assert best.label == "orange" and 0 < best.confidence <= 1 and best.source == "yoloe"


def test_full_hybrid_never_assigns_gender_automatically(illustration):
    from imagecat.detection import YoloDetector

    s = Settings()
    yolo = YoloDetector(s.yolo_model, s.models_dir)
    eng = DetectionEngine(yolo if yolo.available else None, illustration=illustration)
    for name in ("cartoon_boy", "cartoon_girl", "apple_orange"):
        result = eng.detect(pil_to_bgr(fixtures.generate(name)[0]))
        assert not any(d.label in {"boy", "girl", "cartoon boy", "cartoon girl"} for d in result.detections)


def test_illustration_model_not_used_on_photos(illustration):
    try:
        from ultralytics.utils import ASSETS
    except Exception:
        pytest.skip("ultralytics not installed")
    from imagecat.imaging import load_rgb

    eng = DetectionEngine(yolo=None, illustration=illustration)
    result = eng.detect(pil_to_bgr(load_rgb(ASSETS / "bus.jpg")))
    report = next(r for r in result.reports if r.engine == "yoloe")
    assert not report.ran and "not a flat illustration" in report.message
