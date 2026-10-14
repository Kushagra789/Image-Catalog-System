"""Manual corrections: add / edit / delete, validation, persistence, rerun policy."""
import pytest

from imagecat.config import Settings
from imagecat.service import (
    CatalogService,
    ConfirmationRequired,
    RERUN_KEEP_MANUAL,
    RERUN_REPLACE_ALL,
    ValidationError,
)


@pytest.fixture
def cartoon(upload_fixture):
    return upload_fixture("cartoon_boy").record


def test_manual_add(service, cartoon):
    obj = service.add_object(cartoon.id, "Cartoon  Boy", (230, 35, 180, 410), color="Blue", confidence=None)
    assert obj.label == "cartoon boy"          # normalised
    assert obj.color == "blue"
    assert obj.confidence is None              # not invented
    assert obj.source == "manual" and obj.origin == "manual"
    assert obj.is_correction


def test_manual_add_with_user_confidence(service, cartoon):
    obj = service.add_object(cartoon.id, "boy", (10, 10, 50, 50), confidence=0.8)
    assert obj.confidence == 0.8


@pytest.mark.parametrize("bbox", [(0, 0, 0, 10), (-1, 0, 10, 10), (600, 400, 100, 100), ("a", 0, 1, 1)])
def test_invalid_bbox_rejected(service, cartoon, bbox):
    with pytest.raises(ValidationError):
        service.add_object(cartoon.id, "boy", bbox)


@pytest.mark.parametrize("label,conf", [("", None), ("   ", None), ("x", 1.5), ("x", -0.1), ("x", "abc")])
def test_invalid_label_or_confidence_rejected(service, cartoon, label, conf):
    with pytest.raises(ValidationError):
        service.add_object(cartoon.id, label, (1, 1, 10, 10), confidence=conf)


def test_edit_detected_object_keeps_audit_trail(service, cartoon):
    fg = next(o for o in service.db.list_objects(cartoon.id) if o.source == "opencv_foreground")
    edited = service.update_object(fg.id, label="cartoon boy", color="blue",
                                   bbox=(fg.x, fg.y, fg.width - 2, fg.height - 2))
    assert edited.label == "cartoon boy"
    assert edited.color == "blue"
    assert edited.width == fg.width - 2
    assert edited.is_edited and edited.is_correction
    assert edited.source == "opencv_foreground"  # provenance kept
    assert edited.attributes["original"]["label"] == "cartoon character"  # face heuristic
    assert edited.attributes["edit_history"]


def test_relabelling_clears_stale_model_confidence(service, cartoon):
    # Simulate a YOLO detection with a real confidence.
    oid = service.db.insert_object(image_id=cartoon.id, label="person", color="blue", confidence=0.42,
                                   bbox=(230, 35, 180, 410), source="yolo")
    edited = service.update_object(oid, label="cartoon boy")
    assert edited.confidence is None
    assert edited.attributes["original"]["confidence"] == 0.42
    # ... unless the user explicitly provides one.
    edited = service.update_object(oid, confidence=0.9)
    assert edited.confidence == 0.9


def test_edit_validation(service, cartoon):
    obj = service.add_object(cartoon.id, "boy", (10, 10, 50, 50))
    with pytest.raises(ValidationError):
        service.update_object(obj.id, bbox=(630, 10, 50, 50))
    with pytest.raises(ValidationError):
        service.update_object(obj.id, confidence=2)
    assert service.db.get_object(obj.id).bbox.as_tuple() == (10, 10, 50, 50)


def test_delete_and_restore(service, cartoon):
    auto = service.db.list_objects(cartoon.id)[0]
    service.delete_object(auto.id)
    assert auto.id not in [o.id for o in service.db.list_objects(cartoon.id)]
    tomb = service.db.get_object(auto.id)
    assert tomb.is_deleted and tomb.is_correction  # remembered as a correction
    service.restore_object(auto.id)
    assert auto.id in [o.id for o in service.db.list_objects(cartoon.id)]

    manual = service.add_object(cartoon.id, "boy", (10, 10, 50, 50))
    service.delete_object(manual.id)
    assert service.db.get_object(manual.id) is None  # manual objects are removed outright


def test_corrections_persist_across_restarts(tmp_path):
    from tests.conftest import fixture_png, make_settings

    settings = make_settings(tmp_path)
    svc = CatalogService(settings, yolo=None, illustration=None)
    rec = svc.upload(fixture_png("cartoon_girl"), "girl.png").record
    fg = next(o for o in svc.db.list_objects(rec.id) if o.source == "opencv_foreground")
    svc.update_object(fg.id, label="cartoon girl", color="pink")
    added = svc.add_object(rec.id, "hair bow", (300, 30, 40, 30), color="red")
    deleted = next(o for o in svc.db.list_objects(rec.id) if o.source == "opencv_shape")
    svc.delete_object(deleted.id)
    del svc

    # "Restart": brand-new service and DB connection on the same files.
    svc2 = CatalogService(make_settings(tmp_path), yolo=None, illustration=None)
    objs = {o.id: o for o in svc2.db.list_objects(rec.id, include_deleted=True)}
    assert objs[fg.id].label == "cartoon girl" and objs[fg.id].color == "pink" and objs[fg.id].is_edited
    assert objs[added.id].label == "hair bow"
    assert objs[deleted.id].is_deleted
    assert len(svc2.corrections(rec.id)) == 3


def test_rerun_keep_manual_preserves_corrections(service, cartoon):
    fg = next(o for o in service.db.list_objects(cartoon.id) if o.source == "opencv_foreground")
    service.update_object(fg.id, label="cartoon boy")
    shape = next(o for o in service.db.list_objects(cartoon.id) if o.source == "opencv_shape")
    service.delete_object(shape.id)
    manual = service.add_object(cartoon.id, "smile", (300, 125, 40, 15))
    n_before = len(service.db.list_objects(cartoon.id))

    result = service.run_detection(cartoon.id, mode=RERUN_KEEP_MANUAL)
    assert "preserved" in result.message

    objs = service.db.list_objects(cartoon.id, include_deleted=True)
    by_id = {o.id: o for o in objs}
    assert by_id[fg.id].label == "cartoon boy"            # edit kept
    assert by_id[shape.id].is_deleted                       # deletion kept ...
    active = [o for o in objs if not o.is_deleted]
    assert not any(o.bbox.iou(shape.bbox) > 0.5 and o.id != fg.id for o in active)  # ... and not re-added
    assert manual.id in by_id                               # manual object kept
    assert len(active) == n_before                          # no duplicates


def test_replace_all_requires_confirmation(service, cartoon):
    service.add_object(cartoon.id, "cartoon boy", (230, 35, 180, 410))
    with pytest.raises(ConfirmationRequired) as exc:
        service.run_detection(cartoon.id, mode=RERUN_REPLACE_ALL)
    assert exc.value.n_corrections == 1
    assert any(o.label == "cartoon boy" for o in service.db.list_objects(cartoon.id))  # untouched

    service.run_detection(cartoon.id, mode=RERUN_REPLACE_ALL, confirm_replace=True)
    objs = service.db.list_objects(cartoon.id, include_deleted=True)
    assert not any(o.is_correction for o in objs)


def test_replace_all_without_corrections_needs_no_confirmation(service, cartoon):
    result = service.run_detection(cartoon.id, mode=RERUN_REPLACE_ALL)
    assert result.status == "success"


def test_unknown_object_or_image(service):
    with pytest.raises(ValidationError):
        service.update_object(999, label="x")
    with pytest.raises(ValidationError):
        service.add_object(999, "x", (0, 0, 1, 1))
    with pytest.raises(ValidationError):
        service.run_detection(999)


def test_settings_default_paths_inside_project():
    s = Settings()
    assert s.db_path.name == "catalog.db"
