"""Regression tests for issues found during the final review."""
import pytest

from imagecat.composer import Composer
from imagecat.service import ConfirmationRequired, RERUN_KEEP_MANUAL, RERUN_REPLACE_ALL
from tests.conftest import fixture_png


# ------------------------------------------------- corrections & transactions
def test_correction_made_during_detection_is_not_deleted(service, upload_fixture, monkeypatch):
    """Detection takes seconds; a correction made meanwhile (other tab) must survive."""
    rec = upload_fixture("shapes").record
    target = service.db.list_objects(rec.id)[0]
    real_detect = service.engine.detect_path

    def detect_while_user_edits(*args, **kwargs):
        result = real_detect(*args, **kwargs)
        service.update_object(target.id, label="my label")  # user edits mid-detection
        return result

    monkeypatch.setattr(service.engine, "detect_path", detect_while_user_edits)
    service.run_detection(rec.id, mode=RERUN_KEEP_MANUAL)
    survivor = service.db.get_object(target.id)
    assert survivor is not None and survivor.label == "my label" and survivor.is_edited


def test_correction_made_during_detection_blocks_unconfirmed_replace_all(service, upload_fixture, monkeypatch):
    rec = upload_fixture("shapes").record
    target = service.db.list_objects(rec.id)[0]
    real_detect = service.engine.detect_path

    def detect_while_user_edits(*args, **kwargs):
        result = real_detect(*args, **kwargs)
        service.update_object(target.id, label="my label")
        return result

    monkeypatch.setattr(service.engine, "detect_path", detect_while_user_edits)
    with pytest.raises(ConfirmationRequired):
        service.run_detection(rec.id, mode=RERUN_REPLACE_ALL)  # no corrections existed when it started
    assert service.db.get_object(target.id).label == "my label"


def test_failure_while_saving_detection_rolls_back_everything(service, upload_fixture, monkeypatch):
    rec = upload_fixture("shapes").record
    before = [o.to_dict() for o in service.db.list_objects(rec.id, include_deleted=True)]
    status_before = service.db.get_image(rec.id).detection_message

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(service.db, "log_detection_run", boom)
    with pytest.raises(RuntimeError):
        service.run_detection(rec.id, mode=RERUN_KEEP_MANUAL)
    assert [o.to_dict() for o in service.db.list_objects(rec.id, include_deleted=True)] == before
    assert service.db.get_image(rec.id).detection_message == status_before


# --------------------------------------------------------------- provenance
@pytest.fixture
def fruit(service, upload_fixture):
    out = {}
    for name, color in (("apple", "red"), ("orange", "orange")):
        rec = upload_fixture(name).record
        obj = max(service.db.list_objects(rec.id), key=lambda o: o.bbox.area)
        out[name] = (rec, service.update_object(obj.id, label=name, color=color))
    return out


def test_composite_stores_source_snapshot(service, fruit):
    composer = Composer(service)
    record = composer.save(composer.plan("one red apple and one orange"))
    layers = record.composition_metadata["layers"]
    apple_rec, apple_obj = fruit["apple"]
    snap = layers[0]["source_snapshot"]
    assert snap["image"]["id"] == apple_rec.id and snap["image"]["sha256"] == apple_rec.file_hash
    assert snap["object"]["id"] == apple_obj.id and snap["object"]["bbox"] == list(apple_obj.bbox.as_tuple())
    assert snap["object"]["manually_corrected"] is True


def test_provenance_survives_source_deletion(service, fruit):
    composer = Composer(service)
    record = composer.save(composer.plan("one red apple and one orange"))
    apple_rec, _ = fruit["apple"]
    assert [c.id for c in service.db.composites_using(apple_rec.id)] == [record.id]
    service.catalog.delete(apple_rec.id)
    kept = service.db.get_image(record.id)
    assert kept is not None and kept.composition_metadata["layers"][0]["source_snapshot"]["image"]["id"] == apple_rec.id
    report = service.health_report()
    assert report["ok"]  # a deleted source is a warning, not corruption
    assert any(f"Composite #{record.id}" in w for w in report["warnings"])


# ------------------------------------------------------------- health check
def test_health_report_clean_catalog(service, upload_fixture):
    upload_fixture("shapes")
    report = service.health_report()
    assert report["ok"] and report["integrity_check"] == "ok"
    assert report["missing_files"] == 0 and report["orphan_files"] == 0


def test_health_report_detects_missing_and_orphan_files(service, upload_fixture):
    rec = upload_fixture("shapes").record
    service.catalog.path_for(rec).unlink()
    (service.catalog.images_dir / "stray.png").write_bytes(b"x")
    report = service.health_report()
    assert not report["ok"]
    assert any("missing" in p for p in report["problems"])
    assert any("stray.png" in w for w in report["warnings"])
    assert service.db.get_image(rec.id) is not None  # report only: nothing deleted


def test_health_report_detects_bad_boxes(service, upload_fixture):
    rec = upload_fixture("shapes").record
    obj = service.db.list_objects(rec.id)[0]
    service.db.update_object(obj.id, x=rec.width - 1)  # bypasses validation on purpose
    assert any(f"#{obj.id}" in p for p in service.health_report()["problems"])


# -------------------------------------------------------------- fast queries
def test_object_counts_and_label_search(service, upload_fixture):
    shapes = upload_fixture("shapes").record
    blank = upload_fixture("blank").record
    counts = service.db.object_counts()
    assert counts[shapes.id] == len(service.db.list_objects(shapes.id)) and blank.id not in counts
    assert service.db.image_ids_with_label("triang") == {shapes.id}
    service.delete_object(next(o for o in service.db.list_objects(shapes.id) if o.label == "triangle").id)
    assert service.db.image_ids_with_label("triang") == set()  # removed objects are not searchable


def test_existing_catalog_without_new_attributes_still_composes(service, upload_fixture):
    """Records created before today's changes have no snapshot/placement attributes."""
    rec = upload_fixture("orange").record
    obj = max(service.db.list_objects(rec.id), key=lambda o: o.bbox.area)
    service.db.update_object(obj.id, label="orange", attributes={})  # like an old, minimal record
    plan = Composer(service).plan("one orange")
    assert plan.is_complete and plan.layers[0].mask_used in ("key", "grabcut", "rembg", "none")
