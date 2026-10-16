"""Regression tests for the defects found in the independent final code review."""
import io
import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from imagecat.catalog import CatalogError  # noqa: E402
from imagecat.composer import Composer  # noqa: E402
from imagecat.config import Settings  # noqa: E402
from imagecat.service import RERUN_KEEP_MANUAL, CatalogService  # noqa: E402
from tests.conftest import fixture_png  # noqa: E402
from imagecat.ui.editor import form_key  # noqa: E402


# ------------------------------------------------------------------ helpers
def _editor_page(event):
    """Manual Editor with the browser component replaced by a stub."""
    import types

    import streamlit as st_
    import imagecat.ui.bbox_editor as bbe
    from imagecat.ui import editor

    def fake_component():
        def mount(**kwargs):
            st_.session_state["_last_payload"] = kwargs["data"]
            return types.SimpleNamespace(box=st_.session_state.get("_fake_event"))
        return mount

    bbe._component = fake_component
    if event is not None and "_fake_event" not in st_.session_state:
        st_.session_state["_fake_event"] = event
    editor.page()


@pytest.fixture
def app_svc(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGECAT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGECAT_DISABLE_YOLO", "1")
    monkeypatch.setenv("IMAGECAT_DISABLE_ILLUSTRATION_MODEL", "1")
    st.cache_resource.clear()
    svc = CatalogService(Settings())
    svc.upload(fixture_png("shapes"), "shapes.png")
    yield svc
    st.cache_resource.clear()


def _button(at, label):
    return next(b for b in at.button if b.label == label)


# --------------------------------------- HIGH: relabel undone by next Save
def test_relabel_by_button_is_not_reverted_by_next_save(app_svc):
    rec = app_svc.catalog.list_images()[0]
    target = app_svc.db.list_objects(rec.id)[0]
    app_svc.db.update_object(target.id, confidence=0.9, attributes={
        **target.attributes, "alternative_labels": [{"label": "ball", "source": "yolo", "confidence": 0.4}]})
    at = AppTest.from_function(_editor_page, args=(None,), default_timeout=120)
    at.run()
    assert at.text_input(key=f"{form_key(target)}_label").value == target.label
    _button(at, "Use").click().run()
    assert app_svc.db.get_object(target.id).label == "ball"
    _button(at, "Save changes").click().run()  # e.g. to save a box tweak
    assert not at.exception
    saved = app_svc.db.get_object(target.id)
    assert saved.label == "ball"            # was reverted to the old label before the fix
    assert saved.confidence is None         # the old model confidence must not come back


def test_save_then_save_again_keeps_cleared_confidence(app_svc):
    rec = app_svc.catalog.list_images()[0]
    target = app_svc.db.list_objects(rec.id)[0]
    app_svc.db.update_object(target.id, confidence=0.8)
    at = AppTest.from_function(_editor_page, args=(None,), default_timeout=120)
    at.run()
    at.text_input(key=f"{form_key(target)}_label").input("relabelled").run()
    at.checkbox(key=f"{form_key(target)}_has").uncheck().run()
    _button(at, "Save changes").click().run()
    _button(at, "Save changes").click().run()
    saved = app_svc.db.get_object(target.id)
    assert saved.label == "relabelled" and saved.confidence is None


# ------------------------------------------ MEDIUM: canvas snaps back
def test_canvas_payload_follows_the_latest_gesture(app_svc):
    rec = app_svc.catalog.list_images()[0]
    first = app_svc.db.list_objects(rec.id)[0]
    event = {"id": first.id, "x": 21, "y": 31, "width": 101, "height": 91, "action": "move", "seq": 5}
    at = AppTest.from_function(_editor_page, args=(event,), default_timeout=120)
    at.run()
    box = next(b for b in at.session_state["_last_payload"]["boxes"] if b["id"] == first.id)
    assert (box["x"], box["y"], box["width"], box["height"]) == (21, 31, 101, 91)


# ------------------------------- MEDIUM: moved detection comes back on re-run
def test_moved_detection_does_not_reappear_at_original_place(service, upload_fixture):
    rec = upload_fixture("shapes").record
    rect = next(o for o in service.db.list_objects(rec.id) if o.label == "rectangle")
    service.update_object(rect.id, bbox=(560, 400, 50, 50))
    n_before = len(service.db.list_objects(rec.id))
    service.run_detection(rec.id, mode=RERUN_KEEP_MANUAL)
    objs = service.db.list_objects(rec.id)
    assert len(objs) == n_before
    assert not any(o.id != rect.id and o.bbox.iou(rect.bbox) > 0.5 for o in objs)


# ------------------------------------ MEDIUM: cleared layer cell crashes page
def test_cleared_layer_cell_keeps_previous_value(service, upload_fixture):
    from imagecat.ui.composer_page import apply_layer_edits

    rec = upload_fixture("orange").record
    obj = max(service.db.list_objects(rec.id), key=lambda o: o.bbox.area)
    service.update_object(obj.id, label="orange")
    plan = Composer(service).plan("one orange")
    layer = plan.layers[0]
    before = (layer.x, layer.y, layer.scale, layer.z)
    apply_layer_edits(plan, pd.DataFrame([{"layer": 0, "x": math.nan, "y": 7, "scale": math.nan, "z": None}]))
    assert (layer.x, layer.y, layer.scale, layer.z) == (before[0], 7, before[2], before[3])


# -------------------------------------------- MEDIUM: cut-out memory use
def test_flatten_avoids_copies_for_opaque_and_is_exact_for_transparent():
    from imagecat.composer.cutout import _flatten

    opaque = np.dstack([np.full((4, 4, 3), 100, np.uint8), np.full((4, 4), 255, np.uint8)])
    assert np.shares_memory(_flatten(opaque), opaque)  # a view, not a copy
    half = np.dstack([np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 128, np.uint8)])
    assert np.all(np.abs(_flatten(half).astype(int) - 127) <= 1)  # black at 50 % on white = grey


# ------------------------------------ LOW-MEDIUM: stale status after rerun
def _detection_status_probe():
    import streamlit as st_
    from imagecat.ui.common import select_image

    rec = select_image("probe")
    st_.session_state["seen_status"] = rec.detection_status


def test_select_image_returns_fresh_record(app_svc):
    rec = app_svc.catalog.list_images()[0]
    at = AppTest.from_function(_detection_status_probe, default_timeout=120)
    at.run()
    assert at.session_state["seen_status"] == "success"
    app_svc.db.update_image(rec.id, detection_status="failed")
    at.run()
    assert at.session_state["seen_status"] == "failed"


# --------------------------- LOW-MEDIUM: 16-bit PNG stored almost white
def test_16bit_greyscale_png_is_scaled_not_clipped(service):
    grad = np.tile(np.linspace(0, 65535, 256, dtype=np.uint16), (64, 1))
    buf = io.BytesIO()
    Image.fromarray(grad).save(buf, format="PNG")
    rec = service.upload(buf.getvalue(), "depth16.png", auto_detect=False).record
    with Image.open(service.catalog.path_for(rec)) as stored:
        arr = np.asarray(stored.convert("L"))
    assert (arr >= 250).mean() < 0.05            # was ~99.6 % white before the fix
    assert arr[:, 0].mean() < 5 and arr[:, -1].mean() > 250
    assert np.all(np.diff(arr[0].astype(int)) >= 0)  # gradient preserved


# ----------------------------------------- LOW: misleading missing reason
def test_missing_reason_mentions_objects_used_by_earlier_items(service, upload_fixture):
    rec = upload_fixture("orange").record
    obj = max(service.db.list_objects(rec.id), key=lambda o: o.bbox.area)
    service.update_object(obj.id, label="dot")
    plan = Composer(service).plan("a dot and a dot")
    assert len(plan.layers) == 1 and len(plan.missing) == 1
    reason = plan.missing[0].reason
    assert "No object labelled" not in reason and "already used" in reason


# ------------------------------------- review "uncertain" items, confirmed
def test_locked_file_on_delete_deletes_nothing(service, upload_fixture, monkeypatch):
    rec = upload_fixture("shapes").record

    def locked(self, missing_ok=False):
        raise PermissionError("file is being used by another process")

    monkeypatch.setattr(pathlib.Path, "unlink", locked)
    with pytest.raises(CatalogError, match="nothing was deleted"):
        service.catalog.delete(rec.id)
    assert service.db.get_image(rec.id) is not None
    assert service.db.list_objects(rec.id)


def test_label_search_treats_wildcards_literally(service, upload_fixture):
    rec = upload_fixture("shapes").record
    assert service.db.image_ids_with_label("_") == set()
    assert service.db.image_ids_with_label("%") == set()
    service.update_object(service.db.list_objects(rec.id)[0].id, label="red_ball")
    assert service.db.image_ids_with_label("_") == {rec.id}
