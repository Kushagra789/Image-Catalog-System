"""Manual Editor + mouse box editor: Python side, with simulated browser gestures."""
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from imagecat.config import Settings  # noqa: E402
from imagecat.service import CatalogService  # noqa: E402
from tests.conftest import fixture_png  # noqa: E402


def _editor_with_fake_gesture(event):
    """Run the Manual Editor page with the browser component replaced by a stub
    that reports ``event`` (what the JavaScript would send after a drag)."""
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
def svc(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGECAT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGECAT_DISABLE_YOLO", "1")
    monkeypatch.setenv("IMAGECAT_DISABLE_ILLUSTRATION_MODEL", "1")
    st.cache_resource.clear()
    service = CatalogService(Settings())
    service.upload(fixture_png("shapes"), "shapes.png")
    yield service
    st.cache_resource.clear()


def _run(event):
    at = AppTest.from_function(_editor_with_fake_gesture, args=(event,), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _num(at, key):
    return next(n for n in at.number_input if n.key == key).value


def test_payload_contains_all_objects(svc):
    at = _run(None)
    payload = at.session_state["_last_payload"]
    rec = svc.catalog.list_images()[0]
    assert payload["width"] == rec.width and payload["mode"] == "edit"
    assert {b["id"] for b in payload["boxes"]} == {o.id for o in svc.db.list_objects(rec.id)}
    assert payload["image"].startswith("data:image/jpeg;base64,")


def test_drag_moves_box_into_form_and_save_persists(svc):
    rec = svc.catalog.list_images()[0]
    first = svc.db.list_objects(rec.id)[0]
    event = {"id": first.id, "x": 20, "y": 30, "width": 111, "height": 99, "action": "move", "seq": 1}
    at = _run(event)
    k = f"e{first.id}"
    assert (_num(at, f"{k}_x"), _num(at, f"{k}_y"), _num(at, f"{k}_w"), _num(at, f"{k}_h")) == (20, 30, 111, 99)
    assert svc.db.get_object(first.id).bbox.as_tuple() == first.bbox.as_tuple()  # not saved yet
    next(b for b in at.button if b.label == "Save changes").click().run()
    assert not at.exception
    saved = svc.db.get_object(first.id)
    assert saved.bbox.as_tuple() == (20, 30, 111, 99) and saved.is_edited


def test_same_gesture_is_applied_only_once(svc):
    rec = svc.catalog.list_images()[0]
    first = svc.db.list_objects(rec.id)[0]
    at = _run({"id": first.id, "x": 20, "y": 30, "width": 111, "height": 99, "action": "move", "seq": 7})
    k = f"e{first.id}"
    next(n for n in at.number_input if n.key == f"{k}_x").set_value(55).run()  # user types afterwards
    assert _num(at, f"{k}_x") == 55  # the old gesture (same seq) does not overwrite the typed value


def test_clicking_another_box_selects_it(svc):
    rec = svc.catalog.list_images()[0]
    objs = svc.db.list_objects(rec.id)
    target = objs[2]
    at = _run({"id": target.id, **dict(zip(("x", "y", "width", "height"), target.bbox.as_tuple())),
               "action": "select", "seq": 3})
    assert at.session_state[f"edit_obj_{rec.id}"] == target.id
    assert at.session_state["_last_payload"]["selected"] == target.id


def test_drawn_box_creates_new_object(svc):
    rec = svc.catalog.list_images()[0]
    event = {"id": "new", "x": 5, "y": 6, "width": 70, "height": 80, "action": "draw", "seq": 11}
    at = AppTest.from_function(_editor_with_fake_gesture, args=(event,), default_timeout=120)
    at.session_state[f"edit_mode_{rec.id}"] = "Add object"
    at.run()
    assert not at.exception
    assert at.session_state["_last_payload"]["mode"] == "draw"
    at.text_input(key=f"add{rec.id}_label").input("hand drawn").run()
    next(b for b in at.button if b.label == "Add object").click().run()
    assert not at.exception
    new = next(o for o in svc.db.list_objects(rec.id) if o.label == "hand drawn")
    assert new.bbox.as_tuple() == (5, 6, 70, 80) and new.source == "manual"


def test_out_of_range_gesture_is_clamped(svc):
    rec = svc.catalog.list_images()[0]
    first = svc.db.list_objects(rec.id)[0]
    at = _run({"id": first.id, "x": 630, "y": 470, "width": 500, "height": 500, "action": "resize", "seq": 2})
    k = f"e{first.id}"
    x, y, w, h = (_num(at, f"{k}_{f}") for f in ("x", "y", "w", "h"))
    assert x + w <= rec.width and y + h <= rec.height
