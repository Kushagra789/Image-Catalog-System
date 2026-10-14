"""Headless smoke tests of every Streamlit page using streamlit.testing.AppTest."""
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from imagecat.config import Settings  # noqa: E402
from imagecat.service import CatalogService  # noqa: E402
from tests.conftest import fixture_png  # noqa: E402

PAGES = ["dashboard", "upload", "catalog_page", "detection_page", "editor", "composer_page", "metadata"]


def _script(module_name):
    import importlib

    importlib.import_module(f"imagecat.ui.{module_name}").page()


def _app(module_name) -> AppTest:
    at = AppTest.from_function(_script, args=(module_name,), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def populated(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGECAT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGECAT_DISABLE_YOLO", "1")
    monkeypatch.setenv("IMAGECAT_DISABLE_ILLUSTRATION_MODEL", "1")
    st.cache_resource.clear()
    svc = CatalogService(Settings())
    apple = svc.upload(fixture_png("apple"), "apple.png").record
    orange = svc.upload(fixture_png("orange"), "orange.png").record
    for rec, label, color in ((apple, "apple", "red"), (orange, "orange", "orange")):
        obj = max(svc.db.list_objects(rec.id), key=lambda o: o.bbox.area)
        svc.update_object(obj.id, label=label, color=color)
    yield svc
    st.cache_resource.clear()


@pytest.mark.parametrize("module_name", PAGES)
def test_page_renders_with_data(populated, module_name):
    _app(module_name)


@pytest.mark.parametrize("module_name", PAGES)
def test_page_renders_when_empty(tmp_path, monkeypatch, module_name):
    monkeypatch.setenv("IMAGECAT_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("IMAGECAT_DISABLE_YOLO", "1")
    monkeypatch.setenv("IMAGECAT_DISABLE_ILLUSTRATION_MODEL", "1")
    st.cache_resource.clear()
    _app(module_name)
    st.cache_resource.clear()


def test_ui_add_fixtures(populated):
    at = _app("upload")
    _button(at, "Add selected fixtures to catalog").click().run()
    assert not at.exception
    names = {r.name for r in populated.catalog.list_images()}
    assert {"shapes", "overlapping_shapes", "cartoon_boy", "blank"} <= names
    # apple/orange already existed: reported as duplicates rather than re-added
    assert any("identical" in w.value for w in at.warning)


def test_ui_manual_add_object(populated):
    rec = populated.catalog.list_images()[0]
    at = _app("editor")
    at.segmented_control[0].set_value("Add object").run()
    at.text_input(key=f"add{rec.id}_label").input("Stem").run()
    _button(at, "Add object").click().run()
    assert not at.exception
    assert any(o.label == "stem" and o.source == "manual" for o in populated.db.list_objects(rec.id))


def test_ui_replace_all_needs_confirmation(populated):
    at = _app("detection_page")
    at.radio[0].set_value("replace_all").run()
    assert _button(at, "Run detection").disabled  # blocked until the user ticks the confirmation box
    at.checkbox[-1].check().run()
    assert not _button(at, "Run detection").disabled


def test_ui_composite_plan_and_save(populated):
    at = _app("composer_page")
    at.text_area[0].input("Create an image containing one red apple and one orange.").run()
    _button(at, "Find catalog objects & plan").click().run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    _button(at, "Save composite").click().run()
    assert not at.exception
    composites = [r for r in populated.catalog.list_images() if r.is_composite]
    assert len(composites) == 1
    assert len(composites[0].source_asset_ids["object_ids"]) == 2


def test_ui_composite_reports_missing(populated):
    at = _app("composer_page")
    at.text_area[0].input("one banana and one red apple").run()
    _button(at, "Find catalog objects & plan").click().run()
    assert any("banana" in e.value for e in at.error)
    assert _button(at, "Save composite").disabled
