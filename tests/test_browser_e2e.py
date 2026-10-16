"""Real-browser end-to-end tests (Playwright + Chromium).

The headless Streamlit test runner does not reproduce how a real browser keeps
widget values, so a defect slipped through it: a label changed with "Use" /
"Apply" was written back to the old value by the next "Save changes". These
tests start a temporary app instance on a temporary catalog and drive it with
real mouse events.

Optional: skipped automatically when Playwright or its Chromium is not installed
(`pip install playwright` and `python -m playwright install chromium`).
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / ".playwright").exists():
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright"))
sync_api = pytest.importorskip("playwright.sync_api")

from imagecat.config import Settings  # noqa: E402
from imagecat.service import CatalogService  # noqa: E402
from imagecat.ui.editor import form_key  # noqa: E402
from tests.conftest import fixture_png  # noqa: E402

pytestmark = pytest.mark.browser


def test_form_key_changes_with_every_saved_edit(service, upload_fixture):
    rec = upload_fixture("shapes").record
    obj = service.db.list_objects(rec.id)[0]
    k0 = form_key(obj)
    assert form_key(service.db.get_object(obj.id)) == k0          # no change -> same widgets
    service.update_object(obj.id, label="ball")
    k1 = form_key(service.db.get_object(obj.id))
    service.update_object(obj.id, bbox=(1, 1, 20, 20))
    k2 = form_key(service.db.get_object(obj.id))
    assert len({k0, k1, k2}) == 3                                 # every saved edit -> fresh widgets


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    """Temporary app instance on a temporary catalog containing the `shapes` fixture."""
    data = tmp_path_factory.mktemp("browser_e2e") / "data"
    env = {**os.environ, "IMAGECAT_DATA_DIR": str(data), "IMAGECAT_DISABLE_YOLO": "1",
           "IMAGECAT_DISABLE_ILLUSTRATION_MODEL": "1", "PYTHONIOENCODING": "utf-8"}
    settings = Settings()
    settings.data_dir = data
    svc = CatalogService(settings, yolo=None, illustration=None)
    rec = svc.upload(fixture_png("shapes"), "shapes.png").record
    circle = next(o for o in svc.db.list_objects(rec.id) if o.label == "circle")
    # Give the circle an alternative label, as the learned detectors would.
    svc.db.update_object(circle.id, attributes={
        **circle.attributes, "alternative_labels": [{"label": "ball", "source": "yolo", "confidence": 0.46}]})

    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://localhost:{port}"
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"{base}/_stcore/health", timeout=2).read() == b"ok":
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.skip("temporary Streamlit app did not start")
    yield {"base": base, "svc": svc, "image": rec, "circle": circle}
    proc.terminate()
    proc.wait(timeout=30)


@pytest.fixture(scope="module")
def page():
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # browser binaries missing
            pytest.skip(f"Chromium for Playwright not available: {exc}")
        pg = browser.new_context(viewport={"width": 1500, "height": 1100}).new_page()
        pg.set_default_timeout(60_000)
        yield pg
        browser.close()


def _idle(page):
    page.wait_for_timeout(400)
    for _ in range(240):
        busy = (page.locator('[data-testid="stToolbar"]').get_by_role("button", name="Stop").count()
                + page.locator('[data-testid="stSpinner"]').count())
        if not busy:
            page.wait_for_timeout(700)
            if not page.locator('[data-testid="stToolbar"]').get_by_role("button", name="Stop").count():
                return
        page.wait_for_timeout(250)


def _stable_layout(page):
    """Wait until the canvas stops moving (e.g. a success message appearing or disappearing)."""
    last = None
    for _ in range(40):
        box = page.locator("canvas.bbe-canvas").first.bounding_box()
        if box == last:
            return
        last = box
        page.wait_for_timeout(500)


def _canvas_point(page, x, y, width=640):
    box = page.locator("canvas.bbe-canvas").first.bounding_box()
    s = box["width"] / width
    return box["x"] + x * s, box["y"] + y * s


def test_relabel_with_use_survives_move_and_save(app, page):
    svc, c = app["svc"], app["circle"]
    page.goto(f"{app['base']}/editor")
    page.wait_for_selector("canvas.bbe-canvas")
    _idle(page)
    page.wait_for_timeout(800)
    cx, cy = c.x + c.width / 2, c.y + c.height / 2
    page.mouse.click(*_canvas_point(page, cx, cy))                  # select the circle
    _idle(page)
    page.get_by_role("button", name="Use", exact=True).first.click()  # relabel -> "ball"
    _idle(page)
    assert svc.db.get_object(c.id).label == "ball"
    assert page.get_by_label("Label", exact=True).first.input_value() == "ball"  # was "circle" before the fix
    _stable_layout(page)
    sx, sy = _canvas_point(page, cx, cy)
    ex, ey = _canvas_point(page, cx + 20, cy)
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(ex, ey, steps=10)                              # real drag on the canvas
    page.mouse.up()
    _idle(page)
    page.get_by_role("button", name="Save changes").click()
    _idle(page)
    saved = svc.db.get_object(c.id)
    assert saved.label == "ball"                                    # relabel kept
    assert abs(saved.x - (c.x + 20)) <= 2                            # and the moved box saved
