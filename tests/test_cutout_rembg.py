"""Learned background removal (rembg) and the 'auto' method's choices."""
import numpy as np
import pytest

from imagecat.composer import cutout
from imagecat.composer.cutout import extract_cutout, rembg_status

needs_rembg = pytest.mark.skipif(not rembg_status()[0], reason="rembg or its model is not available")


@pytest.fixture
def apple(service, upload_fixture):
    rec = upload_fixture("apple").record
    blob = next(o for o in service.db.list_objects(rec.id) if o.source == "opencv_foreground")
    manual = service.add_object(rec.id, "apple (hand drawn box)", blob.bbox.as_tuple())
    return service, rec, blob, manual


def test_auto_keeps_exact_opencv_outline(apple):
    service, rec, blob, _ = apple
    _, used, _ = extract_cutout(service.catalog.path_for(rec), blob, "auto")
    assert used == "polygon"  # pixel-exact contour on flat art beats any learned mask


def test_auto_never_downloads(apple, monkeypatch):
    service, rec, _, manual = apple
    monkeypatch.setattr(cutout, "rembg_status", lambda *a, **k: (False, "not downloaded"))
    _, used, _ = extract_cutout(service.catalog.path_for(rec), manual, "auto")
    assert used != "rembg"


@needs_rembg
def test_rembg_removes_background_of_box_without_outline(apple):
    service, rec, _, manual = apple
    cut, used, _ = extract_cutout(service.catalog.path_for(rec), manual, "rembg")
    assert used == "rembg"
    alpha = np.asarray(cut)[:, :, 3]
    assert alpha[0, 0] < 50 and alpha[-1, 0] < 50  # white corners removed
    red = np.asarray(cut)[:, :, :3][alpha > 200]
    assert len(red) and (red[:, 0] > 150).mean() > 0.6  # what remains is mostly the red apple


@needs_rembg
def test_auto_uses_rembg_when_no_exact_outline(apple):
    service, rec, _, manual = apple
    _, used, _ = extract_cutout(service.catalog.path_for(rec), manual, "auto")
    assert used == "rembg"


@needs_rembg
def test_rembg_respects_known_outline(service, upload_fixture):
    """With an outline, rembg may not include a neighbouring object in the crop."""
    rec = upload_fixture("apple_orange").record
    apple_blob = next(o for o in service.db.list_objects(rec.id) if o.source == "opencv_foreground")
    wide = service.add_object(rec.id, "apple", (apple_blob.x, apple_blob.y, 640 - apple_blob.x, apple_blob.height))
    service.db.update_object(wide.id, attributes={"polygon": apple_blob.attributes["polygon"],
                                                  "polygon_source": "yolo"})
    obj = service.db.get_object(wide.id)
    cut, used, _ = extract_cutout(service.catalog.path_for(rec), obj, "rembg")
    assert used == "rembg"
    assert cut.width < 0.5 * obj.width  # the orange on the right was excluded
