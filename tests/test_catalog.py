import io

import pytest
from PIL import Image

from imagecat.catalog import CatalogError
from imagecat.models import STATUS_PENDING
from tests.conftest import fixture_png


def test_upload_stores_file_and_metadata(service):
    out = service.upload(fixture_png("apple"), "apple.png", display_name="My apple", auto_detect=False)
    rec = out.record
    assert rec.id > 0
    assert rec.name == "apple"
    assert rec.display_name == "My apple"
    assert (rec.width, rec.height) == (640, 480)
    assert rec.detection_status == STATUS_PENDING
    assert rec.created_at
    assert service.catalog.path_for(rec).exists()
    assert [r.id for r in service.catalog.list_images()] == [rec.id]


def test_duplicate_upload_rejected_unless_allowed(service):
    data = fixture_png("shapes")
    service.upload(data, "shapes.png", auto_detect=False)
    with pytest.raises(CatalogError, match="identical"):
        service.upload(data, "shapes_copy.png", auto_detect=False)
    service.upload(data, "shapes_copy.png", auto_detect=False, allow_duplicates=True)
    assert len(service.catalog.list_images()) == 2


def test_invalid_files_rejected(service):
    with pytest.raises(CatalogError, match="not a readable image"):
        service.upload(b"this is not an image", "broken.png")
    with pytest.raises(CatalogError, match="empty"):
        service.upload(b"", "empty.png")
    with pytest.raises(CatalogError, match="Unsupported"):
        service.upload(b"abc", "notes.txt")
    assert service.catalog.list_images() == []


def test_exif_rotation_is_applied(service):
    img = Image.new("RGB", (300, 100), (200, 30, 30))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    rec = service.upload(buf.getvalue(), "rotated.jpg", auto_detect=False).record
    assert (rec.width, rec.height) == (100, 300)


def test_transparent_png_keeps_alpha(service):
    img = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (20, 20, 60, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rec = service.upload(buf.getvalue(), "sticker.png", auto_detect=False).record
    with Image.open(service.catalog.path_for(rec)) as stored:
        assert stored.mode == "RGBA"


def test_delete_removes_file_objects_and_runs(service, upload_fixture):
    rec = upload_fixture("shapes").record
    path = service.catalog.path_for(rec)
    assert service.db.list_objects(rec.id)
    service.catalog.delete(rec.id)
    assert not path.exists()
    assert service.db.get_image(rec.id) is None
    assert service.db.list_objects(rec.id, include_deleted=True) == []
    assert service.db.list_detection_runs(rec.id) == []
    with pytest.raises(CatalogError):
        service.catalog.delete(rec.id)


def test_rename(service):
    rec = service.upload(fixture_png("orange"), "orange.png", auto_detect=False).record
    service.catalog.rename(rec.id, "  Juicy orange ")
    assert service.db.get_image(rec.id).display_name == "Juicy orange"
    with pytest.raises(CatalogError):
        service.catalog.rename(rec.id, "   ")


def test_long_filenames_are_shortened(service):
    rec = service.upload(fixture_png("apple"), "x" * 200 + ".jpg", auto_detect=False).record
    assert len(rec.name) == 80
    assert rec.display_name == rec.name
