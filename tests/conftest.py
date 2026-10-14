import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imagecat import fixtures  # noqa: E402
from imagecat.config import Settings  # noqa: E402
from imagecat.service import CatalogService  # noqa: E402


def make_settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.data_dir = tmp_path / "data"
    return s


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def service(settings):
    """Service with both learned models disabled: deterministic, offline, fast."""
    return CatalogService(settings, yolo=None, illustration=None)


def fixture_png(name: str) -> bytes:
    img, _ = fixtures.generate(name)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def upload_fixture(service):
    def _upload(name: str, **kwargs):
        return service.upload(fixture_png(name), f"{name}.png", **kwargs)
    return _upload
