"""Image catalog: storing image files on disk and their records in SQLite."""
from __future__ import annotations

import hashlib
import io
import re
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .db import Database
from .models import ImageRecord, STATUS_PENDING

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_PIXELS = 40_000_000  # refuse absurdly large images rather than exhausting memory


class CatalogError(Exception):
    """Raised for user-facing catalog problems (bad file, duplicate, ...)."""


def sanitize_name(filename: str) -> str:
    stem = Path(filename).stem or "image"
    stem = re.sub(r"[^A-Za-z0-9_\- ]+", "_", stem).strip(" _")
    # Web downloads often have very long encoded names; keep names readable.
    return stem[:80].rstrip(" _-") or "image"


class ImageCatalog:
    def __init__(self, db: Database, images_dir: Path):
        self.db = db
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, record: ImageRecord) -> Path:
        return self.images_dir / record.filename

    def add_image_bytes(
        self,
        data: bytes,
        original_filename: str,
        display_name: str | None = None,
        allow_duplicates: bool = False,
    ) -> ImageRecord:
        """Validate, normalise and store an image. Returns the new record.

        Images are EXIF-rotated before saving so stored width/height always match
        the pixel grid that bounding boxes refer to.
        """
        if not data:
            raise CatalogError(f"'{original_filename}' is empty.")
        ext = Path(original_filename).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            raise CatalogError(f"Unsupported file type '{ext}'.")

        file_hash = hashlib.sha256(data).hexdigest()
        if not allow_duplicates:
            existing = self.db.find_image_by_hash(file_hash)
            if existing:
                raise CatalogError(
                    f"'{original_filename}' is identical to image #{existing.id} "
                    f"('{existing.display_name}')."
                )

        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                if im.width * im.height > MAX_PIXELS:
                    raise CatalogError(f"'{original_filename}' is too large ({im.width}x{im.height}).")
                fmt = im.format
                changed = im.getexif().get(0x0112, 1) not in (1, None)  # EXIF orientation
                image = ImageOps.exif_transpose(im) if changed else im.copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise CatalogError(f"'{original_filename}' is not a readable image ({exc}).") from exc

        token = uuid.uuid4().hex[:12]
        if fmt == "JPEG" and not changed:
            filename = f"{token}.jpg"
            (self.images_dir / filename).write_bytes(data)
        else:
            if image.mode in ("I;16", "I;16B", "I;16L", "I"):
                # 16/32-bit greyscale: plain convert() clips everything above 255 to white.
                arr = np.asarray(image, dtype=np.float64)
                hi = arr.max() if arr.max() > 0 else 1.0
                image = Image.fromarray(np.clip(arr * (255.0 / hi), 0, 255).astype(np.uint8), "L")
            # Keep transparency (useful for compositing cartoons); otherwise RGB.
            has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
            image = image.convert("RGBA" if has_alpha else "RGB")
            filename = f"{token}.png"
            image.save(self.images_dir / filename, format="PNG")

        name = sanitize_name(original_filename)
        image_id = self.db.insert_image(
            name=name,
            display_name=(display_name or "").strip() or name,
            filename=filename,
            width=image.width,
            height=image.height,
            file_hash=file_hash,
            detection_status=STATUS_PENDING,
        )
        return self.db.get_image(image_id)  # type: ignore[return-value]

    def add_image_file(self, path: Path | str, display_name: str | None = None,
                       allow_duplicates: bool = False) -> ImageRecord:
        path = Path(path)
        return self.add_image_bytes(path.read_bytes(), path.name, display_name, allow_duplicates)

    def add_pil_image(self, image: Image.Image, name: str, display_name: str | None = None,
                      **record_fields) -> ImageRecord:
        """Store an in-memory image (used for composites)."""
        filename = f"{uuid.uuid4().hex[:12]}.png"
        image.save(self.images_dir / filename, format="PNG")
        data = (self.images_dir / filename).read_bytes()
        image_id = self.db.insert_image(
            name=sanitize_name(name),
            display_name=(display_name or "").strip() or sanitize_name(name),
            filename=filename,
            width=image.width,
            height=image.height,
            file_hash=hashlib.sha256(data).hexdigest(),
            **record_fields,
        )
        return self.db.get_image(image_id)  # type: ignore[return-value]

    def list_images(self) -> list[ImageRecord]:
        return self.db.list_images()

    def get(self, image_id: int) -> ImageRecord | None:
        return self.db.get_image(image_id)

    def rename(self, image_id: int, display_name: str) -> None:
        display_name = display_name.strip()
        if not display_name:
            raise CatalogError("Display name cannot be empty.")
        self.db.update_image(image_id, display_name=display_name)

    def delete(self, image_id: int) -> None:
        """Delete the record (objects cascade) and the stored file."""
        record = self.db.get_image(image_id)
        if record is None:
            raise CatalogError(f"Image #{image_id} does not exist.")
        try:
            self.path_for(record).unlink(missing_ok=True)
        except OSError as exc:  # e.g. the file is open in another program on Windows
            raise CatalogError(f"Could not delete the file for image #{image_id} ({exc}). Close any program "
                               "using it and try again; nothing was deleted.") from exc
        self.db.delete_image(image_id)
