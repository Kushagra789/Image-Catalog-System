"""Drawing bounding boxes and labels for display."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .models import (
    ObjectRecord,
    SOURCE_COLOR_REGION,
    SOURCE_COMPOSITE,
    SOURCE_FOREGROUND,
    SOURCE_MANUAL,
    SOURCE_SHAPE,
    SOURCE_YOLO,
    SOURCE_YOLOE,
)

SOURCE_COLORS = {
    SOURCE_YOLO: (0, 170, 255),
    SOURCE_YOLOE: (0, 90, 220),
    SOURCE_SHAPE: (255, 140, 0),
    SOURCE_COLOR_REGION: (200, 0, 200),
    SOURCE_FOREGROUND: (0, 180, 120),
    SOURCE_MANUAL: (230, 0, 60),
    SOURCE_COMPOSITE: (90, 90, 90),
}
SOURCE_DESCRIPTIONS = {
    SOURCE_YOLO: "YOLO (learned, COCO classes)",
    SOURCE_YOLOE: "YOLOE (learned, illustration vocabulary)",
    SOURCE_SHAPE: "OpenCV geometric shape",
    SOURCE_COLOR_REGION: "OpenCV colour region",
    SOURCE_FOREGROUND: "OpenCV multi-colour foreground blob",
    SOURCE_MANUAL: "Added manually",
    SOURCE_COMPOSITE: "Placed by compositor",
}


def object_caption(obj: ObjectRecord) -> str:
    parts = [f"#{obj.id} {obj.label}"]
    if obj.color:
        parts.append(f"({obj.color})")
    if obj.confidence is not None:
        parts.append(f"{obj.confidence:.2f}")
    if obj.is_edited:
        parts.append("✎")
    return " ".join(parts)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def draw_objects(image: Image.Image, objects: list[ObjectRecord], highlight_id: int | None = None,
                 show_labels: bool = True) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    scale = max(img.size) / 640
    width = max(2, round(2 * scale))
    font = _font(max(11, round(13 * scale)))
    tags: list[tuple[tuple[int, int, int, int], tuple, str]] = []
    placed: list[tuple[int, int, int, int]] = []
    for obj in objects:
        color = SOURCE_COLORS.get(obj.source, (0, 0, 0))
        w = width * 2 if obj.id == highlight_id else width
        if highlight_id is not None and obj.id != highlight_id:
            color = tuple(int(c * 0.5 + 127) for c in color)
        x0, y0 = obj.x, obj.y
        x1, y1 = obj.x + obj.width - 1, obj.y + obj.height - 1
        draw.rectangle((x0, y0, x1, y1), outline=color, width=w)
        if show_labels:
            text = object_caption(obj)
            tb = draw.textbbox((0, 0), text, font=font)
            th, tw = tb[3] - tb[1] + 4, tb[2] - tb[0] + 6
            rect = _place_tag(x0, y0, tw, th, img.size, placed)
            placed.append(rect)
            tags.append((rect, color, text))
    # Tags are drawn after all boxes so no box outline crosses a label.
    for (tx, ty, tx1, ty1), color, text in tags:
        draw.rectangle((tx, ty, tx1, ty1), fill=color)
        draw.text((tx + 3, ty + 1), text, fill=(255, 255, 255), font=font)
    return img


def _place_tag(x0: int, y0: int, tw: int, th: int, size: tuple[int, int],
               placed: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Put a label tag at the box's top-left corner (above it if there is room),
    moving it down in tag-height steps until it no longer covers an earlier tag."""
    img_w, img_h = size
    tx = max(0, min(x0, img_w - tw - 1))
    start = y0 - th if y0 - th >= 0 else y0

    def overlaps(r):
        return any(r[0] < p[2] and p[0] < r[2] and r[1] < p[3] and p[1] < r[3] for p in placed)

    ty = start
    for _ in range(40):
        if ty + th >= img_h:
            break  # no free spot below: fall back to the default position
        rect = (tx, ty, tx + tw, ty + th)
        if not overlaps(rect):
            return rect
        ty += th + 1
    return (tx, start, tx + tw, start + th)


def draw_box_preview(image: Image.Image, bbox: tuple[int, int, int, int], label: str = "") -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox
    lw = max(2, round(max(img.size) / 320))
    draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(230, 0, 60), width=lw)
    if label:
        draw.text((x + 3, max(0, y - 16)), label, fill=(230, 0, 60), font=_font(14))
    return img
