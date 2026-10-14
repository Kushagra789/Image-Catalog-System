"""Shared Streamlit helpers."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from ..composer import Composer
from ..imaging import load_rgb
from ..models import ImageRecord, ObjectRecord
from ..service import CatalogService
from ..visualization import SOURCE_DESCRIPTIONS

STATUS_BADGE = {
    "success": ("green", "detected"),
    "empty": ("orange", "nothing detected"),
    "failed": ("red", "detection failed"),
    "pending": ("gray", "not run"),
    "skipped": ("violet", "composite"),
}


@st.cache_resource(show_spinner="Starting catalog service...")
def get_service() -> CatalogService:
    return CatalogService()


def get_composer() -> Composer:
    return Composer(get_service())


def image_option_label(rec: ImageRecord) -> str:
    return f"#{rec.id} · {rec.display_name}" + (" (composite)" if rec.is_composite else "")


def status_badge(status: str) -> None:
    color, text = STATUS_BADGE.get(status, ("gray", status))
    st.badge(text, color=color)  # type: ignore[arg-type]


def load_image(rec: ImageRecord) -> Image.Image | None:
    path = get_service().catalog.path_for(rec)
    try:
        return load_rgb(path)
    except Exception as exc:
        st.error(f"Image file for #{rec.id} could not be opened: {exc}")
        return None


DISPLAY_MAX_SIDE = 1600


def display_size(image: Image.Image) -> Image.Image:
    """Shrink an (already annotated) image for the browser; boxes are drawn first
    at full resolution, so their positions stay exact."""
    if max(image.size) > DISPLAY_MAX_SIDE:
        image = image.copy()
        image.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))
    return image


@st.cache_data(show_spinner=False, max_entries=512)
def _cached_thumbnail(path: str, mtime: float, size: int) -> Image.Image:
    img = load_rgb(path)
    img.thumbnail((size, size))
    return img


def thumbnail(rec: ImageRecord, size: int = 320) -> Image.Image | None:
    """Small preview, cached per file version (large photos are decoded only once)."""
    path = get_service().catalog.path_for(rec)
    try:
        return _cached_thumbnail(str(path), path.stat().st_mtime, size)
    except Exception as exc:
        st.error(f"Image file for #{rec.id} could not be opened: {exc}")
        return None


def select_image(key: str, label: str = "Image", include_composites: bool = True) -> ImageRecord | None:
    """Image picker that remembers the selection across pages."""
    records = [r for r in get_service().catalog.list_images() if include_composites or not r.is_composite]
    if not records:
        st.info("The catalog is empty. Add images on the **Upload Images** page.")
        return None
    ids = [r.id for r in records]
    shared = st.session_state.get("selected_image_id")
    index = ids.index(shared) if shared in ids else 0
    chosen = st.selectbox(label, records, index=index, format_func=image_option_label, key=key)
    st.session_state["selected_image_id"] = chosen.id
    # After st.rerun() the selectbox can hand back the record object from the
    # previous run (stale status/message); always return the fresh database row.
    return next((r for r in records if r.id == chosen.id), chosen)


def fmt_conf(conf: float | None) -> str:
    return "—" if conf is None else f"{conf:.3f}"


def objects_dataframe(objects: list[ObjectRecord]) -> pd.DataFrame:
    rows = []
    for o in objects:
        status = []
        if o.origin == "manual":
            status.append("manual")
        if o.is_edited:
            status.append("edited")
        if o.is_deleted:
            status.append("deleted")
        if o.attributes.get("occluded"):
            status.append("occluded")
        if o.attributes.get("part_of"):
            status.append(f"part of {o.attributes['part_of']['label']}")
        if o.attributes.get("merged_from"):
            status.append(f"merged {len(o.attributes['merged_from'])}")
        if o.attributes.get("suggested_labels"):
            status.append("unverified suggestion: " + o.attributes["suggested_labels"][0]["label"])
        if o.attributes.get("domain_warning"):
            status.append("verify (illustration)")
        rows.append({
            "id": o.id,
            "label": o.label,
            "color": o.color or "",
            "confidence": fmt_conf(o.confidence),
            "x": o.x, "y": o.y, "width": o.width, "height": o.height,
            "source": o.source,
            "flags": ", ".join(status),
        })
    return pd.DataFrame(rows, columns=["id", "label", "color", "confidence", "x", "y", "width", "height",
                                       "source", "flags"])


def source_legend() -> None:
    from ..visualization import SOURCE_COLORS

    parts = []
    for src, desc in SOURCE_DESCRIPTIONS.items():
        r, g, b = SOURCE_COLORS[src]
        parts.append(f"<span style='display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});"
                     f"border-radius:2px;margin-right:4px;vertical-align:middle'></span>{desc}")
    st.caption(" &nbsp; ".join(parts), unsafe_allow_html=True)


CONFIDENCE_NOTE = (
    "Confidence is shown only when a model produced one (YOLO). OpenCV shape/colour analysis and "
    "composite layers have **no** calibrated confidence, so it is shown as “—” instead of a made-up number."
)
