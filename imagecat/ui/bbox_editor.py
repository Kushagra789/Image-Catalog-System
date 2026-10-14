"""Mouse-driven bounding-box editor built on Streamlit Components v2.

The JavaScript (static/bbox_editor.js) draws the image and boxes on a canvas
and reports each finished gesture (select / move / resize / draw) in image
pixel coordinates. Python applies a gesture only once (tracked by ``seq``).
The numeric inputs on the Manual Editor page stay available for exact values
and as a fallback when custom components are unavailable.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from ..models import ObjectRecord
from ..visualization import SOURCE_COLORS

_JS = (Path(__file__).parent / "static" / "bbox_editor.js").read_text(encoding="utf-8")
_CSS = """
.bbe-root { width: 100%; }
.bbe-canvas { display: block; max-width: 100%; height: auto; touch-action: none; border-radius: 4px; }
.bbe-hint { font-size: 0.8rem; opacity: 0.75; margin-top: 4px; }
"""
_PREVIEW_MAX_SIDE = 1100

AVAILABLE = hasattr(getattr(st, "components", None), "v2")  # Components v2 needs Streamlit >= 1.51


def _component():
    # Registered on every render: the registry belongs to the running Streamlit
    # runtime (not to this module), and re-registering an identical definition
    # is a silent no-op.
    return st.components.v2.component("imagecat_bbox_editor", js=_JS, css=_CSS)


@st.cache_data(show_spinner=False, max_entries=32)
def _encode_preview(path: str, mtime: float) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def editor_boxes(objects: list[ObjectRecord], override: dict[Any, tuple[int, int, int, int]] | None = None,
                 extra: list[dict] | None = None) -> list[dict]:
    """Convert objects (optionally with unsaved form values) into editor boxes."""
    override = override or {}
    out = []
    for o in objects:
        x, y, w, h = override.get(o.id, o.bbox.as_tuple())
        out.append({"id": o.id, "x": x, "y": y, "width": w, "height": h, "label": f"#{o.id} {o.label}",
                    "color": _hex(SOURCE_COLORS.get(o.source, (230, 0, 60)))})
    out.extend(extra or [])
    return out


def bbox_editor(image_path: Path, width: int, height: int, boxes: list[dict], selected: Any,
                mode: str, key: str) -> dict | None:
    """Render the editor. Returns a *new* gesture dict, or None."""
    if not AVAILABLE:
        return None
    payload = {
        "image": _encode_preview(str(image_path), image_path.stat().st_mtime),
        "width": width, "height": height, "boxes": boxes, "selected": selected, "mode": mode,
    }
    # The token changes whenever Python-side state changes, telling the
    # browser to drop its local copy and redraw from these values.
    payload["token"] = hashlib.sha1(json.dumps([boxes, selected, mode], sort_keys=True,
                                               default=str).encode()).hexdigest()
    result = _component()(key=key, data=payload, default={"box": None}, on_box_change=lambda: None)
    event = getattr(result, "box", None)
    if not event or not isinstance(event, dict):
        return None
    seen_key = f"{key}__seq"
    if st.session_state.get(seen_key) == event.get("seq"):
        return None
    st.session_state[seen_key] = event.get("seq")
    return event
