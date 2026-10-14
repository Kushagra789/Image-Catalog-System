import json

import pandas as pd
import streamlit as st

from ..visualization import draw_objects
from .common import (
    display_size,
    CONFIDENCE_NOTE,
    get_service,
    image_option_label,
    load_image,
    objects_dataframe,
    select_image,
    status_badge,
    thumbnail,
)


def page() -> None:
    svc = get_service()
    st.title("Metadata Viewer")
    rec = select_image("meta_image")
    if rec is None:
        return
    meta = svc.image_metadata(rec.id)
    objects = svc.db.list_objects(rec.id, include_deleted=True)

    left, right = st.columns([2, 3])
    with left:
        img = load_image(rec)
        if img is not None:
            st.image(display_size(draw_objects(img, [o for o in objects if not o.is_deleted])), width="stretch")
        st.download_button("Download metadata (JSON)", json.dumps(meta, indent=2, default=str),
                           file_name=f"image_{rec.id}_metadata.json", mime="application/json")
    with right:
        status_badge(rec.detection_status)
        fields = {
            "Image ID": rec.id, "Name": rec.name, "Display name": rec.display_name, "Stored filename": rec.filename,
            "Width": rec.width, "Height": rec.height, "Added (UTC)": rec.created_at,
            "Detection status": rec.detection_status, "Detection message": rec.detection_message or "",
            "Detected at (UTC)": rec.detected_at or "", "SHA-256": rec.file_hash or "",
            "Composite": "yes" if rec.is_composite else "no",
        }
        st.dataframe(pd.DataFrame({"field": list(fields), "value": [str(v) for v in fields.values()]}),
                     hide_index=True)
        if rec.dominant_colors:
            st.caption("Dominant colours: " + ", ".join(
                f"{c['color']} {c['fraction']:.0%}" for c in rec.dominant_colors))

    if rec.is_composite and rec.source_asset_ids:
        st.subheader("Provenance (source assets)")
        ids = rec.source_asset_ids.get("image_ids", [])
        cols = st.columns(max(1, min(6, len(ids))))
        for col, sid in zip(cols, ids):
            src = svc.db.get_image(sid)
            with col:
                if src is None:
                    st.warning(f"Source image #{sid} has since been deleted.")
                else:
                    t = thumbnail(src, 200)
                    if t is not None:
                        st.image(t)
                    st.caption(image_option_label(src))
        st.caption(f"Source object IDs: {rec.source_asset_ids.get('object_ids')}")
        with st.expander("Composition metadata", expanded=True):
            st.json(rec.composition_metadata)

    st.subheader(f"Objects ({len(objects)}, including removed)")
    st.caption(CONFIDENCE_NOTE)
    st.dataframe(objects_dataframe(objects), hide_index=True)
    if objects:
        obj = st.selectbox("Inspect object attributes", objects,
                           format_func=lambda o: f"#{o.id} {o.label} ({o.source})")
        attrs = dict(obj.attributes)
        if "polygon" in attrs:
            attrs["polygon"] = f"<{len(attrs['polygon'])} points>"
        st.json({**obj.to_dict(), "attributes": attrs})

    runs = meta["detection_runs"]
    st.subheader(f"Detection runs ({len(runs)})")
    if runs:
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("run_at", "mode", "status", "num_objects", "message")}
                                   for r in runs]), hide_index=True)

    st.divider()
    if st.button("Prepare full catalog export"):
        export = [svc.image_metadata(r.id) for r in svc.catalog.list_images()]
        st.download_button("Download catalog.json", json.dumps(export, indent=2, default=str),
                           file_name="catalog.json", mime="application/json")
