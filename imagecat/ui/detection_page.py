import pandas as pd
import streamlit as st

from ..service import ConfirmationRequired, RERUN_KEEP_MANUAL, RERUN_REPLACE_ALL
from ..visualization import draw_objects
from .common import (
    display_size,
    CONFIDENCE_NOTE,
    get_service,
    load_image,
    objects_dataframe,
    select_image,
    source_legend,
    status_badge,
)


def page() -> None:
    svc = get_service()
    st.title("Object Detection")
    rec = select_image("det_image")
    if rec is None:
        return

    objects = svc.db.list_objects(rec.id)
    sources = sorted({o.source for o in objects})

    left, right = st.columns([3, 2])
    with left:
        c1, c2 = st.columns(2)
        show_labels = c1.toggle("Show labels", value=True)
        hide_parts = c1.toggle("Hide parts of larger objects", value=False,
                               help="Hides shapes/regions that lie inside a detected whole object (e.g. the "
                                    "head and shirt of a cartoon character). Display only; nothing is deleted.")
        chosen_sources = c2.multiselect("Sources", sources, default=sources) if sources else []
        shown = [o for o in objects if o.source in chosen_sources
                 and not (hide_parts and o.attributes.get("part_of"))]
        img = load_image(rec)
        if img is not None:
            st.image(display_size(draw_objects(img, shown, show_labels=show_labels)), width="stretch")
        source_legend()

    with right:
        st.markdown(f"**#{rec.id} · {rec.display_name}** ({rec.width}×{rec.height})")
        status_badge(rec.detection_status)
        if rec.detection_message:
            box = {"success": st.success, "empty": st.warning, "failed": st.error}.get(rec.detection_status, st.info)
            box(rec.detection_message)
        runs = svc.db.list_detection_runs(rec.id)
        if runs:
            st.caption("Engines on the latest run:")
            for r in runs[0]["details"].get("engines", []):
                st.caption(f"{'✅' if r['ok'] else '⚠️'} {r['engine']}: {r['message']}")
            if runs[0]["details"].get("image_kind"):
                st.caption(f"Image type: **{runs[0]['details']['image_kind']}**")

        corrections = svc.corrections(rec.id)
        with st.container(border=True):
            st.subheader("Run detection again")
            if rec.is_composite:
                st.info("This is a composite: its objects come from the composition itself and are protected.")
            mode = st.radio(
                "What should happen to existing objects?",
                [RERUN_KEEP_MANUAL, RERUN_REPLACE_ALL],
                format_func=lambda m: {
                    RERUN_KEEP_MANUAL: "Keep my corrections (recommended): only refresh automatic detections",
                    RERUN_REPLACE_ALL: "Replace everything, including manual corrections",
                }[m],
            )
            c1, c2 = st.columns(2)
            use_yolo = c1.checkbox("Use YOLO", value=True)
            use_cv = c2.checkbox("Use OpenCV", value=True)
            use_ill = c1.checkbox("Use illustration model (YOLOE)", value=True,
                                  help="Only runs on flat illustrations such as cartoons and clip-art.")
            conf = st.slider("YOLO confidence threshold", 0.05, 0.9, 0.25, 0.05, disabled=not use_yolo)
            confirmed = False
            protected = len(corrections) + sum(1 for o in objects if o.source == "composite")
            if mode == RERUN_REPLACE_ALL and protected:
                st.warning(f"{protected} protected object(s) (manual additions, edits, deletions or composite "
                           "layers) will be permanently discarded.")
                confirmed = st.checkbox(f"I understand: discard {protected} correction(s)")
            disabled = mode == RERUN_REPLACE_ALL and protected > 0 and not confirmed
            if st.button("Run detection", type="primary", disabled=disabled):
                with st.spinner("Detecting objects (the first run loads the models, ~20 s)..."):
                    try:
                        result = svc.run_detection(rec.id, mode=mode, confirm_replace=confirmed,
                                                   use_yolo=use_yolo, use_opencv=use_cv, yolo_confidence=conf,
                                                   use_illustration_model=use_ill)
                        st.session_state["det_flash"] = (result.status, result.message)
                    except ConfirmationRequired as exc:
                        st.session_state["det_flash"] = ("failed", str(exc))
                    except Exception as exc:  # existing objects are untouched on failure
                        st.session_state["det_flash"] = ("failed", f"Detection failed unexpectedly: {exc}. "
                                                                   "Existing objects were not changed.")
                st.rerun()
        flash = st.session_state.pop("det_flash", None)
        if flash:
            {"success": st.success, "empty": st.warning}.get(flash[0], st.error)(flash[1])

    st.subheader(f"Objects ({len(objects)})")
    st.caption(CONFIDENCE_NOTE)
    if objects:
        st.dataframe(objects_dataframe(objects), hide_index=True)
    else:
        st.info("No objects. Re-run detection, or add objects in the **Manual Editor**.")

    if runs:
        with st.expander(f"Detection history ({len(runs)} run(s))"):
            st.dataframe(pd.DataFrame([{k: r[k] for k in ("run_at", "mode", "status", "num_objects", "message")}
                                       for r in runs]), hide_index=True)
