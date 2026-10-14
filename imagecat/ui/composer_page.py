import pandas as pd
import streamlit as st

from ..colors import COLOR_NAMES
from ..composer import LAYOUTS, MASK_METHODS, CompositionError, MissingAssetsError, auto_layout
from ..composer.cutout import rembg_status
from .common import get_composer

EXAMPLES = [
    "Create an image containing one red apple and one orange.",
    "Place an apple on the left, an orange on the right, and a cartoon boy in the center.",
    "A big apple at the top left and a small orange to the right of the apple",
    "A cartoon girl in the foreground and a cartoon boy behind the girl",
    "A cartoon boy and a cartoon girl side by side on a blue background",
    "Overlapping red circle, blue square and green triangle",
]
MASK_HELP = {
    "auto": "Best available: alpha > exact outline > rembg (if downloaded) > outline > key > GrabCut.",
    "alpha": "Use the PNG's own transparency.",
    "polygon": "Use the stored outline (YOLO-seg mask / OpenCV contour).",
    "rembg": "Learned background removal (U2Net, ~176 MB, downloaded on first use).",
    "key": "Remove pixels similar to the box border (flat backgrounds).",
    "grabcut": "OpenCV GrabCut from the bounding box (photos).",
    "none": "Plain rectangular crop (no background removal).",
}


def _plan_controls():
    instruction = st.text_area("Instruction", value=st.session_state.get("comp_instruction", EXAMPLES[0]),
                               height=80, help="Describe which catalog objects to combine.")
    with st.expander("Examples"):
        for ex in EXAMPLES:
            st.code(ex, language=None)
    c1, c2, c3 = st.columns(3)
    width = c1.number_input("Canvas width", 64, 4096, 800, 16)
    height = c2.number_input("Canvas height", 64, 4096, 600, 16)
    bg_choice = c3.selectbox("Background", ["from instruction / white", *COLOR_NAMES, "custom hex"])
    background = None
    if bg_choice == "custom hex":
        background = c3.text_input("Hex colour", "#f0f4ff")
    elif bg_choice != "from instruction / white":
        background = bg_choice
    c4, c5 = st.columns(2)
    layout = c4.selectbox("Layout", ["from instruction / row", *LAYOUTS],
                          help="'spatial' honours positions such as 'on the left' or 'behind the boy'. It is "
                               "chosen automatically when the instruction contains them.")
    mask = c5.selectbox("Background removal", MASK_METHODS, format_func=lambda m: f"{m}: {MASK_HELP[m]}")
    ready, why = rembg_status()
    c5.caption(("✅ " if ready else "ℹ️ ") + why)
    c6, c7 = st.columns(2)
    strict = c6.checkbox("Strict colour matching", value=True,
                         help="If unticked, an object with the right label but a different colour may be used "
                              "(this is recorded in the metadata).")
    include_comp = c7.checkbox("Allow objects from earlier composites", value=False)
    return dict(instruction=instruction, canvas_size=(int(width), int(height)), background=background,
                layout=None if layout.startswith("from") else layout, mask_method=mask, strict_color=strict,
                include_composites=include_comp)


def apply_layer_edits(plan, edited: pd.DataFrame) -> None:
    """Copy the layer table back into the plan. A cleared cell (NaN) keeps the
    previous value instead of crashing the page."""
    for _, row in edited.iterrows():
        layer = plan.layers[int(row["layer"])]
        for field, cast in (("x", int), ("y", int), ("scale", float), ("z", int)):
            value = row.get(field)
            if value is None or pd.isna(value):
                continue
            if field == "scale" and float(value) <= 0:
                continue
            setattr(layer, field, cast(value))


def page() -> None:
    composer = get_composer()
    st.title("Composite Generator")
    st.caption("Builds new images only from **real objects already in the catalog**: cropped, masked, resized "
               "and layered. Nothing is AI-generated. Missing objects are reported, never approximated.")

    opts = _plan_controls()
    if st.button("Find catalog objects & plan", type="primary"):
        st.session_state["comp_instruction"] = opts["instruction"]
        try:
            with st.spinner("Finding objects and cutting them out (background removal can take a few seconds)..."):
                plan = composer.plan(**opts)
            st.session_state["comp_plan"] = plan
            st.session_state["comp_token"] = st.session_state.get("comp_token", 0) + 1
            st.session_state.pop("comp_saved", None)
        except CompositionError as exc:
            st.session_state.pop("comp_plan", None)
            st.error(str(exc))
        except Exception as exc:  # keep the page usable; nothing has been saved
            st.session_state.pop("comp_plan", None)
            st.error(f"Planning failed unexpectedly: {exc}. Nothing was saved.")

    saved = st.session_state.get("comp_saved")
    if saved:
        st.success(f"Saved as catalog image #{saved}. See it in the **Image Catalog** or **Metadata Viewer**.")

    plan = st.session_state.get("comp_plan")
    if plan is None:
        return

    st.divider()
    st.subheader("1 · What was understood")
    st.dataframe(pd.DataFrame([{
        "count": i.count, "colour": i.color or "(any)", "object": i.label,
        "position": i.position or "", "relation": (f"{i.relation['type'].replace('_', ' ')} {i.relation['ref']}"
                                                   if i.relation else ""),
        "size": {None: ""}.get(i.size, f"x{i.size}"), "depth": i.depth or ""} for i in plan.parsed.items]),
        hide_index=True)
    st.caption(f"Layout: **{plan.layout}** · Background: **{plan.background}** · "
               f"Canvas: {plan.canvas_width}×{plan.canvas_height}")
    for w in plan.parsed.warnings:
        st.warning(w)

    st.subheader("2 · Catalog assets")
    if plan.missing:
        for m in plan.missing:
            with st.container(border=True):
                st.error(f"Missing: {m.needed - m.found} × **{m.describe()}**. {m.reason}")
                for s in m.suggestions:
                    st.caption(f"→ {s}")
    if plan.layers:
        cols = st.columns(min(4, len(plan.layers)))
        for i, layer in enumerate(plan.layers):
            with cols[i % len(cols)], st.container(border=True):
                st.image(plan.cutouts[i], width="stretch")
                st.caption(f"Layer {i}: **{layer.label}** ({layer.color or 'no colour'}) from image "
                           f"#{layer.source_image_id}, object #{layer.source_object_id}")
                st.caption(f"match: `{layer.match_type}` · mask: `{layer.mask_used}`"
                           + (f" ({layer.mask_note})" if layer.mask_note else ""))
    else:
        st.warning("None of the requested objects exist in the catalog, so there is nothing to compose. Upload "
                   "images containing them, or label existing objects in the Manual Editor.")
        return

    st.subheader("3 · Position, size and stacking")
    c1, c2 = st.columns([1, 3])
    if c1.button("Reset automatic layout"):
        auto_layout(plan)
        st.session_state["comp_token"] = st.session_state.get("comp_token", 0) + 1
        st.rerun()
    c2.caption("Edit x/y (top-left, pixels), scale (1.0 = original size) and z (higher = in front).")
    df = pd.DataFrame([{"layer": i, "object": f"{l.label} (#{l.source_object_id})", "x": l.x, "y": l.y,
                        "scale": l.scale, "z": l.z, "placed": l.placement} for i, l in enumerate(plan.layers)])
    edited = st.data_editor(
        df, hide_index=True, disabled=["layer", "object", "placed"],
        key=f"layers_{st.session_state.get('comp_token', 0)}",
        column_config={
            "x": st.column_config.NumberColumn(min_value=-4096, max_value=4096, step=1, required=True),
            "y": st.column_config.NumberColumn(min_value=-4096, max_value=4096, step=1, required=True),
            "scale": st.column_config.NumberColumn(min_value=0.02, max_value=10.0, step=0.05, format="%.2f",
                                                   required=True),
            "z": st.column_config.NumberColumn(min_value=-100, max_value=100, step=1, required=True),
        },
    )
    apply_layer_edits(plan, edited)

    image, boxes = composer.render(plan)
    st.image(image, caption="Preview", width="stretch")
    off = [i for i, b in enumerate(boxes) if b is None]
    if off:
        st.warning(f"Layer(s) {off} are completely outside the canvas.")

    st.subheader("4 · Save to catalog")
    name = st.text_input("Display name", value=f"Composite: {plan.instruction[:60]}")
    allow_partial = False
    if plan.missing:
        allow_partial = st.checkbox("Save anyway without the missing objects (they will be listed in the "
                                    "composition metadata)")
    if st.button("Save composite", type="primary", disabled=bool(plan.missing) and not allow_partial):
        try:
            with st.spinner("Rendering and saving the composite..."):
                record = composer.save(plan, display_name=name, allow_partial=allow_partial)
            st.session_state["comp_saved"] = record.id
            st.session_state["selected_image_id"] = record.id
            st.session_state.pop("comp_plan", None)
            st.rerun()
        except MissingAssetsError as exc:
            st.error(str(exc))
        except CompositionError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Saving failed unexpectedly: {exc}")
    with st.expander("Composition metadata that will be stored"):
        st.json(plan.metadata())
