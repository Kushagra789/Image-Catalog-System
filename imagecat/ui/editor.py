import streamlit as st

from ..colors import COLOR_NAMES
from ..service import ValidationError
from ..visualization import draw_box_preview, draw_objects, object_caption
from .bbox_editor import AVAILABLE as MOUSE_EDITOR, bbox_editor, editor_boxes
from .common import CONFIDENCE_NOTE, display_size, get_service, load_image, objects_dataframe, select_image

NONE_COLOR = "(none)"
OTHER_COLOR = "other..."
MODE_EDIT, MODE_ADD, MODE_DELETED = "Edit / remove", "Add object", "Deleted"
BOX_FIELDS = ("x", "y", "w", "h")


def _color_input(current: str | None, key: str) -> str | None:
    options = [NONE_COLOR, *COLOR_NAMES, OTHER_COLOR]
    if current is None:
        index = 0
    elif current in COLOR_NAMES:
        index = options.index(current)
    else:
        index = len(options) - 1
    choice = st.selectbox("Colour", options, index=index, key=f"{key}_sel")
    if choice == OTHER_COLOR:
        return st.text_input("Custom colour name", value=current or "", key=f"{key}_txt") or None
    return None if choice == NONE_COLOR else choice


def _confidence_input(current: float | None, key: str) -> float | None:
    has = st.checkbox("Specify a confidence value", value=current is not None, key=f"{key}_has",
                      help="Leave unticked when no real confidence exists. Never guess one.")
    if not has:
        return None
    return st.number_input("Confidence (0 to 1)", 0.0, 1.0, float(current if current is not None else 0.5),
                           0.01, key=f"{key}_val")


FORM_FIELDS = ("label", "sel", "txt", "has", "val", *BOX_FIELDS)


def _forget_form(object_id) -> None:
    """Drop the edit form's widget state for an object.

    Streamlit keeps a keyed widget's value even when its default changes, so
    after a label is changed by a button (Apply / Use) or saved, the form would
    otherwise still hold the old values, and the next "Save changes" would
    silently write them back over the correction.
    """
    for f in FORM_FIELDS:
        st.session_state.pop(f"e{object_id}_{f}", None)


def _init_box_state(key: str, box: tuple[int, int, int, int]) -> None:
    for field, value in zip(BOX_FIELDS, box):
        st.session_state.setdefault(f"{key}_{field}", int(value))


def _box_state(key: str) -> tuple[int, int, int, int]:
    return tuple(int(st.session_state[f"{key}_{f}"]) for f in BOX_FIELDS)  # type: ignore[return-value]


def _set_box_state(key: str, rec, event: dict) -> None:
    """Copy a mouse gesture into the numeric inputs (before they are drawn)."""
    x = max(0, min(int(event["x"]), rec.width - 1))
    y = max(0, min(int(event["y"]), rec.height - 1))
    w = max(1, min(int(event["width"]), rec.width - x))
    h = max(1, min(int(event["height"]), rec.height - y))
    for field, value in zip(BOX_FIELDS, (x, y, w, h)):
        st.session_state[f"{key}_{field}"] = value


def _bbox_inputs(rec, key: str) -> tuple[int, int, int, int]:
    """Exact numeric box inputs; their values live in session state."""
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("x", 0, rec.width - 1, key=f"{key}_x")
    c2.number_input("y", 0, rec.height - 1, key=f"{key}_y")
    c3.number_input("width", 1, rec.width, key=f"{key}_w")
    c4.number_input("height", 1, rec.height, key=f"{key}_h")
    return _box_state(key)


def _suggestions(svc, obj) -> None:
    """Alternative labels from other detectors and unverified suggestions."""
    sugg = obj.attributes.get("suggested_labels") or []
    alts = obj.attributes.get("alternative_labels") or []
    if not sugg and not alts:
        return
    with st.container(border=True):
        st.markdown("**Other possible labels**")
        for i, s in enumerate(sugg):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"Suggestion: **{s['label']}** (score {s['score']:.2f}, {s['source']})")
            c1.caption("⚠️ " + s.get("warning", "Unverified suggestion."))
            if c2.button("Apply", key=f"sug_{obj.id}_{i}", help="Only apply if you can confirm it yourself."):
                svc.update_object(obj.id, label=s["label"])
                _forget_form(obj.id)
                st.session_state["edit_flash"] = f"Object #{obj.id} relabelled as '{s['label']}' (by you)."
                st.rerun()
        for i, a in enumerate(alts):
            conf = "no confidence" if a.get("confidence") is None else f"{a['confidence']:.2f}"
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"Also detected as **{a['label']}** ({a['source']}, {conf})")
            if c2.button("Use", key=f"alt_{obj.id}_{i}"):
                svc.update_object(obj.id, label=a["label"])
                _forget_form(obj.id)
                st.session_state["edit_flash"] = f"Object #{obj.id} relabelled as '{a['label']}'."
                st.rerun()


def _edit_panel(svc, rec, obj, objects):
    k = f"e{obj.id}"
    st.selectbox("Object", [o.id for o in objects], key=f"edit_obj_{rec.id}",
                 format_func=lambda i: object_caption(next(o for o in objects if o.id == i)))
    label = st.text_input("Label", value=obj.label, key=f"{k}_label")
    color = _color_input(obj.color, k)
    conf = _confidence_input(obj.confidence, k)
    if (obj.origin != "manual" and obj.confidence is not None and conf == obj.confidence
            and label.strip().lower() != obj.label):
        st.caption("ℹ️ The model's confidence refers to the old label. Untick the box to clear it.")
    bbox = _bbox_inputs(rec, k)
    st.caption(f"Source: `{obj.source}` · origin: `{obj.origin}`" + (" · edited" if obj.is_edited else ""))

    c1, c2, c3 = st.columns(3)
    if c1.button("Save changes", type="primary", key=f"{k}_save"):
        try:
            svc.update_object(obj.id, label=label, color=color, confidence=conf, bbox=bbox)
            _forget_form(obj.id)  # e.g. a relabel may have cleared the model confidence
            st.session_state["edit_flash"] = f"Object #{obj.id} saved."
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
    if c2.button("Reset box", key=f"{k}_reset", help="Discard unsaved box changes."):
        for f in BOX_FIELDS:
            st.session_state.pop(f"{k}_{f}", None)
        st.rerun()
    with c3.popover("Remove"):
        st.write(f"Remove **{object_caption(obj)}**?")
        if obj.origin == "manual":
            st.caption("Manually added objects are deleted permanently.")
        else:
            st.caption("Detections are hidden and remembered, so re-running detection will not bring them "
                       "back. Restore them under *Deleted*.")
        if st.button("Confirm removal", key=f"{k}_del", type="primary"):
            svc.delete_object(obj.id)
            st.session_state["edit_flash"] = f"Object #{obj.id} removed."
            st.rerun()
    _suggestions(svc, obj)
    if obj.attributes.get("original"):
        with st.expander("Original detection values"):
            st.json(obj.attributes["original"])
    return bbox, label


def _add_panel(svc, rec, k):
    label = st.text_input("Label", placeholder="e.g. cartoon boy", key=f"{k}_label")
    color = _color_input(None, k)
    conf = _confidence_input(None, k)
    bbox = _bbox_inputs(rec, k)
    if st.button("Add object", type="primary", key=f"{k}_btn"):
        try:
            new = svc.add_object(rec.id, label, bbox, color=color, confidence=conf)
            for f in BOX_FIELDS:
                st.session_state.pop(f"{k}_{f}", None)
            st.session_state["edit_flash"] = f"Added object #{new.id} ({new.label})."
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
    return bbox, label


def page() -> None:
    svc = get_service()
    st.title("Manual Editor")
    st.caption("Corrections are saved to SQLite immediately and stay authoritative: re-running detection in "
               "the default mode never overwrites them.")
    rec = select_image("edit_image")
    if rec is None:
        return
    path = svc.catalog.path_for(rec)
    if not path.exists():
        st.error(f"Image file for #{rec.id} is missing ({path.name}). Run the health check on the Dashboard.")
        return

    objects = svc.db.list_objects(rec.id)
    deleted = [o for o in svc.db.list_objects(rec.id, include_deleted=True) if o.is_deleted]
    flash = st.session_state.pop("edit_flash", None)
    if flash:
        st.success(flash)

    preview_col, controls = st.columns([3, 2])
    with controls:
        mode = st.segmented_control(
            "Action", [MODE_EDIT, MODE_ADD, MODE_DELETED], default=MODE_EDIT if objects else MODE_ADD,
            required=True, key=f"edit_mode_{rec.id}",
            format_func=lambda m: f"{m} ({len(deleted)})" if m == MODE_DELETED else m,
        )
    use_mouse = MOUSE_EDITOR and st.session_state.get("mouse_editor", True)

    if mode == MODE_EDIT and objects:
        sel_key = f"edit_obj_{rec.id}"
        ids = [o.id for o in objects]
        if st.session_state.get(sel_key) not in ids:
            st.session_state[sel_key] = ids[0]
        obj = next(o for o in objects if o.id == st.session_state[sel_key])
        k = f"e{obj.id}"
        _init_box_state(k, obj.bbox.as_tuple())
        if use_mouse:
            with preview_col:
                event = bbox_editor(path, rec.width, rec.height,
                                    editor_boxes(objects, {obj.id: _box_state(k)}), obj.id, "edit",
                                    key=f"bbe_edit_{rec.id}")
            if event and event.get("id") in ids and event["id"] != obj.id:
                st.session_state[sel_key] = event["id"]  # clicked another box: select it
                st.rerun()
            if event and event.get("id") == obj.id and event.get("action") != "select":
                _set_box_state(k, rec, event)
                st.rerun()
        with controls:
            bbox, label = _edit_panel(svc, rec, obj, objects)
        if not use_mouse:
            with preview_col:
                img = load_image(rec)
                if img is None:
                    return
                base = draw_objects(img, objects, highlight_id=obj.id, show_labels=False)
                st.image(display_size(draw_box_preview(base, bbox, label)), width="stretch")
    elif mode in (MODE_ADD, MODE_EDIT):
        if mode == MODE_EDIT:
            controls.info("No objects to edit. Choose **Add object**.")
        k = f"add{rec.id}"
        _init_box_state(k, (rec.width // 4, rec.height // 4, rec.width // 2, rec.height // 2))
        if use_mouse:
            new_box = dict(zip(("x", "y", "width", "height"), _box_state(k)))
            label_now = st.session_state.get(f"{k}_label") or "new object"
            with preview_col:
                event = bbox_editor(path, rec.width, rec.height,
                                    editor_boxes(objects, extra=[{"id": "new", **new_box, "label": label_now,
                                                                  "color": "#e6003c"}]),
                                    "new", "draw", key=f"bbe_add_{rec.id}")
            if event and event.get("id") == "new":
                _set_box_state(k, rec, event)
                st.rerun()
        with controls:
            bbox, label = _add_panel(svc, rec, k)
        if not use_mouse:
            with preview_col:
                img = load_image(rec)
                if img is None:
                    return
                st.image(display_size(draw_box_preview(draw_objects(img, objects, show_labels=False), bbox,
                                                       label or "new object")), width="stretch")
    else:
        with controls:
            if not deleted:
                st.caption("No removed detections.")
            for o in deleted:
                c1, c2 = st.columns([3, 1])
                c1.write(object_caption(o) + f" · {o.source}")
                if c2.button("Restore", key=f"restore_{o.id}"):
                    svc.restore_object(o.id)
                    st.session_state["edit_flash"] = f"Object #{o.id} restored."
                    st.rerun()
        with preview_col:
            img = load_image(rec)
            if img is not None:
                st.image(display_size(draw_objects(img, deleted) if deleted else draw_objects(img, objects)),
                         width="stretch")
            st.caption("Showing removed detections.")

    with preview_col:
        if MOUSE_EDITOR:
            st.toggle("Mouse editing", key="mouse_editor", value=True,
                      help="Turn off to use a static preview with the numeric inputs only.")
            if mode != MODE_DELETED and use_mouse:
                st.caption("The numeric inputs follow your mouse edits. Nothing is saved until you click "
                           "**Save changes** / **Add object**.")
        elif mode != MODE_DELETED:
            st.caption("Red box = the values currently in the form (saved only when you click the button).")

    st.subheader("All objects (including removed)")
    st.caption(CONFIDENCE_NOTE)
    st.dataframe(objects_dataframe(svc.db.list_objects(rec.id, include_deleted=True)), hide_index=True)
