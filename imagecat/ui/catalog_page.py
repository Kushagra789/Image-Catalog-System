import streamlit as st

from ..catalog import CatalogError
from .common import get_service, status_badge, thumbnail

PAGE_SIZE = 24


def _delete_dialog(svc, rec, n_obj: int) -> None:
    corrections = len(svc.corrections(rec.id))
    users = svc.db.composites_using(rec.id)
    st.warning(f"Permanently delete image #{rec.id}, its file and its {n_obj} object(s)? This cannot be undone.")
    if corrections:
        st.error(f"This image has **{corrections} manual correction(s)**; they will be lost.")
    if users:
        st.info("Used as a source by composite(s) " + ", ".join(f"#{c.id}" for c in users)
                + ". Those composites are kept, and their stored provenance snapshot still describes this image.")
    sure = st.checkbox("Yes, delete it", key=f"del_ck_{rec.id}")
    if st.button("Delete permanently", key=f"del_{rec.id}", type="primary", disabled=not sure):
        try:
            svc.catalog.delete(rec.id)
        except CatalogError as exc:
            st.error(str(exc))
            return
        if st.session_state.get("selected_image_id") == rec.id:
            st.session_state.pop("selected_image_id")
        st.toast(f"Deleted image #{rec.id}")
        st.rerun()


def page() -> None:
    svc = get_service()
    st.title("Image Catalog")

    records = svc.catalog.list_images()
    if not records:
        st.info("The catalog is empty. Add images on the **Upload Images** page.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search by name or object label", placeholder="e.g. apple").strip().lower()
    statuses = sorted({r.detection_status for r in records})
    status_filter = c2.multiselect("Detection status", statuses, default=statuses)
    kind = c3.selectbox("Type", ["All", "Originals", "Composites"])

    counts = svc.db.object_counts()
    label_hits = svc.db.image_ids_with_label(query) if query else set()

    def matches(rec) -> bool:
        if rec.detection_status not in status_filter:
            return False
        if kind == "Originals" and rec.is_composite or kind == "Composites" and not rec.is_composite:
            return False
        if not query:
            return True
        return query in rec.display_name.lower() or query in rec.name.lower() or rec.id in label_hits

    shown = [r for r in records if matches(r)]
    pages = max(1, -(-len(shown) // PAGE_SIZE))
    page_no = 1
    if pages > 1:
        page_no = st.number_input(f"Page (of {pages})", 1, pages, 1, key="catalog_page_no")
    visible = shown[(page_no - 1) * PAGE_SIZE: page_no * PAGE_SIZE]
    st.caption(f"{len(shown)} of {len(records)} image(s)" + (f" · page {page_no}/{pages}" if pages > 1 else ""))

    cols = st.columns(4)
    for i, rec in enumerate(visible):
        n_obj = counts.get(rec.id, 0)
        with cols[i % 4], st.container(border=True):
            img = thumbnail(rec, 360)
            if img is not None:
                st.image(img, width="stretch")
            st.markdown(f"**#{rec.id} · {rec.display_name}**")
            st.caption(f"{rec.width}×{rec.height} px · {n_obj} object(s) · "
                       f"{rec.created_at[:19].replace('T', ' ')} UTC")
            status_badge(rec.detection_status)

            b1, b2, b3 = st.columns(3)
            if b1.button("Open", key=f"open_{rec.id}", help="Show in Object Detection"):
                st.session_state["selected_image_id"] = rec.id
                app_pages = st.session_state.get("_pages")
                if app_pages:
                    st.switch_page(app_pages["detection"])
                else:
                    st.info("Selected. Open the Object Detection page to see it.")
            with b2.popover("Rename"):
                new = st.text_input("Display name", value=rec.display_name, key=f"rn_{rec.id}")
                if st.button("Save", key=f"rn_btn_{rec.id}"):
                    try:
                        svc.catalog.rename(rec.id, new)
                        st.rerun()
                    except CatalogError as exc:
                        st.error(str(exc))
            with b3.popover("Delete"):
                _delete_dialog(svc, rec, n_obj)
