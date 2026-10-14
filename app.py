"""Streamlit entry point:  streamlit run app.py"""
import streamlit as st

from imagecat.ui import catalog_page, composer_page, dashboard, detection_page, editor, metadata, upload

st.set_page_config(page_title="Image Catalog System", page_icon="🖼️", layout="wide")

pages = {
    "dashboard": st.Page(dashboard.page, title="Dashboard", icon="📊", url_path="dashboard", default=True),
    "upload": st.Page(upload.page, title="Upload Images", icon="⬆️", url_path="upload"),
    "catalog": st.Page(catalog_page.page, title="Image Catalog", icon="🗂️", url_path="catalog"),
    "detection": st.Page(detection_page.page, title="Object Detection", icon="🎯", url_path="detection"),
    "editor": st.Page(editor.page, title="Manual Editor", icon="✏️", url_path="editor"),
    "composer": st.Page(composer_page.page, title="Composite Generator", icon="🧩", url_path="composer"),
    "metadata": st.Page(metadata.page, title="Metadata Viewer", icon="🧾", url_path="metadata"),
}
st.session_state["_pages"] = pages  # lets pages link to each other via st.switch_page

with st.sidebar:
    st.markdown("### 🖼️ Image Catalog System")
    st.caption("Catalog · Detection · Corrections · Composites")

st.navigation(list(pages.values())).run()
