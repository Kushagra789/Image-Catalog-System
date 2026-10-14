import pandas as pd
import streamlit as st

from ..composer.cutout import rembg_status
from .common import get_service, image_option_label, status_badge, thumbnail

LIMITATIONS = """
* **YOLO (COCO, 80 classes)** recognises common objects such as *person, apple, orange, banana, cup, dog*.
  It is trained on photographs: on cartoons/clip-art it often misses objects or returns wrong labels
  (e.g. a red circle reported as *frisbee*). Those results are flagged *verify (illustration)*.
* **YOLOE (illustration model)** runs on cartoons/clip-art only, restricted to a curated vocabulary. It helps
  (e.g. two real cartoon children found, a clip-art apple named *fruit*), but it still misses line art and
  stylised characters, and its labels must be verified.
* **Cartoon characters** are labelled *cartoon character* (from YOLO/YOLOE person-like labels, or an OpenCV
  heuristic that looks for a face-like pair of eyes). **Boy/girl is never assigned automatically.** The
  model's boy/girl guess is only shown as an *unverified suggestion*, because it is unreliable: in testing it
  called a cartoon boy a girl. Confirm such labels yourself in the **Manual Editor**.
* **OpenCV analysis** only runs on *flat* images. It recognises geometry and colour regions and has no
  semantics. Cartoon parts (head, shirt, legs) appear as separate shapes, marked *part of …*. Hide them with
  *Hide parts of larger objects* on the Object Detection page.
* **Duplicates** from several detectors are merged (overlap + compatible labels). The other labels are kept as
  *alternative labels*, so a wrong merge can be undone with one click in the Manual Editor.
* **Confidence** is stored only when a model produced it. Heuristic detections have none (shown as “—”).
* **Background removal**: rembg (U2Net) when downloaded, otherwise YOLO-seg masks, contours, colour keying or
  GrabCut. Very fine structures, soft shadows and heavily overlapping objects can still be cut imperfectly.
* **Composites** are cut-and-paste of real catalog pixels. Nothing is AI-generated, and missing objects are reported,
  never invented. Spatial words are rule-based: *left/right/center/top/bottom*, *left of / right of /
  above / below / behind / in front of / next to*, *big/small*, *foreground/background*.
"""


def page() -> None:
    svc = get_service()
    st.title("Dashboard")
    st.caption("Overview of the image catalog, detections and corrections.")

    stats = svc.db.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", stats["images"])
    c2.metric("Active objects", stats["objects"])
    c3.metric("Manual corrections", stats["corrections"])
    c4.metric("Composites", stats["composites"])

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Detection engines")
            with st.spinner("Loading detection models (first visit only, can take ~20 s)..."):
                engines = svc.engine.engine_status()
            for name, state in engines.items():
                ok = state == "available"
                st.markdown(f"{'✅' if ok else '⚠️'} **{name.upper()}**: {state}")
            ready, why = rembg_status()
            st.markdown(f"{'✅' if ready else 'ℹ️'} **REMBG** (background removal): {why}")
            if not engines.get("yolo", "").startswith("available"):
                st.info("The app keeps working without YOLO (OpenCV analysis plus manual labelling). "
                        "To enable YOLO, install `ultralytics` and run once with internet so the weights download.")
        with st.container(border=True):
            st.subheader("Detection status")
            by_status = stats["by_status"]
            if by_status:
                st.dataframe(pd.DataFrame({"status": list(by_status), "images": list(by_status.values())}),
                             hide_index=True)
            else:
                st.caption("No images yet.")
    with right:
        with st.container(border=True):
            st.subheader("Objects by source")
            if stats["by_source"]:
                st.dataframe(pd.DataFrame({"source": list(stats["by_source"]),
                                           "objects": list(stats["by_source"].values())}), hide_index=True)
            else:
                st.caption("No objects yet.")
        with st.container(border=True):
            st.subheader("Most common labels")
            if stats["by_label"]:
                st.dataframe(pd.DataFrame({"label": list(stats["by_label"]),
                                           "count": list(stats["by_label"].values())}), hide_index=True)
            else:
                st.caption("No labels yet.")

    st.subheader("Recently added")
    recent = svc.catalog.list_images()[:6]
    if not recent:
        st.info("The catalog is empty. Go to **Upload Images** to add photos, or load the synthetic test images.")
    cols = st.columns(6)
    for col, rec in zip(cols, recent):
        with col:
            img = thumbnail(rec, 240)
            if img is not None:
                st.image(img, width="stretch")
            st.caption(image_option_label(rec))
            status_badge(rec.detection_status)

    with st.container(border=True):
        st.subheader("Database health")
        st.caption("Read-only check: SQLite integrity, foreign keys, missing/orphaned image files, boxes outside "
                   "their image, composites whose sources were deleted. Nothing is changed or repaired.")
        if st.button("Run health check"):
            with st.spinner("Checking database and image folder..."):
                st.session_state["health_report"] = svc.health_report()
        report = st.session_state.get("health_report")
        if report:
            if report["ok"]:
                st.success(f"No problems found ({report['images']} image(s), SQLite integrity: "
                           f"{report['integrity_check']}).")
            for problem in report["problems"]:
                st.error(problem)
            for warning in report["warnings"]:
                st.warning(warning)

    with st.expander("Known limitations (please read)", expanded=False):
        st.markdown(LIMITATIONS)
