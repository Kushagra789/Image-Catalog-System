"""Overlay drawing: label tags must stay readable when boxes share an edge."""
from PIL import Image

from imagecat.models import ObjectRecord
from imagecat.visualization import _place_tag, draw_objects


def _obj(i, box):
    x, y, w, h = box
    return ObjectRecord(id=i, image_id=1, label=f"object {i}", color="red", confidence=None,
                        x=x, y=y, width=w, height=h, source="opencv_shape", origin="auto",
                        is_edited=False, is_deleted=False)


def test_tags_for_boxes_with_same_corner_do_not_overlap():
    placed = []
    for _ in range(4):  # four boxes whose top-left corner is identical
        placed.append(_place_tag(100, 100, 120, 16, (640, 480), placed))
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), (a, b)
        assert 0 <= a[1] and a[3] < 480


def test_draw_objects_runs_on_stacked_boxes():
    img = Image.new("RGB", (200, 150), "white")
    out = draw_objects(img, [_obj(1, (10, 10, 100, 100)), _obj(2, (10, 10, 50, 50)), _obj(3, (150, 140, 50, 10))])
    assert out.size == img.size
