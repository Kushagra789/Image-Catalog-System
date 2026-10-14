"""Composite generation from real catalog objects."""
import numpy as np
import pytest
from PIL import Image

from imagecat.composer import (
    Composer,
    CompositionError,
    MissingAssetsError,
    extract_cutout,
    label_match,
)
from imagecat.models import BBox, SOURCE_COMPOSITE


def _largest(service, image_id):
    return max(service.db.list_objects(image_id), key=lambda o: o.bbox.area)


@pytest.fixture
def fruit_catalog(service, upload_fixture):
    """Apple and orange fixtures, labelled by a user via manual correction
    (the OpenCV detector only knows 'foreground object' / 'circle')."""
    apple = upload_fixture("apple").record
    orange = upload_fixture("orange").record
    a = service.update_object(_largest(service, apple.id).id, label="apple", color="red")
    o = service.update_object(_largest(service, orange.id).id, label="orange", color="orange")
    return {"apple_img": apple, "orange_img": orange, "apple": a, "orange": o}


def test_label_match_rules():
    assert label_match("apple", "apple") == "exact"
    assert label_match("apples", "apple") == "exact"
    assert label_match("boy", "cartoon boy") == "specific"
    assert label_match("apple", "orange") is None
    assert label_match("apple", "pineapple") is None  # whole-word only


def test_spec_example_composite_saved_with_provenance(service, fruit_catalog):
    composer = Composer(service)
    plan = composer.plan("Create an image containing one red apple and one orange.")
    assert plan.is_complete
    assert [l.source_object_id for l in plan.layers] == [fruit_catalog["apple"].id, fruit_catalog["orange"].id]
    assert all(l.mask_used == "polygon" for l in plan.layers)

    record = composer.save(plan)
    assert record.is_composite
    assert record.source_asset_ids["image_ids"] == sorted(
        [fruit_catalog["apple_img"].id, fruit_catalog["orange_img"].id])
    assert record.source_asset_ids["object_ids"] == [fruit_catalog["apple"].id, fruit_catalog["orange"].id]
    meta = record.composition_metadata
    assert meta["instruction"].startswith("Create an image")
    assert meta["complete"] is True and meta["missing"] == []
    assert "no generative AI" in meta["generator"]
    assert len(meta["layers"]) == 2

    objs = service.db.list_objects(record.id)
    assert {o.label for o in objs} == {"apple", "orange"}
    assert all(o.source == SOURCE_COMPOSITE and o.confidence is None for o in objs)
    assert service.catalog.path_for(record).exists()


def test_composite_pixels_come_from_catalog(service, fruit_catalog):
    """The apple layer must contain the apple's actual red pixels."""
    composer = Composer(service)
    plan = composer.plan("one red apple", canvas_size=(640, 480))
    plan.layers[0].scale = 1.0
    plan.layers[0].x, plan.layers[0].y = 0, 0
    img, boxes = composer.render(plan)
    arr = np.asarray(img)
    x, y, w, h = boxes[0]
    region = arr[y:y + h, x:x + w].reshape(-1, 3)
    red = np.all(np.abs(region.astype(int) - (220, 20, 30)) < 3, axis=1).mean()
    assert red > 0.4


def test_missing_assets_reported_and_block_saving(service, fruit_catalog):
    composer = Composer(service)
    plan = composer.plan("one red apple and one banana")
    assert len(plan.layers) == 1
    assert [m.label for m in plan.missing] == ["banana"]
    assert "No object labelled 'banana'" in plan.missing[0].reason
    with pytest.raises(MissingAssetsError):
        composer.save(plan)
    record = composer.save(plan, allow_partial=True)  # explicit user choice
    assert record.composition_metadata["complete"] is False
    assert record.composition_metadata["missing"][0]["label"] == "banana"


def test_nothing_found_cannot_be_saved(service, fruit_catalog):
    composer = Composer(service)
    plan = composer.plan("a purple elephant")
    assert plan.layers == []
    with pytest.raises(CompositionError):
        composer.save(plan, allow_partial=True)


def test_wrong_colour_not_silently_substituted(service, fruit_catalog):
    composer = Composer(service)
    plan = composer.plan("one green apple")
    assert plan.layers == []
    assert "not in colour 'green'" in plan.missing[0].reason
    lenient = composer.plan("one green apple", strict_color=False)
    assert len(lenient.layers) == 1
    assert "color_ignored" in lenient.layers[0].match_type


def test_not_enough_distinct_objects(service, fruit_catalog):
    plan = Composer(service).plan("two apples")
    assert len(plan.layers) == 1
    assert plan.missing[0].needed == 2 and plan.missing[0].found == 1


def test_suggestions_point_to_relabel_candidates(service, upload_fixture):
    upload_fixture("shapes")
    plan = Composer(service).plan("one red apple")
    assert plan.missing
    assert any("relabel" in s for s in plan.missing[0].suggestions)  # red circle exists


def test_overlap_layout_produces_overlapping_layers(service, fruit_catalog):
    composer = Composer(service)
    plan = composer.plan("overlapping red apple and orange")
    assert plan.layout == "overlap"
    _, boxes = composer.render(plan)
    assert BBox(*boxes[0]).iou(BBox(*boxes[1])) > 0
    assert plan.layers[1].z > plan.layers[0].z
    record = composer.save(plan)
    objs = service.db.list_objects(record.id)
    assert objs[0].bbox.iou(objs[1].bbox) > 0


def test_custom_positions_scale_and_z_order(service, fruit_catalog):
    composer = Composer(service)
    plan = composer.plan("a red apple and an orange", canvas_size=(500, 400), background="#102030")
    a, o = plan.layers
    a.x, a.y, a.scale, a.z = 10, 20, 0.5, 2
    o.x, o.y, o.scale, o.z = 30, 40, 0.5, 1  # apple drawn on top
    img, boxes = composer.render(plan)
    assert img.size == (500, 400)
    assert img.getpixel((499, 399)) == (16, 32, 48)
    assert boxes[0][0] >= 10 and boxes[0][2] <= round(a.width * 0.5) + 1


@pytest.mark.parametrize("method", ["polygon", "key", "grabcut", "none"])
def test_cutout_methods(service, fruit_catalog, method):
    rec = fruit_catalog["apple_img"]
    cut, used, _ = extract_cutout(service.catalog.path_for(rec), fruit_catalog["apple"], method)
    alpha = np.asarray(cut)[:, :, 3]
    assert used == method
    if method == "none":
        assert (alpha == 255).all()
    else:
        # Background corners of the apple's box are removed.
        assert alpha[0, 0] < 128 or alpha[-1, -1] < 128
        assert (alpha > 128).mean() > 0.5


def test_transparent_source_uses_alpha(service):
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    img.paste((0, 120, 255, 255), (50, 50, 150, 150))
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rec = service.upload(buf.getvalue(), "blue_sticker.png", auto_detect=False).record
    obj = service.add_object(rec.id, "sticker", (0, 0, 200, 200), color="blue")
    cut, used, _ = extract_cutout(service.catalog.path_for(rec), obj, "auto")
    assert used == "alpha"
    assert cut.size == (100, 100)  # trimmed to visible pixels


def test_composites_excluded_as_sources_by_default(service, fruit_catalog):
    composer = Composer(service)
    composer.save(composer.plan("one red apple"))
    # Only the original apple exists as a source unless composites are included.
    assert len(composer.plan("two red apples").layers) == 1
    assert len(composer.plan("two red apples", include_composites=True).layers) == 2


def test_invalid_options(service, fruit_catalog):
    composer = Composer(service)
    with pytest.raises(CompositionError):
        composer.plan("one apple", background="not-a-colour")
    with pytest.raises(CompositionError):
        composer.plan("one apple", canvas_size=(10, 10))
    with pytest.raises(CompositionError):
        composer.plan("")


def test_ranking_prefers_untruncated_objects(service, upload_fixture):
    rec = upload_fixture("blank", auto_detect=False).record
    # Higher "confidence" but cut off by the left image edge vs. a complete one.
    edge = service.db.insert_object(image_id=rec.id, label="person", color=None, confidence=0.9,
                                    bbox=(0, 100, 80, 200), source="yolo")
    whole = service.db.insert_object(image_id=rec.id, label="person", color=None, confidence=0.6,
                                     bbox=(200, 100, 80, 200), source="yolo")
    plan = Composer(service).plan("one person", mask_method="none")
    assert plan.layers[0].source_object_id == whole
    assert edge != whole


def test_ranking_prefers_fully_visible_shapes(service, upload_fixture):
    """overlapping_shapes has a bigger but partly hidden blue square; the clean
    one from 'shapes' must be chosen."""
    upload_fixture("overlapping_shapes")
    shapes = upload_fixture("shapes").record
    plan = Composer(service).plan("one blue square and one red circle")
    chosen = {l.label: service.db.get_object(l.source_object_id) for l in plan.layers}
    assert chosen["square"].image_id == shapes.id
    assert chosen["square"].attributes["occluded"] is False
    assert chosen["circle"].image_id == shapes.id
