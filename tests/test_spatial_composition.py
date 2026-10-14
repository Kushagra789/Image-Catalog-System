"""Spatial instructions: parsing and layout."""
import pytest

from imagecat.composer import Composer, parse_instruction
from tests.conftest import fixture_png


def _items(text):
    return [(i.label, i.position, i.relation, i.size, i.depth) for i in parse_instruction(text).items]


# ----------------------------------------------------------------- parsing
def test_parse_spec_example():
    p = parse_instruction("Place an apple on the left, an orange on the right, and a cartoon boy in the center.")
    assert p.layout == "spatial"
    assert [(i.count, i.label, i.position) for i in p.items] == [
        (1, "apple", "left"), (1, "orange", "right"), (1, "cartoon boy", "center")]
    assert p.warnings == []


@pytest.mark.parametrize("text,expected", [
    ("an apple at the top left", ("apple", "top-left", None, None, None)),
    ("an apple in the bottom right corner", ("apple", "bottom-right", None, None, None)),
    ("an apple at the top", ("apple", "top", None, None, None)),
    ("an apple in the middle of the image", ("apple", "center", None, None, None)),
    ("an orange to the right of the apple", ("orange", None, {"type": "right_of", "ref": "apple"}, None, None)),
    ("an orange left of the apples", ("orange", None, {"type": "left_of", "ref": "apple"}, None, None)),
    ("an apple above the orange", ("apple", None, {"type": "above", "ref": "orange"}, None, None)),
    ("an apple under the orange", ("apple", None, {"type": "below", "ref": "orange"}, None, None)),
    ("an orange behind the cartoon boy", ("orange", None, {"type": "behind", "ref": "cartoon boy"}, None, None)),
    ("an apple in front of the orange", ("apple", None, {"type": "in_front_of", "ref": "orange"}, None, None)),
    ("a big apple", ("apple", None, None, 1.4, None)),
    ("a tiny orange in the foreground", ("orange", None, None, 0.45, "foreground")),
])
def test_parse_spatial_phrases(text, expected):
    assert _items(text) == [expected]


def test_spatial_words_do_not_break_plain_instructions():
    assert _items("one orange ball") == [("ball", None, None, None, None)]
    p = parse_instruction("Create an image containing one red apple and one orange on a blue background")
    assert p.layout is None and p.background == "blue"


def test_missing_reference_warns():
    p = parse_instruction("an orange to the left of the banana")
    assert any("banana" in w for w in p.warnings)


# ------------------------------------------------------------------ layout
@pytest.fixture
def catalog(service, upload_fixture):
    """apple, orange and cartoon boy objects labelled by a user."""
    out = {}
    for fixture, label in (("apple", "apple"), ("orange", "orange"), ("cartoon_boy", "cartoon boy")):
        rec = upload_fixture(fixture).record
        obj = max(service.db.list_objects(rec.id), key=lambda o: o.bbox.area)
        out[label] = service.update_object(obj.id, label=label)
    return out


def _centre(layer):
    return layer.x + layer.width * layer.scale / 2, layer.y + layer.height * layer.scale / 2


def _by_label(plan):
    return {l.label: l for l in plan.layers}


def test_left_center_right_honoured(service, catalog):
    plan = Composer(service).plan(
        "Place an apple on the left, an orange on the right, and a cartoon boy in the center.", canvas_size=(900, 600))
    assert plan.is_complete and plan.layout == "spatial"
    L = _by_label(plan)
    ax, ox, bx = _centre(L["apple"])[0], _centre(L["orange"])[0], _centre(L["cartoon boy"])[0]
    assert ax < 300 < bx < 600 < ox  # canvas thirds
    assert all("from instruction" in l.placement for l in plan.layers)
    # Metadata records how every layer was placed.
    assert plan.metadata()["layers"][0]["position"] == "left"


def test_vertical_positions(service, catalog):
    plan = Composer(service).plan("an apple at the top and an orange at the bottom", canvas_size=(600, 600))
    L = _by_label(plan)
    assert _centre(L["apple"])[1] < 200 and _centre(L["orange"])[1] > 400


def test_relations(service, catalog):
    composer = Composer(service)
    L = _by_label(composer.plan("an apple in the center and an orange to the right of the apple"))
    assert L["orange"].x >= L["apple"].x + L["apple"].width * L["apple"].scale
    L = _by_label(composer.plan("an orange in the center and an apple above the orange", canvas_size=(800, 800)))
    assert L["apple"].y + L["apple"].height * L["apple"].scale <= L["orange"].y
    L = _by_label(composer.plan("a cartoon boy in the center and an orange behind the boy"))
    assert L["orange"].z < L["cartoon boy"].z


def test_depth_and_size(service, catalog):
    plan = Composer(service).plan("a small apple on the left and a big orange on the right in the foreground")
    L = _by_label(plan)
    assert L["orange"].z > L["apple"].z
    assert L["orange"].height * L["orange"].scale > L["apple"].height * L["apple"].scale


def test_layers_stay_on_canvas(service, catalog):
    plan = Composer(service).plan("a huge apple at the top left and a huge orange at the bottom right",
                                  canvas_size=(300, 200))
    for l in plan.layers:
        assert 0 <= l.x and l.x + l.width * l.scale <= 300 + 1
        assert 0 <= l.y and l.y + l.height * l.scale <= 200 + 1


def test_explicit_layout_overrides_positions_with_warning(service, catalog):
    plan = Composer(service).plan("an apple on the right and an orange on the left", layout="row")
    assert plan.layout == "row"
    assert any("ignored" in w for w in plan.parsed.warnings)


def test_spatial_composite_saved_with_placement_metadata(service, catalog):
    composer = Composer(service)
    record = composer.save(composer.plan("an apple on the left and an orange on the right"))
    layers = record.composition_metadata["layers"]
    assert [l["position"] for l in layers] == ["left", "right"]
    objs = sorted(service.db.list_objects(record.id), key=lambda o: o.x)
    assert [o.label for o in objs] == ["apple", "orange"]
