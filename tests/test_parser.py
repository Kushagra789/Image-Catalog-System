import pytest

from imagecat.composer.parser import parse_instruction, singularize


def _items(text):
    return [(i.count, i.color, i.label) for i in parse_instruction(text).items]


def test_spec_example():
    assert _items("Create an image containing one red apple and one orange.") == [
        (1, "red", "apple"), (1, None, "orange")]


def test_orange_as_colour_vs_fruit():
    assert _items("one orange ball") == [(1, "orange", "ball")]
    assert _items("an orange") == [(1, None, "orange")]


def test_counts_plurals_and_multiword_labels():
    assert _items("Make a picture with two oranges, a cartoon boy and 3 blue circles") == [
        (2, None, "orange"), (1, None, "cartoon boy"), (3, "blue", "circle")]


def test_background_and_layout():
    p = parse_instruction("overlapping red circle and blue square on a yellow background")
    assert p.layout == "overlap"
    assert p.background == "yellow"
    assert [(i.color, i.label) for i in p.items] == [("red", "circle"), ("blue", "square")]


def test_synonym_colours_and_modifiers():
    assert _items("a light grey cat") == [(1, "gray", "cat")]


def test_empty_instruction_warns():
    p = parse_instruction("create an image")
    assert p.items == []
    assert p.warnings


@pytest.mark.parametrize("plural,single", [
    ("apples", "apple"), ("boxes", "box"), ("cherries", "cherry"), ("people", "person"),
    ("glasses", "glasses"), ("bus", "bus"), ("tomatoes", "tomato")])
def test_singularize(plural, single):
    assert singularize(plural) == single
