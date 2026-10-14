"""Reproducible synthetic test images with ground-truth annotations.

These are simple flat drawings, *not* photographs. They exercise the OpenCV
pipeline, manual corrections and composition deterministically. YOLO is trained
on photographs and is not expected to recognise them; upload real photos of an
apple/orange to test semantic detection.

Every fixture returns ``(PIL.Image, [GroundTruth, ...])``. Ground-truth boxes
are measured from the pixels actually drawn for each item, so they are exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

SIZE = (640, 480)
BG = (255, 255, 255)

RED = (220, 20, 30)
ORANGE = (255, 140, 0)
DARK_ORANGE = (235, 120, 0)
GREEN = (40, 160, 40)
BLUE = (30, 70, 220)
YELLOW = (245, 215, 0)
PURPLE = (130, 50, 180)
BROWN = (120, 60, 20)
SKIN = (250, 205, 165)
PINK = (240, 110, 170)
BLACK = (20, 20, 20)
DARK_BLUE = (20, 30, 90)


@dataclass
class GroundTruth:
    label: str
    color: str | None
    bbox: tuple[int, int, int, int]  # x, y, width, height
    kind: str  # "shape", "region", "object"


DrawOp = Callable[[ImageDraw.ImageDraw, tuple], None]


class _Canvas:
    """Draws items and records the exact pixel bbox of each labelled item."""

    def __init__(self, size=SIZE, bg=BG):
        self.image = Image.new("RGB", size, bg)
        self.draw = ImageDraw.Draw(self.image)
        self.truth: list[GroundTruth] = []
        self._groups: dict[str, Image.Image] = {}

    def add(self, op: DrawOp, color, label: str | None = None, color_name: str | None = None,
            kind: str = "shape", group: str | None = None):
        op(self.draw, color)
        if label is None and group is None:
            return
        mask = Image.new("L", self.image.size, 0)
        op(ImageDraw.Draw(mask), 255)
        if group is not None:
            g = self._groups.setdefault(group, Image.new("L", self.image.size, 0))
            g.paste(255, mask=mask)
        if label is not None:
            self.truth.append(GroundTruth(label, color_name, _bbox(mask), kind))

    def add_group_truth(self, group: str, label: str, color_name: str | None):
        self.truth.append(GroundTruth(label, color_name, _bbox(self._groups[group]), "object"))

    def result(self):
        return self.image, self.truth


def _bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(mask) > 0
    ys, xs = np.nonzero(arr)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _ellipse(box):
    return lambda d, c: d.ellipse(box, fill=c)


def _rect(box):
    return lambda d, c: d.rectangle(box, fill=c)


def _poly(points):
    return lambda d, c: d.polygon(points, fill=c)


def _regular_polygon(cx, cy, r, n, rotation=0):
    return lambda d, c: d.regular_polygon((cx, cy, r), n, rotation=rotation, fill=c)


def _blob(circles):
    def op(d, c):
        for cx, cy, r in circles:
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c)
    return op


# ----------------------------------------------------------------- fixtures
def _draw_apple(cv: _Canvas, ox: int, oy: int, group: str):
    cv.add(_ellipse((ox, oy + 20, ox + 170, oy + 180)), RED, group=group)
    cv.add(_rect((ox + 80, oy - 10, ox + 90, oy + 30)), BROWN, group=group)
    cv.add(_ellipse((ox + 92, oy - 5, ox + 150, oy + 25)), GREEN, group=group)


def _draw_orange(cv: _Canvas, ox: int, oy: int, group: str):
    cv.add(_ellipse((ox, oy + 10, ox + 170, oy + 180)), ORANGE, group=group)
    for dx, dy in ((50, 60), (110, 70), (80, 120), (40, 130), (125, 130)):
        cv.add(_ellipse((ox + dx, oy + dy, ox + dx + 6, oy + dy + 6)), DARK_ORANGE, group=group)
    cv.add(_ellipse((ox + 75, oy + 2, ox + 100, oy + 22)), GREEN, group=group)


def apple():
    cv = _Canvas()
    _draw_apple(cv, 235, 140, "apple")
    cv.add_group_truth("apple", "apple", "red")
    return cv.result()


def orange():
    cv = _Canvas()
    _draw_orange(cv, 235, 140, "orange")
    cv.add_group_truth("orange", "orange", "orange")
    return cv.result()


def apple_orange():
    cv = _Canvas()
    _draw_apple(cv, 80, 140, "apple")
    _draw_orange(cv, 380, 140, "orange")
    cv.add_group_truth("apple", "apple", "red")
    cv.add_group_truth("orange", "orange", "orange")
    return cv.result()


def _draw_person(cv: _Canvas, cx: int, group: str, girl: bool):
    if girl:  # long hair behind the head
        cv.add(_ellipse((cx - 70, 40, cx + 70, 230)), YELLOW, group=group)
    cv.add(_ellipse((cx - 55, 50, cx + 55, 160)), SKIN, group=group)  # head
    if not girl:
        cv.add(_ellipse((cx - 58, 40, cx + 58, 100)), BROWN, group=group)  # short hair
        cv.add(_rect((cx - 55, 75, cx + 55, 100)), SKIN, group=group)
    for ex in (cx - 22, cx + 22):
        cv.add(_ellipse((ex - 6, 105, ex + 6, 117)), BLACK, group=group)
    cv.add(_rect((cx - 15, 160, cx + 15, 180)), SKIN, group=group)  # neck
    if girl:
        cv.add(_poly([(cx, 175), (cx - 90, 360), (cx + 90, 360)]), PINK, group=group)  # dress
    else:
        cv.add(_rect((cx - 60, 175, cx + 60, 300)), BLUE, group=group)  # shirt
        cv.add(_rect((cx - 55, 300, cx + 55, 380)), DARK_BLUE, group=group)  # trousers
    arm_top, arm_bottom = 185, 290
    cv.add(_rect((cx - 85, arm_top, cx - 62, arm_bottom)), SKIN, group=group)
    cv.add(_rect((cx + 62, arm_top, cx + 85, arm_bottom)), SKIN, group=group)
    legs_top = 360 if girl else 380
    cv.add(_rect((cx - 40, legs_top, cx - 15, 440)), SKIN, group=group)
    cv.add(_rect((cx + 15, legs_top, cx + 40, 440)), SKIN, group=group)


def cartoon_boy():
    cv = _Canvas()
    _draw_person(cv, 320, "boy", girl=False)
    cv.add_group_truth("boy", "cartoon boy", "blue")
    return cv.result()


def cartoon_girl():
    cv = _Canvas()
    _draw_person(cv, 320, "girl", girl=True)
    cv.add_group_truth("girl", "cartoon girl", "pink")
    return cv.result()


def shapes():
    cv = _Canvas()
    cv.add(_ellipse((50, 60, 190, 200)), RED, "circle", "red")
    cv.add(_rect((260, 60, 400, 200)), BLUE, "square", "blue")
    cv.add(_poly([(540, 60), (620, 200), (460, 200)]), GREEN, "triangle", "green")
    cv.add(_rect((40, 290, 260, 400)), YELLOW, "rectangle", "yellow")
    cv.add(_regular_polygon(410, 350, 75, 5), PURPLE, "pentagon", "purple")
    cv.add(_ellipse((510, 300, 630, 400)), ORANGE, "ellipse", "orange")
    return cv.result()


def colors():
    """Irregular single-colour blobs: should become colour regions, not shapes."""
    cv = _Canvas()
    specs = [
        (RED, "red", [(90, 110, 50), (140, 100, 35), (110, 160, 30)]),
        (GREEN, "green", [(320, 100, 45), (370, 130, 30), (280, 140, 28)]),
        (BLUE, "blue", [(530, 110, 48), (570, 160, 30), (490, 150, 25)]),
        (YELLOW, "yellow", [(100, 340, 45), (150, 370, 35), (60, 390, 25)]),
        (ORANGE, "orange", [(320, 350, 50), (360, 400, 28), (270, 395, 30)]),
        (PURPLE, "purple", [(530, 340, 45), (575, 380, 30), (500, 395, 28)]),
    ]
    for rgb, name, circles in specs:
        cv.add(_blob(circles), rgb, f"{name} region", name, kind="region")
    return cv.result()


def overlapping_shapes():
    """Square partly covered by a circle, circle partly covered by a triangle."""
    cv = _Canvas()
    cv.add(_rect((100, 100, 340, 340)), BLUE, "square", "blue")
    cv.add(_ellipse((270, 240, 450, 420)), RED, "circle", "red")
    cv.add(_poly([(400, 260), (560, 260), (480, 420)]), GREEN, "triangle", "green")
    img, truth = cv.result()
    # Ground truth for occluded items = their *visible* pixel extent.
    arr = np.asarray(img)
    fixed = []
    for gt in truth:
        target = {"blue": BLUE, "red": RED, "green": GREEN}[gt.color]
        ys, xs = np.nonzero(np.all(arr == target, axis=2))
        fixed.append(GroundTruth(gt.label, gt.color,
                                 (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                                  int(ys.max() - ys.min() + 1)), gt.kind))
    return img, fixed


def blank():
    return Image.new("RGB", SIZE, BG), []


FIXTURES: dict[str, Callable[[], tuple[Image.Image, list[GroundTruth]]]] = {
    "apple": apple,
    "orange": orange,
    "cartoon_boy": cartoon_boy,
    "cartoon_girl": cartoon_girl,
    "shapes": shapes,
    "colors": colors,
    "apple_orange": apple_orange,
    "overlapping_shapes": overlapping_shapes,
    "blank": blank,
}


def generate(name: str) -> tuple[Image.Image, list[GroundTruth]]:
    return FIXTURES[name]()


def write_all(out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FIXTURES:
        img, _ = generate(name)
        path = out_dir / f"{name}.png"
        img.save(path, format="PNG")
        paths.append(path)
    return paths
