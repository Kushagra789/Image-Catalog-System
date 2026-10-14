"""Small image I/O helpers shared by detection, composition and the UI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgba(path: Path | str) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return im.convert("RGBA")


def load_rgb(path: Path | str, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Load an image as RGB, flattening any transparency onto ``background``."""
    rgba = load_rgba(path)
    base = Image.new("RGBA", rgba.size, background + (255,))
    return Image.alpha_composite(base, rgba).convert("RGB")


def has_transparency(path: Path | str) -> bool:
    alpha = np.asarray(load_rgba(path))[:, :, 3]
    return bool((alpha < 250).any())


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))
    return np.ascontiguousarray(arr[:, :, ::-1])


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]))
