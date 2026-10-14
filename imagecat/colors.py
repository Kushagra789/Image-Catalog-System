"""Rule-based colour naming in HSV space.

Every pixel gets one of a small set of basic colour names, and an object's colour
is the most frequent name among its pixels. This is deterministic and easy to
explain, but it is a heuristic: lighting, shadows and highlights can shift it.
"""
from __future__ import annotations

import cv2
import numpy as np

COLOR_NAMES = (
    "black", "white", "gray", "red", "orange", "brown", "beige", "yellow",
    "green", "cyan", "blue", "purple", "pink",
)
# Representative RGB value for each name (used for UI swatches and drawing).
COLOR_RGB = {
    "black": (20, 20, 20), "white": (245, 245, 245), "gray": (128, 128, 128),
    "red": (220, 30, 30), "orange": (255, 140, 0), "brown": (139, 69, 19), "beige": (245, 205, 165),
    "yellow": (240, 210, 0), "green": (40, 160, 40), "cyan": (0, 190, 210),
    "blue": (30, 70, 220), "purple": (130, 50, 180), "pink": (240, 110, 170),
}
_NAME_INDEX = {n: i for i, n in enumerate(COLOR_NAMES)}


def name_pixels_bgr(bgr: np.ndarray) -> np.ndarray:
    """Return an array of colour-name indices (into COLOR_NAMES) for BGR pixels.

    Accepts an (H, W, 3) image or an (N, 3) pixel list.
    """
    pixels = bgr.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h = hsv[:, 0] * 2  # degrees 0..358
    s = hsv[:, 1]
    v = hsv[:, 2]

    out = np.full(h.shape, _NAME_INDEX["gray"], dtype=np.int32)
    chromatic = (s >= 50) & (v >= 50)

    hue_name = np.select(
        [
            (h < 15) | (h >= 345),
            h < 40,
            h < 70,
            h < 165,
            h < 200,
            h < 255,
            h < 290,
        ],
        [
            _NAME_INDEX["red"], _NAME_INDEX["orange"], _NAME_INDEX["yellow"],
            _NAME_INDEX["green"], _NAME_INDEX["cyan"], _NAME_INDEX["blue"],
            _NAME_INDEX["purple"],
        ],
        default=_NAME_INDEX["pink"],
    )
    # Dark orange / dark red-orange reads as brown.
    brown = ((h >= 8) & (h < 45)) & (v < 160) & chromatic
    hue_name = np.where(brown, _NAME_INDEX["brown"], hue_name)
    # Light, desaturated orange/yellow (skin tones, sand) reads as beige.
    beige = (h >= 15) & (h < 50) & (s < 115) & (v > 180)
    hue_name = np.where(beige, _NAME_INDEX["beige"], hue_name)
    # Light, desaturated red reads as pink.
    light_red = ((h < 15) | (h >= 345)) & (s < 130) & (v > 180)
    hue_name = np.where(light_red, _NAME_INDEX["pink"], hue_name)

    out = np.where(chromatic, hue_name, out)
    achromatic = ~chromatic
    out = np.where(achromatic & (v < 50), _NAME_INDEX["black"], out)
    out = np.where(achromatic & (v >= 50) & (s < 50) & (v > 200), _NAME_INDEX["white"], out)
    return out.reshape(bgr.shape[:-1])


def name_color_bgr(bgr_pixel) -> str:
    arr = np.array(bgr_pixel, dtype=np.uint8).reshape(1, 3)
    return COLOR_NAMES[int(name_pixels_bgr(arr)[0])]


def name_color_rgb(rgb_pixel) -> str:
    r, g, b = rgb_pixel
    return name_color_bgr((b, g, r))


def dominant_color(
    bgr: np.ndarray,
    mask: np.ndarray | None = None,
    ignore: tuple[str, ...] = (),
) -> tuple[str | None, str | None, float]:
    """Most frequent colour name among (masked) pixels.

    Returns (name, hex of the mean RGB of pixels with that name, fraction).
    ``ignore`` lets callers skip e.g. a known background colour.
    """
    pixels = bgr.reshape(-1, 3) if mask is None else bgr[mask > 0]
    if len(pixels) == 0:
        return None, None, 0.0
    if len(pixels) > 50_000:  # subsample for speed, deterministically
        pixels = pixels[:: len(pixels) // 50_000 + 1]
    names = name_pixels_bgr(pixels.reshape(-1, 1, 3)).reshape(-1)
    counts = np.bincount(names, minlength=len(COLOR_NAMES)).astype(float)
    for n in ignore:
        if counts.sum() - counts[_NAME_INDEX[n]] > 0.15 * counts.sum():
            counts[_NAME_INDEX[n]] = 0
    idx = int(counts.argmax())
    fraction = counts[idx] / max(1.0, float(len(names)))
    mean_bgr = pixels[names == idx].mean(axis=0)
    b, g, r = (int(round(c)) for c in mean_bgr)
    return COLOR_NAMES[idx], f"#{r:02x}{g:02x}{b:02x}", float(fraction)


def dominant_palette(bgr: np.ndarray, top: int = 5) -> list[dict]:
    """Image-level summary: share of each colour name (top N)."""
    small = cv2.resize(bgr, (min(200, bgr.shape[1]), min(200, bgr.shape[0])), interpolation=cv2.INTER_AREA)
    names = name_pixels_bgr(small).reshape(-1)
    counts = np.bincount(names, minlength=len(COLOR_NAMES))
    total = counts.sum()
    order = np.argsort(-counts)[:top]
    return [
        {"color": COLOR_NAMES[i], "fraction": round(float(counts[i] / total), 3)}
        for i in order if counts[i] > 0
    ]
