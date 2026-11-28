"""Resize RGB arrays using Pillow with an explicit aspect-ratio policy."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from PIL import Image

from ._common import validate_rgb


_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _round_dimension(numerator: int, denominator: int) -> int:
    """Round an exact positive ratio to the nearest pixel, half up."""
    return max(1, (2 * numerator + denominator) // (2 * denominator))


def resize_rgb(
    image_rgb: np.ndarray,
    target_width: int,
    target_height: int,
    *,
    interpolation: str = "bilinear",
    aspect_ratio_policy: str = "stretch",
    padding_colour: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    """Return an independent RGB uint8 array at the requested width and height.

    Stretch scales both axes directly. Pad scales to fit and centres the result
    on the supplied RGB colour. Center crop scales to cover and removes the
    excess. For fit or cover, the limiting axis is exact and the other rounds
    to the nearest pixel (half up, at least one). Offsets use floor division:
    an odd surplus adds or removes one extra pixel at the right or bottom.
    Pillow applies the specified nearest, bilinear, or bicubic resampling.
    """
    validate_rgb(image_rgb)
    target_width = _positive_dimension(target_width, "target_width")
    target_height = _positive_dimension(target_height, "target_height")
    if not isinstance(interpolation, str) or interpolation not in _RESAMPLING:
        raise ValueError("interpolation must be 'nearest', 'bilinear', or 'bicubic'.")
    if not isinstance(aspect_ratio_policy, str) or aspect_ratio_policy not in ("stretch", "pad", "center_crop"):
        raise ValueError("aspect_ratio_policy must be 'stretch', 'pad', or 'center_crop'.")
    if (
        not isinstance(padding_colour, Sequence)
        or len(padding_colour) != 3
        or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= 255
            for value in padding_colour
        )
    ):
        raise ValueError("padding_colour must contain three integer RGB values in [0, 255].")

    height, width = image_rgb.shape[:2]
    target_size = (target_width, target_height)
    if (width, height) == target_size:
        return image_rgb.copy()

    resized_size = target_size
    if aspect_ratio_policy != "stretch":
        width_limits_fit = target_width * height <= target_height * width
        use_target_width = width_limits_fit == (aspect_ratio_policy == "pad")
        if use_target_width:
            resized_size = (target_width, _round_dimension(height * target_width, width))
        else:
            resized_size = (_round_dimension(width * target_height, height), target_height)

    with Image.fromarray(image_rgb) as original, original.resize(
        resized_size, resample=_RESAMPLING[interpolation], reducing_gap=None
    ) as resized:
        resized_width, resized_height = resized_size
        if aspect_ratio_policy == "center_crop":
            left = (resized_width - target_width) // 2
            top = (resized_height - target_height) // 2
            with resized.crop((left, top, left + target_width, top + target_height)) as cropped:
                return np.array(cropped, copy=True)
        if aspect_ratio_policy == "pad":
            left = (target_width - resized_width) // 2
            top = (target_height - resized_height) // 2
            canvas = np.empty((target_height, target_width, 3), dtype=np.uint8)
            canvas[:] = padding_colour
            canvas[top:top + resized_height, left:left + resized_width] = np.asarray(resized)
            return canvas
        return np.array(resized, copy=True)
