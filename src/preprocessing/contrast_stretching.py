"""Linear contrast stretching in HSV value or separate RGB channels."""

import numpy as np

from ._common import hsv_value_uint8, replace_hsv_value, rounded_uint8, validate_rgb


def _stretch(channel: np.ndarray, low: int, high: int) -> np.ndarray:
    return 255.0 * (channel.astype(np.float64) - low) / (high - low)


def contrast_stretch(image_rgb: np.ndarray, mode: str = "hsv_value") -> np.ndarray:
    """Stretch V alone or each RGB channel with an explicit mode.

    In 'hsv_value' mode Matplotlib uses float64 hue, saturation and value
    in [0, 1], and only the value component is changed. Constant V returns
    an exact copy. In 'rgb_channels' mode each nonconstant channel spans
    [0, 255] and constant channels retain their original values.
    The output is independent uint8 RGB, rounded with np.rint and clipped.
    """
    validate_rgb(image_rgb)
    if not isinstance(mode, str):
        raise TypeError("mode must be 'hsv_value' or 'rgb_channels'.")
    if mode not in ("hsv_value", "rgb_channels"):
        raise ValueError("mode must be 'hsv_value' or 'rgb_channels'.")

    if mode == "hsv_value":
        value = hsv_value_uint8(image_rgb)
        low, high = int(value.min()), int(value.max())
        if low == high:
            return image_rgb.copy()
        return replace_hsv_value(image_rgb, _stretch(value, low, high))

    stretched = image_rgb.copy()
    for index in range(3):
        channel = image_rgb[..., index]
        low, high = int(channel.min()), int(channel.max())
        if low != high:
            stretched[..., index] = rounded_uint8(_stretch(channel, low, high))
    return stretched
