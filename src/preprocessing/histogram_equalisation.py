"""Histogram equalisation on the HSV value component."""

import numpy as np

from ._common import hsv_value_uint8, replace_hsv_value, validate_rgb


def histogram_equalise(image_rgb: np.ndarray) -> np.ndarray:
    """Equalise HSV V using a 256-bin histogram and cumulative pixel counts.

    Accumulate the histogram by indexed addition on the 0–255 V scale and
    subtract the first occupied cumulative count in the mapping. Constant V
    returns an exact copy. Matplotlib converts float64 RGB/HSV on [0, 1].
    Only V is replaced. The independent RGB output is rounded with np.rint,
    clipped to [0, 255], and converted to uint8.
    """
    validate_rgb(image_rgb)
    value = hsv_value_uint8(image_rgb)
    low = int(value.min())
    if low == int(value.max()):
        return image_rgb.copy()

    counts = np.zeros(256, dtype=np.int64)
    np.add.at(counts, value.ravel(), 1)
    cumulative = np.cumsum(counts, dtype=np.int64)
    first_count = cumulative[low]
    mapping = 255.0 * (cumulative.astype(np.float64) - first_count) / (value.size - first_count)
    return replace_hsv_value(image_rgb, mapping[value])
