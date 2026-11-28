"""Channel-wise median filtering with bounded neighbourhood batches."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._common import rounded_uint8, validate_kernel_size, validate_rgb


_TILE_VALUES = 1_000_000


def median_filter(image_rgb: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Replace each channel value with its local square-window median.

    Input is a nonempty RGB uint8 array and the kernel size must be a positive
    odd integer. NumPy reflect excludes the edge pixel from the reflection.
    Singleton axes repeat their sole value. Kernel size one returns a copy.
    Each tile is evaluated in float64 and rounded with np.rint, clipped to
    [0, 255], and returned in an independent RGB uint8 array.
    """
    validate_rgb(image_rgb)
    size = validate_kernel_size(kernel_size)
    if size == 1:
        return image_rgb.copy()

    height, width, _ = image_rgb.shape
    radius = size // 2
    # Bound the float64 workspace, while still allowing at least one window.
    tile_pixels = max(1, _TILE_VALUES // (size * size))
    tile_width = min(width, tile_pixels)
    tile_height = max(1, tile_pixels // tile_width)
    filtered = np.empty_like(image_rgb)
    for channel in range(3):
        padded = np.pad(image_rgb[..., channel], radius, mode="reflect")
        windows = sliding_window_view(padded, (size, size))
        for top in range(0, height, tile_height):
            for left in range(0, width, tile_width):
                rows = slice(top, top + tile_height)
                cols = slice(left, left + tile_width)
                tile = windows[rows, cols].astype(np.float64, order="C")
                medians = np.median(tile, axis=(-2, -1), overwrite_input=True)
                filtered[rows, cols, channel] = rounded_uint8(medians)
    return filtered
