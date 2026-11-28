"""Mean filtering for the shared RGB representation."""

import numpy as np


def mean_filter(image_rgb: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Average each channel of a nonempty (height, width, 3) uint8 RGB array.

    The kernel is a square of positive odd size. For kernels larger than one,
    each image dimension must be at least two. NumPy's reflect padding mirrors
    neighbouring pixels without repeating the pixel at the edge.

    A float64 summed-area table computes the averages. The result is rounded with
    np.rint (nearest with ties to even), clipped to [0, 255], and returned in a
    new uint8 array with the input shape. A kernel size of one returns a copy.
    """
    if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
        raise TypeError("image_rgb must be a NumPy uint8 array.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or 0 in image_rgb.shape[:2]:
        raise ValueError("image_rgb must have nonempty shape (height, width, 3).")
    if isinstance(kernel_size, (bool, np.bool_)) or not isinstance(kernel_size, (int, np.integer)):
        raise TypeError("kernel_size must be a positive odd integer.")
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if kernel_size == 1:
        return image_rgb.copy()

    height, width, channels = image_rgb.shape
    if min(height, width) < 2:
        raise ValueError("Image height and width must be at least 2 for kernel_size > 1.")

    radius = kernel_size // 2
    area = kernel_size * kernel_size
    filtered = np.empty_like(image_rgb)
    # The top row and left column of the summed-area table represent empty prefixes.
    summed = np.zeros((height + 2 * radius + 1, width + 2 * radius + 1), dtype=np.float64)
    interior = summed[1:, 1:]
    for channel in range(channels):
        interior[:] = np.pad(image_rgb[..., channel], radius, mode="reflect")
        np.cumsum(interior, axis=0, out=interior)
        np.cumsum(interior, axis=1, out=interior)

        window = summed[kernel_size:, kernel_size:] - summed[:-kernel_size, kernel_size:]
        window -= summed[kernel_size:, :-kernel_size]
        window += summed[:-kernel_size, :-kernel_size]
        window /= area
        np.rint(window, out=window)
        np.clip(window, 0, 255, out=window)
        filtered[..., channel] = window
    return filtered
