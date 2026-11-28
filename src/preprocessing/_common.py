"""Shared validation, RGB quantisation, and HSV value helpers."""

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def validate_rgb(image_rgb: np.ndarray) -> None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
        raise TypeError("image_rgb must be a NumPy uint8 array.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or 0 in image_rgb.shape[:2]:
        raise ValueError("image_rgb must have nonempty shape (height, width, 3).")


def validate_kernel_size(kernel_size: int) -> int:
    if isinstance(kernel_size, (bool, np.bool_)) or not isinstance(kernel_size, (int, np.integer)):
        raise TypeError("kernel_size must be a positive odd integer.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    return int(kernel_size)


def rounded_uint8(values: np.ndarray) -> np.ndarray:
    """Round to nearest with ties to even, clip, and return a new uint8 array."""
    rounded = np.rint(values)
    np.clip(rounded, 0, 255, out=rounded)
    return rounded.astype(np.uint8)


def hsv_value_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """Get HSV V on the 0–255 scale from the maximum RGB component."""
    return image_rgb.max(axis=2)


def replace_hsv_value(image_rgb: np.ndarray, new_value: np.ndarray) -> np.ndarray:
    """Use Matplotlib's float64 RGB/HSV conversion with channels in [0, 1]."""
    hsv = rgb_to_hsv(image_rgb.astype(np.float64) / 255.0)
    hsv[..., 2] = new_value / 255.0
    rgb = hsv_to_rgb(hsv)
    rgb *= 255.0
    return rounded_uint8(rgb)
