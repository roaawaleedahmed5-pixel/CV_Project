"""Separable Gaussian filtering for the shared RGB representation."""

from numbers import Real

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._common import rounded_uint8, validate_kernel_size, validate_rgb


def gaussian_kernel(kernel_size: int = 7, sigma: float = 1.0) -> np.ndarray:
    """Return symmetric, unit-sum float64 weights on a finite 1-D grid."""
    size = validate_kernel_size(kernel_size)
    if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, Real):
        raise TypeError("sigma must be a finite positive real number.")
    try:
        sigma = float(sigma)
    except OverflowError as error:
        raise ValueError("sigma must be a finite positive real number.") from error
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive real number.")

    radius = size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    # Very small sigma makes the off-centre weights tend to zero.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-0.5 * np.square(offsets / sigma))
    return weights / weights.sum()


def _convolve_axis(plane: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    radius = len(weights) // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(plane, padding, mode="reflect")
    windows = sliding_window_view(padded, len(weights), axis=axis)
    # The window view only adds strides. The contraction accumulates in float64.
    return np.einsum("ijk,k->ij", windows, weights, dtype=np.float64, optimize=False)


def gaussian_filter(image_rgb: np.ndarray, kernel_size: int = 7, sigma: float = 1.0) -> np.ndarray:
    """Smooth each RGB channel with a finite separable Gaussian.

    Input is a nonempty RGB uint8 array. Use a positive odd kernel size and
    finite positive sigma in pixels. NumPy reflect padding mirrors the border
    without repeating its edge pixel. A singleton axis repeats its sole value.
    Kernel size one returns a copy. Otherwise both passes use float64 and the
    independent uint8 output is rounded with np.rint and clipped to [0, 255].
    """
    validate_rgb(image_rgb)
    weights = gaussian_kernel(kernel_size, sigma)
    if weights.size == 1:
        return image_rgb.copy()

    filtered = np.empty_like(image_rgb)
    for channel in range(3):
        horizontal = _convolve_axis(image_rgb[..., channel], weights, axis=1)
        vertical = _convolve_axis(horizontal, weights, axis=0)
        filtered[..., channel] = rounded_uint8(vertical)
    return filtered
