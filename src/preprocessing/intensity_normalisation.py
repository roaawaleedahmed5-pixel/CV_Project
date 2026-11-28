"""Per-image z-score normalisation with shared RGB statistics."""

from numbers import Real

import numpy as np

from ._common import validate_rgb


def normalise_intensity(image_rgb: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Return a float32 z-score array with the same shape as the input RGB uint8.

    Compute one float64 mean and population standard deviation across every
    value in the image, including all RGB channels. When std <= epsilon,
    return float32 zeros. Epsilon must be finite and nonnegative. Values are
    signed and retain their normalised range for downstream numeric use.
    The result has independent storage, and the input is preserved.
    """
    validate_rgb(image_rgb)
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a finite nonnegative real number.")
    try:
        epsilon = float(epsilon)
    except OverflowError as error:
        raise ValueError("epsilon must be a finite nonnegative real number.") from error
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite nonnegative real number.")

    values = image_rgb.astype(np.float64)
    mean = values.mean()
    std = values.std(ddof=0)
    if std <= epsilon:
        return np.zeros(image_rgb.shape, dtype=np.float32)
    values -= mean
    values /= std
    return values.astype(np.float32)
