"""Load the shared RGB representation without changing the source image."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFile


def load_rgb(image_path: str | Path) -> np.ndarray:
    """Fully decode an RGB or opaque RGBA image as a writable uint8 RGB array."""
    path = Path(image_path)
    if path.name.startswith("._"):
        raise ValueError(f"Metadata filename is not a photograph: {path}")
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError("Pillow's LOAD_TRUNCATED_IMAGES must be False for strict decoding.")

    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            pixels = np.asarray(image) if mode in {"RGB", "RGBA"} else None
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Image file not found: {path}") from error
    except (OSError, ValueError, EOFError, SyntaxError, Image.DecompressionBombError) as error:
        raise OSError(f"Cannot decode image {path}: {type(error).__name__}: {error}") from error

    if pixels is None:
        raise ValueError(f"Unsupported colour mode {mode!r} for {path}; expected RGB or opaque RGBA.")
    if mode == "RGBA":
        alpha = pixels[..., 3]
        opaque = np.iinfo(alpha.dtype).max
        nonopaque = np.count_nonzero(alpha != opaque)
        if nonopaque:
            raise ValueError(
                f"Nonopaque alpha in {path}: {nonopaque} of {alpha.size} pixels "
                f"have alpha below {opaque}."
            )
    return pixels[..., :3].copy()
