import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from src.data_loader import load_rgb


class LoadRGBTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    def save_pixels(self, filename, values):
        path = self.root / filename
        with Image.fromarray(values) as image:
            image.save(path)
        return path

    def test_rgb_pixels_shape_and_independent_writable_output(self):
        path = self.save_pixels("rgb.png", self.rgb)
        source = path.read_bytes()
        result = load_rgb(path)
        np.testing.assert_array_equal(result, self.rgb)
        self.assertEqual(result.shape, (2, 3, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(result.flags.writeable and result.flags.owndata)
        result[0, 0] = (100, 101, 102)
        self.assertEqual(path.read_bytes(), source)
        np.testing.assert_array_equal(load_rgb(path), self.rgb)

    def test_opaque_rgba_preserves_rgb_pixels(self):
        alpha = np.full((2, 3, 1), 255, dtype=np.uint8)
        path = self.save_pixels("opaque.png", np.concatenate((self.rgb, alpha), axis=2))
        result = load_rgb(path)
        np.testing.assert_array_equal(result, self.rgb)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(result.flags.writeable and result.flags.owndata)

    def test_rejects_nonopaque_alpha(self):
        alpha = np.full((2, 3, 1), 255, dtype=np.uint8)
        alpha[0, 1] = 254
        path = self.save_pixels("partial.png", np.concatenate((self.rgb, alpha), axis=2))
        with self.assertRaisesRegex(ValueError, "Nonopaque alpha.*partial.png"):
            load_rgb(path)

    def test_rejects_unsupported_mode_and_metadata_names(self):
        grayscale = self.save_pixels("gray.png", self.rgb[..., 0])
        metadata = self.save_pixels("._rgb.png", self.rgb)
        with self.assertRaisesRegex(ValueError, "Unsupported colour mode 'L'"):
            load_rgb(grayscale)
        with self.assertRaisesRegex(ValueError, "Metadata filename"):
            load_rgb(metadata)

    def test_reports_missing_and_undecodable_files(self):
        corrupt = self.root / "corrupt.jpg"
        corrupt.write_bytes(b"invalid image bytes")
        with self.assertRaisesRegex(FileNotFoundError, "absent.jpg"):
            load_rgb(self.root / "absent.jpg")
        with self.assertRaisesRegex(OSError, "Cannot decode image.*corrupt.jpg"):
            load_rgb(corrupt)
