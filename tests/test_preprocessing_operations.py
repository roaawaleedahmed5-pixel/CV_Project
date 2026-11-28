import unittest
from unittest.mock import patch

import numpy as np

from src.preprocessing.contrast_stretching import contrast_stretch
from src.preprocessing.gaussian_filter import gaussian_filter, gaussian_kernel
from src.preprocessing.histogram_equalisation import histogram_equalise
from src.preprocessing.median_filter import median_filter


def rgb_from_plane(plane):
    return np.repeat(np.asarray(plane, dtype=np.uint8)[..., None], 3, axis=2)


class GaussianTests(unittest.TestCase):
    def test_kernel_symmetry_and_unit_sum(self):
        for sigma in (1.0, 1.5):
            with self.subTest(sigma=sigma):
                weights = gaussian_kernel(7, sigma)
                self.assertEqual(weights.shape, (7,))
                self.assertEqual(weights.dtype, np.float64)
                np.testing.assert_array_equal(weights, weights[::-1])
                self.assertAlmostEqual(float(weights.sum()), 1.0)
                self.assertGreater(weights[3], weights[0])

    def test_preserves_constant_values_on_a_singleton_axis(self):
        image = np.full((1, 3, 3), [2, 127, 255], dtype=np.uint8)
        np.testing.assert_array_equal(gaussian_filter(image), image)

    def test_impulse_matches_expected_weights(self):
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        image[3, 3, 0] = 255
        result = gaussian_filter(image, kernel_size=3, sigma=1)
        edge_weight = np.exp(-0.5)
        expected = np.zeros_like(result)
        expected[2:5, 2:5, 0] = np.rint(
            255 / (1 + 2 * edge_weight) ** 2
            * np.array([[edge_weight**2, edge_weight, edge_weight**2],
                        [edge_weight, 1, edge_weight],
                        [edge_weight**2, edge_weight, edge_weight**2]])
        )
        np.testing.assert_array_equal(result, expected)

    def test_rejects_invalid_sigma(self):
        image = rgb_from_plane([[0]])
        for sigma in (0, -1, np.inf, np.nan):
            with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, "sigma"):
                gaussian_filter(image, kernel_size=1, sigma=sigma)
        for sigma in (True, "1", 1j):
            with self.subTest(sigma=sigma), self.assertRaisesRegex(TypeError, "sigma"):
                gaussian_filter(image, sigma=sigma)


class MedianTests(unittest.TestCase):
    def test_removes_impulse_and_preserves_independent_channel_medians(self):
        image = np.full((5, 5, 3), [10, 20, 0], dtype=np.uint8)
        image[..., 2] = [0, 50, 100, 150, 200]
        image[2, 2, :2] = [255, 0]
        expected = np.full_like(image, [10, 20, 0])
        expected[..., 2] = [50, 50, 100, 150, 150]
        # Force tile boundaries across both axes on this small fixture.
        with patch("src.preprocessing.median_filter._TILE_VALUES", 18):
            result = median_filter(image)
        np.testing.assert_array_equal(result, expected)


class ContrastTests(unittest.TestCase):
    def test_rgb_endpoints_and_constant_channel(self):
        image = np.array([[[10, 200, 80], [20, 200, 120], [30, 200, 160]]], dtype=np.uint8)
        expected = [[[0, 200, 0], [128, 200, 128], [255, 200, 255]]]
        np.testing.assert_array_equal(contrast_stretch(image, mode="rgb_channels"), expected)

    def test_hsv_stretches_value_and_keeps_colour_ratios(self):
        image = np.array([[[20, 5, 0], [90, 22, 0], [160, 40, 0]]], dtype=np.uint8)
        expected = [[[0, 0, 0], [128, 31, 0], [255, 64, 0]]]
        np.testing.assert_array_equal(contrast_stretch(image), expected)

    def test_rejects_unsupported_mode(self):
        image = rgb_from_plane([[0]])
        with self.assertRaisesRegex(ValueError, "mode"):
            contrast_stretch(image, mode="rgb")
        with self.assertRaisesRegex(TypeError, "mode"):
            contrast_stretch(image, mode=None)


class EqualisationTests(unittest.TestCase):
    def test_matches_hand_calculated_cumulative_counts(self):
        image = rgb_from_plane([[30, 30, 80], [150, 150, 230]])
        # Occupied CDF counts are 2, 3, 5, 6. Subtract 2 and divide by 6 - 2.
        expected = rgb_from_plane([[0, 0, 64], [191, 191, 255]])
        np.testing.assert_array_equal(histogram_equalise(image), expected)

    def test_equalises_value_and_keeps_colour_ratios(self):
        image = np.array([[[60, 20, 0], [120, 40, 0], [180, 60, 0]]], dtype=np.uint8)
        expected = [[[0, 0, 0], [128, 42, 0], [255, 85, 0]]]
        np.testing.assert_array_equal(histogram_equalise(image), expected)


class PreprocessingContractTests(unittest.TestCase):
    operations = (gaussian_filter, median_filter, contrast_stretch, histogram_equalise)

    def test_preserves_shape_dtype_range_and_input(self):
        image = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)[:, ::-1]
        original = image.copy()
        image.flags.writeable = False
        for operation in self.operations:
            with self.subTest(operation=operation.__name__):
                result = operation(image)
                self.assertEqual(result.shape, image.shape)
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(0 <= int(result.min()) <= int(result.max()) <= 255)
                self.assertFalse(np.shares_memory(result, image))
                np.testing.assert_array_equal(image, original)

    def test_rejects_invalid_formats_for_every_operation(self):
        for operation in self.operations:
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(TypeError, "NumPy uint8"):
                    operation(np.zeros((3, 3, 3), dtype=np.float64))
                for shape in ((3, 3), (3, 3, 4), (0, 3, 3)):
                    with self.assertRaisesRegex(ValueError, "nonempty shape"):
                        operation(np.zeros(shape, dtype=np.uint8))

    def test_spatial_filters_reject_invalid_kernels_and_copy_for_one(self):
        image = rgb_from_plane([[7]])
        for operation in (gaussian_filter, median_filter):
            with self.subTest(operation=operation.__name__):
                for size in (0, -3, 2):
                    with self.assertRaisesRegex(ValueError, "positive odd integer"):
                        operation(image, kernel_size=size)
                for size in (True, 3.0):
                    with self.assertRaisesRegex(TypeError, "positive odd integer"):
                        operation(image, kernel_size=size)
                result = operation(image, kernel_size=1)
                np.testing.assert_array_equal(result, image)
                self.assertFalse(np.shares_memory(result, image))

    def test_constant_hsv_value_returns_original_colours_in_a_copy(self):
        # The colours differ, but every pixel has V = 64.
        image = np.array([[[7, 64, 1], [64, 18, 12], [9, 20, 64]]], dtype=np.uint8)
        for operation in (contrast_stretch, histogram_equalise):
            with self.subTest(operation=operation.__name__):
                result = operation(image)
                np.testing.assert_array_equal(result, image)
                self.assertFalse(np.shares_memory(result, image))
