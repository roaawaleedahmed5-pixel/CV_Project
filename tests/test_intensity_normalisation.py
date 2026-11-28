import unittest

import numpy as np

from src.preprocessing.intensity_normalisation import normalise_intensity


class IntensityNormalisationTests(unittest.TestCase):
    def test_hand_calculated_example_uses_shared_channel_statistics(self):
        image = np.array([[[0, 1, 2]]], dtype=np.uint8)
        expected = np.array([[[-1, 0, 1]]]) / np.sqrt(2 / 3)
        np.testing.assert_allclose(normalise_intensity(image), expected, rtol=1e-7)

    def test_global_moments_shape_dtype_finiteness_and_preservation(self):
        image = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)[:, ::-1]
        original = image.copy()
        image.flags.writeable = False
        result = normalise_intensity(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertFalse(np.shares_memory(result, image))
        self.assertAlmostEqual(float(result.mean(dtype=np.float64)), 0.0, places=7)
        self.assertAlmostEqual(float(result.std(dtype=np.float64, ddof=0)), 1.0, places=7)
        np.testing.assert_array_equal(image, original)

    def test_constant_returns_zeros_and_epsilon_can_include_nonconstant_images(self):
        constant = np.full((3, 4, 3), 120, dtype=np.uint8)
        result = normalise_intensity(constant, epsilon=0)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.zeros_like(constant))
        image = np.array([[[0, 2, 0], [2, 0, 2]]], dtype=np.uint8)
        np.testing.assert_array_equal(normalise_intensity(image, epsilon=1), np.zeros_like(image))

    def test_positive_affine_transform_without_clipping_or_rounding(self):
        image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        transformed = 3 * image.astype(np.int16) + 20
        self.assertLessEqual(int(transformed.max()), 255)
        np.testing.assert_allclose(
            normalise_intensity(transformed.astype(np.uint8)), normalise_intensity(image), atol=1e-7,
        )

    def test_invalid_arguments(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(TypeError, "NumPy uint8"):
            normalise_intensity(image.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "nonempty shape"):
            normalise_intensity(image[:0])
        for epsilon in (-1, np.nan, np.inf):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(ValueError, "epsilon"):
                normalise_intensity(image, epsilon)
        for epsilon in (True, "1", None):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(TypeError, "epsilon"):
                normalise_intensity(image, epsilon)
