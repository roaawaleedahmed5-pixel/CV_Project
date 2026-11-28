import unittest

import numpy as np

from src.preprocessing.mean_filter import mean_filter


class MeanFilterTests(unittest.TestCase):
    def setUp(self):
        plane = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
        self.image = np.repeat(plane[..., None], 3, axis=2)

    def test_hand_calculated_centre_and_reflected_corner(self):
        result = mean_filter(self.image)
        np.testing.assert_array_equal(result[1, 1], [5, 5, 5])
        # The reflected top-left window is [[5, 4, 5], [2, 1, 2], [5, 4, 5]].
        np.testing.assert_array_equal(result[0, 0], [4, 4, 4])

    def test_constant_images_even_with_kernel_larger_than_image(self):
        image = np.empty((2, 4, 3), dtype=np.uint8)
        image[:] = [11, 127, 255]
        np.testing.assert_array_equal(mean_filter(image, 7), image)

    def test_channels_are_averaged_independently(self):
        self.image[..., 1] = 200
        self.image[..., 2] = 0
        self.image[1, 1, 2] = 90
        np.testing.assert_array_equal(mean_filter(self.image)[1, 1], [5, 200, 10])

    def test_values_near_255_do_not_overflow(self):
        image = np.full((3, 3, 3), 255, dtype=np.uint8)
        image[1, 1] = 247
        # (8 * 255 + 247) / 9 rounds to 254.
        np.testing.assert_array_equal(mean_filter(image)[1, 1], [254, 254, 254])

    def test_preserves_shape_dtype_and_input(self):
        image = self.image[:, ::-1]
        before = image.copy()
        image.flags.writeable = False
        result = mean_filter(image, np.int64(3))
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(np.shares_memory(result, image))
        np.testing.assert_array_equal(image, before)

    def test_kernel_one_returns_independent_copy_for_single_pixel(self):
        image = self.image[:1, :1]
        result = mean_filter(image, 1)
        np.testing.assert_array_equal(result, image)
        self.assertFalse(np.shares_memory(result, image))
        result[0, 0] = 0
        np.testing.assert_array_equal(image[0, 0], [1, 1, 1])

    def test_invalid_kernels(self):
        for kernel in (0, -3, 2):
            with self.subTest(kernel=kernel), self.assertRaisesRegex(ValueError, "positive odd integer"):
                mean_filter(self.image, kernel)
        for kernel in (3.0, "3", True, np.bool_(True), None):
            with self.subTest(kernel=kernel), self.assertRaisesRegex(TypeError, "positive odd integer"):
                mean_filter(self.image, kernel)

    def test_invalid_image_formats(self):
        for image in (self.image.tolist(), self.image.astype(np.float64)):
            with self.subTest(image_type=type(image)), self.assertRaisesRegex(TypeError, "NumPy uint8"):
                mean_filter(image)
        for shape in ((3, 3), (3, 3, 4), (0, 3, 3), (3, 0, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "nonempty shape"):
                mean_filter(np.zeros(shape, dtype=np.uint8))

    def test_kernel_larger_than_one_requires_both_dimensions_at_least_two(self):
        for shape in ((1, 3, 3), (3, 1, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "at least 2"):
                mean_filter(np.zeros(shape, dtype=np.uint8))
