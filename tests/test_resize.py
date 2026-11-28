import unittest

import numpy as np

from src.preprocessing.resize import resize_rgb


class ResizeTests(unittest.TestCase):
    def test_target_shape_dtype_and_input_preservation(self):
        image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[:, ::-1]
        original = image.copy()
        image.flags.writeable = False
        for interpolation in ("nearest", "bilinear", "bicubic"):
            with self.subTest(interpolation=interpolation):
                output = resize_rgb(image, 7, 5, interpolation=interpolation)
                self.assertEqual(output.shape, (5, 7, 3))
                self.assertEqual(output.dtype, np.uint8)
                self.assertTrue(output.flags.writeable)
                self.assertFalse(np.shares_memory(output, image))
                np.testing.assert_array_equal(image, original)

    def test_identical_size_stretch_preserves_pixels_in_an_independent_array(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        resized = resize_rgb(image, 4, 2)
        np.testing.assert_array_equal(resized, image)
        self.assertFalse(np.shares_memory(resized, image))

    def test_known_nearest_neighbour_enlargement(self):
        image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
        expected = image.repeat(2, axis=0).repeat(2, axis=1)
        np.testing.assert_array_equal(resize_rgb(image, 4, 4, interpolation="nearest"), expected)

    def test_constant_image_for_all_interpolations_and_policies(self):
        colour = (13, 150, 250)
        image = np.full((5, 8, 3), colour, dtype=np.uint8)
        for interpolation in ("nearest", "bilinear", "bicubic"):
            for policy in ("stretch", "pad", "center_crop"):
                with self.subTest(interpolation=interpolation, policy=policy):
                    output = resize_rgb(
                        image, 3, 4, interpolation=interpolation,
                        aspect_ratio_policy=policy, padding_colour=colour,
                    )
                    np.testing.assert_array_equal(output, np.full((4, 3, 3), colour, dtype=np.uint8))

    def test_pads_with_requested_colour_and_extra_row_at_bottom(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        colour = (17, 18, 19)
        expected = np.full((5, 4, 3), colour, dtype=np.uint8)
        expected[1:3] = image
        np.testing.assert_array_equal(
            resize_rgb(image, 4, 5, aspect_ratio_policy="pad", padding_colour=colour), expected,
        )

    def test_pad_rounds_half_up(self):
        image = np.full((1, 2, 3), 200, dtype=np.uint8)
        result = resize_rgb(image, 3, 4, aspect_ratio_policy="pad")
        np.testing.assert_array_equal(result[:, 0, 0], [0, 200, 200, 0])

    def test_center_crop_discards_odd_surplus_at_the_right_or_bottom(self):
        image = np.arange(30, dtype=np.uint8).reshape(2, 5, 3)
        np.testing.assert_array_equal(
            resize_rgb(image, 2, 2, aspect_ratio_policy="center_crop"), image[:, 1:3],
        )
        portrait = image.transpose(1, 0, 2)
        np.testing.assert_array_equal(
            resize_rgb(portrait, 2, 2, aspect_ratio_policy="center_crop"), portrait[1:3],
        )

    def test_invalid_arguments(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(TypeError, "NumPy uint8"):
            resize_rgb(image.astype(float), 2, 2)
        with self.assertRaisesRegex(ValueError, "nonempty shape"):
            resize_rgb(image[:, :, 0], 2, 2)
        for width, height, error_type in ((True, 2, TypeError), (2, 1.5, TypeError), (0, 2, ValueError), (2, -1, ValueError)):
            with self.subTest(width=width, height=height), self.assertRaisesRegex(error_type, "positive integer"):
                resize_rgb(image, width, height)
        for settings in (
            {"interpolation": "lanczos"}, {"aspect_ratio_policy": "fit"},
            {"padding_colour": (0, 0)}, {"padding_colour": (0, 0, 256)}, {"padding_colour": (0, 0, True)},
        ):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, next(iter(settings))):
                resize_rgb(image, 2, 2, **settings)
