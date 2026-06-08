import unittest

import numpy as np

from perfect_pixel.adaptive_sampling import sample_adaptive


class AdaptiveSamplingTests(unittest.TestCase):
    def test_preserves_uniform_cell_colors(self):
        pixel_art = np.array(
            [
                [[220, 40, 30], [20, 160, 90]],
                [[35, 70, 210], [240, 190, 45]],
            ],
            dtype=np.uint8,
        )
        image = np.repeat(np.repeat(pixel_art, 8, axis=0), 8, axis=1)

        output = sample_adaptive(image, [0, 8, 16], [0, 8, 16])

        np.testing.assert_array_equal(output, pixel_art)

    def test_downweights_boundary_color_bleed(self):
        target = np.array([210, 55, 35], dtype=np.uint8)
        bleed = np.array([30, 75, 205], dtype=np.uint8)
        image = np.full((10, 10, 3), bleed, dtype=np.uint8)
        image[2:8, 2:8] = target

        output = sample_adaptive(image, [0, 10], [0, 10])[0, 0].astype(np.float32)
        raw_mean = image.reshape(-1, 3).mean(axis=0)

        self.assertLess(
            np.linalg.norm(output - target.astype(np.float32)),
            np.linalg.norm(raw_mean - target.astype(np.float32)),
        )

    def test_supports_float_images(self):
        image = np.full((6, 6, 3), 0.25, dtype=np.float32)

        output = sample_adaptive(image, [0, 6], [0, 6])

        np.testing.assert_allclose(output[0, 0], [0.25, 0.25, 0.25], atol=1e-5)

    def test_dense_grid_fast_path_preserves_dimensions(self):
        image = np.zeros((1254, 1254, 3), dtype=np.uint8)
        coords = np.rint(np.linspace(0, 1254, 314)).astype(int)

        output = sample_adaptive(image, coords, coords)

        self.assertEqual(output.shape[:2], (313, 313))


if __name__ == "__main__":
    unittest.main()
