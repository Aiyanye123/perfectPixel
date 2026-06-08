import unittest

import numpy as np

from perfect_pixel import perfect_pixel, perfect_pixel_noCV2


class GridRefinementTests(unittest.TestCase):
    def test_opencv_refinement_preserves_requested_grid_count(self):
        image = self._detailed_square_image()

        x_coords, y_coords = perfect_pixel.refine_grids(image, 313, 313, 0.25)

        self.assertEqual(len(x_coords), 314)
        self.assertEqual(len(y_coords), 314)
        self.assertEqual((x_coords[0], x_coords[-1]), (0, 1254))
        self.assertEqual((y_coords[0], y_coords[-1]), (0, 1254))
        self.assertTrue(np.all(np.diff(x_coords) > 0))
        self.assertTrue(np.all(np.diff(y_coords) > 0))

    def test_numpy_refinement_preserves_requested_grid_count(self):
        image = self._detailed_square_image()

        x_coords, y_coords = perfect_pixel_noCV2.refine_grids(image, 313, 313, 0.25)

        self.assertEqual(len(x_coords), 314)
        self.assertEqual(len(y_coords), 314)
        self.assertTrue(np.all(np.diff(x_coords) > 0))
        self.assertTrue(np.all(np.diff(y_coords) > 0))

    def test_manual_square_grid_produces_square_output(self):
        image = self._detailed_square_image()

        width, height, output = perfect_pixel.get_perfect_pixel(
            image,
            sample_method="center",
            grid_size=(313, 313),
            refine_intensity=0.25,
        )

        self.assertEqual((width, height), (313, 313))
        self.assertEqual(output.shape[:2], (313, 313))

    @staticmethod
    def _detailed_square_image():
        y, x = np.indices((1254, 1254))
        base = ((x * 17 + y * 29) % 256).astype(np.uint8)
        return np.stack((base, np.roll(base, 3, axis=0), np.roll(base, 5, axis=1)), axis=2)


if __name__ == "__main__":
    unittest.main()
