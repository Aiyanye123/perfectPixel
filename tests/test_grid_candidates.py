import unittest

import numpy as np

from perfect_pixel.grid_candidates import rank_grid_candidates


class GridCandidateTests(unittest.TestCase):
    def test_ranking_prefers_source_grid_over_denser_harmonic(self):
        rng = np.random.default_rng(7)
        pixel_art = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        image = np.repeat(np.repeat(pixel_art, 8, axis=0), 8, axis=1)

        ranking = rank_grid_candidates(
            image,
            (
                (16, 16, "fft"),
                (32, 32, "gradient"),
            ),
        )

        self.assertEqual(
            (ranking["best"]["grid_width"], ranking["best"]["grid_height"]),
            (16, 16),
        )
        self.assertIn("complexity", ranking["best"])

    def test_ranking_returns_explainable_top_three(self):
        pixel_art = np.zeros((24, 24, 3), dtype=np.uint8)
        pixel_art[4:20, 6:18] = (220, 70, 40)
        image = np.repeat(np.repeat(pixel_art, 6, axis=0), 6, axis=1)

        ranking = rank_grid_candidates(image, ((24, 24, "fft"),))

        self.assertLessEqual(len(ranking["alternatives"]), 3)
        self.assertGreaterEqual(ranking["confidence"], 0.0)
        self.assertLessEqual(ranking["confidence"], 1.0)
        self.assertIn("render_error", ranking["best"])
        self.assertIn("complexity", ranking["best"])
        self.assertIn("sources", ranking["best"])

    def test_ranking_preserves_detailed_foreground_on_plain_background(self):
        rng = np.random.default_rng(11)
        image = np.full((512, 512, 3), 248, dtype=np.uint8)
        detail = rng.integers(20, 220, size=(64, 64, 3), dtype=np.uint8)
        image[128:384, 128:384] = np.repeat(np.repeat(detail, 4, axis=0), 4, axis=1)

        ranking = rank_grid_candidates(
            image,
            (
                (24, 24, "fft"),
                (128, 128, "gradient"),
            ),
        )

        self.assertGreaterEqual(ranking["best"]["grid_width"], 96)
        self.assertGreaterEqual(ranking["best"]["grid_height"], 96)

    def test_top_three_includes_a_detailed_escape_candidate(self):
        image = np.full((512, 512, 3), 245, dtype=np.uint8)
        image[96:416, 96:416:2] = 25

        ranking = rank_grid_candidates(image, ((24, 24, "fft"),))

        self.assertTrue(
            any(candidate["grid_width"] >= 96 for candidate in ranking["alternatives"])
        )


if __name__ == "__main__":
    unittest.main()
