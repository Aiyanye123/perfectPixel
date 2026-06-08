
"""
Perfect Pixel: A library for auto grid detection and pixel art refinement.
"""

__version__ = "0.1.2"

from .perfect_pixel_noCV2 import get_perfect_pixel as _get_perfect_pixel_numpy
from .perfect_pixel_noCV2 import detect_grid_candidates as _detect_grid_candidates_numpy

try:
    import cv2
    from .perfect_pixel import get_perfect_pixel as _get_perfect_pixel_opencv
    from .perfect_pixel import detect_grid_candidates as _detect_grid_candidates_opencv
    get_perfect_pixel = _get_perfect_pixel_opencv
    detect_grid_candidates = _detect_grid_candidates_opencv
except ImportError:
    _get_perfect_pixel_opencv = None
    _detect_grid_candidates_opencv = None
    get_perfect_pixel = _get_perfect_pixel_numpy
    detect_grid_candidates = _detect_grid_candidates_numpy

__all__ = ["detect_grid_candidates", "get_perfect_pixel"]
