from __future__ import annotations

import numpy as np


def sample_adaptive(image: np.ndarray, x_coords, y_coords) -> np.ndarray:
    """Recover cell colors while down-weighting boundary bleed and color outliers."""
    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)
    if _use_dense_grid_fast_path(x, y):
        return _sample_centers(image, x, y)

    source = image.astype(np.float32)
    if source.ndim == 2:
        source = source[..., None]

    height, width = source.shape[:2]
    channels = source.shape[2]
    output = np.empty((len(y) - 1, len(x) - 1, channels), dtype=np.float32)

    for row in range(len(y) - 1):
        y0, y1 = _bounded_cell(y[row], y[row + 1], height)
        for column in range(len(x) - 1):
            x0, x1 = _bounded_cell(x[column], x[column + 1], width)
            cell = source[y0:y1, x0:x1]
            output[row, column] = _robust_cell_color(cell)

    if image.dtype == np.uint8:
        output = np.clip(np.rint(output), 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return output[..., 0]
    return output


def _use_dense_grid_fast_path(x: np.ndarray, y: np.ndarray) -> bool:
    return float(np.median(np.diff(x))) <= 5.0 or float(np.median(np.diff(y))) <= 5.0


def _sample_centers(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centers_x = np.clip((x[1:] + x[:-1]) // 2, 0, image.shape[1] - 1)
    centers_y = np.clip((y[1:] + y[:-1]) // 2, 0, image.shape[0] - 1)
    return image[centers_y[:, None], centers_x[None, :]]


def _bounded_cell(start: int, end: int, limit: int) -> tuple[int, int]:
    start = int(np.clip(start, 0, limit))
    end = int(np.clip(end, 0, limit))
    if end <= start:
        end = min(start + 1, limit)
    return start, end


def _robust_cell_color(cell: np.ndarray) -> np.ndarray:
    height, width = cell.shape[:2]
    pixels = cell.reshape(-1, cell.shape[2])
    if len(pixels) == 0:
        return np.zeros(cell.shape[2], dtype=np.float32)
    if height <= 2 or width <= 2:
        return pixels[len(pixels) // 2]

    spatial_weights = _spatial_weights(height, width).reshape(-1)
    estimate = np.asarray(
        [_weighted_median(pixels[:, channel], spatial_weights) for channel in range(pixels.shape[1])],
        dtype=np.float32,
    )
    value_scale = 255.0 if float(np.max(pixels)) > 2.0 else 1.0

    for _ in range(3):
        distances = np.sqrt(np.sum((pixels - estimate) ** 2, axis=1))
        scale = _weighted_median(distances, spatial_weights) * 2.5 + value_scale * 0.012
        robust_weights = 1.0 / (1.0 + (distances / scale) ** 2)
        estimate = _weighted_mean(pixels, spatial_weights * robust_weights)
    return estimate


def _spatial_weights(height: int, width: int) -> np.ndarray:
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    center_y = np.sin(np.pi * y)
    center_x = np.sin(np.pi * x)
    center_weight = np.sqrt(center_y[:, None] * center_x[None, :])
    return 0.08 + 0.92 * center_weight


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 1e-8:
        return np.mean(values, axis=0)
    return np.sum(values * weights[:, None], axis=0) / total


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] * 0.5, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])
