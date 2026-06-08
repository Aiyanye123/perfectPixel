from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


COMMON_GRID_SIZES = (16, 24, 32, 48, 64, 96, 128, 160, 192, 256)


def rank_grid_candidates(
    image: np.ndarray,
    seeds: Iterable[tuple[int | None, int | None, str]],
    min_cell_size: float = 4.0,
    max_cell_ratio: float = 1.5,
    max_candidates: int = 18,
    top_k: int = 3,
) -> dict:
    """Generate and rank plausible regular grids using inexpensive global scores."""
    height, width = image.shape[:2]
    sources: dict[tuple[int, int], set[str]] = defaultdict(set)
    seed_list = [
        (int(grid_w), int(grid_h), source)
        for grid_w, grid_h, source in seeds
        if grid_w is not None and grid_h is not None
    ]

    for grid_w, grid_h, source in seed_list:
        _add_candidate_family(sources, grid_w, grid_h, source)

    for grid_w, _, width_source in seed_list:
        for _, grid_h, height_source in seed_list:
            if width_source != height_source:
                _add_candidate(
                    sources, grid_w, grid_h, f"mixed:{width_source}+{height_source}"
                )

    for common_w in COMMON_GRID_SIZES:
        common_h = max(2, int(round(common_w * height / width)))
        _add_candidate(sources, common_w, common_h, "common")

    detail_w = max(2, int(width // min_cell_size))
    detail_h = max(2, int(height // min_cell_size))
    _add_candidate(sources, detail_w, detail_h, "detail_limit")

    valid = [
        (grid_w, grid_h, sorted(candidate_sources))
        for (grid_w, grid_h), candidate_sources in sources.items()
        if _is_valid_candidate(
            grid_w, grid_h, width, height, min_cell_size, max_cell_ratio
        )
    ]
    if not valid:
        return {"best": None, "alternatives": [], "confidence": 0.0}

    proxy = _make_proxy(image, max_side=640)
    gray = _to_gray(proxy)
    saliency = _saliency_weights(proxy)
    grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    profile_x = grad_x.mean(axis=0)
    profile_y = grad_y.mean(axis=1)

    quick_ranked = []
    for grid_w, grid_h, candidate_sources in valid:
        x_score = _axis_grid_score(profile_x, grid_w)
        y_score = _axis_grid_score(profile_y, grid_h)
        quick_score = (
            0.55 * (x_score["boundary_loss"] + y_score["boundary_loss"]) / 2
            + 0.25 * (x_score["interior_ratio"] + y_score["interior_ratio"]) / 2
            + 0.20 * (x_score["irregularity"] + y_score["irregularity"]) / 2
        )
        quick_ranked.append(
            {
                "grid_width": grid_w,
                "grid_height": grid_h,
                "sources": candidate_sources,
                "boundary_loss": (x_score["boundary_loss"] + y_score["boundary_loss"]) / 2,
                "internal_edge_ratio": (
                    x_score["interior_ratio"] + y_score["interior_ratio"]
                )
                / 2,
                "grid_irregularity": (x_score["irregularity"] + y_score["irregularity"]) / 2,
                "quick_score": quick_score,
            }
        )

    quick_ranked.sort(key=lambda item: (item["quick_score"], -len(item["sources"])))
    finalists = quick_ranked[:max_candidates]
    detail_candidates = sorted(
        quick_ranked,
        key=lambda item: item["grid_width"] * item["grid_height"],
        reverse=True,
    )[:4]
    finalist_keys = {(item["grid_width"], item["grid_height"]) for item in finalists}
    finalists.extend(
        item
        for item in detail_candidates
        if (item["grid_width"], item["grid_height"]) not in finalist_keys
    )
    pixel_count = max(1, height * width)

    for candidate in finalists:
        grid_w = candidate["grid_width"]
        grid_h = candidate["grid_height"]
        render_error, render_model = _fast_render_error(proxy, grid_w, grid_h, saliency)
        complexity = min(1.0, grid_w * grid_h / pixel_count)
        detail_loss = _detail_loss(proxy, grid_w, grid_h)
        candidate["render_error"] = render_error
        candidate["render_model"] = render_model
        candidate["complexity"] = complexity
        candidate["detail_loss"] = detail_loss
        candidate["score"] = (
            0.56 * render_error
            + 0.14 * candidate["boundary_loss"]
            + 0.06 * candidate["internal_edge_ratio"]
            + 0.06 * candidate["grid_irregularity"]
            + 0.04 * complexity
            + 0.14 * detail_loss
        )

    finalists.sort(key=lambda item: (item["score"], -len(item["sources"])))
    finalists = _prefer_detail_when_scores_are_close(finalists)
    selected = [_public_candidate(item) for item in _select_diverse_candidates(finalists, top_k)]
    return {
        "best": selected[0] if selected else None,
        "alternatives": selected,
        "confidence": _estimate_confidence(selected),
    }


def _add_candidate_family(
    sources: dict[tuple[int, int], set[str]], grid_w: int, grid_h: int, source: str
) -> None:
    for factor, suffix in ((0.5, "half"), (1.0, "base"), (2.0, "double")):
        width = max(2, int(round(grid_w * factor)))
        height = max(2, int(round(grid_h * factor)))
        _add_candidate(sources, width, height, f"{source}:{suffix}")
        if factor == 1.0:
            for delta_w, delta_h in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                _add_candidate(sources, width + delta_w, height + delta_h, f"{source}:nearby")


def _add_candidate(
    sources: dict[tuple[int, int], set[str]], grid_w: int, grid_h: int, source: str
) -> None:
    if grid_w >= 2 and grid_h >= 2:
        sources[(grid_w, grid_h)].add(source)


def _is_valid_candidate(
    grid_w: int,
    grid_h: int,
    image_w: int,
    image_h: int,
    min_cell_size: float,
    max_cell_ratio: float,
) -> bool:
    if grid_w < 2 or grid_h < 2:
        return False
    cell_w = image_w / grid_w
    cell_h = image_h / grid_h
    return (
        cell_w >= min_cell_size
        and cell_h >= min_cell_size
        and cell_w / cell_h <= max_cell_ratio
        and cell_h / cell_w <= max_cell_ratio
    )


def _make_proxy(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    stride = max(1, int(math.ceil(max(height, width) / max_side)))
    proxy = image[::stride, ::stride]
    return proxy.astype(np.float32) / 255.0 if image.dtype == np.uint8 else proxy.astype(np.float32)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    return (
        0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    ).astype(np.float32)


def _axis_grid_score(profile: np.ndarray, grid_count: int) -> dict[str, float]:
    length = len(profile)
    if length < 3 or grid_count < 2:
        return {"boundary_loss": 1.0, "interior_ratio": 1.0, "irregularity": 1.0}

    mean_gradient = float(profile.mean()) + 1e-8
    cell_size = length / grid_count
    radius = max(1, int(round(min(3.0, cell_size * 0.25))))
    ideal = np.arange(1, grid_count, dtype=np.float32) * cell_size
    snapped = []
    boundary_values = []
    for position in ideal:
        center = int(round(position))
        left = max(0, center - radius)
        right = min(length, center + radius + 1)
        if right <= left:
            continue
        local = profile[left:right]
        offset = int(np.argmax(local))
        snapped.append(left + offset)
        boundary_values.append(float(local[offset]))

    centers = np.clip(
        np.rint((np.arange(grid_count, dtype=np.float32) + 0.5) * cell_size).astype(int),
        0,
        length - 1,
    )
    boundary_ratio = np.mean(boundary_values) / mean_gradient if boundary_values else 0.0
    interior_ratio = float(np.mean(profile[centers]) / mean_gradient)
    if len(snapped) >= 2:
        irregularity = float(
            np.mean(np.abs(np.diff(np.asarray(snapped, dtype=np.float32)) - cell_size))
            / max(cell_size, 1e-8)
        )
    else:
        irregularity = 1.0

    return {
        "boundary_loss": float(1.0 - np.clip(boundary_ratio / 3.0, 0.0, 1.0)),
        "interior_ratio": float(np.clip(interior_ratio / 3.0, 0.0, 1.0)),
        "irregularity": float(np.clip(irregularity, 0.0, 1.0)),
    }


def _fast_render_error(
    image: np.ndarray, grid_w: int, grid_h: int, saliency: np.ndarray
) -> tuple[float, str]:
    height, width = image.shape[:2]
    if grid_w > width or grid_h > height:
        return 1.0, "proxy_too_small"
    center_x = np.clip(
        np.rint((np.arange(grid_w, dtype=np.float32) + 0.5) * width / grid_w).astype(int),
        0,
        width - 1,
    )
    center_y = np.clip(
        np.rint((np.arange(grid_h, dtype=np.float32) + 0.5) * height / grid_h).astype(int),
        0,
        height - 1,
    )
    low_resolution = image[center_y[:, None], center_x[None, :]]
    map_x = np.clip((np.arange(width) * grid_w / width).astype(int), 0, grid_w - 1)
    map_y = np.clip((np.arange(height) * grid_h / height).astype(int), 0, grid_h - 1)
    rendered = low_resolution[map_y[:, None], map_x[None, :]]

    models = {
        "nearest": rendered,
        "box_blur": _box_blur(rendered),
    }
    best_error = float("inf")
    best_model = "nearest"
    for model_name, model_image in models.items():
        error = _best_shifted_mae(image, model_image, saliency)
        if error < best_error:
            best_error = error
            best_model = model_name
    return float(np.clip(best_error, 0.0, 1.0)), best_model


def _detail_loss(image: np.ndarray, grid_w: int, grid_h: int) -> float:
    gray = _to_gray(image)
    source_gradient = (
        np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()
    )
    if source_gradient <= 1e-8:
        return 0.0

    height, width = gray.shape
    if grid_w > width or grid_h > height:
        return 1.0
    center_x = np.clip(
        np.rint((np.arange(grid_w, dtype=np.float32) + 0.5) * width / grid_w).astype(int),
        0,
        width - 1,
    )
    center_y = np.clip(
        np.rint((np.arange(grid_h, dtype=np.float32) + 0.5) * height / grid_h).astype(int),
        0,
        height - 1,
    )
    low_resolution = gray[center_y[:, None], center_x[None, :]]
    map_x = np.clip((np.arange(width) * grid_w / width).astype(int), 0, grid_w - 1)
    map_y = np.clip((np.arange(height) * grid_h / height).astype(int), 0, grid_h - 1)
    rendered = low_resolution[map_y[:, None], map_x[None, :]]
    rendered_gradient = (
        np.abs(np.diff(rendered, axis=1)).mean() + np.abs(np.diff(rendered, axis=0)).mean()
    )
    return float(np.clip(1.0 - rendered_gradient / source_gradient, 0.0, 1.0))


def _box_blur(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    else:
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for y_offset in range(3):
        for x_offset in range(3):
            output += padded[
                y_offset : y_offset + image.shape[0],
                x_offset : x_offset + image.shape[1],
            ]
    return output / 9.0


def _saliency_weights(image: np.ndarray) -> np.ndarray:
    color_image = image[..., None] if image.ndim == 2 else image
    gray = _to_gray(image)
    gradient = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gradient += np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gradient /= float(np.percentile(gradient, 95)) + 1e-8

    border = np.concatenate(
        (
            color_image[0],
            color_image[-1],
            color_image[:, 0],
            color_image[:, -1],
        ),
        axis=0,
    )
    background = np.median(border, axis=0)
    foreground = np.sqrt(np.sum((color_image - background) ** 2, axis=2))
    foreground /= float(np.percentile(foreground, 90)) + 1e-8
    return 0.15 + 0.45 * np.clip(foreground, 0.0, 1.0) + 0.40 * np.clip(gradient, 0.0, 1.0)


def _best_shifted_mae(
    reference: np.ndarray, rendered: np.ndarray, weights: np.ndarray
) -> float:
    best = float("inf")
    for shift_y in (-1, 0, 1):
        for shift_x in (-1, 0, 1):
            ref_y, out_y = _overlap_slices(reference.shape[0], shift_y)
            ref_x, out_x = _overlap_slices(reference.shape[1], shift_x)
            difference = np.abs(reference[ref_y, ref_x] - rendered[out_y, out_x])
            if difference.ndim == 3:
                difference = difference.mean(axis=2)
            local_weights = weights[ref_y, ref_x]
            error = float(
                np.sum(difference * local_weights) / (np.sum(local_weights) + 1e-8)
            )
            best = min(best, error)
    return best


def _prefer_detail_when_scores_are_close(candidates: list[dict]) -> list[dict]:
    if len(candidates) < 2:
        return candidates
    best_score = candidates[0]["score"]
    current = candidates[0]
    competitive = [
        candidate
        for candidate in candidates
        if candidate["score"] <= best_score + max(0.018, best_score * 0.12)
        and (
            candidate["detail_loss"] + 0.06 < current["detail_loss"]
            or candidate["render_error"] + 0.015 < current["render_error"]
        )
    ]
    if not competitive:
        return candidates

    conservative = max(
        competitive,
        key=lambda item: (
            item["grid_width"] * item["grid_height"],
            -item["score"],
        ),
    )
    if conservative is candidates[0]:
        return candidates
    return [conservative] + [candidate for candidate in candidates if candidate is not conservative]


def _select_diverse_candidates(candidates: list[dict], top_k: int) -> list[dict]:
    selected = list(candidates[:top_k])
    if len(selected) < top_k or not candidates:
        return selected

    best_area = candidates[0]["grid_width"] * candidates[0]["grid_height"]
    detailed = [
        candidate
        for candidate in candidates
        if candidate["grid_width"] * candidate["grid_height"] >= best_area * 4
    ]
    if not detailed:
        return selected

    detail_alternative = min(detailed, key=lambda item: item["score"])
    if detail_alternative not in selected:
        selected[-1] = detail_alternative
    return selected


def _overlap_slices(length: int, shift: int) -> tuple[slice, slice]:
    if shift < 0:
        return slice(0, length + shift), slice(-shift, length)
    if shift > 0:
        return slice(shift, length), slice(0, length - shift)
    return slice(0, length), slice(0, length)


def _public_candidate(candidate: dict) -> dict:
    return {
        "grid_width": int(candidate["grid_width"]),
        "grid_height": int(candidate["grid_height"]),
        "score": round(float(candidate["score"]), 5),
        "render_error": round(float(candidate["render_error"]), 5),
        "render_model": candidate["render_model"],
        "boundary_alignment": round(1.0 - float(candidate["boundary_loss"]), 5),
        "internal_edge_ratio": round(float(candidate["internal_edge_ratio"]), 5),
        "grid_irregularity": round(float(candidate["grid_irregularity"]), 5),
        "complexity": round(float(candidate["complexity"]), 5),
        "detail_loss": round(float(candidate["detail_loss"]), 5),
        "sources": candidate["sources"],
    }


def _estimate_confidence(candidates: list[dict]) -> float:
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return 0.35
    best, second = candidates[:2]
    gap = max(0.0, second["score"] - best["score"])
    relative_gap = np.clip(gap / max(second["score"], 0.05), 0.0, 1.0)
    detector_agreement = np.clip(len(best["sources"]) / 3.0, 0.0, 1.0)
    return round(float(0.75 * relative_gap + 0.25 * detector_agreement), 4)
