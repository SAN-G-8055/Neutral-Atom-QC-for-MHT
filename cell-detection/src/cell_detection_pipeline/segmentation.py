"""Deterministic, CPU-friendly cell segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config import SegmentationConfig


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    corrected_image: np.ndarray
    otsu_threshold: float
    high_threshold: float
    low_threshold: float
    seed_count: int
    detection_count: int


def otsu_threshold(
    image: np.ndarray,
    *,
    bins: int = 256,
    percentile_low: float = 0.1,
    percentile_high: float = 99.9,
) -> float:
    """Compute Otsu's threshold after clipping extreme histogram tails."""

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot threshold an image with no finite values")
    low, high = np.percentile(finite, (percentile_low, percentile_high))
    if high <= low:
        return float(low)
    clipped = np.clip(finite, low, high)
    histogram, edges = np.histogram(clipped, bins=bins, range=(low, high))
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_left = np.cumsum(histogram, dtype=np.float64)
    weighted_sum_left = np.cumsum(histogram * centers, dtype=np.float64)
    total_weight = float(weight_left[-1])
    total_sum = float(weighted_sum_left[-1])
    weight_right = total_weight - weight_left

    valid = (weight_left > 0) & (weight_right > 0)
    score = np.full(centers.shape, -np.inf, dtype=np.float64)
    mean_left = np.zeros_like(centers, dtype=np.float64)
    mean_right = np.zeros_like(centers, dtype=np.float64)
    mean_left[valid] = weighted_sum_left[valid] / weight_left[valid]
    mean_right[valid] = (total_sum - weighted_sum_left[valid]) / weight_right[valid]
    score[valid] = weight_left[valid] * weight_right[valid] * (mean_left[valid] - mean_right[valid]) ** 2
    if not np.isfinite(score).any():
        return float(np.median(finite))
    return float(centers[int(np.argmax(score))])


def _connectivity_structure(connectivity: int) -> np.ndarray:
    return ndi.generate_binary_structure(2, 2 if connectivity == 8 else 1)


def _filter_and_relabel(
    labels: np.ndarray,
    minimum_area: int,
    maximum_area: int,
) -> np.ndarray:
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero((counts >= minimum_area) & (counts <= maximum_area))
    keep = keep[keep != 0]
    mapping = np.zeros(counts.size, dtype=np.int32)
    mapping[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    return mapping[labels]


def segment_cells(image: np.ndarray, config: SegmentationConfig | None = None) -> SegmentationResult:
    """Segment bright phase-contrast cells and return one integer label per object.

    The algorithm is deliberately lightweight: Gaussian denoising, smooth-background
    subtraction, Otsu-derived high-confidence seeds, hysteresis support growth, and
    nearest-seed partitioning. It requires no trained model and is deterministic.
    """

    cfg = config or SegmentationConfig()
    raw = np.asarray(image)
    if raw.ndim != 2:
        raise ValueError(f"Expected a two-dimensional grayscale image, got {raw.shape}")
    if raw.size == 0:
        raise ValueError("Cannot segment an empty image")

    work = raw.astype(np.float32, copy=False)
    smoothed = ndi.gaussian_filter(work, sigma=cfg.gaussian_sigma)
    background = ndi.gaussian_filter(smoothed, sigma=cfg.background_sigma)
    corrected = smoothed - background

    threshold = otsu_threshold(
        corrected,
        bins=cfg.histogram_bins,
        percentile_low=cfg.histogram_percentile_low,
        percentile_high=cfg.histogram_percentile_high,
    )
    high_threshold = threshold * cfg.high_threshold_factor
    low_threshold = min(high_threshold, threshold * cfg.low_threshold_factor)

    high_mask = corrected > high_threshold
    if cfg.opening_size > 1:
        high_mask = ndi.binary_opening(
            high_mask,
            structure=np.ones((cfg.opening_size, cfg.opening_size), dtype=bool),
        )
    if cfg.closing_size > 1:
        high_mask = ndi.binary_closing(
            high_mask,
            structure=np.ones((cfg.closing_size, cfg.closing_size), dtype=bool),
        )
    high_mask = ndi.binary_fill_holes(high_mask)

    structure = _connectivity_structure(cfg.connectivity)
    seed_labels, _ = ndi.label(high_mask, structure=structure)
    seed_labels = _filter_and_relabel(
        seed_labels,
        cfg.min_seed_area_px,
        cfg.max_seed_area_px,
    )
    seed_count = int(seed_labels.max(initial=0))
    if seed_count == 0:
        empty = np.zeros(raw.shape, dtype=np.int32)
        return SegmentationResult(
            labels=empty,
            corrected_image=corrected,
            otsu_threshold=threshold,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            seed_count=0,
            detection_count=0,
        )

    support = corrected > low_threshold
    if cfg.closing_size > 1:
        support = ndi.binary_closing(
            support,
            structure=np.ones((cfg.closing_size, cfg.closing_size), dtype=bool),
        )
    support = ndi.binary_fill_holes(support)

    support_labels, _ = ndi.label(support, structure=structure)
    touched_support = np.unique(support_labels[seed_labels > 0])
    keep_support = np.zeros(int(support_labels.max(initial=0)) + 1, dtype=bool)
    keep_support[touched_support] = True
    keep_support[0] = False
    support = keep_support[support_labels]

    nearest_seed_indices = ndi.distance_transform_edt(
        seed_labels == 0,
        return_distances=False,
        return_indices=True,
    )
    grown_labels = seed_labels[tuple(nearest_seed_indices)]
    grown_labels = np.where(support, grown_labels, 0)
    labels = _filter_and_relabel(
        grown_labels.astype(np.int32, copy=False),
        cfg.min_final_area_px,
        cfg.max_final_area_px,
    )

    return SegmentationResult(
        labels=labels,
        corrected_image=corrected,
        otsu_threshold=threshold,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        seed_count=seed_count,
        detection_count=int(labels.max(initial=0)),
    )
