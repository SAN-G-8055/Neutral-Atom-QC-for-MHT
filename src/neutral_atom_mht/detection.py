"""Deterministic conversion of one microscopy frame into detection events.

A *detection event* is one positive final instance label in one frame.  Its
position is the geometric centroid in zero-based image coordinates: ``x_px`` is
the column and ``y_px`` is the row.  Predicted identifiers are unique only
inside a frame; human-gold identifiers retain their source labels, but matching
uses neither kind as a cross-frame track identity.

The detector deliberately uses a short, inspectable sequence of operations:

1. suppress pixel noise with a narrow Gaussian filter;
2. subtract a broad Gaussian background estimate;
3. create high-confidence cell seeds from an Otsu-derived threshold;
4. discard seed components outside the declared area range;
5. grow seeds only inside connected low-threshold support regions; and
6. discard final instances outside the declared area range.

No learned model, random choice, or gold annotation is used by detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy import ndimage as ndi


def _finite_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))


def _integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validate_event_scope(sequence: Any, frame: Any, source: Any) -> None:
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must not be empty")
    if not _integer(frame):
        raise ValueError("frame must be an integer")
    if frame < 0:
        raise ValueError("frame must be non-negative")
    if not isinstance(source, str) or not source:
        raise ValueError("source must not be empty")


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Frozen parameters of the sequence-01 detector.

    These defaults predate the sequence-01 evaluation.  Keeping them here,
    rather than in a second JSON file, makes the evaluated method unambiguous.
    """

    gaussian_sigma_px: float = 1.0
    background_sigma_px: float = 8.0
    high_threshold_factor: float = 1.2
    low_threshold_factor: float = 0.04
    opening_size_px: int = 2
    closing_size_px: int = 3
    min_seed_area_px: int = 40
    max_seed_area_px: int = 1_000
    min_detection_area_px: int = 40
    max_detection_area_px: int = 1_500

    def __post_init__(self) -> None:
        real_fields = {
            "gaussian_sigma_px": self.gaussian_sigma_px,
            "background_sigma_px": self.background_sigma_px,
            "high_threshold_factor": self.high_threshold_factor,
            "low_threshold_factor": self.low_threshold_factor,
        }
        invalid_real = [name for name, value in real_fields.items() if not _finite_real(value)]
        if invalid_real:
            raise ValueError(f"configuration values must be finite real numbers: {', '.join(invalid_real)}")
        integer_fields = {
            "opening_size_px": self.opening_size_px,
            "closing_size_px": self.closing_size_px,
            "min_seed_area_px": self.min_seed_area_px,
            "max_seed_area_px": self.max_seed_area_px,
            "min_detection_area_px": self.min_detection_area_px,
            "max_detection_area_px": self.max_detection_area_px,
        }
        invalid_integer = [name for name, value in integer_fields.items() if not _integer(value)]
        if invalid_integer:
            raise ValueError(f"configuration values must be integers: {', '.join(invalid_integer)}")
        for name, value in real_fields.items():
            object.__setattr__(self, name, float(value))
        for name, value in integer_fields.items():
            object.__setattr__(self, name, int(value))
        if self.gaussian_sigma_px < 0:
            raise ValueError("gaussian_sigma_px must be non-negative")
        if self.background_sigma_px <= self.gaussian_sigma_px:
            raise ValueError("background_sigma_px must exceed gaussian_sigma_px")
        if self.high_threshold_factor <= 0:
            raise ValueError("high_threshold_factor must be positive")
        if not 0 <= self.low_threshold_factor <= self.high_threshold_factor:
            raise ValueError("low_threshold_factor must lie between 0 and high_threshold_factor")
        if self.opening_size_px < 1 or self.closing_size_px < 1:
            raise ValueError("morphology sizes must be positive")
        if not 0 < self.min_seed_area_px <= self.max_seed_area_px:
            raise ValueError("seed area limits are invalid")
        if not 0 < self.min_detection_area_px <= self.max_detection_area_px:
            raise ValueError("detection area limits are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Detection:
    """One frame-local point observation derived from one labelled instance."""

    sequence: str
    frame: int
    detection_id: int
    x_px: float
    y_px: float
    area_px: int
    source: str

    def __post_init__(self) -> None:
        _validate_event_scope(self.sequence, self.frame, self.source)
        if not _integer(self.detection_id):
            raise ValueError("detection_id must be an integer")
        if self.detection_id < 1:
            raise ValueError("detection_id must be positive")
        if not _finite_real(self.x_px) or not _finite_real(self.y_px):
            raise ValueError("detection coordinates must be finite")
        if not _integer(self.area_px):
            raise ValueError("area_px must be an integer")
        if self.area_px < 1:
            raise ValueError("area_px must be positive")
        object.__setattr__(self, "frame", int(self.frame))
        object.__setattr__(self, "detection_id", int(self.detection_id))
        object.__setattr__(self, "x_px", float(self.x_px))
        object.__setattr__(self, "y_px", float(self.y_px))
        object.__setattr__(self, "area_px", int(self.area_px))

    @property
    def key(self) -> tuple[str, int, int]:
        """The source event key; prediction IDs have no cross-frame meaning."""

        return self.sequence, self.frame, self.detection_id


@dataclass(frozen=True, slots=True)
class DetectionDiagnostics:
    otsu_threshold: float
    high_threshold: float
    low_threshold: float
    seed_count: int
    detection_count: int


@dataclass(frozen=True, slots=True)
class DetectionResult:
    sequence: str
    frame: int
    labels: np.ndarray
    detections: tuple[Detection, ...]
    diagnostics: DetectionDiagnostics

    def __post_init__(self) -> None:
        if self.labels.ndim != 2:
            raise ValueError("labels must be a two-dimensional array")
        if len(self.detections) != self.diagnostics.detection_count:
            raise ValueError("diagnostic detection count does not match the event table")


def _otsu_threshold(image: np.ndarray) -> float:
    """Otsu threshold after fixed 0.1/99.9-percentile tail clipping."""

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot threshold an image with no finite values")
    low, high = np.percentile(finite, (0.1, 99.9))
    if high <= low:
        return float(low)
    histogram, edges = np.histogram(np.clip(finite, low, high), bins=256, range=(low, high))
    centers = (edges[:-1] + edges[1:]) * 0.5
    left_count = np.cumsum(histogram, dtype=np.float64)
    left_sum = np.cumsum(histogram * centers, dtype=np.float64)
    total_count = float(left_count[-1])
    total_sum = float(left_sum[-1])
    right_count = total_count - left_count
    valid = (left_count > 0) & (right_count > 0)
    score = np.full(centers.shape, -np.inf, dtype=np.float64)
    left_mean = np.zeros_like(centers, dtype=np.float64)
    right_mean = np.zeros_like(centers, dtype=np.float64)
    left_mean[valid] = left_sum[valid] / left_count[valid]
    right_mean[valid] = (total_sum - left_sum[valid]) / right_count[valid]
    score[valid] = left_count[valid] * right_count[valid] * (left_mean[valid] - right_mean[valid]) ** 2
    return float(centers[int(np.argmax(score))]) if np.isfinite(score).any() else float(np.median(finite))


def _filter_and_relabel(labels: np.ndarray, minimum_area: int, maximum_area: int) -> np.ndarray:
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero((counts >= minimum_area) & (counts <= maximum_area))
    keep = keep[keep != 0]
    mapping = np.zeros(counts.size, dtype=np.int32)
    mapping[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    return mapping[labels]


def _grow_seeds_within_support(seed_labels: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Assign support pixels only to seeds in the same support component."""

    structure = ndi.generate_binary_structure(2, 2)
    support_labels, _ = ndi.label(support, structure=structure)
    touched_components = np.unique(support_labels[seed_labels > 0])
    touched_components = touched_components[touched_components > 0]
    component_slices = ndi.find_objects(support_labels)
    grown = np.zeros(seed_labels.shape, dtype=np.int32)

    for component_id in touched_components:
        box = component_slices[int(component_id) - 1]
        if box is None:
            continue
        component = support_labels[box] == component_id
        local_seeds = seed_labels[box]
        seed_ids = np.unique(local_seeds[component])
        seed_ids = seed_ids[seed_ids > 0]
        local_output = grown[box]
        if seed_ids.size == 1:
            local_output[component] = int(seed_ids[0])
            continue

        # Only seed pixels from this connected support component are zeros in
        # the distance transform, so assignment cannot jump across background.
        distance_input = np.ones(component.shape, dtype=bool)
        distance_input[component & (local_seeds > 0)] = False
        nearest = ndi.distance_transform_edt(
            distance_input,
            return_distances=False,
            return_indices=True,
        )
        assigned = local_seeds[tuple(nearest)]
        local_output[component] = assigned[component]

    return grown


def detections_from_label_image(
    labels: np.ndarray,
    *,
    sequence: str,
    frame: int,
    source: str,
) -> tuple[Detection, ...]:
    """Convert positive labels to geometric-centroid detection events.

    Original label values are retained as ``detection_id``.  This is important
    for gold tracking masks, whose IDs persist across frames.
    """

    _validate_event_scope(sequence, frame, source)
    frame = int(frame)
    instance_labels = np.asarray(labels)
    if instance_labels.ndim != 2:
        raise ValueError(f"expected a two-dimensional label image, got {instance_labels.shape}")
    if not np.issubdtype(instance_labels.dtype, np.integer):
        raise ValueError("label images must have an integer dtype")
    if np.any(instance_labels < 0):
        raise ValueError("label images cannot contain negative values")

    original_ids = np.unique(instance_labels)
    original_ids = original_ids[original_ids > 0]
    if original_ids.size == 0:
        return ()

    dense_labels = np.zeros(instance_labels.shape, dtype=np.int32)
    foreground = instance_labels > 0
    dense_labels[foreground] = np.searchsorted(original_ids, instance_labels[foreground]) + 1
    dense_ids = np.arange(1, original_ids.size + 1, dtype=np.int32)
    areas = np.bincount(dense_labels.ravel(), minlength=original_ids.size + 1)[1:]
    centroids_yx = ndi.center_of_mass(
        np.ones(instance_labels.shape, dtype=np.uint8),
        dense_labels,
        dense_ids,
    )

    return tuple(
        Detection(
            sequence=sequence,
            frame=frame,
            detection_id=int(original_id),
            x_px=float(centroid[1]),
            y_px=float(centroid[0]),
            area_px=int(area),
            source=source,
        )
        for original_id, area, centroid in zip(original_ids, areas, centroids_yx, strict=True)
    )


def detect_frame(
    image: np.ndarray,
    *,
    sequence: str,
    frame: int,
    config: DetectionConfig | None = None,
) -> DetectionResult:
    """Detect all cell events in one grayscale frame."""

    _validate_event_scope(sequence, frame, "prediction")
    frame = int(frame)
    cfg = config or DetectionConfig()
    raw = np.asarray(image)
    if raw.ndim != 2:
        raise ValueError(f"expected one two-dimensional grayscale frame, got {raw.shape}")
    if raw.size == 0:
        raise ValueError("cannot detect cells in an empty image")
    if not np.isfinite(raw).all():
        raise ValueError("image contains non-finite values")

    work = raw.astype(np.float32, copy=False)
    smoothed = ndi.gaussian_filter(work, sigma=cfg.gaussian_sigma_px)
    background = ndi.gaussian_filter(smoothed, sigma=cfg.background_sigma_px)
    corrected = smoothed - background

    otsu = _otsu_threshold(corrected)
    high_threshold = otsu * cfg.high_threshold_factor
    low_threshold = min(high_threshold, otsu * cfg.low_threshold_factor)

    seeds = corrected > high_threshold
    if cfg.opening_size_px > 1:
        seeds = ndi.binary_opening(
            seeds,
            structure=np.ones((cfg.opening_size_px, cfg.opening_size_px), dtype=bool),
        )
    if cfg.closing_size_px > 1:
        seeds = ndi.binary_closing(
            seeds,
            structure=np.ones((cfg.closing_size_px, cfg.closing_size_px), dtype=bool),
        )
    seeds = ndi.binary_fill_holes(seeds)
    seed_labels, _ = ndi.label(seeds, structure=ndi.generate_binary_structure(2, 2))
    seed_labels = _filter_and_relabel(
        seed_labels,
        cfg.min_seed_area_px,
        cfg.max_seed_area_px,
    )
    seed_count = int(seed_labels.max(initial=0))

    if seed_count == 0:
        labels = np.zeros(raw.shape, dtype=np.int32)
    else:
        support = corrected > low_threshold
        if cfg.closing_size_px > 1:
            support = ndi.binary_closing(
                support,
                structure=np.ones((cfg.closing_size_px, cfg.closing_size_px), dtype=bool),
            )
        support = ndi.binary_fill_holes(support)
        grown = _grow_seeds_within_support(seed_labels, support)
        labels = _filter_and_relabel(
            grown,
            cfg.min_detection_area_px,
            cfg.max_detection_area_px,
        )

    detections = detections_from_label_image(
        labels,
        sequence=sequence,
        frame=frame,
        source="prediction",
    )
    diagnostics = DetectionDiagnostics(
        otsu_threshold=otsu,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        seed_count=seed_count,
        detection_count=len(detections),
    )
    return DetectionResult(
        sequence=sequence,
        frame=frame,
        labels=labels,
        detections=detections,
        diagnostics=diagnostics,
    )
