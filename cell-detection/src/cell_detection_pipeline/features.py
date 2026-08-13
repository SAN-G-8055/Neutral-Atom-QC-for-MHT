"""Convert instance labels into tabular detection records."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


def detections_from_labels(
    labels: np.ndarray,
    image: np.ndarray | None = None,
    *,
    dataset: str = "",
    sequence: str = "",
    frame: int = 0,
    source: str = "prediction",
    image_name: str = "",
) -> list[dict[str, Any]]:
    """Extract geometric centroids and object features from a label image.

    Coordinates use zero-based image indices: ``x`` is column and ``y`` is row.
    Bounding-box maxima are exclusive so slices can be reconstructed directly.
    """

    instance_labels = np.asarray(labels)
    if instance_labels.ndim != 2:
        raise ValueError(f"Expected two-dimensional labels, got {instance_labels.shape}")
    intensity = np.ones(instance_labels.shape, dtype=np.float32) if image is None else np.asarray(image)
    if intensity.shape != instance_labels.shape:
        raise ValueError("Image and label shapes differ")

    ids = np.unique(instance_labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return []

    # Reference masks may have sparse or very large persistent tracking IDs. Remap
    # them only for measurement so SciPy never allocates or divides across absent IDs;
    # records still retain the original IDs below.
    dense_mapping = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    dense_mapping[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    dense_labels = dense_mapping[instance_labels]
    dense_ids = np.arange(1, ids.size + 1, dtype=np.int32)

    ones = np.ones(instance_labels.shape, dtype=np.float32)
    centers = np.asarray(ndi.center_of_mass(ones, dense_labels, dense_ids), dtype=np.float64)
    areas = np.asarray(ndi.sum(ones, dense_labels, dense_ids), dtype=np.int64)
    totals = np.asarray(ndi.sum(intensity, dense_labels, dense_ids), dtype=np.float64)
    squared_totals = np.asarray(ndi.sum(intensity.astype(np.float64) ** 2, dense_labels, dense_ids))
    means = totals / areas
    variances = np.maximum(squared_totals / areas - means**2, 0.0)
    stds = np.sqrt(variances)
    maxima = np.asarray(ndi.maximum(intensity, dense_labels, dense_ids), dtype=np.float64)
    x_coordinates = np.arange(instance_labels.shape[1], dtype=np.float64)[None, :]
    y_coordinates = np.arange(instance_labels.shape[0], dtype=np.float64)[:, None]
    weighted_x_sums = np.asarray(ndi.sum(intensity * x_coordinates, dense_labels, dense_ids))
    weighted_y_sums = np.asarray(ndi.sum(intensity * y_coordinates, dense_labels, dense_ids))
    weighted_x = np.divide(
        weighted_x_sums,
        totals,
        out=centers[:, 1].copy(),
        where=totals != 0,
    )
    weighted_y = np.divide(
        weighted_y_sums,
        totals,
        out=centers[:, 0].copy(),
        where=totals != 0,
    )
    boxes = ndi.find_objects(dense_labels)

    records: list[dict[str, Any]] = []
    for index, label_id in enumerate(ids):
        box = boxes[index]
        if box is None:
            continue
        y_slice, x_slice = box
        object_weighted_x = weighted_x[index]
        object_weighted_y = weighted_y[index]
        records.append(
            {
                "dataset": dataset,
                "sequence": sequence,
                "frame": int(frame),
                "detection_id": int(label_id),
                "x": float(centers[index, 1]),
                "y": float(centers[index, 0]),
                "area_px": int(areas[index]),
                "mean_intensity": float(means[index]),
                "std_intensity": float(stds[index]),
                "max_intensity": float(maxima[index]),
                "integrated_intensity": float(totals[index]),
                "intensity_weighted_x": float(object_weighted_x),
                "intensity_weighted_y": float(object_weighted_y),
                "bbox_x_min": int(x_slice.start),
                "bbox_y_min": int(y_slice.start),
                "bbox_x_max_exclusive": int(x_slice.stop),
                "bbox_y_max_exclusive": int(y_slice.stop),
                "source": source,
                "image": image_name,
            }
        )
    return records
