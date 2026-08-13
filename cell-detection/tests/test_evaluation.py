from __future__ import annotations

import numpy as np

from cell_detection_pipeline.evaluation import binary_mask_metrics, match_centroids


def _record(x: float, y: float) -> dict[str, float]:
    return {"x": x, "y": y}


def test_centroid_matching_is_one_to_one_and_gated() -> None:
    predicted = [_record(0, 0), _record(0.5, 0), _record(20, 20)]
    reference = [_record(0, 0), _record(20, 21), _record(100, 100)]

    metrics = match_centroids(predicted, reference, max_distance_px=2)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3


def test_binary_metrics() -> None:
    predicted = np.array([[1, 1, 0], [0, 0, 0]])
    reference = np.array([[1, 0, 0], [1, 0, 0]])

    metrics = binary_mask_metrics(predicted, reference)

    assert metrics["intersection_px"] == 1
    assert metrics["union_px"] == 3
    assert metrics["dice"] == 0.5
    assert metrics["iou"] == 1 / 3
