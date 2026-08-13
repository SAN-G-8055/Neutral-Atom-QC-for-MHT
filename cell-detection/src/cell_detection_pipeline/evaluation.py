"""One-to-one centroid and mask comparison metrics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _points(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not records:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([(float(row["x"]), float(row["y"])) for row in records], dtype=np.float64)


def match_centroids(
    predicted: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    max_distance_px: float = 10.0,
) -> dict[str, Any]:
    """Match detections one-to-one, maximizing gated matches then minimizing distance."""

    if max_distance_px <= 0:
        raise ValueError("max_distance_px must be positive")
    predicted_points = _points(predicted)
    reference_points = _points(reference)
    n_predicted, n_reference = len(predicted_points), len(reference_points)

    if n_predicted == 0 or n_reference == 0:
        true_positive = 0
        distances = np.empty(0, dtype=np.float64)
    else:
        distance = np.linalg.norm(
            predicted_points[:, None, :] - reference_points[None, :, :],
            axis=2,
        )
        size = n_predicted + n_reference
        disallowed = max_distance_px * (size + 2) * 10.0
        unmatched = max_distance_px + np.finfo(np.float64).eps * 100
        cost = np.full((size, size), disallowed, dtype=np.float64)
        cost[:n_predicted, :n_reference] = np.where(
            distance <= max_distance_px,
            distance,
            disallowed,
        )
        cost[np.arange(n_predicted), n_reference + np.arange(n_predicted)] = unmatched
        cost[n_predicted + np.arange(n_reference), np.arange(n_reference)] = unmatched
        cost[n_predicted:, n_reference:] = 0.0
        rows, columns = linear_sum_assignment(cost)
        real = (rows < n_predicted) & (columns < n_reference)
        real_rows = rows[real]
        real_columns = columns[real]
        accepted = distance[real_rows, real_columns] <= max_distance_px
        distances = distance[real_rows[accepted], real_columns[accepted]]
        true_positive = int(distances.size)

    false_positive = n_predicted - true_positive
    false_negative = n_reference - true_positive
    precision = true_positive / n_predicted if n_predicted else 0.0
    recall = true_positive / n_reference if n_reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "max_distance_px": float(max_distance_px),
        "predicted_count": n_predicted,
        "reference_count": n_reference,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_localization_error_px": float(distances.mean()) if distances.size else None,
        "median_localization_error_px": float(np.median(distances)) if distances.size else None,
        "rmse_localization_px": float(np.sqrt(np.mean(distances**2))) if distances.size else None,
        "max_localization_error_px": float(distances.max()) if distances.size else None,
    }


def binary_mask_metrics(predicted_labels: np.ndarray, reference_labels: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted_labels) > 0
    reference = np.asarray(reference_labels) > 0
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference masks have different shapes")
    intersection = int(np.logical_and(predicted, reference).sum())
    union = int(np.logical_or(predicted, reference).sum())
    predicted_area = int(predicted.sum())
    reference_area = int(reference.sum())
    denominator = predicted_area + reference_area
    return {
        "predicted_foreground_px": predicted_area,
        "reference_foreground_px": reference_area,
        "intersection_px": intersection,
        "union_px": union,
        "dice": 2 * intersection / denominator if denominator else 1.0,
        "iou": intersection / union if union else 1.0,
    }


def instance_iou_metrics(
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Report one-to-one instance matches using maximum total intersection-over-union."""

    predicted = np.asarray(predicted_labels)
    reference = np.asarray(reference_labels)
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference masks have different shapes")
    predicted_ids = np.unique(predicted)
    predicted_ids = predicted_ids[predicted_ids > 0]
    reference_ids = np.unique(reference)
    reference_ids = reference_ids[reference_ids > 0]
    if predicted_ids.size == 0 or reference_ids.size == 0:
        matched = 0
        matched_ious = np.empty(0, dtype=np.float64)
    else:
        predicted_index = {int(label): index for index, label in enumerate(predicted_ids)}
        reference_index = {int(label): index for index, label in enumerate(reference_ids)}
        intersections = np.zeros((predicted_ids.size, reference_ids.size), dtype=np.int64)
        both = (predicted > 0) & (reference > 0)
        pairs, counts = np.unique(
            np.column_stack((predicted[both], reference[both])),
            axis=0,
            return_counts=True,
        )
        for pair, count in zip(pairs, counts, strict=True):
            intersections[predicted_index[int(pair[0])], reference_index[int(pair[1])]] = int(count)
        predicted_areas = np.asarray([(predicted == label).sum() for label in predicted_ids])
        reference_areas = np.asarray([(reference == label).sum() for label in reference_ids])
        unions = predicted_areas[:, None] + reference_areas[None, :] - intersections
        ious = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)
        rows, columns = linear_sum_assignment(1.0 - ious)
        assigned = ious[rows, columns]
        matched_ious = assigned[assigned >= iou_threshold]
        matched = int(matched_ious.size)

    precision = matched / predicted_ids.size if predicted_ids.size else 0.0
    recall = matched / reference_ids.size if reference_ids.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou_threshold": float(iou_threshold),
        "predicted_count": int(predicted_ids.size),
        "reference_count": int(reference_ids.size),
        "matched_count": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_of_matches": float(matched_ious.mean()) if matched_ious.size else None,
    }
