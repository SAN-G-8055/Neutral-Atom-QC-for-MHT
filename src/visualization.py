"""Diagnostic plots for cell detections and their reference evaluation.

The functions in this module accept mappings keyed by integer frame number.  This
keeps plotting independent of image I/O and lets callers select a few informative
frames without loading an entire sequence into memory at once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


class DetectionLike(Protocol):
    """Minimum detection interface required by :func:`save_detection_overview`."""

    frame: int
    detection_id: int
    x_px: float
    y_px: float


class MatchLike(Protocol):
    """Minimum matched-pair interface required by the visualizations."""

    predicted_id: int
    reference_id: int
    distance_px: float


class FrameEvaluationLike(Protocol):
    """Minimum per-frame evaluation interface required by this module."""

    frame: int
    matches: Sequence[MatchLike]
    unmatched_predicted_ids: Sequence[int]
    unmatched_reference_ids: Sequence[int]
    precision: float
    recall: float
    f1: float


_MATCH_COLOR = "#1B9E77"
_FALSE_POSITIVE_COLOR = "#E67E22"
_FALSE_NEGATIVE_COLOR = "#CC2CA8"


def _ensure_output_parent(output: Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _index_detections(
    detections: Sequence[DetectionLike],
    *,
    frame: int,
    kind: str,
) -> dict[int, DetectionLike]:
    indexed: dict[int, DetectionLike] = {}
    for detection in detections:
        if int(detection.frame) != frame:
            raise ValueError(
                f"{kind.capitalize()} detection {detection.detection_id} is stored "
                f"under frame {frame}, but detection.frame is {detection.frame}"
            )
        detection_id = int(detection.detection_id)
        if detection_id in indexed:
            raise ValueError(
                f"Frame {frame} has duplicate {kind} detection_id {detection_id}"
            )
        indexed[detection_id] = detection
    return indexed


def _validate_evaluation_key(frame: int, evaluation: FrameEvaluationLike) -> None:
    if int(evaluation.frame) != frame:
        raise ValueError(
            f"Evaluation mapping key {frame} does not match evaluation.frame "
            f"{evaluation.frame}"
        )


def save_detection_overview(
    images: Mapping[int, np.ndarray],
    predictions: Mapping[int, Sequence[DetectionLike]],
    references: Mapping[int, Sequence[DetectionLike]],
    evaluations: Mapping[int, FrameEvaluationLike],
    output: Path,
    *,
    frames: Sequence[int] | None = None,
    columns: int = 4,
    marker_size: float = 72.0,
    dpi: int = 150,
) -> None:
    """Save raw frames with predicted detections classified by evaluation result.

    Parameters
    ----------
    images:
        ``frame -> 2-D image`` mapping. Images are displayed with a grayscale
        colormap; coordinates use ``x=column`` and ``y=row``.
    predictions, references:
        ``frame -> detections`` mappings. Each detection supplies
        ``detection_id``, ``x_px`` and ``y_px`` attributes.
    evaluations:
        ``frame -> FrameEvaluation`` mapping. Matched predicted IDs are drawn as
        green circles, false positives as orange circles, and missed reference
        IDs as magenta crosses.
    output:
        Destination image path. Its parent directory is created if needed.
    frames:
        Ordered subset to show. By default, all frames in ``images`` are shown in
        ascending order.
    columns:
        Maximum number of panels per row.

    The figure is always closed after saving, including when saving raises.
    """

    if columns < 1:
        raise ValueError("columns must be at least 1")
    if marker_size <= 0:
        raise ValueError("marker_size must be positive")

    selected_frames = (
        sorted(int(frame) for frame in images)
        if frames is None
        else [int(frame) for frame in frames]
    )
    if not selected_frames:
        raise ValueError("At least one frame is required")
    if len(set(selected_frames)) != len(selected_frames):
        raise ValueError("frames must not contain duplicates")

    missing_images = [frame for frame in selected_frames if frame not in images]
    missing_evaluations = [frame for frame in selected_frames if frame not in evaluations]
    if missing_images:
        raise KeyError(f"No image supplied for frame(s): {missing_images}")
    if missing_evaluations:
        raise KeyError(f"No evaluation supplied for frame(s): {missing_evaluations}")

    panel_columns = min(columns, len(selected_frames))
    panel_rows = (len(selected_frames) + panel_columns - 1) // panel_columns
    figure, axes = plt.subplots(
        panel_rows,
        panel_columns,
        figsize=(4.0 * panel_columns, 3.7 * panel_rows),
        squeeze=False,
    )

    try:
        for axis, frame in zip(axes.flat, selected_frames, strict=False):
            image = np.asarray(images[frame])
            if image.ndim != 2:
                raise ValueError(
                    f"Frame {frame} image must be two-dimensional, got shape {image.shape}"
                )

            evaluation = evaluations[frame]
            _validate_evaluation_key(frame, evaluation)
            predicted = _index_detections(
                predictions.get(frame, ()), frame=frame, kind="predicted"
            )
            reference = _index_detections(
                references.get(frame, ()), frame=frame, kind="reference"
            )

            matched_predicted_ids = [
                int(match.predicted_id) for match in evaluation.matches
            ]
            matched_reference_ids = [
                int(match.reference_id) for match in evaluation.matches
            ]
            matched_ids = set(matched_predicted_ids)
            matched_gold_ids = set(matched_reference_ids)
            false_positive_ids = {
                int(detection_id)
                for detection_id in evaluation.unmatched_predicted_ids
            }
            false_negative_ids = {
                int(detection_id)
                for detection_id in evaluation.unmatched_reference_ids
            }
            if len(matched_ids) != len(matched_predicted_ids):
                raise ValueError(f"Frame {frame} matches a predicted ID more than once")
            if len(matched_gold_ids) != len(matched_reference_ids):
                raise ValueError(f"Frame {frame} matches a reference ID more than once")
            if matched_ids & false_positive_ids:
                raise ValueError(
                    f"Frame {frame} classifies a predicted ID as both matched and "
                    "false positive"
                )
            if matched_gold_ids & false_negative_ids:
                raise ValueError(
                    f"Frame {frame} classifies a reference ID as both matched and missed"
                )

            unknown_predicted = (matched_ids | false_positive_ids) - predicted.keys()
            unknown_reference = (
                matched_gold_ids | false_negative_ids
            ) - reference.keys()
            if unknown_predicted:
                raise ValueError(
                    f"Frame {frame} evaluation refers to unknown predicted ID(s): "
                    f"{sorted(unknown_predicted)}"
                )
            if unknown_reference:
                raise ValueError(
                    f"Frame {frame} evaluation refers to unknown reference ID(s): "
                    f"{sorted(unknown_reference)}"
                )
            omitted_predicted = predicted.keys() - (matched_ids | false_positive_ids)
            omitted_reference = reference.keys() - (matched_gold_ids | false_negative_ids)
            if omitted_predicted:
                raise ValueError(
                    f"Frame {frame} evaluation omits predicted ID(s): {sorted(omitted_predicted)}"
                )
            if omitted_reference:
                raise ValueError(
                    f"Frame {frame} evaluation omits reference ID(s): {sorted(omitted_reference)}"
                )

            axis.imshow(image, cmap="gray", interpolation="nearest")
            _scatter_detections(
                axis,
                [predicted[detection_id] for detection_id in sorted(matched_ids)],
                marker="o",
                color=_MATCH_COLOR,
                size=marker_size,
                hollow=True,
            )
            _scatter_detections(
                axis,
                [predicted[detection_id] for detection_id in sorted(false_positive_ids)],
                marker="o",
                color=_FALSE_POSITIVE_COLOR,
                size=marker_size,
                hollow=True,
            )
            _scatter_detections(
                axis,
                [reference[detection_id] for detection_id in sorted(false_negative_ids)],
                marker="x",
                color=_FALSE_NEGATIVE_COLOR,
                size=marker_size,
                hollow=False,
            )

            predicted_count = len(evaluation.matches) + len(
                evaluation.unmatched_predicted_ids
            )
            reference_count = len(evaluation.matches) + len(
                evaluation.unmatched_reference_ids
            )
            axis.set_title(
                f"Frame {frame} | predicted {predicted_count}, gold {reference_count}"
                f" | F1 {evaluation.f1:.3f}",
                fontsize=10,
            )
            axis.set_axis_off()

        for unused_axis in axes.flat[len(selected_frames) :]:
            unused_axis.set_visible(False)

        legend_handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor=_MATCH_COLOR,
                markeredgewidth=1.8,
                label="Matched prediction",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor=_FALSE_POSITIVE_COLOR,
                markeredgewidth=1.8,
                label="False positive",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="x",
                color=_FALSE_NEGATIVE_COLOR,
                markeredgewidth=1.8,
                label="Missed gold detection",
            ),
        ]
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=3,
            frameon=False,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        figure.savefig(_ensure_output_parent(output), dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def _scatter_detections(
    axis: Axes,
    detections: Sequence[DetectionLike],
    *,
    marker: str,
    color: str,
    size: float,
    hollow: bool,
) -> None:
    if not detections:
        return
    x = [float(detection.x_px) for detection in detections]
    y = [float(detection.y_px) for detection in detections]
    if hollow:
        axis.scatter(
            x,
            y,
            s=size,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
        )
    else:
        axis.scatter(x, y, s=size, marker=marker, c=color, linewidths=1.8)


def save_per_frame_performance(
    evaluations: Mapping[int, FrameEvaluationLike],
    output: Path,
    *,
    dpi: int = 150,
) -> None:
    """Save per-frame metric and object-count trends.

    ``evaluations`` maps frame number to its evaluation. Predicted and gold counts
    are derived respectively as ``matches + false positives`` and ``matches +
    false negatives``. Evaluation mapping keys must agree with each object's
    ``frame`` attribute.

    The figure is always closed after saving, including when saving raises.
    """

    if not evaluations:
        raise ValueError("At least one frame evaluation is required")

    frames = sorted(int(frame) for frame in evaluations)
    ordered_evaluations = [evaluations[frame] for frame in frames]
    for frame, evaluation in zip(frames, ordered_evaluations, strict=True):
        _validate_evaluation_key(frame, evaluation)

    precision = [float(evaluation.precision) for evaluation in ordered_evaluations]
    recall = [float(evaluation.recall) for evaluation in ordered_evaluations]
    f1 = [float(evaluation.f1) for evaluation in ordered_evaluations]
    predicted_counts = [
        len(evaluation.matches) + len(evaluation.unmatched_predicted_ids)
        for evaluation in ordered_evaluations
    ]
    reference_counts = [
        len(evaluation.matches) + len(evaluation.unmatched_reference_ids)
        for evaluation in ordered_evaluations
    ]

    figure, (metric_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 2)},
    )
    try:
        metric_axis.plot(frames, precision, marker="o", markersize=3, label="Precision")
        metric_axis.plot(frames, recall, marker="o", markersize=3, label="Recall")
        metric_axis.plot(frames, f1, marker="o", markersize=3, label="F1")
        metric_axis.set_ylabel("Score")
        metric_axis.set_ylim(-0.02, 1.02)
        metric_axis.grid(alpha=0.25)
        metric_axis.legend(frameon=False, ncol=3, loc="lower right")
        metric_axis.set_title("Per-frame detection performance")

        count_axis.plot(
            frames,
            predicted_counts,
            color=_FALSE_POSITIVE_COLOR,
            marker="o",
            markersize=3,
            label="Predicted",
        )
        count_axis.plot(
            frames,
            reference_counts,
            color=_FALSE_NEGATIVE_COLOR,
            marker="o",
            markersize=3,
            label="Gold standard",
        )
        count_axis.set_xlabel("Frame")
        count_axis.set_ylabel("Detection count")
        count_axis.grid(alpha=0.25)
        count_axis.legend(frameon=False, ncol=2, loc="best")

        figure.tight_layout()
        figure.savefig(_ensure_output_parent(output), dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
