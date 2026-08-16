"""Auditable, frame-safe evaluation of centroid detections.

A true positive is one predicted detection paired with one reference detection
from the same sequence and frame whose Euclidean centroid distance is less than
or equal to ``max_distance_px``.  Matching is lexicographic: it first maximizes
the number of valid pairs and then minimizes their total distance.  Every
remaining prediction is a false positive and every remaining reference is a
false negative.

At sequence level, ``precision``, ``recall``, and ``f1`` are micro-averaged
from the summed counts.  The corresponding ``macro_*`` values are unweighted
means of the per-frame scores.  Localization RMSE is calculated over all true
positive pair distances and is ``None`` when there are no matches.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import Detection


DEFAULT_MAX_DISTANCE_PX = 10.0


@dataclass(frozen=True, slots=True)
class Match:
    """One auditable prediction/reference correspondence."""

    sequence: str
    frame: int
    predicted_id: int
    reference_id: int
    distance_px: float

    @property
    def predicted_detection_id(self) -> int:
        """Explicit alias for consumers that use the full detection ID name."""

        return self.predicted_id

    @property
    def prediction_id(self) -> int:
        """Alias using the noun form of ``predicted_id``."""

        return self.predicted_id

    @property
    def reference_detection_id(self) -> int:
        """Explicit alias for consumers that use the full detection ID name."""

        return self.reference_id


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    """Counts, scores, and complete matching audit data for one frame."""

    sequence: str
    frame: int
    max_distance_px: float
    predicted_count: int
    reference_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    localization_rmse_px: float | None
    matches: tuple[Match, ...]
    unmatched_predicted_ids: tuple[int, ...]
    unmatched_reference_ids: tuple[int, ...]

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives

    @property
    def true_positive(self) -> int:
        return self.true_positives

    @property
    def false_positive(self) -> int:
        return self.false_positives

    @property
    def false_negative(self) -> int:
        return self.false_negatives

    @property
    def rmse_localization_px(self) -> float | None:
        """Compatibility alias with the metric name in the previous project."""

        return self.localization_rmse_px

    @property
    def false_positive_ids(self) -> tuple[int, ...]:
        """IDs of unmatched predictions, named by their evaluated outcome."""

        return self.unmatched_predicted_ids

    @property
    def false_negative_ids(self) -> tuple[int, ...]:
        """IDs of unmatched references, named by their evaluated outcome."""

        return self.unmatched_reference_ids


@dataclass(frozen=True, slots=True)
class SequenceEvaluation:
    """Micro and macro detection scores for all observed frames in a sequence."""

    sequence: str
    max_distance_px: float
    frames: tuple[FrameEvaluation, ...]
    predicted_count: int
    reference_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    localization_rmse_px: float | None

    @property
    def frame_evaluations(self) -> tuple[FrameEvaluation, ...]:
        return self.frames

    @property
    def matches(self) -> tuple[Match, ...]:
        """All matches in deterministic frame/ID order for audit and plotting."""

        return tuple(match for frame in self.frames for match in frame.matches)

    @property
    def micro_precision(self) -> float:
        return self.precision

    @property
    def micro_recall(self) -> float:
        return self.recall

    @property
    def micro_f1(self) -> float:
        return self.f1

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives

    @property
    def true_positive(self) -> int:
        return self.true_positives

    @property
    def false_positive(self) -> int:
        return self.false_positives

    @property
    def false_negative(self) -> int:
        return self.false_negatives

    @property
    def rmse_localization_px(self) -> float | None:
        return self.localization_rmse_px


def _resolve_gate(max_distance_px: float | None, gate_px: float | None) -> float:
    if max_distance_px is not None and gate_px is not None:
        raise TypeError("Specify only one of max_distance_px and gate_px")
    value = DEFAULT_MAX_DISTANCE_PX if max_distance_px is None and gate_px is None else (
        gate_px if gate_px is not None else max_distance_px
    )
    try:
        gate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("max_distance_px must be a finite positive number") from error
    if not math.isfinite(gate) or gate <= 0.0:
        raise ValueError("max_distance_px must be a finite positive number")
    return gate


def _validate_detections(detections: Sequence[Detection], role: str) -> None:
    for detection in detections:
        if not math.isfinite(float(detection.x_px)) or not math.isfinite(float(detection.y_px)):
            raise ValueError(
                f"{role} detection {detection.detection_id} in "
                f"{detection.sequence!r} frame {detection.frame} has a non-finite centroid"
            )


def _validate_unique_ids(detections: Sequence[Detection], role: str) -> None:
    ids = [detection.detection_id for detection in detections]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{role} detection IDs must be unique within each sequence and frame")


def _scores(
    true_positives: int,
    predicted_count: int,
    reference_count: int,
) -> tuple[float, float, float]:
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / reference_count if reference_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _localization_rmse(matches: Sequence[Match]) -> float | None:
    if not matches:
        return None
    return math.sqrt(math.fsum(match.distance_px**2 for match in matches) / len(matches))


def _match_single_frame(
    predicted: Sequence[Detection],
    reference: Sequence[Detection],
    max_distance_px: float,
) -> tuple[Match, ...]:
    """Apply lexicographic bipartite matching to one sequence/frame group."""

    if not predicted or not reference:
        return ()

    predicted_ordered = tuple(sorted(predicted, key=lambda detection: detection.detection_id))
    reference_ordered = tuple(sorted(reference, key=lambda detection: detection.detection_id))
    predicted_xy = np.asarray(
        [(detection.x_px, detection.y_px) for detection in predicted_ordered],
        dtype=np.float64,
    )
    reference_xy = np.asarray(
        [(detection.x_px, detection.y_px) for detection in reference_ordered],
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        difference = predicted_xy[:, None, :] - reference_xy[None, :, :]
        distances = np.hypot(difference[..., 0], difference[..., 1])
    allowed = distances <= max_distance_px

    # The assignment has min(n_predicted, n_reference) slots.  Disallowed slots
    # cost zero, while each allowed edge receives a reward larger than every
    # possible sum of normalized distances.  One additional allowed edge must
    # therefore beat any distance improvement, which gives exact cardinality-
    # first behavior; among equal-cardinality assignments the remaining term is
    # total Euclidean distance.
    assignment_size = min(len(predicted_ordered), len(reference_ordered))
    cardinality_reward = float(assignment_size + 1)
    cost = np.zeros(distances.shape, dtype=np.float64)
    cost[allowed] = distances[allowed] / max_distance_px - cardinality_reward
    row_indices, column_indices = linear_sum_assignment(cost)

    matches = [
        Match(
            sequence=predicted_ordered[row].sequence,
            frame=predicted_ordered[row].frame,
            predicted_id=predicted_ordered[row].detection_id,
            reference_id=reference_ordered[column].detection_id,
            distance_px=float(distances[row, column]),
        )
        for row, column in zip(row_indices, column_indices, strict=True)
        if allowed[row, column]
    ]
    return tuple(sorted(matches, key=lambda match: (match.predicted_id, match.reference_id)))


def match_detections(
    predicted: Iterable[Detection],
    reference: Iterable[Detection],
    max_distance_px: float | None = None,
    *,
    gate_px: float | None = None,
) -> tuple[Match, ...]:
    """Match detections without ever crossing sequence or frame boundaries.

    Unlike :func:`evaluate_sequence`, this helper may receive several sequences;
    each ``(sequence, frame)`` group is matched independently.
    """

    gate = _resolve_gate(max_distance_px, gate_px)
    predicted_tuple = tuple(predicted)
    reference_tuple = tuple(reference)
    _validate_detections(predicted_tuple, "Predicted")
    _validate_detections(reference_tuple, "Reference")

    predicted_by_key: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    reference_by_key: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for detection in predicted_tuple:
        predicted_by_key[(detection.sequence, detection.frame)].append(detection)
    for detection in reference_tuple:
        reference_by_key[(detection.sequence, detection.frame)].append(detection)

    matches: list[Match] = []
    for key in sorted(predicted_by_key.keys() | reference_by_key.keys()):
        predicted_frame = predicted_by_key[key]
        reference_frame = reference_by_key[key]
        _validate_unique_ids(predicted_frame, "Predicted")
        _validate_unique_ids(reference_frame, "Reference")
        matches.extend(_match_single_frame(predicted_frame, reference_frame, gate))
    return tuple(matches)


def evaluate_frame(
    predicted: Iterable[Detection],
    reference: Iterable[Detection],
    max_distance_px: float | None = None,
    *,
    gate_px: float | None = None,
    sequence: str | None = None,
    frame: int | None = None,
) -> FrameEvaluation:
    """Evaluate exactly one sequence/frame, rejecting mixed-frame input.

    ``sequence`` and ``frame`` are normally inferred.  They can be supplied to
    label an empty frame; without explicit labels a fully empty frame uses ``""``
    and ``-1`` respectively.
    """

    gate = _resolve_gate(max_distance_px, gate_px)
    predicted_tuple = tuple(predicted)
    reference_tuple = tuple(reference)
    _validate_detections(predicted_tuple, "Predicted")
    _validate_detections(reference_tuple, "Reference")
    keys = {(detection.sequence, detection.frame) for detection in predicted_tuple + reference_tuple}
    if len(keys) > 1:
        raise ValueError("evaluate_frame accepts detections from exactly one sequence and frame")

    inferred_sequence, inferred_frame = next(iter(keys), ("", -1))
    resolved_sequence = inferred_sequence if sequence is None else sequence
    resolved_frame = inferred_frame if frame is None else frame
    if keys and (resolved_sequence, resolved_frame) != (inferred_sequence, inferred_frame):
        raise ValueError("Explicit sequence/frame does not match the detections")

    _validate_unique_ids(predicted_tuple, "Predicted")
    _validate_unique_ids(reference_tuple, "Reference")
    matches = _match_single_frame(predicted_tuple, reference_tuple, gate)
    matched_predicted_ids = {match.predicted_id for match in matches}
    matched_reference_ids = {match.reference_id for match in matches}
    unmatched_predicted_ids = tuple(
        sorted(
            detection.detection_id
            for detection in predicted_tuple
            if detection.detection_id not in matched_predicted_ids
        )
    )
    unmatched_reference_ids = tuple(
        sorted(
            detection.detection_id
            for detection in reference_tuple
            if detection.detection_id not in matched_reference_ids
        )
    )
    true_positives = len(matches)
    predicted_count = len(predicted_tuple)
    reference_count = len(reference_tuple)
    precision, recall, f1 = _scores(true_positives, predicted_count, reference_count)
    return FrameEvaluation(
        sequence=resolved_sequence,
        frame=resolved_frame,
        max_distance_px=gate,
        predicted_count=predicted_count,
        reference_count=reference_count,
        true_positives=true_positives,
        false_positives=predicted_count - true_positives,
        false_negatives=reference_count - true_positives,
        precision=precision,
        recall=recall,
        f1=f1,
        localization_rmse_px=_localization_rmse(matches),
        matches=matches,
        unmatched_predicted_ids=unmatched_predicted_ids,
        unmatched_reference_ids=unmatched_reference_ids,
    )


def evaluate_sequence(
    predicted: Iterable[Detection],
    reference: Iterable[Detection],
    max_distance_px: float | None = None,
    *,
    gate_px: float | None = None,
    sequence: str | None = None,
) -> SequenceEvaluation:
    """Evaluate all observed frames belonging to one sequence."""

    gate = _resolve_gate(max_distance_px, gate_px)
    predicted_tuple = tuple(predicted)
    reference_tuple = tuple(reference)
    _validate_detections(predicted_tuple, "Predicted")
    _validate_detections(reference_tuple, "Reference")
    observed_sequences = {
        detection.sequence for detection in predicted_tuple + reference_tuple
    }
    if len(observed_sequences) > 1:
        raise ValueError("evaluate_sequence accepts detections from exactly one sequence")
    inferred_sequence = next(iter(observed_sequences), "")
    resolved_sequence = inferred_sequence if sequence is None else sequence
    if observed_sequences and resolved_sequence != inferred_sequence:
        raise ValueError("Explicit sequence does not match the detections")

    predicted_by_frame: dict[int, list[Detection]] = defaultdict(list)
    reference_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in predicted_tuple:
        predicted_by_frame[detection.frame].append(detection)
    for detection in reference_tuple:
        reference_by_frame[detection.frame].append(detection)

    frames = tuple(
        evaluate_frame(
            predicted_by_frame[frame_number],
            reference_by_frame[frame_number],
            gate,
            sequence=resolved_sequence,
            frame=frame_number,
        )
        for frame_number in sorted(predicted_by_frame.keys() | reference_by_frame.keys())
    )
    predicted_count = sum(frame.predicted_count for frame in frames)
    reference_count = sum(frame.reference_count for frame in frames)
    true_positives = sum(frame.true_positives for frame in frames)
    false_positives = sum(frame.false_positives for frame in frames)
    false_negatives = sum(frame.false_negatives for frame in frames)
    precision, recall, f1 = _scores(true_positives, predicted_count, reference_count)
    frame_count = len(frames)
    all_matches = tuple(match for frame_result in frames for match in frame_result.matches)
    return SequenceEvaluation(
        sequence=resolved_sequence,
        max_distance_px=gate,
        frames=frames,
        predicted_count=predicted_count,
        reference_count=reference_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_precision=(
            math.fsum(frame_result.precision for frame_result in frames) / frame_count
            if frame_count
            else 0.0
        ),
        macro_recall=(
            math.fsum(frame_result.recall for frame_result in frames) / frame_count
            if frame_count
            else 0.0
        ),
        macro_f1=(
            math.fsum(frame_result.f1 for frame_result in frames) / frame_count
            if frame_count
            else 0.0
        ),
        localization_rmse_px=_localization_rmse(all_matches),
    )


__all__ = [
    "DEFAULT_MAX_DISTANCE_PX",
    "FrameEvaluation",
    "Match",
    "SequenceEvaluation",
    "evaluate_frame",
    "evaluate_sequence",
    "match_detections",
]
