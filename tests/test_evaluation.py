"""Check frame-safe matching and detection figures of merit."""

from __future__ import annotations

import math

import pytest

from neutral_atom_mht.detection import Detection
from neutral_atom_mht.evaluation import evaluate_frame, evaluate_sequence, match_detections


def _detection(
    detection_id: int,
    x_px: float,
    y_px: float = 0.0,
    *,
    sequence: str = "01",
    frame: int = 0,
    source: str = "prediction",
) -> Detection:
    return Detection(
        sequence=sequence,
        frame=frame,
        detection_id=detection_id,
        x_px=x_px,
        y_px=y_px,
        area_px=1,
        source=source,
    )


def test_matching_maximizes_cardinality_before_minimizing_distance() -> None:
    # Three zero-distance pairs look cheaper to a naïve gated Hungarian match,
    # but the shifted chain is a valid four-pair assignment at the inclusive gate.
    predicted = [_detection(index, x) for index, x in enumerate((1, 2, 3, 4), 10)]
    reference = [
        _detection(index, x, source="gold")
        for index, x in enumerate((1, 2, 3, 0), 20)
    ]

    result = evaluate_frame(predicted, reference, max_distance_px=1.0)

    assert result.true_positives == 4
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == result.recall == result.f1 == 1.0
    assert [match.distance_px for match in result.matches] == [1.0, 1.0, 1.0, 1.0]


def test_matches_are_one_to_one_and_expose_audit_ids() -> None:
    predicted = [_detection(11, 0.0), _detection(12, 0.25), _detection(13, 10.0)]
    reference = [_detection(21, 0.0, source="gold")]

    result = evaluate_frame(predicted, reference, gate_px=1.0)

    assert result.true_positives == 1
    assert result.matches[0].predicted_id == 11
    assert result.matches[0].reference_id == 21
    assert result.matches[0].distance_px == 0.0
    assert result.unmatched_predicted_ids == (12, 13)
    assert result.unmatched_reference_ids == ()
    assert result.localization_rmse_px == 0.0


def test_sequence_evaluation_never_matches_across_frames() -> None:
    predicted = [_detection(1, 5.0, frame=0)]
    reference = [_detection(2, 5.0, frame=1, source="gold")]

    result = evaluate_sequence(predicted, reference, max_distance_px=1.0)

    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.matches == ()
    assert [frame.frame for frame in result.frames] == [0, 1]
    assert match_detections(predicted, reference, 1.0) == ()


def test_sequence_reports_micro_primary_macro_per_frame_and_global_rmse() -> None:
    predicted = [
        _detection(1, 0.0, frame=0),
        _detection(2, 20.0, frame=0),
        _detection(3, 3.0, 4.0, frame=1),
    ]
    reference = [
        _detection(101, 0.0, frame=0, source="gold"),
        _detection(102, 0.0, frame=1, source="gold"),
        _detection(103, 20.0, frame=1, source="gold"),
    ]

    result = evaluate_sequence(predicted, reference, max_distance_px=5.0)

    assert (result.true_positives, result.false_positives, result.false_negatives) == (2, 1, 1)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.micro_precision == result.precision
    assert result.macro_precision == pytest.approx(0.75)
    assert result.macro_recall == pytest.approx(0.75)
    assert result.macro_f1 == pytest.approx(2 / 3)
    assert result.localization_rmse_px == pytest.approx(math.sqrt((0.0**2 + 5.0**2) / 2))


def test_gate_is_inclusive_and_must_be_finite_and_positive() -> None:
    predicted = [_detection(1, 0.0, 0.0)]
    reference = [_detection(2, 3.0, 4.0, source="gold")]

    result = evaluate_frame(predicted, reference, max_distance_px=5.0)

    assert result.true_positives == 1
    assert result.matches[0].distance_px == 5.0
    for invalid_gate in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite positive"):
            evaluate_frame(predicted, reference, max_distance_px=invalid_gate)


def test_empty_sides_have_defined_zero_scores_and_no_localization_rmse() -> None:
    empty = evaluate_frame([], [], sequence="01", frame=7, max_distance_px=2.0)
    predicted_only = evaluate_frame(
        [_detection(1, 0.0, frame=7)],
        [],
        max_distance_px=2.0,
    )
    reference_only = evaluate_frame(
        [],
        [_detection(2, 0.0, frame=7, source="gold")],
        max_distance_px=2.0,
    )

    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)
    assert empty.localization_rmse_px is None
    assert (predicted_only.true_positives, predicted_only.false_positives) == (0, 1)
    assert (reference_only.true_positives, reference_only.false_negatives) == (0, 1)


def test_mixed_frames_and_sequences_are_rejected_by_scoped_evaluators() -> None:
    with pytest.raises(ValueError, match="one sequence and frame"):
        evaluate_frame(
            [_detection(1, 0.0, frame=0)],
            [_detection(2, 0.0, frame=1, source="gold")],
        )
    with pytest.raises(ValueError, match="one sequence"):
        evaluate_sequence(
            [_detection(1, 0.0, sequence="01")],
            [_detection(2, 0.0, sequence="02", source="gold")],
        )


def test_non_finite_coordinates_and_duplicate_ids_fail_clearly() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_frame([_detection(1, float("nan"))], [])
    with pytest.raises(ValueError, match="IDs must be unique"):
        evaluate_frame([_detection(1, 0.0), _detection(1, 1.0)], [])
