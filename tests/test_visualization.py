"""Check that diagnostic figures are validated, saved, and closed cleanly."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neutral_atom_mht.visualization import (
    save_detection_overview,
    save_per_frame_performance,
)


@dataclass(frozen=True)
class Detection:
    frame: int
    detection_id: int
    x_px: float
    y_px: float


@dataclass(frozen=True)
class Match:
    predicted_id: int
    reference_id: int
    distance_px: float


@dataclass(frozen=True)
class FrameEvaluation:
    frame: int
    matches: tuple[Match, ...]
    false_positive_ids: tuple[int, ...]
    false_negative_ids: tuple[int, ...]
    precision: float
    recall: float
    f1: float


def _example_data():
    images = {
        0: np.zeros((24, 32), dtype=np.uint8),
        2: np.full((24, 32), 127, dtype=np.uint8),
    }
    predictions = {
        0: [Detection(0, 10, 5.0, 6.0), Detection(0, 11, 22.0, 15.0)],
        2: [Detection(2, 20, 12.0, 10.0)],
    }
    references = {
        0: [Detection(0, 1, 5.5, 6.0), Detection(0, 2, 14.0, 13.0)],
        2: [Detection(2, 3, 12.5, 10.0)],
    }
    evaluations = {
        0: FrameEvaluation(
            frame=0,
            matches=(Match(10, 1, 0.5),),
            false_positive_ids=(11,),
            false_negative_ids=(2,),
            precision=0.5,
            recall=0.5,
            f1=0.5,
        ),
        2: FrameEvaluation(
            frame=2,
            matches=(Match(20, 3, 0.5),),
            false_positive_ids=(),
            false_negative_ids=(),
            precision=1.0,
            recall=1.0,
            f1=1.0,
        ),
    }
    return images, predictions, references, evaluations


def test_save_detection_overview_writes_headless_figure_and_closes_it(tmp_path):
    images, predictions, references, evaluations = _example_data()
    output = tmp_path / "figures" / "detections.png"

    save_detection_overview(
        images,
        predictions,
        references,
        evaluations,
        output,
        frames=[2, 0],
        columns=2,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert plt.get_fignums() == []


def test_save_per_frame_performance_writes_headless_figure_and_closes_it(tmp_path):
    _, _, _, evaluations = _example_data()
    output = tmp_path / "performance.png"

    save_per_frame_performance(evaluations, output)

    assert output.is_file()
    assert output.stat().st_size > 0
    assert plt.get_fignums() == []


def test_overview_rejects_evaluation_ids_absent_from_detections(tmp_path):
    images, predictions, references, evaluations = _example_data()
    evaluations[0] = FrameEvaluation(
        frame=0,
        matches=(Match(999, 1, 0.5),),
        false_positive_ids=(11,),
        false_negative_ids=(2,),
        precision=0.5,
        recall=0.5,
        f1=0.5,
    )

    with pytest.raises(ValueError, match="unknown predicted ID"):
        save_detection_overview(
            images,
            predictions,
            references,
            evaluations,
            tmp_path / "invalid.png",
            frames=[0],
        )

    assert plt.get_fignums() == []


def test_overview_rejects_non_one_to_one_matches(tmp_path):
    images, predictions, references, evaluations = _example_data()
    evaluations[0] = FrameEvaluation(
        frame=0,
        matches=(Match(10, 1, 0.5), Match(10, 2, 9.0)),
        false_positive_ids=(11,),
        false_negative_ids=(),
        precision=1.0,
        recall=1.0,
        f1=1.0,
    )

    with pytest.raises(ValueError, match="predicted ID more than once"):
        save_detection_overview(
            images,
            predictions,
            references,
            evaluations,
            tmp_path / "invalid.png",
            frames=[0],
        )

    assert plt.get_fignums() == []


def test_performance_requires_at_least_one_evaluation(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        save_per_frame_performance({}, tmp_path / "empty.png")


def test_overview_rejects_events_omitted_from_evaluation(tmp_path):
    images, predictions, references, evaluations = _example_data()
    evaluations[0] = FrameEvaluation(
        frame=0,
        matches=(Match(10, 1, 0.5),),
        false_positive_ids=(),
        false_negative_ids=(2,),
        precision=1.0,
        recall=0.5,
        f1=2 / 3,
    )

    with pytest.raises(ValueError, match="omits predicted ID"):
        save_detection_overview(
            images,
            predictions,
            references,
            evaluations,
            tmp_path / "invalid.png",
            frames=[0],
        )
