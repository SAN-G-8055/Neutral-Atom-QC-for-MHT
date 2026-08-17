"""Check that diagnostic figures are validated, saved, and closed cleanly."""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neutral_atom_mht.detection import Detection
from neutral_atom_mht.evaluation import Match, evaluate_frame
from neutral_atom_mht.visualization import (
    save_detection_overview,
    save_per_frame_performance,
)


def _detection(
    frame: int,
    detection_id: int,
    x_px: float,
    y_px: float,
    *,
    source: str = "prediction",
) -> Detection:
    return Detection("01", frame, detection_id, x_px, y_px, 1, source)


def _example_data():
    images = {
        0: np.zeros((24, 32), dtype=np.uint8),
        2: np.full((24, 32), 127, dtype=np.uint8),
    }
    predictions = {
        0: [_detection(0, 10, 5.0, 6.0), _detection(0, 11, 22.0, 15.0)],
        2: [_detection(2, 20, 12.0, 10.0)],
    }
    references = {
        0: [
            _detection(0, 1, 5.5, 6.0, source="gold"),
            _detection(0, 2, 14.0, 13.0, source="gold"),
        ],
        2: [_detection(2, 3, 12.5, 10.0, source="gold")],
    }
    evaluations = {
        frame: evaluate_frame(predictions[frame], references[frame], max_distance_px=1.0)
        for frame in images
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
    evaluations[0] = replace(
        evaluations[0],
        matches=(Match("01", 0, 999, 1, 0.5),),
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
    evaluations[0] = replace(
        evaluations[0],
        matches=(Match("01", 0, 10, 1, 0.5), Match("01", 0, 10, 2, 9.0)),
        unmatched_reference_ids=(),
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
    evaluations[0] = replace(
        evaluations[0],
        unmatched_predicted_ids=(),
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
