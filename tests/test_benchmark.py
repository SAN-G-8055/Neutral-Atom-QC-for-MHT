"""Check end-to-end detection benchmark orchestration, provenance, and integrity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark import _validate_frame_arrays, run_detection_benchmark
from cell_data import DATASET_NAME, raw_frame_path


def test_raw_and_gold_must_share_one_coordinate_domain() -> None:
    raw = np.zeros((8, 10), dtype=np.uint8)
    gold = np.zeros((7, 10), dtype=np.uint16)

    with pytest.raises(ValueError, match="shapes differ"):
        _validate_frame_arrays(raw, gold, frame=3)


@pytest.mark.parametrize(
    ("raw_dtype", "gold_dtype", "message"),
    [
        (np.uint16, np.uint16, "raw image must be uint8"),
        (np.uint8, np.uint8, "gold tracking mask must be uint16"),
    ],
)
def test_sequence_01_tiff_types_are_part_of_the_contract(
    raw_dtype: np.dtype,
    gold_dtype: np.dtype,
    message: str,
) -> None:
    raw = np.zeros((576, 720), dtype=raw_dtype)
    gold = np.zeros((576, 720), dtype=gold_dtype)

    with pytest.raises(ValueError, match=message):
        _validate_frame_arrays(raw, gold, frame=0)


def test_sequence_01_frame_shape_is_fixed() -> None:
    raw = np.zeros((8, 10), dtype=np.uint8)
    gold = np.zeros((8, 10), dtype=np.uint16)

    with pytest.raises(ValueError, match="must have shape"):
        _validate_frame_arrays(raw, gold, frame=0)


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ((), "at least one frame"),
        ((-1,), "frames must lie"),
        ((300,), "frames must lie"),
    ],
)
def test_programmatic_frame_selection_rejects_invalid_values(
    tmp_path: Path,
    frames: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_detection_benchmark(tmp_path, tmp_path / "output", frames=frames)


def test_one_real_frame_produces_the_complete_artifact_contract(tmp_path: Path) -> None:
    dataset_root = Path("data") / DATASET_NAME
    if not raw_frame_path(dataset_root, 0).is_file():
        pytest.skip("the gitignored sequence-01 dataset is not available")

    output = tmp_path / "benchmark"
    summary = run_detection_benchmark(dataset_root, output, frames=(0,))

    assert summary["dataset"]["frame_selection"]["count"] == 1
    assert summary["primary_metrics"]["predicted_count"] > 0
    artifacts = {path.name: path for path in output.iterdir()}
    assert set(artifacts) == {
        "detections.csv",
        "gold_events.csv",
        "matches.csv",
        "per_frame_metrics.csv",
        "summary.json",
        "detections_overview.png",
        "performance_over_time.png",
    }
    assert all(path.stat().st_size > 0 for path in artifacts.values())
    saved_summary = json.loads(artifacts["summary.json"].read_text(encoding="utf-8"))
    assert saved_summary["dataset"] == summary["dataset"]
    assert saved_summary["primary_metrics"] == summary["primary_metrics"]
