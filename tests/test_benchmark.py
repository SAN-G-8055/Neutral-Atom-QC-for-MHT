"""Check end-to-end detection benchmark orchestration, provenance, and integrity."""

from __future__ import annotations

import numpy as np
import pytest

from neutral_atom_mht.benchmark import _validate_frame_arrays, run_detection_benchmark


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


def test_programmatic_frame_selection_rejects_fractional_frames(tmp_path) -> None:
    with pytest.raises(ValueError, match="frames must be integers"):
        run_detection_benchmark(tmp_path, tmp_path / "output", frames=[1.9])
