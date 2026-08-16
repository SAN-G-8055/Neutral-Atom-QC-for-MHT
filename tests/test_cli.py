from pathlib import Path
from argparse import ArgumentTypeError

import pytest

from neutral_atom_mht.cli import _default_output_dir, _parse_frames


def test_frame_parser_accepts_ranges_and_deduplicates() -> None:
    assert _parse_frames("0-2,2,5") == (0, 1, 2, 5)


def test_frame_parser_rejects_descending_or_out_of_range_values() -> None:
    with pytest.raises(ArgumentTypeError, match="descending"):
        _parse_frames("5-2")
    with pytest.raises(ArgumentTypeError, match="0-299"):
        _parse_frames("300")
    with pytest.raises(ArgumentTypeError, match="0-299"):
        _parse_frames("0-1000000000")


def test_subset_default_cannot_overwrite_curated_full_run() -> None:
    full = _default_output_dir(tuple(range(300)))
    subset = _default_output_dir(tuple(range(10)))

    assert full == Path("artifacts/detection/sequence_01")
    assert subset == Path("outputs/detection/sequence_01_frames_000-009")


def test_large_sparse_selection_has_a_bounded_stable_output_name() -> None:
    frames = tuple(range(0, 300, 2))

    first = _default_output_dir(frames)
    second = _default_output_dir(frames)

    assert first == second
    assert first.parent == Path("outputs/detection")
    assert first.name.startswith("sequence_01_frames_000-298_150-frames_")
    assert len(first.name) < 80
