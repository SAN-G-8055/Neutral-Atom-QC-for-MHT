"""Check command-line parsing and safe, bounded output-path choices."""

from pathlib import Path
from argparse import ArgumentTypeError

import pytest

from neutral_atom_mht import cli
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

    assert full == Path("artifacts")
    assert subset == Path("outputs/detection/sequence_01_frames_000-009")


def test_large_sparse_selection_has_a_bounded_stable_output_name() -> None:
    frames = tuple(range(0, 300, 2))

    first = _default_output_dir(frames)
    second = _default_output_dir(frames)

    assert first == second
    assert first.parent == Path("outputs/detection")
    assert first.name.startswith("sequence_01_frames_000-298_150-frames_")
    assert len(first.name) < 80


def test_run_command_passes_parsed_values_to_the_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "results"
    received: dict[str, object] = {}

    def fake_benchmark(dataset, output, *, frames, progress):
        received.update(
            dataset=dataset,
            output=output,
            frames=frames,
            progress=progress,
        )
        return {
            "primary_metrics": {
                "f1": 0.8,
                "precision": 0.75,
                "recall": 0.857,
                "true_positives": 6,
                "false_positives": 2,
                "false_negatives": 1,
            }
        }

    monkeypatch.setattr(cli, "run_detection_benchmark", fake_benchmark)

    exit_code = cli.main(
        [
            "run",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--frames",
            "1-2,4",
        ]
    )

    assert exit_code == 0
    assert received == {
        "dataset": dataset_root,
        "output": output_dir,
        "frames": (1, 2, 4),
        "progress": print,
    }
    output = capsys.readouterr().out
    assert "F1=0.800 precision=0.750 recall=0.857" in output
    assert f"artifacts written to {output_dir}" in output


def test_prepare_data_command_reports_the_prepared_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    dataset_root = data_dir / "PhC-C2DL-PSC"
    monkeypatch.setattr(cli, "prepare_sequence_01", lambda output: dataset_root)

    assert cli.main(["prepare-data", "--data-dir", str(data_dir)]) == 0
    assert capsys.readouterr().out.strip() == f"sequence 01 ready at {dataset_root}"
