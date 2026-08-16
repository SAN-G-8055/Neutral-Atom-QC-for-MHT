"""Command-line entry point for the cleaned detection stage."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from pathlib import Path

from .data import DATASET_NAME, FRAME_COUNT, prepare_sequence_01
from .pipeline import run_detection_benchmark


def _parse_frames(value: str) -> tuple[int, ...]:
    frames: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", maxsplit=1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise argparse.ArgumentTypeError(f"descending frame range: {token}")
            if start < 0 or stop >= FRAME_COUNT:
                raise argparse.ArgumentTypeError(
                    f"frames must be a non-empty subset of 0-{FRAME_COUNT - 1}"
                )
            frames.update(range(start, stop + 1))
        else:
            frame = int(token)
            if not 0 <= frame < FRAME_COUNT:
                raise argparse.ArgumentTypeError(
                    f"frames must be a non-empty subset of 0-{FRAME_COUNT - 1}"
                )
            frames.add(frame)
    if not frames or min(frames) < 0 or max(frames) >= FRAME_COUNT:
        raise argparse.ArgumentTypeError(f"frames must be a non-empty subset of 0-{FRAME_COUNT - 1}")
    return tuple(sorted(frames))


def _default_output_dir(frames: tuple[int, ...]) -> Path:
    complete = tuple(range(FRAME_COUNT))
    if frames == complete:
        return Path("artifacts") / "detection" / "sequence_01"
    continuous = frames == tuple(range(frames[0], frames[-1] + 1))
    if continuous:
        label = f"{frames[0]:03d}-{frames[-1]:03d}"
    else:
        encoded = ",".join(str(frame) for frame in frames).encode("ascii")
        fingerprint = hashlib.sha256(encoded).hexdigest()[:8]
        label = f"{frames[0]:03d}-{frames[-1]:03d}_{len(frames)}-frames_{fingerprint}"
    return Path("outputs") / "detection" / f"sequence_01_frames_{label}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cell-detect",
        description="Detect sequence-01 cells and compare them with human GT/TRA events.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-data",
        help="download only raw sequence 01 and its human tracking gold",
    )
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))

    run = commands.add_parser("run", help="run the reproducible detection benchmark")
    run.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data") / DATASET_NAME,
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="destination (full run defaults to artifacts/; subsets default to ignored outputs/)",
    )
    run.add_argument(
        "--frames",
        type=_parse_frames,
        default=tuple(range(FRAME_COUNT)),
        metavar="RANGE",
        help="comma-separated frames/ranges (default: 0-299)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        dataset_root = prepare_sequence_01(args.data_dir)
        print(f"sequence 01 ready at {dataset_root}")
        return 0
    output_dir = args.output_dir or _default_output_dir(args.frames)
    summary = run_detection_benchmark(
        args.dataset_root,
        output_dir,
        frames=args.frames,
        progress=print,
    )
    metrics = summary["primary_metrics"]
    print(
        "F1={f1:.3f} precision={precision:.3f} recall={recall:.3f} "
        "TP/FP/FN={true_positives}/{false_positives}/{false_negatives}".format(**metrics)
    )
    print(f"artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
