"""Locate, load, and verify the local sequence-01 microscopy TIFF files.

The project expects raw sequence ``01`` and its human ``GT/TRA`` annotations
to already be present below the supplied dataset root. Dataset acquisition is
deliberately outside the package.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image


DATASET_NAME = "PhC-C2DL-PSC"
SEQUENCE = "01"
FRAME_COUNT = 300
IMAGE_SHAPE = (576, 720)
CANONICAL_RAW_SHA256 = "46c15979d995a6e8f3bbbed78652965c7575fba8f4d49da87493903e051b90fa"
CANONICAL_GOLD_SHA256 = "4795100971222e24686c8dae8532c24d4c99d7401c08b418937b2968dd56f01b"


def raw_frame_path(dataset_root: str | Path, frame: int) -> Path:
    _validate_frame(frame)
    return Path(dataset_root) / SEQUENCE / f"t{frame:03d}.tif"


def gold_tracking_path(dataset_root: str | Path, frame: int) -> Path:
    _validate_frame(frame)
    return Path(dataset_root) / f"{SEQUENCE}_GT" / "TRA" / f"man_track{frame:03d}.tif"


def _validate_frame(frame: int) -> None:
    if not 0 <= frame < FRAME_COUNT:
        raise ValueError(f"frame must be in [0, {FRAME_COUNT - 1}], got {frame}")


def load_tiff(path: str | Path) -> np.ndarray:
    """Load one two-dimensional TIFF without changing its stored values."""

    image_path = Path(path)
    with Image.open(image_path) as opened:
        array = np.asarray(opened).copy()
    if array.ndim != 2:
        raise ValueError(f"expected a two-dimensional TIFF, got {array.shape}: {image_path}")
    return array


def paired_frame_paths(dataset_root: str | Path) -> Iterator[tuple[int, Path, Path]]:
    """Yield all 300 raw/gold path pairs, failing clearly on missing data."""

    root = Path(dataset_root)
    missing: list[Path] = []
    pairs: list[tuple[int, Path, Path]] = []
    for frame in range(FRAME_COUNT):
        raw = raw_frame_path(root, frame)
        gold = gold_tracking_path(root, frame)
        if not raw.is_file():
            missing.append(raw)
        if not gold.is_file():
            missing.append(gold)
        pairs.append((frame, raw, gold))
    if missing:
        preview = ", ".join(str(path) for path in missing[:4])
        suffix = " ..." if len(missing) > 4 else ""
        raise FileNotFoundError(f"sequence 01 is incomplete; missing {len(missing)} files: {preview}{suffix}")
    yield from pairs


def verify_sequence_01(dataset_root: str | Path) -> tuple[str, str]:
    """Validate completeness and byte-for-byte canonical source fingerprints."""

    from artifact_io import sha256_files

    pairs = tuple(paired_frame_paths(dataset_root))
    raw_hash = sha256_files(raw for _, raw, _ in pairs)
    gold_hash = sha256_files(gold for _, _, gold in pairs)
    if raw_hash != CANONICAL_RAW_SHA256:
        raise ValueError(
            "raw sequence 01 does not match the canonical project fingerprint; "
            f"expected {CANONICAL_RAW_SHA256}, got {raw_hash}"
        )
    if gold_hash != CANONICAL_GOLD_SHA256:
        raise ValueError(
            "sequence-01 human tracking gold does not match the canonical project fingerprint; "
            f"expected {CANONICAL_GOLD_SHA256}, got {gold_hash}"
        )
    return raw_hash, gold_hash
