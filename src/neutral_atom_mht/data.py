"""The single microscopy dataset retained by the project: sequence 01.

Only raw sequence ``01`` and its human ``GT/TRA`` annotations are needed for
event detection.  The challenge test set has no public gold annotations, while
``ST`` and ``ERR_SEG`` are non-human segmentation products, so none of those are
downloaded by :func:`prepare_sequence_01`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing
from numbers import Integral
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image


DATASET_NAME = "PhC-C2DL-PSC"
SEQUENCE = "01"
FRAME_COUNT = 300
IMAGE_SHAPE = (576, 720)
TRAINING_ARCHIVE_URL = (
    "https://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip"
)
TRAINING_ARCHIVE_SIZE_BYTES = 145_227_316
CANONICAL_EXTRACTED_SIZE_BYTES = 62_440_750
MAX_ARCHIVE_MEMBER_SIZE_BYTES = 1_000_000
CANONICAL_RAW_SHA256 = "46c15979d995a6e8f3bbbed78652965c7575fba8f4d49da87493903e051b90fa"
CANONICAL_GOLD_SHA256 = "4795100971222e24686c8dae8532c24d4c99d7401c08b418937b2968dd56f01b"


def raw_frame_path(dataset_root: str | Path, frame: int) -> Path:
    _validate_frame(frame)
    return Path(dataset_root) / SEQUENCE / f"t{frame:03d}.tif"


def gold_tracking_path(dataset_root: str | Path, frame: int) -> Path:
    _validate_frame(frame)
    return Path(dataset_root) / f"{SEQUENCE}_GT" / "TRA" / f"man_track{frame:03d}.tif"


def _validate_frame(frame: int) -> None:
    if not isinstance(frame, Integral) or isinstance(frame, bool):
        raise ValueError(f"frame must be an integer in [0, {FRAME_COUNT - 1}], got {frame!r}")
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


def _safe_archive_target(staging_root: Path, member_name: str) -> Path:
    """Resolve an archive member below staging on POSIX and Windows."""

    if "\\" in member_name:
        raise ValueError(f"unsafe archive member uses a backslash: {member_name}")
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member: {member_name}")
    resolved_root = staging_root.resolve()
    target = resolved_root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(resolved_root):
        raise ValueError(f"archive member escapes staging directory: {member_name}")
    return target


def verify_sequence_01(dataset_root: str | Path) -> tuple[str, str]:
    """Validate completeness and byte-for-byte canonical source fingerprints."""

    from .io import sha256_files

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


def prepare_sequence_01(output_root: str | Path) -> Path:
    """Download and keep only raw sequence 01 plus human tracking gold.

    The 139 MB source archive is temporary.  Extraction is staged and validated
    before the selected 60 MB subset is moved into ``output_root``.  Existing
    complete data is left untouched; an incomplete target is never overwritten.
    """

    destination_parent = Path(output_root)
    dataset_root = destination_parent / DATASET_NAME
    if dataset_root.exists():
        verify_sequence_01(dataset_root)
        return dataset_root

    destination_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sequence_01_", dir=destination_parent) as temporary:
        temporary_root = Path(temporary)
        archive_path = temporary_root / f"{DATASET_NAME}.zip"
        request = Request(TRAINING_ARCHIVE_URL, headers={"User-Agent": "neutral-atom-mht/0.1"})
        with closing(urlopen(request, timeout=60)) as response, archive_path.open("wb") as output:
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and int(declared_size) != TRAINING_ARCHIVE_SIZE_BYTES:
                raise ValueError(
                    f"official archive size changed: expected {TRAINING_ARCHIVE_SIZE_BYTES}, "
                    f"server declared {declared_size}"
                )
            received = 0
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > TRAINING_ARCHIVE_SIZE_BYTES:
                    raise ValueError("official archive exceeded its canonical size")
                output.write(block)
            if received != TRAINING_ARCHIVE_SIZE_BYTES:
                raise ValueError(
                    f"incomplete official archive: expected {TRAINING_ARCHIVE_SIZE_BYTES} bytes, "
                    f"received {received}"
                )

        expected_names = [
            *(f"{DATASET_NAME}/{SEQUENCE}/t{frame:03d}.tif" for frame in range(FRAME_COUNT)),
            *(f"{DATASET_NAME}/{SEQUENCE}_GT/TRA/man_track{frame:03d}.tif" for frame in range(FRAME_COUNT)),
            f"{DATASET_NAME}/{SEQUENCE}_GT/TRA/man_track.txt",
        ]
        with zipfile.ZipFile(archive_path) as archive:
            available = set(archive.namelist())
            missing = sorted(set(expected_names) - available)
            if missing:
                raise ValueError(
                    f"official archive is missing {len(missing)} required sequence-01 files"
                )
            members = [archive.getinfo(member_name) for member_name in expected_names]
            if any(member.file_size > MAX_ARCHIVE_MEMBER_SIZE_BYTES for member in members):
                raise ValueError("official archive contains an oversized selected member")
            uncompressed_size = sum(member.file_size for member in members)
            if uncompressed_size != CANONICAL_EXTRACTED_SIZE_BYTES:
                raise ValueError(
                    f"selected archive contents changed size: expected "
                    f"{CANONICAL_EXTRACTED_SIZE_BYTES}, got {uncompressed_size}"
                )
            for member in members:
                target = _safe_archive_target(temporary_root, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    written = 0
                    while block := source.read(1024 * 1024):
                        written += len(block)
                        if written > member.file_size:
                            raise ValueError(f"archive member expanded past its declared size: {member.filename}")
                        output.write(block)
                    if written != member.file_size:
                        raise ValueError(f"archive member was truncated: {member.filename}")

        staged_dataset = temporary_root / DATASET_NAME
        verify_sequence_01(staged_dataset)
        staged_dataset.replace(dataset_root)

    return dataset_root
