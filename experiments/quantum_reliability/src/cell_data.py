"""Locate and load the local sequence-01 microscopy TIFF files.

The project expects raw sequence ``01`` frames to already be present below
the supplied dataset root. Dataset acquisition is deliberately outside the
package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


DATASET_NAME = "PhC-C2DL-PSC"
SEQUENCE = "01"
FRAME_COUNT = 300


def raw_frame_path(dataset_root: str | Path, frame: int) -> Path:
    _validate_frame(frame)
    return Path(dataset_root) / SEQUENCE / f"t{frame:03d}.tif"


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
