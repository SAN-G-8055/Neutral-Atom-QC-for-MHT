"""Check local sequence-01 path conventions and TIFF loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from cell_data import FRAME_COUNT, load_tiff, raw_frame_path


def test_sequence_01_path_convention() -> None:
    root = Path("data") / "PhC-C2DL-PSC"

    assert raw_frame_path(root, 7) == root / "01" / "t007.tif"


def test_frame_range_is_exactly_zero_through_299() -> None:
    assert FRAME_COUNT == 300
    with pytest.raises(ValueError):
        raw_frame_path("data", -1)
    with pytest.raises(ValueError):
        raw_frame_path("data", 300)


def test_load_tiff_preserves_local_pixel_values(tmp_path) -> None:
    pixels = np.array([[0, 17], [255, 42]], dtype=np.uint8)
    image_path = tmp_path / "frame.tif"
    Image.fromarray(pixels).save(image_path)

    loaded = load_tiff(image_path)

    assert np.array_equal(loaded, pixels)
    assert loaded.dtype == pixels.dtype


def test_load_tiff_rejects_non_two_dimensional_images(tmp_path) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    image_path = tmp_path / "color.tif"
    Image.fromarray(pixels).save(image_path)

    with pytest.raises(ValueError, match="two-dimensional"):
        load_tiff(image_path)
