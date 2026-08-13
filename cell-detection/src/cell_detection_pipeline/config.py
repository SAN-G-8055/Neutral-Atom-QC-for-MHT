"""Configuration for the deterministic segmentation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters for background correction, seed extraction, and cell growth."""

    gaussian_sigma: float = 1.0
    background_sigma: float = 8.0
    histogram_bins: int = 256
    histogram_percentile_low: float = 0.1
    histogram_percentile_high: float = 99.9
    high_threshold_factor: float = 1.2
    low_threshold_factor: float = 0.04
    opening_size: int = 2
    closing_size: int = 3
    min_seed_area_px: int = 40
    max_seed_area_px: int = 1000
    min_final_area_px: int = 40
    max_final_area_px: int = 1500
    connectivity: int = 8

    def __post_init__(self) -> None:
        if self.gaussian_sigma < 0 or self.background_sigma <= 0:
            raise ValueError("Gaussian sigma values must be non-negative, with background_sigma > 0")
        if self.background_sigma <= self.gaussian_sigma:
            raise ValueError("background_sigma must be greater than gaussian_sigma")
        if not 0 <= self.histogram_percentile_low < self.histogram_percentile_high <= 100:
            raise ValueError("Histogram percentiles must be ordered within [0, 100]")
        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2")
        if self.high_threshold_factor <= 0:
            raise ValueError("high_threshold_factor must be positive")
        if not 0 <= self.low_threshold_factor <= self.high_threshold_factor:
            raise ValueError("low_threshold_factor must be between 0 and high_threshold_factor")
        if self.opening_size < 1 or self.closing_size < 1:
            raise ValueError("Morphology sizes must be positive")
        if not 0 < self.min_seed_area_px <= self.max_seed_area_px:
            raise ValueError("Seed area bounds are invalid")
        if not 0 < self.min_final_area_px <= self.max_final_area_px:
            raise ValueError("Final area bounds are invalid")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SegmentationConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
