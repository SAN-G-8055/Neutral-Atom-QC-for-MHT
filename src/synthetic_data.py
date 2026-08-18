"""Generate deterministic synthetic tracking images in the local CTC layout.

The generator owns scene motion and rendering only.  Detection, graph
construction, tracking, and visualization stay with their existing modules
and workflows.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SYNTHETIC_DATA_ROOT = Path("data") / "synthetic"

_BACKGROUND_LEVEL = 135
_BACKGROUND_SHADING = 12.0
_BLOB_AMPLITUDE = 95.0
_BLOB_LONG_AXIS_PX = 3.4
_BLOB_SHORT_AXIS_PX = 1.7
_MARKER_RADIUS_PX = 1
_REGION_FRACTION = 0.40
_TURN_SIGMA_RAD = 0.16
_MAX_TRACKING_LABEL = int(np.iinfo(np.uint16).max)
# Stable, distinct stream identifiers.  These IDs retain the versioned quantum
# smoke topology while keeping the effects statistically and operationally
# separate.
_DROPOUT_STREAM_ID = 17
_AMPLITUDE_STREAM_ID = 118
_CLUTTER_STREAM_ID = 228
_SENSOR_NOISE_STREAM_ID = 324


@dataclass(frozen=True, slots=True)
class SyntheticDataConfig:
    """Configuration for one reproducible synthetic tracking sequence.

    ``noise`` is the convenient coupled difficulty control retained for the
    small demo.  Any ``*_override`` value replaces just that derived effect,
    which lets factorial benchmarks distinguish motion, dropout, clutter, and
    sensor noise.
    """

    noise: float = 0.6
    frame_count: int = 40
    object_count: int = 55
    seed: int = 0
    dataset_name: str = "SYN-MHT"
    sequence: str = "01"
    image_shape: tuple[int, int] = (576, 720)
    speed_px_per_frame_override: float | None = None
    detection_probability_override: float | None = None
    clutter_per_frame_override: float | None = None
    pixel_noise_sigma_override: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("noise must lie in [0, 1]")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if self.object_count < 1:
            raise ValueError("object_count must be positive")
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("image_shape must contain two positive dimensions")
        if (
            self.speed_px_per_frame_override is not None
            and (
                not np.isfinite(self.speed_px_per_frame_override)
                or self.speed_px_per_frame_override <= 0.0
            )
        ):
            raise ValueError(
                "speed_px_per_frame_override must be finite and positive"
            )
        if (
            self.detection_probability_override is not None
            and (
                not np.isfinite(self.detection_probability_override)
                or not 0.0 <= self.detection_probability_override <= 1.0
            )
        ):
            raise ValueError(
                "detection_probability_override must be finite and lie in [0, 1]"
            )
        if (
            self.clutter_per_frame_override is not None
            and (
                not np.isfinite(self.clutter_per_frame_override)
                or self.clutter_per_frame_override < 0.0
            )
        ):
            raise ValueError(
                "clutter_per_frame_override must be finite and non-negative"
            )
        if (
            self.pixel_noise_sigma_override is not None
            and (
                not np.isfinite(self.pixel_noise_sigma_override)
                or self.pixel_noise_sigma_override < 0.0
            )
        ):
            raise ValueError(
                "pixel_noise_sigma_override must be finite and non-negative"
            )
        if self.object_count > _MAX_TRACKING_LABEL:
            raise ValueError(
                f"object_count cannot exceed the uint16 label limit "
                f"({_MAX_TRACKING_LABEL})"
            )
        if self.object_count > self.height * self.width:
            raise ValueError("object_count cannot exceed the available image pixels")
        for name, value in (
            ("dataset_name", self.dataset_name),
            ("sequence", self.sequence),
        ):
            if not value or value in {".", ".."} or Path(value).name != value:
                raise ValueError(f"{name} must be one directory name")

    @property
    def height(self) -> int:
        return self.image_shape[0]

    @property
    def width(self) -> int:
        return self.image_shape[1]

    @property
    def speed_px_per_frame(self) -> float:
        if self.speed_px_per_frame_override is not None:
            return float(self.speed_px_per_frame_override)
        return 2.5 + 14.0 * self.noise

    @property
    def detection_probability(self) -> float:
        if self.detection_probability_override is not None:
            return float(self.detection_probability_override)
        return 0.98 - 0.36 * self.noise

    @property
    def clutter_per_frame(self) -> float:
        if self.clutter_per_frame_override is not None:
            return float(self.clutter_per_frame_override)
        return 36.0 * self.noise

    @property
    def pixel_noise_sigma(self) -> float:
        if self.pixel_noise_sigma_override is not None:
            return float(self.pixel_noise_sigma_override)
        return 1.5 + 6.0 * self.noise


# Small deterministic scene validated against the real quantum simulator.
QUANTUM_DEMO_DATA_CONFIG = SyntheticDataConfig(
    noise=0.1,
    frame_count=8,
    object_count=4,
    seed=0,
    dataset_name="SYN-MHT-QUANTUM-v1",
    sequence="01",
    image_shape=(256, 320),
)


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Paths and loaders for one generated synthetic sequence."""

    root: Path
    config: SyntheticDataConfig

    @property
    def raw_directory(self) -> Path:
        return self.root / self.config.sequence

    @property
    def tracking_directory(self) -> Path:
        return self.root / f"{self.config.sequence}_GT" / "TRA"

    @property
    def track_manifest_path(self) -> Path:
        return self.tracking_directory / "man_track.txt"

    def raw_frame_path(self, frame: int) -> Path:
        self._check_frame(frame)
        return self.raw_directory / f"t{frame:03d}.tif"

    def tracking_frame_path(self, frame: int) -> Path:
        self._check_frame(frame)
        return self.tracking_directory / f"man_track{frame:03d}.tif"

    def load_frame(self, frame: int) -> np.ndarray:
        return self._load(self.raw_frame_path(frame))

    def load_tracking_labels(self, frame: int) -> np.ndarray:
        return self._load(self.tracking_frame_path(frame))

    def _check_frame(self, frame: int) -> None:
        if not 0 <= frame < self.config.frame_count:
            raise ValueError(
                f"frame must lie in [0, {self.config.frame_count - 1}]"
            )

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image).copy()


class SyntheticDataGenerator:
    """Render a configured scene without changing global random state."""

    def __init__(self, config: SyntheticDataConfig | None = None) -> None:
        self.config = config or SyntheticDataConfig()

    def generate(
        self,
        output_root: str | Path = DEFAULT_SYNTHETIC_DATA_ROOT,
    ) -> SyntheticDataset:
        dataset = SyntheticDataset(
            root=Path(output_root) / self.config.dataset_name,
            config=self.config,
        )
        if dataset.root.exists() and (
            not dataset.root.is_dir() or any(dataset.root.iterdir())
        ):
            raise FileExistsError(
                f"refusing to overwrite nonempty synthetic dataset: {dataset.root}"
            )
        dataset.raw_directory.mkdir(parents=True, exist_ok=True)
        dataset.tracking_directory.mkdir(parents=True, exist_ok=True)

        for frame, (image, labels) in enumerate(self.iter_frames()):
            Image.fromarray(image).save(dataset.raw_frame_path(frame))
            Image.fromarray(labels).save(dataset.tracking_frame_path(frame))

        final_frame = self.config.frame_count - 1
        manifest = "".join(
            f"{object_id} 0 {final_frame} 0\n"
            for object_id in range(1, self.config.object_count + 1)
        )
        dataset.track_manifest_path.write_text(manifest, encoding="utf-8")
        return dataset

    def iter_frames(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield deterministic image/label pairs without writing TIFF files.

        Streaming is useful for large parameter sweeps: only one rendered frame
        is resident at a time, while :meth:`generate` remains the convenient
        Cell Tracking Challenge-compatible on-disk interface.
        """

        trajectory_rng = np.random.default_rng(self.config.seed)
        trajectories = self._generate_trajectories(trajectory_rng)
        background = self._background()

        # Preserve the exact byte stream of the original/default datasets,
        # including the versioned quantum demo. Benchmark scenarios set every
        # override explicitly and take the independent-stream path below.
        if self._uses_legacy_random_stream():
            for frame in range(self.config.frame_count):
                yield self._render_frame_legacy(
                    frame,
                    trajectories,
                    background,
                    trajectory_rng,
                )
            return

        # Keep each benchmark effect on its own deterministic stream.  Paired
        # scenarios can then vary motion, dropout, clutter, or sensor noise
        # without silently changing unrelated latent draws.
        dropout_rng = self._effect_rng(_DROPOUT_STREAM_ID)
        amplitude_rng = self._effect_rng(_AMPLITUDE_STREAM_ID)
        clutter_rng = self._effect_rng(_CLUTTER_STREAM_ID)
        sensor_noise_rng = self._effect_rng(_SENSOR_NOISE_STREAM_ID)
        for frame in range(self.config.frame_count):
            yield self._render_frame(
                frame,
                trajectories,
                background,
                dropout_rng=dropout_rng,
                amplitude_rng=amplitude_rng,
                clutter_rng=clutter_rng,
                sensor_noise_rng=sensor_noise_rng,
            )

    def _uses_legacy_random_stream(self) -> bool:
        return all(
            value is None
            for value in (
                self.config.speed_px_per_frame_override,
                self.config.detection_probability_override,
                self.config.clutter_per_frame_override,
                self.config.pixel_noise_sigma_override,
            )
        )

    def _effect_rng(self, stream_id: int) -> np.random.Generator:
        """Return a stable effect-specific stream derived from the scene seed."""

        return np.random.default_rng(
            np.random.SeedSequence((self.config.seed, stream_id))
        )

    def _generate_trajectories(self, rng: np.random.Generator) -> np.ndarray:
        config = self.config
        box_width = _REGION_FRACTION * config.width
        box_height = _REGION_FRACTION * config.height
        x_min = (config.width - box_width) / 2.0
        y_min = (config.height - box_height) / 2.0
        bounds = ((x_min, x_min + box_width), (y_min, y_min + box_height))

        positions = np.column_stack(
            (
                rng.uniform(*bounds[0], config.object_count),
                rng.uniform(*bounds[1], config.object_count),
            )
        )
        headings = rng.uniform(0.0, 2.0 * np.pi, config.object_count)
        speeds = np.full(config.object_count, config.speed_px_per_frame)
        trajectories = np.empty(
            (config.frame_count, config.object_count, 2),
            dtype=float,
        )

        for frame in range(config.frame_count):
            trajectories[frame] = positions
            headings += rng.normal(0.0, _TURN_SIGMA_RAD, config.object_count)
            speeds = np.clip(
                speeds
                + rng.normal(
                    0.0,
                    0.08 * config.speed_px_per_frame,
                    config.object_count,
                ),
                0.35 * config.speed_px_per_frame,
                1.8 * config.speed_px_per_frame,
            )
            positions += np.column_stack(
                (speeds * np.cos(headings), speeds * np.sin(headings))
            )
            self._reflect(positions, headings, bounds)

        return trajectories

    @staticmethod
    def _reflect(
        positions: np.ndarray,
        headings: np.ndarray,
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        for dimension, (lower, upper) in enumerate(bounds):
            outside = (positions[:, dimension] < lower) | (
                positions[:, dimension] > upper
            )
            while np.any(outside):
                below = positions[:, dimension] < lower
                above = positions[:, dimension] > upper
                positions[below, dimension] = (
                    2.0 * lower - positions[below, dimension]
                )
                positions[above, dimension] = (
                    2.0 * upper - positions[above, dimension]
                )
                reflected = below | above
                if dimension == 0:
                    headings[reflected] = np.pi - headings[reflected]
                else:
                    headings[reflected] = -headings[reflected]
                outside = (positions[:, dimension] < lower) | (
                    positions[:, dimension] > upper
                )

    def _background(self) -> np.ndarray:
        height, width = self.config.image_shape
        yy, xx = np.mgrid[0:height, 0:width]
        return (
            _BACKGROUND_LEVEL
            + _BACKGROUND_SHADING
            * np.sin(2.0 * np.pi * xx / (2.3 * width))
            * np.cos(2.0 * np.pi * yy / (1.9 * height))
        )

    def _render_frame(
        self,
        frame: int,
        trajectories: np.ndarray,
        background: np.ndarray,
        *,
        dropout_rng: np.random.Generator,
        amplitude_rng: np.random.Generator,
        clutter_rng: np.random.Generator,
        sensor_noise_rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        image = background.copy()
        labels = np.zeros(config.image_shape, dtype=np.uint16)
        free_label_pixels = config.height * config.width

        for object_index, (x, y) in enumerate(trajectories[frame]):
            object_id = object_index + 1
            previous = trajectories[max(0, frame - 1), object_index]
            angle = (
                np.arctan2(y - previous[1], x - previous[0])
                if frame > 0
                else 0.0
            )
            row, column = int(round(y)), int(round(x))
            used_pixels = self._place_tracking_marker(
                labels,
                row,
                column,
                object_id,
                free_pixels=free_label_pixels,
                remaining_objects=config.object_count - object_id,
            )
            free_label_pixels -= used_pixels

            # Draw amplitude even for a dropped object so changing dropout does
            # not shift later objects' latent brightness values.
            amplitude = _BLOB_AMPLITUDE * amplitude_rng.uniform(0.85, 1.15)
            if dropout_rng.random() <= config.detection_probability:
                self._add_blob(
                    image,
                    x,
                    y,
                    angle,
                    amplitude,
                )

        for _ in range(clutter_rng.poisson(config.clutter_per_frame)):
            self._add_blob(
                image,
                clutter_rng.uniform(0.0, config.width),
                clutter_rng.uniform(0.0, config.height),
                clutter_rng.uniform(0.0, 2.0 * np.pi),
                _BLOB_AMPLITUDE * clutter_rng.uniform(0.6, 1.0),
            )

        image += sensor_noise_rng.normal(
            0.0,
            config.pixel_noise_sigma,
            image.shape,
        )
        return np.clip(image, 0.0, 255.0).astype(np.uint8), labels

    def _render_frame_legacy(
        self,
        frame: int,
        trajectories: np.ndarray,
        background: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render the pre-override byte stream for cached dataset compatibility."""

        config = self.config
        image = background.copy()
        labels = np.zeros(config.image_shape, dtype=np.uint16)
        free_label_pixels = config.height * config.width

        for object_index, (x, y) in enumerate(trajectories[frame]):
            object_id = object_index + 1
            previous = trajectories[max(0, frame - 1), object_index]
            angle = (
                np.arctan2(y - previous[1], x - previous[0])
                if frame > 0
                else 0.0
            )
            row, column = int(round(y)), int(round(x))
            used_pixels = self._place_tracking_marker(
                labels,
                row,
                column,
                object_id,
                free_pixels=free_label_pixels,
                remaining_objects=config.object_count - object_id,
            )
            free_label_pixels -= used_pixels

            if rng.random() <= config.detection_probability:
                self._add_blob(
                    image,
                    x,
                    y,
                    angle,
                    _BLOB_AMPLITUDE * rng.uniform(0.85, 1.15),
                )

        for _ in range(rng.poisson(config.clutter_per_frame)):
            self._add_blob(
                image,
                rng.uniform(0.0, config.width),
                rng.uniform(0.0, config.height),
                rng.uniform(0.0, 2.0 * np.pi),
                _BLOB_AMPLITUDE * rng.uniform(0.6, 1.0),
            )

        image += rng.normal(0.0, config.pixel_noise_sigma, image.shape)
        return np.clip(image, 0.0, 255.0).astype(np.uint8), labels

    @classmethod
    def _place_tracking_marker(
        cls,
        labels: np.ndarray,
        row: int,
        column: int,
        object_id: int,
        *,
        free_pixels: int,
        remaining_objects: int,
    ) -> int:
        marker_width = 2 * _MARKER_RADIUS_PX + 1
        marker_area = marker_width**2
        if free_pixels - marker_area >= remaining_objects:
            center = cls._nearest_free_area(
                labels,
                row,
                column,
                radius=_MARKER_RADIUS_PX,
            )
            if center is not None:
                center_row, center_column = center
                radius = _MARKER_RADIUS_PX
                labels[
                    center_row - radius : center_row + radius + 1,
                    center_column - radius : center_column + radius + 1,
                ] = object_id
                return marker_area

        pixel = cls._nearest_free_area(labels, row, column, radius=0)
        if pixel is None:
            raise RuntimeError("no free pixel remains for a tracking marker")
        labels[pixel] = object_id
        return 1

    @staticmethod
    def _nearest_free_area(
        labels: np.ndarray,
        target_row: int,
        target_column: int,
        *,
        radius: int,
    ) -> tuple[int, int] | None:
        height, width = labels.shape
        if height < 2 * radius + 1 or width < 2 * radius + 1:
            return None

        minimum_row = minimum_column = radius
        maximum_row = height - radius - 1
        maximum_column = width - radius - 1
        start = (
            min(max(target_row, minimum_row), maximum_row),
            min(max(target_column, minimum_column), maximum_column),
        )
        queue = [
            (
                (start[0] - target_row) ** 2
                + (start[1] - target_column) ** 2,
                start[0],
                start[1],
            )
        ]
        visited = {start}

        while queue:
            _, row, column = heappop(queue)
            area = labels[
                row - radius : row + radius + 1,
                column - radius : column + radius + 1,
            ]
            if not np.any(area):
                return row, column

            for next_row, next_column in (
                (row - 1, column),
                (row, column - 1),
                (row, column + 1),
                (row + 1, column),
            ):
                candidate = (next_row, next_column)
                if (
                    minimum_row <= next_row <= maximum_row
                    and minimum_column <= next_column <= maximum_column
                    and candidate not in visited
                ):
                    visited.add(candidate)
                    heappush(
                        queue,
                        (
                            (next_row - target_row) ** 2
                            + (next_column - target_column) ** 2,
                            next_row,
                            next_column,
                        ),
                    )

        return None

    @staticmethod
    def _add_blob(
        image: np.ndarray,
        x: float,
        y: float,
        angle: float,
        amplitude: float,
    ) -> None:
        radius = int(
            np.ceil(3.2 * max(_BLOB_LONG_AXIS_PX, _BLOB_SHORT_AXIS_PX))
        ) + 1
        height, width = image.shape
        x_start, x_stop = max(0, int(x) - radius), min(
            width, int(x) + radius + 1
        )
        y_start, y_stop = max(0, int(y) - radius), min(
            height, int(y) + radius + 1
        )
        if x_start >= x_stop or y_start >= y_stop:
            return

        yy, xx = np.mgrid[y_start:y_stop, x_start:x_stop]
        dx, dy = xx - x, yy - y
        cosine, sine = np.cos(-angle), np.sin(-angle)
        longitudinal = dx * cosine - dy * sine
        transverse = dx * sine + dy * cosine
        image[y_start:y_stop, x_start:x_stop] += amplitude * np.exp(
            -0.5
            * (
                (longitudinal / _BLOB_LONG_AXIS_PX) ** 2
                + (transverse / _BLOB_SHORT_AXIS_PX) ** 2
            )
        )


__all__ = [
    "DEFAULT_SYNTHETIC_DATA_ROOT",
    "QUANTUM_DEMO_DATA_CONFIG",
    "SyntheticDataConfig",
    "SyntheticDataGenerator",
    "SyntheticDataset",
]
