"""Check deterministic synthetic sequence generation and its CTC file layout."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from neutral_atom_mht import (
    ClassicalSolver,
    DEFAULT_SYNTHETIC_DATA_ROOT,
    HPC,
    HPCConfig,
    QUANTUM_DEMO_DATA_CONFIG,
    QuantumSolver,
    SyntheticDataConfig,
    SyntheticDataGenerator,
)


def _small_config(**changes: object) -> SyntheticDataConfig:
    values = {
        "noise": 0.25,
        "frame_count": 3,
        "object_count": 4,
        "seed": 17,
        "dataset_name": "TINY-MHT",
        "sequence": "01",
        "image_shape": (64, 80),
    }
    values.update(changes)
    return SyntheticDataConfig(**values)


def test_generator_writes_ctc_sequence_and_loads_tiffs(tmp_path: Path) -> None:
    config = _small_config()
    dataset = SyntheticDataGenerator(config).generate(tmp_path)

    assert dataset.root == tmp_path / config.dataset_name
    assert dataset.raw_directory == dataset.root / "01"
    assert dataset.tracking_directory == dataset.root / "01_GT" / "TRA"
    assert {path.name for path in dataset.raw_directory.iterdir()} == {
        "t000.tif",
        "t001.tif",
        "t002.tif",
    }
    assert {path.name for path in dataset.tracking_directory.iterdir()} == {
        "man_track000.tif",
        "man_track001.tif",
        "man_track002.tif",
        "man_track.txt",
    }

    image = dataset.load_frame(1)
    labels = dataset.load_tracking_labels(1)
    assert image.shape == config.image_shape
    assert labels.shape == config.image_shape
    assert image.dtype == np.uint8
    assert labels.dtype == np.uint16
    assert dataset.track_manifest_path.read_text(encoding="utf-8") == (
        "1 0 2 0\n2 0 2 0\n3 0 2 0\n4 0 2 0\n"
    )


def test_same_seed_produces_identical_images_and_labels(tmp_path: Path) -> None:
    config = _small_config(frame_count=2, object_count=3)
    first = SyntheticDataGenerator(config).generate(tmp_path / "first")
    second = SyntheticDataGenerator(config).generate(tmp_path / "second")

    for frame in range(config.frame_count):
        assert np.array_equal(first.load_frame(frame), second.load_frame(frame))
        assert np.array_equal(
            first.load_tracking_labels(frame),
            second.load_tracking_labels(frame),
        )


def test_default_configs_preserve_the_legacy_cached_byte_stream() -> None:
    digest = sha256()
    for image, labels in SyntheticDataGenerator(_small_config()).iter_frames():
        digest.update(image.tobytes())
        digest.update(labels.tobytes())

    assert digest.hexdigest() == (
        "7f1dd4f9b84593b12e2b91f4028e1edeea987ed5be71a4ccb75cad9887706878"
    )


def test_override_scenarios_keep_amplitude_draws_paired_across_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _small_config(
        noise=0.0,
        frame_count=4,
        object_count=8,
        speed_px_per_frame_override=2.5,
        clutter_per_frame_override=0.0,
        pixel_noise_sigma_override=0.0,
    )

    def captured_blobs(config: SyntheticDataConfig) -> tuple[tuple[float, ...], ...]:
        calls: list[tuple[float, ...]] = []

        def capture(
            image: np.ndarray,
            x: float,
            y: float,
            angle: float,
            amplitude: float,
        ) -> None:
            calls.append((x, y, angle, amplitude))

        monkeypatch.setattr(
            SyntheticDataGenerator,
            "_add_blob",
            staticmethod(capture),
        )
        tuple(
            SyntheticDataGenerator(config).iter_frames()
        )
        return tuple(calls)

    always_visible = replace(common, detection_probability_override=1.0)
    partly_visible = replace(common, detection_probability_override=0.5)
    all_blobs = captured_blobs(always_visible)
    retained_blobs = captured_blobs(partly_visible)

    assert len(all_blobs) == common.frame_count * common.object_count
    assert 0 < len(retained_blobs) < len(all_blobs)
    assert set(retained_blobs) < set(all_blobs)


def test_streamed_frames_match_the_written_dataset(tmp_path: Path) -> None:
    config = _small_config(frame_count=2)
    streamed = tuple(SyntheticDataGenerator(config).iter_frames())
    dataset = SyntheticDataGenerator(config).generate(tmp_path)

    assert len(streamed) == config.frame_count
    for frame, (image, labels) in enumerate(streamed):
        assert np.array_equal(image, dataset.load_frame(frame))
        assert np.array_equal(labels, dataset.load_tracking_labels(frame))


def test_difficulty_controls_can_be_varied_independently(tmp_path: Path) -> None:
    config = _small_config(
        noise=0.0,
        speed_px_per_frame_override=9.5,
        detection_probability_override=0.45,
        clutter_per_frame_override=7.0,
        pixel_noise_sigma_override=3.25,
    )

    assert config.speed_px_per_frame == 9.5
    assert config.detection_probability == 0.45
    assert config.clutter_per_frame == 7.0
    assert config.pixel_noise_sigma == 3.25
    assert len(tuple(SyntheticDataGenerator(config).iter_frames())) == 3


def test_quantum_demo_preset_is_small_versioned_and_nontrivial(
    tmp_path: Path,
) -> None:
    config = QUANTUM_DEMO_DATA_CONFIG

    assert config == SyntheticDataConfig(
        noise=0.1,
        frame_count=8,
        object_count=4,
        seed=0,
        dataset_name="SYN-MHT-QUANTUM-v1",
        sequence="01",
        image_shape=(256, 320),
    )

    dataset = SyntheticDataGenerator(config).generate(tmp_path)
    tracker = HPC(HPCConfig(), sequence=config.sequence)
    classical = ClassicalSolver(maximum_component_nodes=8)
    quantum = QuantumSolver(maximum_component_nodes=8)
    simulated_shapes: list[tuple[int, int]] = []
    maximum_component_size = 0

    for frame in range(config.frame_count):
        prepared = tracker.prepare_frame(dataset.load_frame(frame), frame=frame)
        components = quantum.prepare(prepared.solver_input())
        maximum_component_size = max(
            (
                maximum_component_size,
                *(len(component.node_ids) for component in components),
            )
        )
        simulated_shapes.extend(
            (len(component.node_ids), len(component.edges))
            for component in components
            if not quantum._is_clique(component)
        )
        tracker.advance(prepared, tracker.solve(prepared, classical))

    assert maximum_component_size == 5
    assert simulated_shapes == [(5, 5)]


def test_existing_dataset_is_not_partially_overwritten(tmp_path: Path) -> None:
    three_frames = _small_config(frame_count=3)
    original = SyntheticDataGenerator(three_frames).generate(tmp_path)
    original_bytes = {
        path.name: path.read_bytes() for path in original.raw_directory.iterdir()
    }

    one_frame = _small_config(frame_count=1)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        SyntheticDataGenerator(one_frame).generate(tmp_path)

    assert {
        path.name: path.read_bytes() for path in original.raw_directory.iterdir()
    } == original_bytes
    assert set(original_bytes) == {"t000.tif", "t001.tif", "t002.tif"}


def test_ground_truth_remains_present_at_the_noisiest_setting(
    tmp_path: Path,
) -> None:
    config = _small_config(noise=1.0, frame_count=5, object_count=1)
    dataset = SyntheticDataGenerator(config).generate(tmp_path)

    for frame in range(config.frame_count):
        labels = dataset.load_tracking_labels(frame)
        assert set(np.unique(labels)) == {0, 1}
        assert np.count_nonzero(labels == 1) == 9


def test_colliding_trajectories_keep_every_ground_truth_id(
    tmp_path: Path,
) -> None:
    config = _small_config(
        noise=1.0,
        frame_count=4,
        object_count=20,
        image_shape=(6, 6),
    )
    dataset = SyntheticDataGenerator(config).generate(tmp_path)
    expected_ids = set(range(1, config.object_count + 1))

    for frame in range(config.frame_count):
        labels = dataset.load_tracking_labels(frame)
        assert set(np.unique(labels)) - {0} == expected_ids


def test_configuration_and_frame_bounds_are_clear(tmp_path: Path) -> None:
    assert DEFAULT_SYNTHETIC_DATA_ROOT == Path("data") / "synthetic"
    with pytest.raises(ValueError, match="noise"):
        _small_config(noise=1.1)
    with pytest.raises(ValueError, match="directory name"):
        _small_config(dataset_name="../outside")
    with pytest.raises(ValueError, match="uint16"):
        _small_config(object_count=65_536, image_shape=(256, 256))
    with pytest.raises(ValueError, match="available image pixels"):
        _small_config(object_count=17, image_shape=(4, 4))
    with pytest.raises(ValueError, match="speed_px_per_frame_override"):
        _small_config(speed_px_per_frame_override=0.0)
    with pytest.raises(ValueError, match="detection_probability_override"):
        _small_config(detection_probability_override=1.1)
    with pytest.raises(ValueError, match="clutter_per_frame_override"):
        _small_config(clutter_per_frame_override=-1.0)
    with pytest.raises(ValueError, match="pixel_noise_sigma_override"):
        _small_config(pixel_noise_sigma_override=-1.0)

    dataset = SyntheticDataGenerator(_small_config()).generate(tmp_path)
    with pytest.raises(ValueError, match="frame"):
        dataset.load_frame(-1)
    with pytest.raises(ValueError, match="frame"):
        dataset.load_tracking_labels(dataset.config.frame_count)
