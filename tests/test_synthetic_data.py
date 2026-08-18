"""Check deterministic synthetic sequence generation and its CTC file layout."""

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

    dataset = SyntheticDataGenerator(_small_config()).generate(tmp_path)
    with pytest.raises(ValueError, match="frame"):
        dataset.load_frame(-1)
    with pytest.raises(ValueError, match="frame"):
        dataset.load_tracking_labels(dataset.config.frame_count)
