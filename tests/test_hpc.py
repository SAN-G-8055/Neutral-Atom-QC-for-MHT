"""Exercise the readable HPC stages and the state-safe solver round trip."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neutral_atom_mht.classical_solver import ClassicalSolver
from neutral_atom_mht.detection import DetectionConfig
from neutral_atom_mht.filtering import FilterConfig
from neutral_atom_mht.gating import GateConfig
from neutral_atom_mht.graph import GraphCluster
from neutral_atom_mht.hpc import (
    HPC,
    HPCConfig,
    ObservedFrame,
    PreparedFrame,
    SequenceResult,
    hpc,
)
from neutral_atom_mht.likelihood import BayesianConfig
from neutral_atom_mht.models import Observation
from neutral_atom_mht.neutral_atom import QuantumSolver


def configured_hpc() -> HPC:
    """Create a permissive tracker suitable for small deterministic examples."""

    return HPC(
        HPCConfig(
            seconds_per_frame=1.0,
            initial_velocity_std=2.0,
            observation_variance_px2=1.5,
            detection=DetectionConfig(
                min_seed_area_px=12,
                max_seed_area_px=500,
                min_detection_area_px=20,
                max_detection_area_px=800,
            ),
            filtering=FilterConfig(
                acceleration_std=0.1,
                minimum_posterior=1e-4,
                maximum_misses=2,
            ),
            gating=GateConfig(mahalanobis_sq=20.0),
            bayesian=BayesianConfig(clutter_spatial_density=1e-4),
        ),
        sequence="synthetic",
    )


def cell_image(*, centre_x: float = 34.0, centre_y: float = 30.0) -> np.ndarray:
    """Return one smooth synthetic cell on a weak sloping background."""

    y, x = np.mgrid[:64, :72]
    image = 60.0 + 0.05 * x + 0.03 * y
    image += 155.0 * np.exp(
        -(((x - centre_x) / 6.0) ** 2 + ((y - centre_y) / 5.0) ** 2) / 2.0
    )
    return np.clip(image, 0, 255).astype(np.uint8)


class AlternateClassicalSolver(ClassicalSolver):
    """Use the exact algorithm under another name for a fair input comparison."""

    @property
    def solver_name(self) -> str:
        return "classical_exact_copy"


def test_public_class_name_and_lowercase_requested_alias_are_both_available() -> None:
    assert hpc is HPC


def test_observe_keeps_detection_diagnostics_beside_tracking_observations() -> None:
    tracker = configured_hpc()

    observed = tracker.observe(cell_image(), frame=4)

    assert isinstance(observed, ObservedFrame)
    assert observed.sequence == "synthetic"
    assert observed.frame == 4
    assert observed.detection.diagnostics.detection_count == 1
    assert len(observed.observations) == 1
    observation = observed.observations[0]
    detection = observed.detection.detections[0]
    assert observation.position == (detection.x_px, detection.y_px)
    assert observation.covariance == ((1.5, 0.0), (0.0, 1.5))


def test_preparation_calls_the_same_public_stages_a_user_can_inspect(
    tmp_path: Path,
) -> None:
    tracker = configured_hpc()
    solver = ClassicalSolver()
    tracker.step_observations(
        (Observation(frame=0, observation_id=1, x=10.0, y=20.0),),
        solver,
        frame=0,
    )
    observations = (
        Observation(frame=1, observation_id=1, x=10.2, y=20.0),
        Observation(frame=1, observation_id=2, x=11.0, y=20.0),
    )

    predicted = tracker.predict(frame=1)
    gated = tracker.gate(predicted, observations)
    calculated = tracker.calculate_weights(predicted, gated)
    hypotheses = tracker.filter_hypotheses(calculated)
    graph = tracker.encode_graph(hypotheses)
    clusters = tracker.cluster(graph)
    prepared = tracker.prepare_observations(observations, frame=1)

    assert isinstance(prepared, PreparedFrame)
    assert prepared.predicted_tracks == predicted
    assert prepared.gated_associations == gated
    assert prepared.hypotheses == hypotheses
    assert prepared.graph == graph
    assert prepared.clusters == clusters

    altered_graph = replace(
        graph,
        nodes=(
            replace(graph.nodes[0], weight=graph.nodes[0].weight + 1.0),
            *graph.nodes[1:],
        ),
    )
    with pytest.raises(ValueError, match="exactly encode"):
        replace(prepared, graph=altered_graph)

    split_clusters = tuple(
        GraphCluster(index, (node_id,))
        for index, node_id in enumerate(graph.node_ids)
    )
    with pytest.raises(ValueError, match="connected components"):
        replace(prepared, clusters=split_clusters)

    assert set(tracker.graph_embedding(graph)) == set(graph.node_ids)
    figure = tracker.visualize_graph(graph, tmp_path / "graph.png")
    assert figure.is_file() and figure.stat().st_size > 0


def test_prepare_frame_retains_the_interpretable_image_boundary() -> None:
    tracker = configured_hpc()

    prepared = tracker.prepare_frame(cell_image(), frame=0)

    assert prepared.observed_frame is not None
    assert prepared.observed_frame.observations == prepared.observations
    assert prepared.observed_frame.detection.labels.shape == (64, 72)


def test_solver_result_is_read_only_until_bayesian_update_and_advance() -> None:
    tracker = configured_hpc()
    solver = ClassicalSolver()
    tracker.step_observations(
        (Observation(frame=0, observation_id=1, x=0.0, y=0.0),),
        solver,
        frame=0,
    )
    prepared = tracker.prepare_observations(
        (
            Observation(frame=1, observation_id=1, x=0.1, y=0.0),
            Observation(frame=1, observation_id=2, x=2.0, y=0.0),
        ),
        frame=1,
    )
    before = tracker.tracks

    run = tracker.solve(prepared, solver)
    updated, assigned = tracker.bayesian_update(prepared, run)
    filtered = tracker.filter_tracks(updated)

    assert tracker.tracks == before
    assert run.successful
    assert assigned == frozenset({1})
    assert filtered[0].hits == 2

    result = tracker.advance(prepared, run)
    assert result.assigned_observation_ids == (1,)
    assert result.tracks == tracker.tracks
    assert len(result.tracks) == 2
    assert all(not hasattr(track, "family") for track in result.tracks)

    with pytest.raises(ValueError, match="stale"):
        tracker.advance(prepared, run)


def test_compare_uses_identical_inputs_and_never_selects_state_for_the_user() -> None:
    tracker = configured_hpc()
    solver = ClassicalSolver()
    tracker.step_observations(
        (Observation(frame=0, observation_id=1, x=0.0, y=0.0),),
        solver,
        frame=0,
    )
    prepared = tracker.prepare_observations(
        (Observation(frame=1, observation_id=1, x=0.25, y=0.0),),
        frame=1,
    )
    before = tracker.tracks

    comparison = tracker.compare(
        prepared,
        (solver, AlternateClassicalSolver()),
    )

    assert tracker.tracks == before
    assert comparison.input_fingerprints
    assert comparison.run("classical_exact").input_fingerprints == (
        comparison.input_fingerprints
    )
    assert comparison.run("classical_exact_copy").input_fingerprints == (
        comparison.input_fingerprints
    )
    assert len(comparison.rows()) == 2


def test_configuration_change_invalidates_an_old_prepared_frame() -> None:
    tracker = configured_hpc()
    solver = ClassicalSolver()
    tracker.step_observations(
        (Observation(frame=0, observation_id=1, x=0.0, y=0.0),),
        solver,
        frame=0,
    )
    prepared = tracker.prepare_observations(
        (Observation(frame=1, observation_id=1, x=0.1, y=0.0),),
        frame=1,
    )
    run = tracker.solve(prepared, solver)
    tracker.config = replace(
        tracker.config,
        bayesian=BayesianConfig(detection_probability=0.75),
    )

    with pytest.raises(ValueError, match="stale"):
        tracker.advance(prepared, run)


def test_unimplemented_neutral_atom_run_cannot_change_tracking_state() -> None:
    tracker = configured_hpc()
    classical = ClassicalSolver()
    tracker.step_observations(
        (Observation(frame=0, observation_id=1, x=0.0, y=0.0),),
        classical,
        frame=0,
    )
    prepared = tracker.prepare_observations(
        (Observation(frame=1, observation_id=1, x=0.1, y=0.0),),
        frame=1,
    )
    before = tracker.tracks

    neutral_atom_run = tracker.solve(prepared, QuantumSolver())

    assert not neutral_atom_run.successful
    assert neutral_atom_run.results[0].status == "not_implemented"
    with pytest.raises(ValueError, match="cannot advance"):
        tracker.advance(prepared, neutral_atom_run)
    assert tracker.tracks == before


def test_run_sequence_consumes_an_image_stream_once_and_returns_frozen_history() -> None:
    tracker = configured_hpc()
    consumed: list[int] = []

    def images():
        for frame in range(3):
            consumed.append(frame)
            yield cell_image(centre_x=34.0 + 0.4 * frame)

    result = tracker.run_sequence(images(), ClassicalSolver(), start_frame=0)

    assert isinstance(result, SequenceResult)
    assert consumed == [0, 1, 2]
    assert tuple(step.frame for step in result.steps) == (0, 1, 2)
    assert result.final_tracks == tracker.tracks == result.steps[-1].tracks
    assert result.solver_name == "classical_exact"
    assert tracker.last_frame == 2
    assert len(result.final_tracks) == 1
    assert result.final_tracks[0].hits == 3


def test_run_sequence_continues_after_the_last_applied_frame() -> None:
    tracker = configured_hpc()
    tracker.step(cell_image(), ClassicalSolver(), frame=3)

    result = tracker.run_sequence(
        (cell_image(centre_x=34.4), cell_image(centre_x=34.8)),
        ClassicalSolver(),
    )

    assert tuple(step.frame for step in result.steps) == (4, 5)
    assert tracker.last_frame == 5


def test_explicit_frame_mapping_is_sorted_and_cannot_mix_with_start_frame() -> None:
    tracker = configured_hpc()
    frames = {1: cell_image(centre_x=34.4), 0: cell_image()}

    result = tracker.run_sequence(frames, ClassicalSolver())

    assert tuple(step.frame for step in result.steps) == (0, 1)
    with pytest.raises(ValueError, match="start_frame"):
        HPC(configured_hpc().config).run_sequence(
            frames,
            ClassicalSolver(),
            start_frame=0,
        )


@pytest.mark.parametrize("frame", [2, 1])
def test_frames_must_advance_strictly_after_a_step(frame: int) -> None:
    tracker = configured_hpc()
    tracker.step_observations(
        (Observation(frame=2, observation_id=1, x=0.0, y=0.0),),
        ClassicalSolver(),
        frame=2,
    )

    with pytest.raises(ValueError, match="follow the last applied frame"):
        tracker.prepare_observations((), frame=frame)
