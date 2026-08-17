from __future__ import annotations

from dataclasses import replace
import importlib.util

import pytest

from neutral_atom_mht.backends.base import SolverInput, SolverResult
from neutral_atom_mht.backends.classical import ClassicalBackend
from neutral_atom_mht.backends.neutral_atom import NeutralAtomBackend
from neutral_atom_mht.tracking.filtering import FilterConfig
from neutral_atom_mht.tracking.gating import GateConfig
from neutral_atom_mht.tracking.interface import STAGE_ORDER, TrackingConfig, TrackingInterface
from neutral_atom_mht.tracking.likelihood import BayesianConfig
from neutral_atom_mht.tracking.models import Observation


class RecordingBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.inputs: list[SolverInput] = []
        self._delegate = ClassicalBackend()

    def solve(self, solver_input: SolverInput) -> SolverResult:
        self.inputs.append(solver_input)
        result = self._delegate.solve(solver_input)
        return replace(result, backend=self.name)


class InvalidFailureBackend:
    name = "invalid_failure"

    def solve(self, solver_input: SolverInput) -> SolverResult:
        return SolverResult(
            problem_id="wrong-problem",
            input_fingerprint=solver_input.fingerprint,
            backend=self.name,
            selected_ids=(),
            objective=999.0,
            feasible=True,
            status="unsupported_size",
            runtime_seconds=0.0,
            diagnostics={},
        )


def interface() -> TrackingInterface:
    return TrackingInterface(
        TrackingConfig(
            seconds_per_frame=1.0,
            initial_velocity_std=2.0,
            filtering=FilterConfig(
                acceleration_std=0.1,
                minimum_posterior=1e-4,
                maximum_misses=2,
            ),
            gating=GateConfig(mahalanobis_sq=20.0),
            bayesian=BayesianConfig(clutter_spatial_density=1e-4),
        )
    )


def test_first_frame_initializes_tracks_without_manufacturing_global_hypotheses() -> None:
    tracker = interface()
    result = tracker.step(
        0,
        (Observation(0, 1, 10.0, 20.0), Observation(0, 2, 50.0, 60.0)),
        ClassicalBackend(),
    )

    assert len(result.tracks) == 2
    assert result.backend_run.results == ()
    assert result.stage_order == STAGE_ORDER
    assert all(not hasattr(track, "family") for track in result.tracks)


def test_comparison_passes_identical_frozen_inputs_and_does_not_mutate_state() -> None:
    tracker = interface()
    tracker.step(0, (Observation(0, 1, 0.0, 0.0),), ClassicalBackend())
    original_tracks = tracker.tracks
    prepared = tracker.prepare(
        1,
        (Observation(1, 1, 0.5, 0.0), Observation(1, 2, 1.0, 0.0)),
    )
    classical = RecordingBackend("classical_test")
    quantum = RecordingBackend("qutip_test")

    comparison = tracker.compare_prepared(prepared, (classical, quantum))

    assert tracker.tracks == original_tracks
    assert comparison.input_fingerprints
    assert classical.inputs == quantum.inputs
    assert classical.inputs[0] is not quantum.inputs[0]
    assert classical.inputs[0].to_dict() == quantum.inputs[0].to_dict()
    assert set(comparison.rows()[0]) == set(comparison.rows()[1])


def test_only_explicitly_chosen_backend_advances_shared_bayesian_state() -> None:
    tracker = interface()
    tracker.step(0, (Observation(0, 1, 0.0, 0.0),), ClassicalBackend())
    prepared = tracker.prepare(
        1,
        (Observation(1, 1, 0.1, 0.0), Observation(1, 2, 2.0, 0.0)),
    )
    comparison = tracker.compare_prepared(
        prepared,
        (RecordingBackend("classical_test"), RecordingBackend("qutip_test")),
    )

    advanced = tracker.advance(prepared, comparison.run("classical_test"))

    assert advanced.assigned_observation_ids == (1,)
    assert len(advanced.tracks) == 2  # selected old track plus new unassigned observation
    old_track = next(track for track in advanced.tracks if track.track_id == 1)
    assert old_track.hits == 2
    assert old_track.observation_history[-1] == (1, 1)


def test_unsuccessful_container_result_cannot_mutate_tracker() -> None:
    tracker = interface()
    tracker.step(0, (Observation(0, 1, 0.0, 0.0),), ClassicalBackend())
    prepared = tracker.prepare(1, (Observation(1, 1, 0.2, 0.0),))
    run = tracker.solve_prepared(prepared, ClassicalBackend(maximum_nodes=0 + 1))
    # Replace a valid result with an explicit transparent failure.
    failed = replace(
        run.results[0],
        selected_ids=(),
        objective=0.0,
        status="embedding_error",
    )
    failed_run = replace(run, results=(failed,))

    before = tracker.tracks
    with pytest.raises(ValueError, match="cannot advance"):
        tracker.advance(prepared, failed_run)
    assert tracker.tracks == before


def test_unsuccessful_container_result_still_must_satisfy_common_contract() -> None:
    tracker = interface()
    tracker.step(0, (Observation(0, 1, 0.0, 0.0),), ClassicalBackend())
    prepared = tracker.prepare(1, (Observation(1, 1, 0.2, 0.0),))

    with pytest.raises(ValueError, match="problem_id"):
        tracker.solve_prepared(prepared, InvalidFailureBackend())


def test_prepared_step_cannot_be_applied_after_tracker_state_changes() -> None:
    tracker = interface()
    tracker.step(0, (Observation(0, 1, 0.0, 0.0),), ClassicalBackend())
    stale = tracker.prepare(1, (Observation(1, 1, 0.1, 0.0),))
    fresh = tracker.prepare(1, (Observation(1, 1, 0.2, 0.0),))
    tracker.advance(fresh, tracker.solve_prepared(fresh, ClassicalBackend()))

    with pytest.raises(ValueError, match="stale"):
        tracker.advance(stale, tracker.solve_prepared(stale, ClassicalBackend()))


def test_prepared_step_cannot_be_applied_after_config_changes() -> None:
    tracker = interface()
    tracker.step(0, (Observation(0, 1, 0.0, 0.0),), ClassicalBackend())
    prepared = tracker.prepare(1, (Observation(1, 1, 0.1, 0.0),))
    run = tracker.solve_prepared(prepared, ClassicalBackend())

    tracker.config = TrackingConfig(
        seconds_per_frame=1.0,
        gating=GateConfig(),
        bayesian=BayesianConfig(detection_probability=0.75),
    )

    with pytest.raises(ValueError, match="stale"):
        tracker.advance(prepared, run)


def test_long_run_of_hits_keeps_a_valid_finite_bayesian_state() -> None:
    tracker = interface()
    backend = ClassicalBackend()

    for frame in range(20):
        result = tracker.step(
            frame,
            (Observation(frame, 1, 0.0, 0.0),),
            backend,
        )

    assert len(result.tracks) == 1
    assert result.tracks[0].hits == 20
    assert 0.0 < result.tracks[0].posterior_probability < 1.0


@pytest.mark.quantum
@pytest.mark.skipif(importlib.util.find_spec("qutip") is None, reason="optional qutip is absent")
def test_real_classical_and_qutip_containers_share_one_preprocessed_problem() -> None:
    tracker = interface()
    tracker.step(0, (Observation(0, 1, 0.0, 0.0),), ClassicalBackend())
    prepared = tracker.prepare(
        1,
        (Observation(1, 1, 0.25, 0.0), Observation(1, 2, 1.25, 0.0)),
    )

    comparison = tracker.compare_prepared(
        prepared,
        (ClassicalBackend(), NeutralAtomBackend(maximum_simulation_atoms=3)),
    )

    classical = comparison.run("classical_exact").results[0]
    quantum = comparison.run("neutral_atom_qutip").results[0]
    assert classical.input_fingerprint == quantum.input_fingerprint
    assert classical.status == "optimal"
    assert quantum.status == "simulated"
    assert set(classical.to_dict()) == set(quantum.to_dict())
    assert quantum.feasible
