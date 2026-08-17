"""Stateful interface that gives both backends the same frozen problem.

Every frame executes the stages explicitly and in one place:

``predict -> gate -> likelihood -> filter candidates -> encode -> cluster -> solve``

Only after a caller chooses one backend run does the interface apply the shared
Bayesian/Kalman update and track filter.  Running a comparison is therefore
read-only: classical and quantum solvers see byte-identical input fingerprints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite

import numpy as np

from neutral_atom_mht.backends.base import (
    SolverBackend,
    SolverInput,
    SolverResult,
    validate_result,
)
from neutral_atom_mht.graph import ConflictGraph, GraphCluster, cluster_graph, encode_conflict_graph

from .filtering import (
    FilterConfig,
    filter_association_hypotheses,
    filter_tracks,
    predict_tracks,
)
from .gating import GateConfig, gate_observations
from .likelihood import (
    BayesianConfig,
    apply_bayesian_updates,
    calculate_association_hypotheses,
    log_odds_to_probability,
    probability_to_log_odds,
)
from .models import AssociationHypothesis, GatedAssociation, Observation, TrackState


STAGE_ORDER = (
    "predict",
    "gate",
    "calculate_likelihoods",
    "filter_candidates",
    "encode_graph",
    "cluster_graph",
    "solve",
    "bayesian_update",
    "filter_tracks",
)
SUCCESS_STATUSES = frozenset({"optimal", "simulated"})


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    seconds_per_frame: float = 1.0
    initial_velocity_std: float = 10.0
    minimum_hypothesis_weight: float = 0.0
    filtering: FilterConfig = field(default_factory=FilterConfig)
    gating: GateConfig = field(default_factory=GateConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.filtering, FilterConfig):
            raise TypeError("filtering must be a FilterConfig")
        if not isinstance(self.gating, GateConfig):
            raise TypeError("gating must be a GateConfig")
        if not isinstance(self.bayesian, BayesianConfig):
            raise TypeError("bayesian must be a BayesianConfig")
        seconds = float(self.seconds_per_frame)
        velocity_std = float(self.initial_velocity_std)
        minimum_weight = float(self.minimum_hypothesis_weight)
        if not isfinite(seconds) or seconds <= 0.0:
            raise ValueError("seconds_per_frame must be finite and positive")
        if not isfinite(velocity_std) or velocity_std <= 0.0:
            raise ValueError("initial_velocity_std must be finite and positive")
        if not isfinite(minimum_weight) or minimum_weight < 0.0:
            raise ValueError("minimum_hypothesis_weight must be finite and non-negative")
        object.__setattr__(self, "seconds_per_frame", seconds)
        object.__setattr__(self, "initial_velocity_std", velocity_std)
        object.__setattr__(self, "minimum_hypothesis_weight", minimum_weight)


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """Immutable classical preprocessing output, ready for either solver."""

    frame: int
    observations: tuple[Observation, ...]
    predicted_tracks: tuple[TrackState, ...]
    gated_associations: tuple[GatedAssociation, ...]
    hypotheses: tuple[AssociationHypothesis, ...]
    graph: ConflictGraph
    clusters: tuple[GraphCluster, ...]
    source_state_fingerprint: str
    stage_order: tuple[str, ...] = STAGE_ORDER[:6]

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise ValueError("frame must be a non-negative integer")
        if any(item.frame != self.frame for item in self.observations):
            raise ValueError("prepared observations must share the frame")
        if any(item.frame != self.frame for item in self.predicted_tracks):
            raise ValueError("prepared tracks must share the frame")
        if any(item.frame != self.frame for item in self.gated_associations):
            raise ValueError("prepared gates must share the frame")
        if any(item.frame != self.frame for item in self.hypotheses):
            raise ValueError("prepared hypotheses must share the frame")
        if encode_conflict_graph(self.hypotheses) != self.graph:
            raise ValueError("prepared graph must encode exactly the supplied hypotheses")
        if cluster_graph(self.graph) != self.clusters:
            raise ValueError("prepared clusters must be the graph connected components")
        if self.stage_order != STAGE_ORDER[:6]:
            raise ValueError("prepared stage_order is fixed by the public contract")
        if (
            len(self.source_state_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.source_state_fingerprint)
        ):
            raise ValueError("source_state_fingerprint must be a SHA-256 hex digest")

    def solver_inputs(self) -> tuple[SolverInput, ...]:
        return tuple(
            SolverInput(
                problem_id=f"frame-{self.frame:04d}-cluster-{cluster.cluster_id:03d}",
                frame=self.frame,
                graph=self.graph,
                cluster=cluster,
            )
            for cluster in self.clusters
        )


@dataclass(frozen=True, slots=True)
class BackendRun:
    backend: str
    results: tuple[SolverResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")
        if any(result.backend != self.backend for result in self.results):
            raise ValueError("every result backend must match the run backend")
        problem_ids = [result.problem_id for result in self.results]
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("a backend run cannot repeat a problem_id")

    @property
    def input_fingerprints(self) -> tuple[str, ...]:
        return tuple(result.input_fingerprint for result in self.results)

    @property
    def selected_ids(self) -> tuple[int, ...]:
        return tuple(sorted(node_id for result in self.results for node_id in result.selected_ids))


@dataclass(frozen=True, slots=True)
class BackendComparison:
    frame: int
    input_fingerprints: tuple[str, ...]
    runs: tuple[BackendRun, ...]

    def run(self, backend: str) -> BackendRun:
        matches = [run for run in self.runs if run.backend == backend]
        if len(matches) != 1:
            raise KeyError(backend)
        return matches[0]

    def rows(self) -> tuple[dict[str, object], ...]:
        """Return a notebook-friendly table with identical common columns."""

        return tuple(
            result.to_dict()
            for run in self.runs
            for result in run.results
        )


@dataclass(frozen=True, slots=True)
class TrackingStepResult:
    frame: int
    tracks: tuple[TrackState, ...]
    assigned_observation_ids: tuple[int, ...]
    backend_run: BackendRun
    stage_order: tuple[str, ...] = STAGE_ORDER


class TrackingInterface:
    """One-track-per-object interface with interchangeable solver backends."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._tracks: tuple[TrackState, ...] = ()
        self._next_track_id = 1

    @property
    def tracks(self) -> tuple[TrackState, ...]:
        return self._tracks

    def prepare(
        self,
        frame: int,
        observations: tuple[Observation, ...],
    ) -> PreparedStep:
        """Run all shared preprocessing without mutating tracker state."""

        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("frame must be a non-negative integer")
        observations = tuple(sorted(observations, key=lambda item: item.observation_id))
        if any(item.frame != frame for item in observations):
            raise ValueError("every observation must belong to the requested frame")
        observation_ids = [item.observation_id for item in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation IDs must be unique within a frame")
        if self._tracks:
            predicted = predict_tracks(
                self._tracks,
                frame=frame,
                seconds_per_frame=self.config.seconds_per_frame,
                config=self.config.filtering,
            )
        else:
            predicted = ()
        gated = gate_observations(predicted, observations, self.config.gating)
        calculated = calculate_association_hypotheses(
            predicted, gated, self.config.bayesian
        )
        hypotheses = filter_association_hypotheses(
            calculated,
            minimum_weight=self.config.minimum_hypothesis_weight,
        )
        graph = encode_conflict_graph(hypotheses)
        clusters = cluster_graph(graph)
        return PreparedStep(
            frame=frame,
            observations=observations,
            predicted_tracks=predicted,
            gated_associations=gated,
            hypotheses=hypotheses,
            graph=graph,
            clusters=clusters,
            source_state_fingerprint=self._state_fingerprint(),
        )

    def solve_prepared(
        self,
        prepared: PreparedStep,
        backend: SolverBackend,
    ) -> BackendRun:
        """Send each frozen cluster to one backend without updating tracks."""

        results: list[SolverResult] = []
        for solver_input in prepared.solver_inputs():
            result = backend.solve(solver_input)
            validate_result(solver_input, result)
            if result.status not in SUCCESS_STATUSES and result.selected_ids:
                raise ValueError("an unsuccessful backend result cannot select hypotheses")
            results.append(result)
        return BackendRun(backend=backend.name, results=tuple(results))

    def compare_prepared(
        self,
        prepared: PreparedStep,
        backends: tuple[SolverBackend, ...],
    ) -> BackendComparison:
        """Run multiple containers on the exact same immutable inputs."""

        if len(backends) < 2:
            raise ValueError("comparison requires at least two backends")
        names = [backend.name for backend in backends]
        if len(names) != len(set(names)):
            raise ValueError("backend names must be unique")
        expected = tuple(item.fingerprint for item in prepared.solver_inputs())
        runs = tuple(self.solve_prepared(prepared, backend) for backend in backends)
        if any(run.input_fingerprints != expected for run in runs):
            raise RuntimeError("backends did not receive identical solver inputs")
        return BackendComparison(
            frame=prepared.frame,
            input_fingerprints=expected,
            runs=runs,
        )

    def advance(
        self,
        prepared: PreparedStep,
        backend_run: BackendRun,
    ) -> TrackingStepResult:
        """Advance state with one explicitly chosen backend result."""

        solver_inputs = prepared.solver_inputs()
        if prepared.source_state_fingerprint != self._state_fingerprint():
            raise ValueError("prepared step is stale relative to current tracker state")
        if len(backend_run.results) != len(solver_inputs):
            raise ValueError("backend run does not cover every prepared cluster")
        for solver_input, result in zip(solver_inputs, backend_run.results, strict=True):
            if result.status not in SUCCESS_STATUSES:
                raise ValueError(
                    f"cannot advance from backend status {result.status!r}; "
                    "choose a successful backend"
                )
            validate_result(solver_input, result)
        updated, assigned = apply_bayesian_updates(
            prepared.predicted_tracks,
            prepared.observations,
            prepared.hypotheses,
            backend_run.selected_ids,
            self.config.bayesian,
        )
        retained = list(filter_tracks(updated, self.config.filtering))
        for observation in prepared.observations:
            if observation.observation_id not in assigned:
                retained.append(self._initialize_track(observation))
        self._tracks = filter_tracks(tuple(retained), self.config.filtering)
        return TrackingStepResult(
            frame=prepared.frame,
            tracks=self._tracks,
            assigned_observation_ids=tuple(sorted(assigned)),
            backend_run=backend_run,
        )

    def step(
        self,
        frame: int,
        observations: tuple[Observation, ...],
        backend: SolverBackend,
    ) -> TrackingStepResult:
        prepared = self.prepare(frame, observations)
        run = self.solve_prepared(prepared, backend)
        return self.advance(prepared, run)

    def _initialize_track(self, observation: Observation) -> TrackState:
        position_covariance = np.asarray(observation.covariance)
        covariance = np.zeros((4, 4), dtype=float)
        covariance[:2, :2] = position_covariance
        covariance[2:, 2:] = self.config.initial_velocity_std**2 * np.eye(2)
        log_odds = probability_to_log_odds(
            self.config.bayesian.initial_existence_probability
        )
        track = TrackState(
            track_id=self._next_track_id,
            frame=observation.frame,
            state=(observation.x, observation.y, 0.0, 0.0),
            covariance=tuple(map(tuple, covariance)),
            log_odds=log_odds,
            posterior_probability=log_odds_to_probability(log_odds),
            hits=1,
            misses=0,
            observation_history=((observation.frame, observation.observation_id),),
        )
        self._next_track_id += 1
        return track

    def _state_fingerprint(self) -> str:
        payload = {
            "config": asdict(self.config),
            "next_track_id": self._next_track_id,
            "tracks": [asdict(track) for track in self._tracks],
        }
        return sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
