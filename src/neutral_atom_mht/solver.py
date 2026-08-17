"""Define the one input/output language spoken by every solver.

The tracking controller produces a weighted conflict-graph problem and should
not need to know how that problem is solved.  This module provides immutable,
JSON-safe records for that hand-off and a template :class:`Solver` class that
all concrete solvers inherit.  The template measures runtime, computes the
objective from the original weights, and validates feasibility so a solver
cannot quietly change the problem or return a differently shaped result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import fsum, isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping, final

from .graph import ConflictGraph, GraphCluster, GraphNode


SCHEMA_VERSION = "2.0"
SUCCESS_STATUSES = frozenset({"optimal", "completed"})


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _freeze_json(value: Any, path: str = "diagnostics") -> Any:
    """Validate and recursively freeze a JSON-compatible value."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} mapping keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_json(item, f"{path}.{key}")
                for key, item in sorted(value.items())
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} cannot contain non-finite numbers")
        return value
    raise ValueError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    """Copy frozen JSON data into ordinary containers for serialization."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _normalized_ids(values: Iterable[int], name: str) -> tuple[int, ...]:
    ids = tuple(values)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids):
        raise ValueError(f"{name} must contain non-negative integers")
    normalized = tuple(sorted(ids))
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class SolverInput:
    """One canonical graph component handed unchanged to a solver.

    A full frame may contain several disconnected graph components.  Each
    component becomes one ``SolverInput`` so it can be solved independently.
    The fingerprint covers every meaningful field and lets comparisons prove
    that different solvers received byte-identical logical problems.
    """

    problem_id: str
    frame: int
    graph: ConflictGraph
    cluster: GraphCluster
    fingerprint: str = ""

    def __post_init__(self) -> None:
        _non_empty_string(self.problem_id, "problem_id")
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise ValueError("frame must be a non-negative integer")
        if not isinstance(self.graph, ConflictGraph):
            raise TypeError("graph must be a ConflictGraph")
        if not isinstance(self.cluster, GraphCluster):
            raise TypeError("cluster must be a GraphCluster")
        graph_ids = set(self.graph.node_ids)
        cluster_ids = set(self.cluster.node_ids)
        if not cluster_ids or not cluster_ids <= graph_ids:
            raise ValueError("cluster must contain known graph nodes")

        canonical = self.to_dict(include_fingerprint=False)
        expected = sha256(
            json.dumps(
                canonical,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError("solver-input fingerprint does not match its contents")
        object.__setattr__(self, "fingerprint", expected)

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        """Return the component nodes in canonical node-ID order."""

        allowed = set(self.cluster.node_ids)
        return tuple(node for node in self.graph.nodes if node.node_id in allowed)

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return only graph edges internal to this component."""

        allowed = set(self.cluster.node_ids)
        return tuple(
            edge
            for edge in self.graph.edges
            if edge[0] in allowed and edge[1] in allowed
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        """Return the canonical JSON-safe representation used for hashing."""

        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "problem_id": self.problem_id,
            "frame": self.frame,
            "cluster_id": self.cluster.cluster_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class SolverSelection:
    """The small decision a concrete solver returns to the template.

    Concrete solvers choose node IDs and describe how that choice was made.
    They do not calculate their own objective or construct a ``SolverResult``;
    those shared responsibilities remain in :class:`Solver`.
    """

    selected_ids: tuple[int, ...] = ()
    status: str = "completed"
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_ids",
            _normalized_ids(self.selected_ids, "selected_ids"),
        )
        _non_empty_string(self.status, "status")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", _freeze_json(self.diagnostics))


@dataclass(frozen=True, slots=True)
class SolverResult:
    """The common immutable result returned by classical and future solvers."""

    problem_id: str
    input_fingerprint: str
    solver_name: str
    selected_ids: tuple[int, ...]
    objective: float
    feasible: bool
    status: str
    runtime_seconds: float
    diagnostics: Mapping[str, object]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_string(self.problem_id, "problem_id")
        _non_empty_string(self.input_fingerprint, "input_fingerprint")
        _non_empty_string(self.solver_name, "solver_name")
        _non_empty_string(self.status, "status")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        object.__setattr__(
            self,
            "selected_ids",
            _normalized_ids(self.selected_ids, "selected_ids"),
        )
        objective = float(self.objective)
        runtime = float(self.runtime_seconds)
        if not isfinite(objective):
            raise ValueError("objective must be finite")
        if not isfinite(runtime) or runtime < 0.0:
            raise ValueError("runtime_seconds must be finite and non-negative")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "runtime_seconds", runtime)
        if not isinstance(self.feasible, bool):
            raise ValueError("feasible must be boolean")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", _freeze_json(self.diagnostics))

    @property
    def successful(self) -> bool:
        """Whether the status permits the tracking controller to advance."""

        return self.status in SUCCESS_STATUSES

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe representation for tables or storage."""

        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "input_fingerprint": self.input_fingerprint,
            "solver_name": self.solver_name,
            "selected_ids": list(self.selected_ids),
            "objective": self.objective,
            "feasible": self.feasible,
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
            "diagnostics": _thaw_json(self.diagnostics),
        }


def validate_result(solver_input: SolverInput, result: SolverResult) -> None:
    """Check a result against the exact immutable problem originally supplied."""

    if not isinstance(solver_input, SolverInput):
        raise TypeError("solver_input must be a SolverInput")
    if not isinstance(result, SolverResult):
        raise TypeError("result must be a SolverResult")
    if result.problem_id != solver_input.problem_id:
        raise ValueError("result problem_id does not match solver input")
    if result.input_fingerprint != solver_input.fingerprint:
        raise ValueError("result input_fingerprint does not match solver input")

    selected = set(result.selected_ids)
    if not selected <= set(solver_input.cluster.node_ids):
        raise ValueError("result selected unknown nodes")
    if any(left in selected and right in selected for left, right in solver_input.edges):
        raise ValueError("result is not an independent set")
    if not result.feasible:
        raise ValueError("a returned selection must be marked feasible")

    objective = fsum(
        node.weight for node in solver_input.nodes if node.node_id in selected
    )
    if result.objective != objective:
        raise ValueError("result objective was not computed from original input weights")
    if not result.successful and result.selected_ids:
        raise ValueError("an unsuccessful result cannot select graph nodes")


@dataclass(frozen=True, slots=True)
class SolverRun:
    """All component results produced by one solver for one prepared frame."""

    solver_name: str
    results: tuple[SolverResult, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.solver_name, "solver_name")
        results = tuple(self.results)
        if any(not isinstance(result, SolverResult) for result in results):
            raise TypeError("results must contain only SolverResult instances")
        if any(result.solver_name != self.solver_name for result in results):
            raise ValueError("every result must match the run solver_name")
        problem_ids = tuple(result.problem_id for result in results)
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("a solver run cannot repeat a problem_id")
        object.__setattr__(self, "results", results)

    @property
    def input_fingerprints(self) -> tuple[str, ...]:
        return tuple(result.input_fingerprint for result in self.results)

    @property
    def selected_ids(self) -> tuple[int, ...]:
        ids = tuple(node_id for result in self.results for node_id in result.selected_ids)
        return tuple(sorted(ids))

    @property
    def successful(self) -> bool:
        return all(result.successful for result in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "solver_name": self.solver_name,
            "successful": self.successful,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class SolverComparison:
    """Read-only results from multiple solvers given the same inputs."""

    input_fingerprints: tuple[str, ...]
    runs: tuple[SolverRun, ...]

    def __post_init__(self) -> None:
        fingerprints = tuple(self.input_fingerprints)
        runs = tuple(self.runs)
        if len(runs) < 2:
            raise ValueError("a solver comparison requires at least two runs")
        if any(not isinstance(run, SolverRun) for run in runs):
            raise TypeError("runs must contain only SolverRun instances")
        names = tuple(run.solver_name for run in runs)
        if len(names) != len(set(names)):
            raise ValueError("comparison solver names must be unique")
        if any(run.input_fingerprints != fingerprints for run in runs):
            raise ValueError("comparison runs must use identical input fingerprints")
        object.__setattr__(self, "input_fingerprints", fingerprints)
        object.__setattr__(self, "runs", runs)

    @classmethod
    def from_runs(cls, runs: Iterable[SolverRun]) -> SolverComparison:
        """Construct a comparison and derive its expected fingerprints."""

        normalized = tuple(runs)
        fingerprints = normalized[0].input_fingerprints if normalized else ()
        return cls(input_fingerprints=fingerprints, runs=normalized)

    def run(self, solver_name: str) -> SolverRun:
        matches = tuple(run for run in self.runs if run.solver_name == solver_name)
        if len(matches) != 1:
            raise KeyError(solver_name)
        return matches[0]

    def rows(self) -> tuple[dict[str, object], ...]:
        """Return notebook-friendly rows with identical common columns."""

        return tuple(
            result.to_dict()
            for run in self.runs
            for result in run.results
        )


class Solver(ABC):
    """Template shared by every maximum-weight-independent-set solver.

    Subclasses implement only :meth:`_select`.  The public methods are final so
    runtime measurement, objective calculation, validation, and result fields
    remain identical regardless of the implementation used to choose nodes.
    """

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """Return the stable name written into every result."""

    @property
    def name(self) -> str:
        """Short convenience alias used in diagrams and interactive work."""

        return self.solver_name

    @final
    def solve(self, solver_input: SolverInput) -> SolverResult:
        """Solve one component and construct the validated common result."""

        if not isinstance(solver_input, SolverInput):
            raise TypeError("solver_input must be a SolverInput")
        solver_name = _non_empty_string(self.solver_name, "solver_name")
        started = perf_counter()
        selection = self._select(solver_input)
        runtime_seconds = perf_counter() - started
        if not isinstance(selection, SolverSelection):
            raise TypeError("_select() must return a SolverSelection")

        selected = set(selection.selected_ids)
        unknown = selected - set(solver_input.cluster.node_ids)
        if unknown:
            raise ValueError(f"solver selected unknown node IDs: {sorted(unknown)}")
        feasible = not any(
            left in selected and right in selected for left, right in solver_input.edges
        )
        objective = fsum(
            node.weight for node in solver_input.nodes if node.node_id in selected
        )
        result = SolverResult(
            problem_id=solver_input.problem_id,
            input_fingerprint=solver_input.fingerprint,
            solver_name=solver_name,
            selected_ids=selection.selected_ids,
            objective=objective,
            feasible=feasible,
            status=selection.status,
            runtime_seconds=runtime_seconds,
            diagnostics=selection.diagnostics,
        )
        validate_result(solver_input, result)
        return result

    @final
    def solve_all(self, solver_inputs: Iterable[SolverInput]) -> SolverRun:
        """Solve every component in order using the same result contract."""

        inputs = tuple(solver_inputs)
        if any(not isinstance(item, SolverInput) for item in inputs):
            raise TypeError("solver_inputs must contain only SolverInput instances")
        problem_ids = tuple(item.problem_id for item in inputs)
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("solver_inputs cannot repeat a problem_id")
        return SolverRun(
            solver_name=self.solver_name,
            results=tuple(self.solve(item) for item in inputs),
        )

    @abstractmethod
    def _select(self, solver_input: SolverInput) -> SolverSelection:
        """Choose an independent set without constructing the public result."""


def compare_solvers(
    solver_inputs: Iterable[SolverInput],
    solvers: Iterable[Solver],
) -> SolverComparison:
    """Run multiple solvers on the same immutable tuple of component inputs."""

    inputs = tuple(solver_inputs)
    selected_solvers = tuple(solvers)
    if len(selected_solvers) < 2:
        raise ValueError("comparison requires at least two solvers")
    if any(not isinstance(solver, Solver) for solver in selected_solvers):
        raise TypeError("solvers must contain only Solver instances")
    names = tuple(solver.solver_name for solver in selected_solvers)
    if len(names) != len(set(names)):
        raise ValueError("comparison solver names must be unique")
    return SolverComparison.from_runs(
        solver.solve_all(inputs) for solver in selected_solvers
    )


__all__ = [
    "SCHEMA_VERSION",
    "SUCCESS_STATUSES",
    "Solver",
    "SolverComparison",
    "SolverInput",
    "SolverResult",
    "SolverRun",
    "SolverSelection",
    "compare_solvers",
    "validate_result",
]
