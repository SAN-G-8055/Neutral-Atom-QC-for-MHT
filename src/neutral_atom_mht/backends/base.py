"""One immutable input/output contract for classical and quantum solvers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum, isfinite
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from neutral_atom_mht.graph import ConflictGraph, GraphCluster


SCHEMA_VERSION = "1.0"


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
    """Return detached built-in containers suitable for JSON serialization."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SolverInput:
    """A frozen cluster handed unchanged to either backend."""

    problem_id: str
    frame: int
    graph: ConflictGraph
    cluster: GraphCluster
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise ValueError("frame must be a non-negative integer")
        graph_ids = set(self.graph.node_ids)
        cluster_ids = set(self.cluster.node_ids)
        if not cluster_ids or not cluster_ids <= graph_ids:
            raise ValueError("cluster must contain known graph nodes")
        canonical = self.to_dict(include_fingerprint=False)
        expected = sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError("solver-input fingerprint does not match its contents")
        object.__setattr__(self, "fingerprint", expected)

    @property
    def nodes(self) -> tuple[object, ...]:
        allowed = set(self.cluster.node_ids)
        return tuple(node for node in self.graph.nodes if node.node_id in allowed)

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        allowed = set(self.cluster.node_ids)
        return tuple(
            edge for edge in self.graph.edges if edge[0] in allowed and edge[1] in allowed
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "problem_id": self.problem_id,
            "frame": self.frame,
            "cluster_id": self.cluster.cluster_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "weight": node.weight,
                    "track_id": node.track_id,
                    "observation_id": node.observation_id,
                    "posterior_probability": node.posterior_probability,
                }
                for node in self.nodes
            ],
            "edges": [list(edge) for edge in self.edges],
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class SolverResult:
    """The common result schema returned by every backend."""

    problem_id: str
    input_fingerprint: str
    backend: str
    selected_ids: tuple[int, ...]
    objective: float
    feasible: bool
    status: str
    runtime_seconds: float
    diagnostics: Mapping[str, object]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("problem_id", "input_fingerprint", "backend", "status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        ids = tuple(self.selected_ids)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids):
            raise ValueError("selected_ids must contain non-negative integers")
        if len(ids) != len(set(ids)) or ids != tuple(sorted(ids)):
            raise ValueError("selected_ids must be unique and sorted")
        object.__setattr__(self, "selected_ids", ids)
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

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "input_fingerprint": self.input_fingerprint,
            "backend": self.backend,
            "selected_ids": list(self.selected_ids),
            "objective": self.objective,
            "feasible": self.feasible,
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
            "diagnostics": _thaw_json(self.diagnostics),
        }


def validate_result(solver_input: SolverInput, result: SolverResult) -> None:
    """Validate backend output against the original Bayesian problem."""

    if result.problem_id != solver_input.problem_id:
        raise ValueError("result problem_id does not match solver input")
    if result.input_fingerprint != solver_input.fingerprint:
        raise ValueError("result input_fingerprint does not match solver input")
    selected = set(result.selected_ids)
    if not selected <= set(solver_input.cluster.node_ids):
        raise ValueError("result selected unknown hypotheses")
    if any(left in selected and right in selected for left, right in solver_input.edges):
        raise ValueError("result is not an independent set")
    objective = fsum(
        node.weight for node in solver_input.nodes if node.node_id in selected
    )
    if not isfinite(result.objective) or result.objective != objective:
        raise ValueError("result objective was not computed from original input weights")
    if not result.feasible:
        raise ValueError("a selected independent set must be marked feasible")


@runtime_checkable
class SolverBackend(Protocol):
    """Structural interface implemented by both solver containers."""

    name: str

    def solve(self, solver_input: SolverInput) -> SolverResult:
        ...
