"""Document and format the future neutral-atom solver boundary.

There is intentionally no quantum simulation or hardware model in this file.
``QuantumSolver`` currently converts the shared weighted-graph input into a
small neutral-atom request record and returns an honest ``not_implemented``
result.  A future implementation can send :class:`NeutralAtomInput` to a
simulator or device, put the chosen graph-node IDs in
:class:`NeutralAtomOutput`, and pass that response through ``format_output``.

That future implementation belongs only in the protected selection hook.  It
must not change graph weights, repair edges, update tracks, or construct a
different result schema: the inherited :class:`~neutral_atom_mht.solver.Solver`
template performs those shared checks after the adapter returns its selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

from .solver import (
    SUCCESS_STATUSES,
    Solver,
    SolverInput,
    SolverSelection,
    _freeze_json,
    _non_empty_string,
    _normalized_ids,
    _thaw_json,
)


NEUTRAL_ATOM_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class NeutralAtomInput:
    """Portable weighted-graph request for a future manual implementation.

    ``nodes`` contains ``(node_id, weight)`` pairs.  ``edges`` contains the
    mutually exclusive node pairs.  These are the only optimization data a
    neutral-atom implementation may use; the echoed fingerprint binds the
    request to the controller's immutable ``SolverInput``.
    """

    problem_id: str
    input_fingerprint: str
    frame: int
    nodes: tuple[tuple[int, float], ...]
    edges: tuple[tuple[int, int], ...]
    schema_version: str = NEUTRAL_ATOM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_string(self.problem_id, "problem_id")
        _non_empty_string(self.input_fingerprint, "input_fingerprint")
        if self.schema_version != NEUTRAL_ATOM_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {NEUTRAL_ATOM_SCHEMA_VERSION}"
            )
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise ValueError("frame must be a non-negative integer")

        normalized_nodes: list[tuple[int, float]] = []
        for item in tuple(self.nodes):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("each neutral-atom node must contain ID and weight")
            node_id, weight = item
            if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
                raise ValueError("neutral-atom node IDs must be non-negative integers")
            numeric_weight = float(weight)
            if isinstance(weight, bool) or not isfinite(numeric_weight):
                raise ValueError("neutral-atom node weights must be finite numbers")
            normalized_nodes.append((node_id, numeric_weight))
        normalized_nodes.sort(key=lambda item: item[0])
        if len(normalized_nodes) != len({item[0] for item in normalized_nodes}):
            raise ValueError("neutral-atom node IDs must be unique")
        known = {item[0] for item in normalized_nodes}

        normalized_edges: list[tuple[int, int]] = []
        for edge in tuple(self.edges):
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ValueError("each neutral-atom edge must contain two node IDs")
            left, right = edge
            if any(isinstance(value, bool) or not isinstance(value, int) for value in edge):
                raise ValueError("neutral-atom edge endpoints must be integers")
            if left == right:
                raise ValueError("neutral-atom edges cannot be self-loops")
            if left not in known or right not in known:
                raise ValueError("neutral-atom edges must reference known node IDs")
            normalized_edges.append((min(left, right), max(left, right)))
        normalized_edges.sort()
        if len(normalized_edges) != len(set(normalized_edges)):
            raise ValueError("neutral-atom edges must be unique")

        object.__setattr__(self, "nodes", tuple(normalized_nodes))
        object.__setattr__(self, "edges", tuple(normalized_edges))

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(node_id for node_id, _ in self.nodes)

    def to_dict(self) -> dict[str, object]:
        """Return a transport-friendly request with explicit node fields."""

        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "input_fingerprint": self.input_fingerprint,
            "frame": self.frame,
            "nodes": [
                {"node_id": node_id, "weight": weight}
                for node_id, weight in self.nodes
            ],
            "edges": [list(edge) for edge in self.edges],
        }


@dataclass(frozen=True, slots=True)
class NeutralAtomOutput:
    """Minimal response a future neutral-atom implementation must provide."""

    problem_id: str
    input_fingerprint: str
    selected_ids: tuple[int, ...]
    status: str = "completed"
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = NEUTRAL_ATOM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_string(self.problem_id, "problem_id")
        _non_empty_string(self.input_fingerprint, "input_fingerprint")
        _non_empty_string(self.status, "status")
        if self.schema_version != NEUTRAL_ATOM_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {NEUTRAL_ATOM_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "selected_ids",
            _normalized_ids(self.selected_ids, "selected_ids"),
        )
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", _freeze_json(self.diagnostics))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "input_fingerprint": self.input_fingerprint,
            "selected_ids": list(self.selected_ids),
            "status": self.status,
            "diagnostics": _thaw_json(self.diagnostics),
        }


class QuantumSolver(Solver):
    """Format the neutral-atom boundary without claiming an implementation."""

    @property
    def solver_name(self) -> str:
        return "neutral_atom"

    def format_input(self, solver_input: SolverInput) -> NeutralAtomInput:
        """Translate the common solver problem into the portable request."""

        if not isinstance(solver_input, SolverInput):
            raise TypeError("solver_input must be a SolverInput")
        return NeutralAtomInput(
            problem_id=solver_input.problem_id,
            input_fingerprint=solver_input.fingerprint,
            frame=solver_input.frame,
            nodes=tuple((node.node_id, node.weight) for node in solver_input.nodes),
            edges=solver_input.edges,
        )

    def format_output(
        self,
        solver_input: SolverInput,
        output: NeutralAtomOutput,
    ) -> SolverSelection:
        """Validate a future external response and expose its graph selection.

        The external implementation must echo the request identity.  Unknown
        node IDs, conflicting selections, and selections attached to failure
        statuses are rejected before they can reach the tracking controller.
        """

        if not isinstance(solver_input, SolverInput):
            raise TypeError("solver_input must be a SolverInput")
        if not isinstance(output, NeutralAtomOutput):
            raise TypeError("output must be a NeutralAtomOutput")
        if output.problem_id != solver_input.problem_id:
            raise ValueError("neutral-atom output problem_id does not match input")
        if output.input_fingerprint != solver_input.fingerprint:
            raise ValueError("neutral-atom output fingerprint does not match input")
        selected = set(output.selected_ids)
        unknown = selected - set(solver_input.cluster.node_ids)
        if unknown:
            raise ValueError(
                f"neutral-atom output selected unknown node IDs: {sorted(unknown)}"
            )
        if any(left in selected and right in selected for left, right in solver_input.edges):
            raise ValueError("neutral-atom output is not an independent set")
        if output.status not in SUCCESS_STATUSES and output.selected_ids:
            raise ValueError("an unsuccessful neutral-atom output cannot select nodes")
        return SolverSelection(
            selected_ids=output.selected_ids,
            status=output.status,
            diagnostics=output.diagnostics,
        )

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        request = self.format_input(solver_input)
        return SolverSelection(
            status="not_implemented",
            diagnostics={
                "message": (
                    "Neutral-atom execution is intentionally not implemented; "
                    "use format_input() and format_output() when adding it."
                ),
                "formatted_input": request.to_dict(),
            },
        )


# The descriptive alias reads naturally without preserving the old backend API.
NeutralAtomSolver = QuantumSolver


__all__ = [
    "NEUTRAL_ATOM_SCHEMA_VERSION",
    "NeutralAtomInput",
    "NeutralAtomOutput",
    "NeutralAtomSolver",
    "QuantumSolver",
]
