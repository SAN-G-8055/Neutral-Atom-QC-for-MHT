"""Define the unimplemented neutral-atom solver boundary.

This file contains no simulation, pulse model, device parameters, or fallback
optimizer. ``QuantumSolver`` only converts a shared ``SolverInput`` into a
portable request and converts a future external response into the common
``SolverSelection`` format. The inherited solver template remains responsible
for objective calculation and feasibility validation.

To implement the quantum path later, override ``_select()``, send the result of
``format_input()`` to the chosen simulator or device, and pass its response to
``format_output()``. Tracking and Bayesian updates do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .solver import SUCCESS_STATUSES, Solver, SolverInput, SolverSelection


@dataclass(frozen=True, slots=True)
class NeutralAtomInput:
    """JSON-ready weighted graph sent to a future neutral-atom implementation."""

    problem_id: str
    input_fingerprint: str
    frame: int
    nodes: tuple[tuple[int, float], ...]
    edges: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, object]:
        """Return the transport representation with explicit node fields."""

        return {
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
    """Small response record expected from a future external implementation."""

    problem_id: str
    input_fingerprint: str
    selected_ids: tuple[int, ...]
    status: str = "completed"
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class QuantumSolver(Solver):
    """Format neutral-atom I/O while reporting that solving is not implemented."""

    @property
    def solver_name(self) -> str:
        return "neutral_atom"

    def format_input(self, solver_input: SolverInput) -> NeutralAtomInput:
        """Translate one immutable graph component into the portable request."""

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
        """Validate response identity and convert it to the shared selection."""

        if not isinstance(solver_input, SolverInput):
            raise TypeError("solver_input must be a SolverInput")
        if not isinstance(output, NeutralAtomOutput):
            raise TypeError("output must be a NeutralAtomOutput")
        if output.problem_id != solver_input.problem_id:
            raise ValueError("neutral-atom output problem_id does not match input")
        if output.input_fingerprint != solver_input.fingerprint:
            raise ValueError("neutral-atom output fingerprint does not match input")

        selection = SolverSelection(
            selected_ids=output.selected_ids,
            status=output.status,
            diagnostics=output.diagnostics,
        )
        selected = set(selection.selected_ids)
        unknown = selected - set(solver_input.cluster.node_ids)
        if unknown:
            raise ValueError(f"neutral-atom output selected unknown nodes: {sorted(unknown)}")
        if any(left in selected and right in selected for left, right in solver_input.edges):
            raise ValueError("neutral-atom output is not an independent set")
        if selection.status not in SUCCESS_STATUSES and selection.selected_ids:
            raise ValueError("an unsuccessful neutral-atom output cannot select nodes")
        return selection

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        return SolverSelection(
            status="not_implemented",
            diagnostics={
                "message": (
                    "Neutral-atom execution is not implemented; use format_input() "
                    "and format_output() when adding it."
                )
            },
        )


__all__ = ["NeutralAtomInput", "NeutralAtomOutput", "QuantumSolver"]
