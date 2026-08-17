"""Provide the small exact classical reference solver.

The controller has already turned tracking choices into a weighted conflict
graph before this class is called.  ``ClassicalSolver`` therefore does one job:
find the maximum-weight independent set of each connected component.  Its
explicit node limit makes the exponential cost visible, and it never rounds
weights, substitutes a heuristic, or calls the neutral-atom adapter.
"""

from __future__ import annotations

from functools import lru_cache
from math import fsum

from .solver import Solver, SolverInput, SolverSelection


class ClassicalSolver(Solver):
    """Solve a bounded conflict-graph component exactly and deterministically."""

    def __init__(self, maximum_nodes: int = 30) -> None:
        if (
            isinstance(maximum_nodes, bool)
            or not isinstance(maximum_nodes, int)
            or maximum_nodes < 1
        ):
            raise ValueError("maximum_nodes must be a positive integer")
        self.maximum_nodes = maximum_nodes

    @property
    def solver_name(self) -> str:
        return "classical_exact"

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        nodes = tuple(sorted(solver_input.nodes, key=lambda item: item.node_id))
        if len(nodes) > self.maximum_nodes:
            return SolverSelection(
                status="unsupported_size",
                diagnostics={
                    "node_count": len(nodes),
                    "maximum_nodes": self.maximum_nodes,
                    "optimal": False,
                },
            )

        node_ids = tuple(node.node_id for node in nodes)
        weights = {node.node_id: node.weight for node in nodes}
        neighbors = {node_id: set() for node_id in node_ids}
        for left, right in solver_input.edges:
            neighbors[left].add(right)
            neighbors[right].add(left)

        @lru_cache(maxsize=None)
        def optimize(remaining: frozenset[int]) -> tuple[float, tuple[int, ...]]:
            if not remaining:
                return 0.0, ()

            vertex = min(remaining)
            excluded_weight, excluded = optimize(remaining - {vertex})

            include_remaining = remaining - {vertex} - neighbors[vertex]
            _, suffix = optimize(include_remaining)
            included = tuple(sorted((vertex, *suffix)))
            included_weight = fsum(weights[node_id] for node_id in included)

            if included_weight > excluded_weight:
                return included_weight, included
            if excluded_weight > included_weight:
                return excluded_weight, excluded
            return included_weight, min(included, excluded)

        _, selected = optimize(frozenset(node_ids))
        return SolverSelection(
            selected_ids=selected,
            status="optimal",
            diagnostics={
                "node_count": len(nodes),
                "edge_count": len(solver_input.edges),
                "optimal": True,
                "states_evaluated": optimize.cache_info().currsize,
            },
        )


__all__ = ["ClassicalSolver"]
