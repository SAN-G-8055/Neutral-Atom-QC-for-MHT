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

from solver import ComponentSolver, SolverInput, SolverSelection


class ClassicalSolver(ComponentSolver):
    """Solve every bounded graph component exactly and deterministically."""

    def __init__(self, maximum_component_nodes: int = 30) -> None:
        super().__init__("classical_exact", maximum_component_nodes)

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        components = self._components(solver_input)
        problem_diagnostics = self._problem_diagnostics(solver_input, components)
        oversized = self._oversized_component_ids(components)
        if oversized:
            return SolverSelection(
                status="unsupported_size",
                diagnostics={
                    **problem_diagnostics,
                    "oversized_component_ids": oversized,
                    "optimal": False,
                },
            )

        nodes_by_id = {node.node_id: node for node in solver_input.nodes}
        selected_ids: list[int] = []
        component_diagnostics: list[dict[str, int]] = []
        for component in components:
            node_ids = component.node_ids
            weights = {node_id: nodes_by_id[node_id].weight for node_id in node_ids}
            neighbors = {node_id: set() for node_id in node_ids}
            component_edges = self._component_edges(solver_input, component)
            for left, right in component_edges:
                neighbors[left].add(right)
                neighbors[right].add(left)

            @lru_cache(maxsize=None)
            def optimize(
                remaining: frozenset[int],
            ) -> tuple[float, tuple[int, ...]]:
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

            _, component_selection = optimize(frozenset(node_ids))
            selected_ids.extend(component_selection)
            component_diagnostics.append(
                {
                    "component_id": component.cluster_id,
                    "node_count": len(node_ids),
                    "edge_count": len(component_edges),
                    "states_evaluated": optimize.cache_info().currsize,
                }
            )

        return SolverSelection(
            selected_ids=tuple(selected_ids),
            status="optimal",
            diagnostics={
                **problem_diagnostics,
                "components": component_diagnostics,
                "optimal": True,
                "states_evaluated": sum(
                    component["states_evaluated"]
                    for component in component_diagnostics
                ),
            },
        )


__all__ = ["ClassicalSolver"]
