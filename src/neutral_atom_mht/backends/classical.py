"""Exact classical maximum-weight-independent-set backend."""

from __future__ import annotations

from functools import lru_cache
from math import fsum
from time import perf_counter

from .base import SolverInput, SolverResult, validate_result


class ClassicalBackend:
    """Solve one cluster exactly with deterministic dynamic programming.

    The explicit size limit prevents an exponential computation from being
    mistaken for a scalable classical baseline.  There is no silent heuristic
    or quantum fallback.
    """

    name = "classical_exact"

    def __init__(self, *, maximum_nodes: int = 30) -> None:
        if (
            isinstance(maximum_nodes, bool)
            or not isinstance(maximum_nodes, int)
            or maximum_nodes < 1
        ):
            raise ValueError("maximum_nodes must be a positive integer")
        self.maximum_nodes = maximum_nodes

    def solve(self, solver_input: SolverInput) -> SolverResult:
        started = perf_counter()
        nodes = tuple(sorted(solver_input.nodes, key=lambda item: item.node_id))
        if len(nodes) > self.maximum_nodes:
            return SolverResult(
                problem_id=solver_input.problem_id,
                input_fingerprint=solver_input.fingerprint,
                backend=self.name,
                selected_ids=(),
                objective=0.0,
                feasible=True,
                status="unsupported_size",
                runtime_seconds=perf_counter() - started,
                diagnostics={
                    "node_count": len(nodes),
                    "maximum_nodes": self.maximum_nodes,
                    "optimal": False,
                },
            )
        node_ids = tuple(node.node_id for node in nodes)
        weights = {node.node_id: node.weight for node in nodes}
        neighbours = {node_id: set() for node_id in node_ids}
        for left, right in solver_input.edges:
            neighbours[left].add(right)
            neighbours[right].add(left)

        @lru_cache(maxsize=None)
        def optimize(remaining: frozenset[int]) -> tuple[float, tuple[int, ...]]:
            if not remaining:
                return 0.0, ()
            vertex = min(remaining)
            excluded_weight, excluded = optimize(remaining - {vertex})
            include_remaining = remaining - {vertex} - neighbours[vertex]
            _, suffix = optimize(include_remaining)
            included = tuple(sorted((vertex, *suffix)))
            included_weight = fsum(weights[node_id] for node_id in included)
            if included_weight > excluded_weight:
                return included_weight, included
            if excluded_weight > included_weight:
                return excluded_weight, excluded
            return included_weight, min(included, excluded)

        _, selected = optimize(frozenset(node_ids))
        objective = fsum(weights[node_id] for node_id in selected)
        result = SolverResult(
            problem_id=solver_input.problem_id,
            input_fingerprint=solver_input.fingerprint,
            backend=self.name,
            selected_ids=selected,
            objective=objective,
            feasible=True,
            status="optimal",
            runtime_seconds=perf_counter() - started,
            diagnostics={
                "node_count": len(nodes),
                "edge_count": len(solver_input.edges),
                "optimal": True,
                "states_evaluated": optimize.cache_info().currsize,
            },
        )
        validate_result(solver_input, result)
        return result
