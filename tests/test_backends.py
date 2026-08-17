from __future__ import annotations

import json
from math import fsum
from types import MappingProxyType

import pytest

from neutral_atom_mht.backends.base import SolverInput, SolverResult, validate_result
from neutral_atom_mht.backends.classical import ClassicalBackend
from neutral_atom_mht.graph import ConflictGraph, GraphCluster, GraphNode


def paper_problem() -> SolverInput:
    graph = ConflictGraph(
        nodes=tuple(
            GraphNode(node_id, weight, node_id, node_id)
            for node_id, weight in {
                2: 3.4,
                3: 9.1,
                4: 7.5,
                5: 4.8,
                6: 10.1,
            }.items()
        ),
        edges=((2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6)),
    )
    return SolverInput(
        problem_id="paper-figure-5",
        frame=0,
        graph=graph,
        cluster=GraphCluster(0, graph.node_ids),
    )


def test_solver_input_has_stable_json_safe_fingerprint() -> None:
    first = paper_problem()
    second = paper_problem()

    assert first.fingerprint == second.fingerprint
    assert json.loads(json.dumps(first.to_dict()))["fingerprint"] == first.fingerprint


def test_exact_classical_backend_reproduces_worked_mwis() -> None:
    solver_input = paper_problem()
    result = ClassicalBackend().solve(solver_input)

    assert result.selected_ids == (3, 6)
    assert result.objective == pytest.approx(19.2)
    assert result.status == "optimal"
    assert result.feasible
    validate_result(solver_input, result)


def test_classical_solver_uses_exact_float_weights_without_rounding() -> None:
    graph = ConflictGraph(
        nodes=(
            GraphNode(0, 1.0004, 0, 0),
            GraphNode(1, 1.0005, 1, 1),
        ),
        edges=((0, 1),),
    )
    solver_input = SolverInput("near-tie", 1, graph, GraphCluster(0, (0, 1)))

    result = ClassicalBackend().solve(solver_input)

    assert result.selected_ids == (1,)
    assert result.objective == 1.0005


def test_classical_solver_keeps_a_sub_picoweight_strict_improvement() -> None:
    graph = ConflictGraph(
        nodes=(
            GraphNode(0, 1.0, 0, 0),
            GraphNode(1, 1.0 + 5e-13, 1, 1),
        ),
        edges=((0, 1),),
    )
    solver_input = SolverInput("sub-picoweight", 1, graph, GraphCluster(0, (0, 1)))

    result = ClassicalBackend().solve(solver_input)

    assert result.selected_ids == (1,)
    assert result.objective == 1.0 + 5e-13


def test_classical_objective_uses_the_canonical_high_accuracy_sum() -> None:
    graph = ConflictGraph(
        nodes=tuple(
            GraphNode(index, weight, index, index)
            for index, weight in enumerate((1e16, 1.0, -1e16, 2.0))
        )
    )
    solver_input = SolverInput("sum", 1, graph, GraphCluster(0, graph.node_ids))

    result = ClassicalBackend().solve(solver_input)

    selected_weights = [graph.node(node_id).weight for node_id in result.selected_ids]
    assert result.objective == fsum(selected_weights)


def test_solver_result_diagnostics_are_deeply_immutable_and_export_detached() -> None:
    solver_input = paper_problem()
    result = SolverResult(
        problem_id=solver_input.problem_id,
        input_fingerprint=solver_input.fingerprint,
        backend="test",
        selected_ids=(),
        objective=0.0,
        feasible=True,
        status="test",
        runtime_seconds=0.0,
        diagnostics={"nested": {"items": [1, 2]}},
    )

    assert isinstance(result.diagnostics["nested"], MappingProxyType)
    with pytest.raises(TypeError):
        result.diagnostics["nested"]["changed"] = True  # type: ignore[index]
    exported = result.to_dict()
    exported["diagnostics"]["nested"]["items"].append(3)  # type: ignore[index,union-attr]
    assert result.diagnostics["nested"]["items"] == (1, 2)


def test_backend_reports_unsupported_size_without_silent_fallback() -> None:
    solver_input = paper_problem()
    result = ClassicalBackend(maximum_nodes=4).solve(solver_input)

    assert result.status == "unsupported_size"
    assert result.selected_ids == ()
    assert result.diagnostics["optimal"] is False


def test_result_validation_rejects_backend_specific_objective_reweighting() -> None:
    solver_input = paper_problem()
    invalid = SolverResult(
        problem_id=solver_input.problem_id,
        input_fingerprint=solver_input.fingerprint,
        backend="bad",
        selected_ids=(3, 6),
        objective=999.0,
        feasible=True,
        status="claimed",
        runtime_seconds=0.0,
        diagnostics={},
    )

    with pytest.raises(ValueError, match="original input weights"):
        validate_result(solver_input, invalid)


def test_result_validation_rejects_conflicting_vertices() -> None:
    solver_input = paper_problem()
    invalid = SolverResult(
        problem_id=solver_input.problem_id,
        input_fingerprint=solver_input.fingerprint,
        backend="bad",
        selected_ids=(2, 3),
        objective=12.5,
        feasible=True,
        status="claimed",
        runtime_seconds=0.0,
        diagnostics={},
    )

    with pytest.raises(ValueError, match="independent"):
        validate_result(solver_input, invalid)
