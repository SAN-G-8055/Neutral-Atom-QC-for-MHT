"""Verify the documented neutral-atom boundary without simulating physics.

The neutral-atom class is deliberately an input/output adapter at this stage.
These tests make the placeholder status explicit and show how a future manual
implementation can validate an external response through the same base solver
contract used by the exact classical implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType

import pytest

from neutral_atom_mht.graph import ConflictGraph, GraphCluster, GraphNode
from neutral_atom_mht.neutral_atom import (
    NeutralAtomInput,
    NeutralAtomOutput,
    NeutralAtomSolver,
    QuantumSolver,
)
from neutral_atom_mht.solver import Solver, SolverInput


def neutral_problem() -> SolverInput:
    graph = ConflictGraph(
        nodes=(
            GraphNode(1, 2.0, 1, 1),
            GraphNode(2, 3.5, 1, 2),
            GraphNode(3, 4.0, 2, 3),
        ),
        edges=((1, 2),),
    )
    return SolverInput(
        "neutral-adapter-example",
        4,
        graph,
        GraphCluster(0, graph.node_ids),
    )


class ManuallyImplementedQuantumSolver(QuantumSolver):
    """Stand in for the small protected hook a future implementation fills."""

    def __init__(self, output: NeutralAtomOutput) -> None:
        self.output = output

    def _select(self, solver_input: SolverInput):  # type: ignore[no-untyped-def]
        return self.format_output(solver_input, self.output)


def test_quantum_solver_inherits_the_shared_solver_template() -> None:
    assert issubclass(QuantumSolver, Solver)
    assert NeutralAtomSolver is QuantumSolver


def test_format_input_is_a_small_deterministic_transport_record() -> None:
    solver_input = neutral_problem()

    request = QuantumSolver().format_input(solver_input)

    assert isinstance(request, NeutralAtomInput)
    assert request.problem_id == solver_input.problem_id
    assert request.input_fingerprint == solver_input.fingerprint
    assert request.node_ids == (1, 2, 3)
    assert request.nodes == ((1, 2.0), (2, 3.5), (3, 4.0))
    assert request.edges == ((1, 2),)
    assert request.to_dict()["nodes"][1] == {"node_id": 2, "weight": 3.5}


def test_default_quantum_solve_returns_an_honest_common_placeholder() -> None:
    solver_input = neutral_problem()

    result = QuantumSolver().solve(solver_input)

    assert result.problem_id == solver_input.problem_id
    assert result.input_fingerprint == solver_input.fingerprint
    assert result.solver_name == "neutral_atom"
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    assert result.status == "not_implemented"
    assert not result.successful
    assert result.diagnostics["formatted_input"]["input_fingerprint"] == (
        solver_input.fingerprint
    )


def test_format_output_supports_a_future_manual_selection() -> None:
    solver_input = neutral_problem()
    output = NeutralAtomOutput(
        problem_id=solver_input.problem_id,
        input_fingerprint=solver_input.fingerprint,
        selected_ids=(2, 3),
        diagnostics={"source": "manual-test"},
    )

    result = ManuallyImplementedQuantumSolver(output).solve(solver_input)

    assert result.status == "completed"
    assert result.successful
    assert result.selected_ids == (2, 3)
    assert result.objective == 7.5
    assert result.diagnostics["source"] == "manual-test"


def test_neutral_output_diagnostics_are_deeply_immutable() -> None:
    solver_input = neutral_problem()
    output = NeutralAtomOutput(
        solver_input.problem_id,
        solver_input.fingerprint,
        (),
        diagnostics={"nested": {"values": [1]}},
    )

    assert isinstance(output.diagnostics["nested"], MappingProxyType)
    with pytest.raises(TypeError):
        output.diagnostics["nested"]["changed"] = True  # type: ignore[index]


@pytest.mark.parametrize("field", ["problem_id", "input_fingerprint"])
def test_format_output_rejects_a_response_for_another_problem(field: str) -> None:
    solver_input = neutral_problem()
    values = {
        "problem_id": solver_input.problem_id,
        "input_fingerprint": solver_input.fingerprint,
        "selected_ids": (),
    }
    values[field] = "different"
    output = NeutralAtomOutput(**values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="does not match"):
        QuantumSolver().format_output(solver_input, output)


def test_format_output_rejects_unknown_or_conflicting_nodes() -> None:
    solver_input = neutral_problem()
    unknown = NeutralAtomOutput(
        solver_input.problem_id,
        solver_input.fingerprint,
        (99,),
    )
    conflicting = NeutralAtomOutput(
        solver_input.problem_id,
        solver_input.fingerprint,
        (1, 2),
    )

    with pytest.raises(ValueError, match="unknown"):
        QuantumSolver().format_output(solver_input, unknown)
    with pytest.raises(ValueError, match="independent set"):
        QuantumSolver().format_output(solver_input, conflicting)


def test_format_output_rejects_selected_nodes_on_a_failure_status() -> None:
    solver_input = neutral_problem()
    output = NeutralAtomOutput(
        solver_input.problem_id,
        solver_input.fingerprint,
        (2,),
        status="device_error",
    )

    with pytest.raises(ValueError, match="unsuccessful"):
        QuantumSolver().format_output(solver_input, output)


def test_neutral_atom_module_has_no_physics_or_vendor_dependency_imports() -> None:
    source_path = Path(__file__).parents[1] / "src" / "neutral_atom_mht" / "neutral_atom.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    )

    assert imported_roots.isdisjoint({"numpy", "scipy", "qutip", "pulser", "pasqal"})

