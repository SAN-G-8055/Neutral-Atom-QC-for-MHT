"""Opt-in smoke test for the real Pulser/QuTiP simulation stack.

The original 40,000 ns pulse and coordinate search are too expensive for the
ordinary unit suite. After installing ``.[quantum]``, run this test explicitly
from PowerShell with::

    $env:NEUTRAL_ATOM_INTEGRATION="1"
    python -m pytest tests/test_neutral_atom_integration.py -q
"""

from __future__ import annotations

import os

import pytest

from graph import ConflictGraph, GraphNode
from neutral_atom import QuantumSolver
from solver import SolverInput, validate_result


@pytest.mark.skipif(
    os.environ.get("NEUTRAL_ATOM_INTEGRATION") != "1",
    reason=(
        "set NEUTRAL_ATOM_INTEGRATION=1 to run the expensive Pulser/QuTiP smoke test"
    ),
)
def test_real_pulser_qutip_three_node_path_satisfies_the_solver_contract() -> None:
    pytest.importorskip("pulser")
    pytest.importorskip("pulser_simulation")
    solver_input = SolverInput(
        "real-pulser-path",
        0,
        ConflictGraph(
            nodes=(
                GraphNode(1, 2.0, 1, 1),
                GraphNode(2, 4.0, 2, 2),
                GraphNode(3, 3.0, 3, 3),
            ),
            edges=((1, 2), (2, 3)),
        ),
    )

    result = QuantumSolver().solve(solver_input)

    assert result.status == "completed"
    assert result.successful and result.feasible
    assert set(result.selected_ids) <= set(solver_input.graph.node_ids)
    validate_result(solver_input, result)
