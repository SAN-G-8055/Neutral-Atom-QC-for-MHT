"""Exercise neutral-atom orchestration without requiring Pulser or QuTiP.

The concrete runner owns the physical simulation. These tests inject a small
recording runner so they can protect the common ``SolverInput``/``SolverResult``
contract, component ordering, and sampled-bitstring decoding in the ordinary
test environment.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neutral_atom_mht.graph import ConflictGraph, GraphNode
from neutral_atom_mht.neutral_atom import (
    NeutralAtomComponent,
    NeutralAtomConfig,
    NeutralAtomRun,
    QuantumSolver,
)
from neutral_atom_mht.solver import Solver, SolverInput, validate_result


class ScriptedRunner:
    """Return declared component runs while retaining every received problem."""

    def __init__(self, runs: dict[int, NeutralAtomRun]) -> None:
        self.runs = runs
        self.components: list[NeutralAtomComponent] = []

    def execute(self, component: NeutralAtomComponent) -> NeutralAtomRun:
        self.components.append(component)
        return self.runs[component.component_id]


def disconnected_problem() -> SolverInput:
    """Two paths with deliberately sparse, noncontiguous node identifiers."""

    graph = ConflictGraph(
        nodes=(
            GraphNode(309, 4.0, 6, 9),
            GraphNode(11, 4.0, 1, 1),
            GraphNode(205, 6.0, 5, 8),
            GraphNode(47, 5.0, 3, 3),
            GraphNode(101, 3.0, 4, 7),
            GraphNode(29, 7.0, 2, 2),
        ),
        edges=((205, 309), (29, 47), (101, 205), (11, 29)),
    )
    return SolverInput("quantum-disconnected", 4, graph)


def path_problem() -> SolverInput:
    graph = ConflictGraph(
        nodes=(
            GraphNode(11, 4.0, 1, 1),
            GraphNode(29, 7.0, 2, 2),
            GraphNode(47, 5.0, 3, 3),
        ),
        edges=((11, 29), (29, 47)),
    )
    return SolverInput("quantum-path", 2, graph)


def component_run(
    component_id: int,
    node_ids: tuple[int, ...],
    bitstring_counts: tuple[tuple[str, int], ...],
    *,
    atom_order: tuple[str, ...] | None = None,
) -> NeutralAtomRun:
    qubit_ids = tuple(f"q{index}" for index in range(len(node_ids)))
    return NeutralAtomRun(
        component_id=component_id,
        node_ids=node_ids,
        atom_order=qubit_ids if atom_order is None else atom_order,
        bitstring_counts=bitstring_counts,
        coordinates=tuple(
            (float(index), float(index + 1)) for index in range(len(node_ids))
        ),
        mapping_cost=0.125,
        mapping_success=True,
    )


def test_quantum_solver_inherits_the_shared_solver_template() -> None:
    assert issubclass(QuantumSolver, Solver)


def test_quantum_solver_clusters_one_full_input_and_decodes_original_ids() -> None:
    """A frequent infeasible sample must not displace the best feasible one."""

    solver_input = disconnected_problem()
    runner = ScriptedRunner(
        {
            0: component_run(
                0,
                (11, 29, 47),
                (
                    ("001", 30),
                    ("011", 100),  # q0/node 11 and q1/node 29 conflict.
                    ("110", 2),  # q2/node 47 and q0/node 11 are best.
                ),
                atom_order=("q2", "q0", "q1"),
            ),
            1: component_run(1, (101, 205, 309), (("010", 5),)),
        }
    )

    result = QuantumSolver(runner=runner).solve(solver_input)

    assert result.problem_id == solver_input.problem_id
    assert result.input_fingerprint == solver_input.fingerprint
    assert result.solver_name == "neutral_atom"
    assert result.status == "completed"
    assert result.successful and result.feasible
    assert result.selected_ids == (11, 47, 205)
    assert result.objective == 15.0
    validate_result(solver_input, result)

    assert [component.component_id for component in runner.components] == [0, 1]
    first, second = runner.components
    assert first.node_ids == (11, 29, 47)
    assert first.qubit_ids == ("q0", "q1", "q2")
    assert first.weights == (4.0, 7.0, 5.0)
    assert first.edges == ((11, 29), (29, 47))
    np.testing.assert_array_equal(
        first.matrix,
        np.array(
            (
                (4.0, 1.0, 0.0),
                (1.0, 7.0, 1.0),
                (0.0, 1.0, 5.0),
            )
        ),
    )
    assert second.node_ids == (101, 205, 309)
    assert second.weights == (3.0, 6.0, 4.0)
    assert second.edges == ((101, 205), (205, 309))

    assert len(result.diagnostics["components"]) == 2
    json.dumps(result.to_dict()["diagnostics"], allow_nan=False)


def test_equal_weight_samples_use_a_deterministic_node_id_tie_break() -> None:
    solver_input = SolverInput(
        "quantum-tie",
        1,
        ConflictGraph(
            nodes=(
                GraphNode(4, 2.0, 1, 1),
                GraphNode(9, 2.0, 2, 2),
                GraphNode(12, 2.0, 3, 3),
            ),
            edges=((4, 9), (9, 12)),
        ),
    )
    runner = ScriptedRunner(
        {
            0: component_run(
                0,
                (4, 9, 12),
                (("010", 7), ("100", 7)),
                atom_order=("q2", "q0", "q1"),
            )
        }
    )

    result = QuantumSolver(runner=runner).solve(solver_input)

    assert result.selected_ids == (4,)
    assert result.objective == 2.0
    assert result.status == "completed"


def test_component_with_no_valid_feasible_sample_fails_the_whole_result() -> None:
    runner = ScriptedRunner(
        {
            0: component_run(
                0,
                (11, 29, 47),
                (
                    ("10x", 2),  # Malformed.
                    ("110", 3),  # Nodes 11 and 29 conflict.
                ),
            )
        }
    )

    result = QuantumSolver(runner=runner).solve(path_problem())

    assert result.status == "no_feasible_sample"
    assert not result.successful
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    assert result.diagnostics["failed_component_id"] == 0
    assert "no valid feasible sample" in result.diagnostics["message"]


def test_negative_weights_are_rejected_only_for_simulated_components() -> None:
    runner = ScriptedRunner({})
    solver_input = SolverInput(
        "quantum-negative-path",
        2,
        ConflictGraph(
            nodes=(
                GraphNode(11, 4.0, 1, 1),
                GraphNode(29, -1.0, 2, 2),
                GraphNode(47, 5.0, 3, 3),
            ),
            edges=((11, 29), (29, 47)),
        ),
    )

    result = QuantumSolver(runner=runner).solve(solver_input)

    assert result.status == "unsupported_weights"
    assert not result.successful
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    assert runner.components == []
    assert result.diagnostics["negative_weight_component_ids"] == (0,)


def test_empty_graph_completes_without_calling_the_runner() -> None:
    runner = ScriptedRunner({})
    solver_input = SolverInput("quantum-empty", 0, ConflictGraph(()))

    result = QuantumSolver(runner=runner).solve(solver_input)

    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.status == "completed"
    assert result.successful and result.feasible
    assert runner.components == []
    assert result.diagnostics["components"] == ()


def test_singleton_components_are_resolved_without_calling_the_runner() -> None:
    runner = ScriptedRunner({})
    solver_input = SolverInput(
        "quantum-singletons",
        3,
        ConflictGraph(
            nodes=(
                GraphNode(2, 3.0, 1, 1),
                GraphNode(7, 5.0, 2, 2),
            )
        ),
    )

    result = QuantumSolver(runner=runner).solve(solver_input)

    assert result.selected_ids == (2, 7)
    assert result.objective == 8.0
    assert result.status == "completed"
    assert runner.components == []
    json.dumps(result.to_dict()["diagnostics"], allow_nan=False)


def test_maximum_component_size_is_an_atomic_unsuccessful_result() -> None:
    solver_input = SolverInput(
        "quantum-oversized",
        5,
        ConflictGraph(
            nodes=(
                GraphNode(1, 2.0, 1, 1),
                GraphNode(2, 3.0, 2, 2),
                GraphNode(10, 4.0, 3, 3),
                GraphNode(11, 5.0, 4, 4),
                GraphNode(12, 6.0, 5, 5),
            ),
            edges=((1, 2), (10, 11), (11, 12)),
        ),
    )
    runner = ScriptedRunner(
        {0: component_run(0, (1, 2), (("01", 1),))}
    )

    result = QuantumSolver(
        config=NeutralAtomConfig(maximum_component_nodes=2),
        runner=runner,
    ).solve(solver_input)

    assert result.status == "unsupported_size"
    assert not result.successful
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    # The smaller first component is not executed before the later cap failure.
    assert runner.components == []
    assert result.diagnostics["maximum_component_nodes"] == 2


@pytest.mark.parametrize(
    "bad_run",
    (
        component_run(7, (11, 29, 47), (("101", 1),)),
        component_run(0, (11, 29, 99), (("101", 1),)),
        component_run(
            0,
            (11, 29, 47),
            (("101", 1),),
            atom_order=("q0", "q1", "q9"),
        ),
    ),
)
def test_runner_output_must_match_the_requested_component(
    bad_run: NeutralAtomRun,
) -> None:
    runner = ScriptedRunner({0: bad_run})

    with pytest.raises(ValueError, match="component|node|atom"):
        QuantumSolver(runner=runner).solve(path_problem())


def test_configuration_defaults_preserve_the_quantum_attempt_parameters() -> None:
    config = NeutralAtomConfig()

    assert config.random_seed == 0
    assert config.mapping_tolerance == 1e-6
    assert config.mapping_max_iterations == 200_000
    assert config.pulse_duration_ns == 40_000
    assert config.interaction_scale == 10.0
    assert config.maximum_component_nodes == 16


@pytest.mark.parametrize("seed", (-1, 2**32))
def test_configuration_rejects_seeds_outside_numpy_range(seed: int) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        NeutralAtomConfig(random_seed=seed)
