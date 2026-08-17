from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
import math

import pytest

from neutral_atom_mht.backends.base import SolverInput, validate_result
from neutral_atom_mht.backends.neutral_atom import (
    MAXIMUM_EXHAUSTIVE_ENERGY_AUDIT_ATOMS,
    PULSER_MWIS_TUTORIAL_URL,
    PULSER_WEIGHTED_ANALOG_DEVICE_DOCS_URL,
    AdiabaticPulse,
    NeutralAtomBackend,
    PasqalParameters,
    embed_unit_disk,
)
from neutral_atom_mht.graph import ConflictGraph, GraphCluster, GraphNode


def _problem(
    node_count: int,
    edges: tuple[tuple[int, int], ...] = (),
    *,
    weights: tuple[float, ...] | None = None,
) -> SolverInput:
    weights = weights or tuple(float(index + 1) for index in range(node_count))
    nodes = tuple(
        GraphNode(
            node_id=index + 1,
            weight=weights[index],
            track_id=index + 1,
            observation_id=index + 1,
        )
        for index in range(node_count)
    )
    graph = ConflictGraph(nodes, edges)
    return SolverInput("qutip-test", 4, graph, GraphCluster(0, graph.node_ids))


def test_pasqal_defaults_match_the_official_weighted_device_reference() -> None:
    parameters = PasqalParameters()

    assert parameters.rydberg_level == 75
    assert parameters.c6_over_hbar_rad_per_us_um6 == 12_241_414.53
    assert parameters.minimum_atom_spacing_um == 5.0
    assert parameters.maximum_atoms == 256
    assert parameters.maximum_radial_distance_um == 80.0
    assert parameters.maximum_duration_us == 6.0
    assert parameters.maximum_runs == 500
    assert parameters.maximum_omega_rad_per_us == pytest.approx(4.0 * math.pi)
    assert parameters.maximum_abs_detuning_rad_per_us == pytest.approx(20.0 * math.pi)
    assert parameters.minimum_average_omega_rad_per_us == pytest.approx(0.6 * math.pi)
    assert parameters.dmm_bottom_detuning_rad_per_us == pytest.approx(-20.0 * math.pi)
    assert parameters.requires_layout
    assert parameters.minimum_layout_traps == 150
    assert parameters.maximum_layout_traps == 512
    assert parameters.minimum_layout_filling == pytest.approx(0.35)
    assert parameters.maximum_layout_filling == pytest.approx(0.5)
    assert parameters.source_urls == (
        PULSER_WEIGHTED_ANALOG_DEVICE_DOCS_URL,
        PULSER_MWIS_TUTORIAL_URL,
    )
    assert all(url.startswith("https://") for url in parameters.source_urls)
    serialized = json.loads(json.dumps(parameters.to_dict()))
    assert serialized["rydberg_level"] == 75
    assert serialized["requires_layout"] is True
    assert serialized["maximum_layout_traps"] == 512
    with pytest.raises(FrozenInstanceError):
        parameters.maximum_atoms = 81  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rydberg_level", 0),
        ("maximum_atoms", True),
        ("maximum_runs", 0),
        ("c6_over_hbar_rad_per_us_um6", 0.0),
        ("minimum_atom_spacing_um", math.inf),
        ("maximum_duration_us", -1.0),
        ("maximum_omega_rad_per_us", math.nan),
        ("dmm_bottom_detuning_rad_per_us", 0.0),
        ("requires_layout", "yes"),
        ("minimum_layout_traps", 513),
        ("maximum_layout_filling", 1.1),
        ("source_urls", ("http://not-authoritative.example",)),
    ],
)
def test_pasqal_parameters_validate_every_physical_field(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(PasqalParameters(), **{field: value})


def test_reference_pulse_is_four_microseconds_and_stays_inside_caps() -> None:
    pulse = AdiabaticPulse()
    parameters = PasqalParameters()

    pulse.validate_against(parameters)
    assert pulse.duration_us == 4.0
    assert pulse.omega(0.0) == pytest.approx(0.0)
    assert pulse.omega(pulse.ramp_duration_us) == pytest.approx(pulse.omega_peak_rad_per_us)
    assert pulse.omega(pulse.duration_us / 2.0) == pytest.approx(pulse.omega_peak_rad_per_us)
    assert pulse.omega(pulse.duration_us) == pytest.approx(0.0)
    assert pulse.detuning(0.0) == pytest.approx(-pulse.detuning_span_rad_per_us)
    assert pulse.detuning(pulse.duration_us / 2.0) == pytest.approx(0.0)
    assert pulse.detuning(pulse.duration_us) == pytest.approx(pulse.detuning_span_rad_per_us)
    assert pulse.dmm_detuning(2.0) == pytest.approx(-pulse.detuning_span_rad_per_us)
    assert pulse.average_omega_rad_per_us == pytest.approx(1.8 * math.pi)
    assert len(pulse.times_us) == pulse.time_steps
    assert pulse.times_us[0] == 0.0
    assert pulse.times_us[-1] == pulse.duration_us


def test_pulse_rejects_invalid_shape_time_and_device_caps() -> None:
    with pytest.raises(ValueError, match="ramps"):
        AdiabaticPulse(duration_us=1.0, ramp_duration_us=0.6)
    with pytest.raises(ValueError, match="time_us"):
        AdiabaticPulse().omega(-0.1)
    with pytest.raises(ValueError, match="duration"):
        AdiabaticPulse(duration_us=6.1).validate_against(PasqalParameters())
    with pytest.raises(ValueError, match="Omega"):
        AdiabaticPulse(omega_peak_rad_per_us=4.1 * math.pi).validate_against(
            PasqalParameters()
        )
    with pytest.raises(ValueError, match="average Omega"):
        AdiabaticPulse(omega_peak_rad_per_us=0.5 * math.pi).validate_against(
            PasqalParameters()
        )
    with pytest.raises(ValueError, match="detuning"):
        AdiabaticPulse(detuning_span_rad_per_us=20.1 * math.pi).validate_against(
            PasqalParameters()
        )


@pytest.mark.parametrize(
    "edges",
    [
        (),
        ((1, 2),),
        ((1, 2), (2, 3)),
        ((1, 2), (1, 3), (2, 3)),
    ],
)
def test_small_graph_embedding_has_exact_unit_disk_fidelity(
    edges: tuple[tuple[int, int], ...],
) -> None:
    node_count = 3 if any(3 in edge for edge in edges) or not edges else 2
    solver_input = _problem(node_count, edges, weights=(1.0,) * node_count)

    first = embed_unit_disk(solver_input)
    second = embed_unit_disk(solver_input)

    assert first == second
    assert first.exact_fidelity
    assert first.realized_edges == tuple(sorted(edges))
    assert first.missing_edges == ()
    assert first.spurious_edges == ()
    assert first.spacing_valid
    assert first.radius_valid
    assert first.topology_fidelity
    assert first.energy_audit_complete
    assert first.weighted_objective_fidelity
    assert first.constraint_radius_um > PasqalParameters().minimum_atom_spacing_um
    if edges:
        assert first.minimum_edge_interaction_rad_per_us is not None
        assert (
            first.minimum_edge_interaction_rad_per_us
            >= first.detuning_penalty_rad_per_us
        )
        assert first.minimum_interaction_to_detuning_ratio > 1.0
    assert json.loads(json.dumps(first.to_dict()))["exact_fidelity"] is True


def test_final_hamiltonian_audit_rejects_nonedge_tail_that_changes_mwis() -> None:
    solver_input = _problem(
        3,
        ((1, 2), (2, 3)),
        weights=(1.0, 0.4, 0.01),
    )

    embedding = embed_unit_disk(solver_input)

    assert embedding.topology_fidelity
    assert embedding.energy_audit_complete
    assert embedding.abstract_optimal_node_ids == (1, 3)
    assert embedding.physical_ground_node_ids == (1,)
    assert embedding.maximum_nonedge_interaction_rad_per_us is not None
    assert embedding.maximum_nonedge_to_minimum_reward_ratio > 1.0
    assert embedding.weighted_objective_fidelity is False
    assert embedding.exact_fidelity is False

    result = NeutralAtomBackend(maximum_simulation_atoms=3).solve(solver_input)
    assert result.status == "embedding_error"
    assert result.diagnostics["embedding"]["abstract_optimal_node_ids"] == (1, 3)
    assert result.diagnostics["embedding"]["physical_ground_node_ids"] == (1,)


def test_energy_audit_does_not_widen_a_real_tiny_weight_into_a_tie() -> None:
    solver_input = _problem(
        3,
        ((1, 2), (2, 3)),
        weights=(1.0, 0.4, 5e-11),
    )

    embedding = embed_unit_disk(solver_input)

    assert embedding.abstract_optimal_node_ids == (1, 3)
    assert embedding.physical_ground_node_ids == (1,)
    assert embedding.weighted_objective_fidelity is False
    assert embedding.exact_fidelity is False


def test_unrepresentable_dmm_weight_range_fails_transparently() -> None:
    solver_input = _problem(
        3,
        ((1, 2), (2, 3)),
        weights=(1.0, 0.4, 5e-324),
    )

    embedding = embed_unit_disk(solver_input)
    assert embedding.energy_audit_complete is False
    assert embedding.weighted_objective_fidelity is False
    assert embedding.exact_fidelity is False
    assert embedding.abstract_optimal_node_ids is None
    assert embedding.physical_ground_node_ids is None

    result = NeutralAtomBackend(maximum_simulation_atoms=3).solve(solver_input)
    assert result.status == "invalid_weights"
    assert "dynamic range" in result.diagnostics["reason"]


def test_energy_audit_and_simulation_caps_are_explicitly_bounded() -> None:
    solver_input = _problem(2, (), weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="exhaustive-audit"):
        embed_unit_disk(
            solver_input,
            maximum_energy_audit_atoms=MAXIMUM_EXHAUSTIVE_ENERGY_AUDIT_ATOMS + 1,
        )
    with pytest.raises(ValueError, match="exhaustive-energy-audit"):
        NeutralAtomBackend(
            maximum_simulation_atoms=MAXIMUM_EXHAUSTIVE_ENERGY_AUDIT_ATOMS + 1
        )


def test_backend_reports_embedding_error_instead_of_solving_a_changed_graph() -> None:
    solver_input = _problem(2, ((1, 2),))
    parameters = replace(PasqalParameters(), c6_over_hbar_rad_per_us_um6=1.0)
    backend = NeutralAtomBackend(parameters=parameters, maximum_simulation_atoms=2)

    result = backend.solve(solver_input)

    assert result.status == "embedding_error"
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.diagnostics["embedding"]["exact_fidelity"] is False
    validate_result(solver_input, result)


def test_backend_reports_state_vector_size_limit_without_fallback() -> None:
    solver_input = _problem(
        4,
        tuple((left, right) for left in range(1, 5) for right in range(left + 1, 5)),
    )
    backend = NeutralAtomBackend(maximum_simulation_atoms=3)

    result = backend.solve(solver_input)

    assert result.status == "unsupported_size"
    assert result.diagnostics["reason"] == "state_vector_simulation_limit"
    assert result.selected_ids == ()
    assert result.input_fingerprint == solver_input.fingerprint
    validate_result(solver_input, result)


def test_backend_reports_missing_qutip_in_common_schema(monkeypatch) -> None:
    from neutral_atom_mht.backends import neutral_atom

    def unavailable() -> object:
        raise ModuleNotFoundError("No module named 'qutip'")

    monkeypatch.setattr(neutral_atom, "_import_qutip", unavailable)
    solver_input = _problem(1)

    result = NeutralAtomBackend(maximum_simulation_atoms=1).solve(solver_input)

    assert result.status == "dependency_missing"
    assert result.backend == "neutral_atom_qutip"
    assert result.problem_id == solver_input.problem_id
    assert result.input_fingerprint == solver_input.fingerprint
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    assert result.diagnostics["distribution_retained"] is False
    assert result.diagnostics["basis_convention"] == {
        "ground": "|0>",
        "rydberg": "|1>",
    }
    assert result.diagnostics["hardware_execution_readiness"] == {
        "validated": False,
        "reason": (
            "direct-coordinate QuTiP emulation does not validate the required "
            "hardware register layout and filling constraints"
        ),
        "requires_layout": True,
    }
    assert json.loads(json.dumps(result.to_dict()))["status"] == "dependency_missing"
    validate_result(solver_input, result)


def test_invalid_dmm_weights_fail_transparently_before_optional_import() -> None:
    solver_input = _problem(1, weights=(-1.0,))

    result = NeutralAtomBackend(maximum_simulation_atoms=1).solve(solver_input)

    assert result.status == "invalid_weights"
    assert "strictly positive" in result.diagnostics["reason"]
    assert result.selected_ids == ()


@pytest.mark.skipif(importlib.util.find_spec("qutip") is None, reason="optional qutip is absent")
@pytest.mark.parametrize(
    ("node_count", "edges"),
    [
        (1, ()),
        (2, ((1, 2),)),
        (3, ((1, 2), (2, 3))),
    ],
)
def test_direct_qutip_simulation_returns_one_feasible_common_result(
    node_count: int, edges: tuple[tuple[int, int], ...]
) -> None:
    solver_input = _problem(node_count, edges)

    result = NeutralAtomBackend(maximum_simulation_atoms=3).solve(solver_input)

    assert result.status == "simulated"
    assert result.feasible
    assert result.diagnostics["distribution_retained"] is False
    assert result.diagnostics["state_dimension"] == 2**node_count
    assert 0.0 <= result.diagnostics["selected_probability"] <= 1.0
    validate_result(solver_input, result)
