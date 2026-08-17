r"""Direct QuTiP simulation of a Pasqal-style neutral-atom MWIS register.

The physical reference values are the public ``WeightedAnalogDevice``
specification in Pasqal's Pulser project:

* https://docs.pasqal.com/pulser/apidoc/_autosummary/pulser.WeightedAnalogDevice/
* https://docs.pasqal.com/pulser/tutorials/mwis/

Pulser is not a runtime dependency here.  The Hamiltonian is built directly in
QuTiP, which is imported lazily so the classical pipeline remains usable without
the optional quantum environment.  Units are explicit throughout: positions in
micrometres, time in microseconds, and angular frequencies in radians per
microsecond.  With ``|g> = |0>`` and ``|r> = |1>``, the simulated Hamiltonian is

.. math::

   H/\hbar = \sum_{i<j} C_6/r_{ij}^6 n_i n_j
             + \Omega(t)/2 \sum_i X_i
             - \sum_i [\delta(t) + \epsilon_i\delta_{DMM}(t)] n_i.

Only one answer crosses the backend boundary: the highest-probability feasible
bitstring.  The evolved distribution is not retained as a family of
global hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
from math import ceil, cos, fsum, isfinite, pi, sin, sqrt
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .base import SolverInput, SolverResult, validate_result


PULSER_WEIGHTED_ANALOG_DEVICE_DOCS_URL = (
    "https://docs.pasqal.com/pulser/apidoc/_autosummary/"
    "pulser.WeightedAnalogDevice/"
)
PULSER_MWIS_TUTORIAL_URL = (
    "https://docs.pasqal.com/pulser/tutorials/mwis/"
)
QUTIP_TIME_DEPENDENT_DYNAMICS_URL = (
    "https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-time.html"
)
MAXIMUM_EXHAUSTIVE_ENERGY_AUDIT_ATOMS = 16


def _positive_real(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive real number")
    return float(value)


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


@dataclass(frozen=True, slots=True)
class PasqalParameters:
    """Published Pulser ``WeightedAnalogDevice`` reference limits.

    These are reference-device parameters, not a claim about the live state of
    a particular cloud QPU.  They are stored in every result for reproducible
    interpretation of the simulation.  The direct-coordinate QuTiP emulator
    cannot validate the device's register-layout and filling constraints.
    """

    rydberg_level: int = 75
    c6_over_hbar_rad_per_us_um6: float = 12_241_414.53
    minimum_atom_spacing_um: float = 5.0
    maximum_atoms: int = 256
    maximum_radial_distance_um: float = 80.0
    maximum_duration_us: float = 6.0
    maximum_runs: int = 500
    maximum_omega_rad_per_us: float = 4.0 * pi
    maximum_abs_detuning_rad_per_us: float = 20.0 * pi
    minimum_average_omega_rad_per_us: float = 0.6 * pi
    dmm_bottom_detuning_rad_per_us: float = -20.0 * pi
    requires_layout: bool = True
    minimum_layout_traps: int = 150
    maximum_layout_traps: int = 512
    minimum_layout_filling: float = 0.35
    maximum_layout_filling: float = 0.5
    source_urls: tuple[str, ...] = (
        PULSER_WEIGHTED_ANALOG_DEVICE_DOCS_URL,
        PULSER_MWIS_TUTORIAL_URL,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rydberg_level", _positive_integer(self.rydberg_level, "rydberg_level")
        )
        object.__setattr__(
            self, "maximum_atoms", _positive_integer(self.maximum_atoms, "maximum_atoms")
        )
        object.__setattr__(
            self, "maximum_runs", _positive_integer(self.maximum_runs, "maximum_runs")
        )
        object.__setattr__(
            self,
            "minimum_layout_traps",
            _positive_integer(self.minimum_layout_traps, "minimum_layout_traps"),
        )
        object.__setattr__(
            self,
            "maximum_layout_traps",
            _positive_integer(self.maximum_layout_traps, "maximum_layout_traps"),
        )
        for field_name in (
            "c6_over_hbar_rad_per_us_um6",
            "minimum_atom_spacing_um",
            "maximum_radial_distance_um",
            "maximum_duration_us",
            "maximum_omega_rad_per_us",
            "maximum_abs_detuning_rad_per_us",
            "minimum_average_omega_rad_per_us",
            "minimum_layout_filling",
            "maximum_layout_filling",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_real(getattr(self, field_name), field_name),
            )
        if self.minimum_atom_spacing_um >= 2.0 * self.maximum_radial_distance_um:
            raise ValueError("the radial limit must accommodate at least two atoms")
        dmm_bottom = float(self.dmm_bottom_detuning_rad_per_us)
        if not isfinite(dmm_bottom) or dmm_bottom >= 0.0:
            raise ValueError("dmm_bottom_detuning_rad_per_us must be finite and negative")
        object.__setattr__(self, "dmm_bottom_detuning_rad_per_us", dmm_bottom)
        if not isinstance(self.requires_layout, bool):
            raise ValueError("requires_layout must be boolean")
        if self.minimum_layout_traps > self.maximum_layout_traps:
            raise ValueError("minimum_layout_traps cannot exceed maximum_layout_traps")
        if not 0.0 < self.minimum_layout_filling <= self.maximum_layout_filling <= 1.0:
            raise ValueError("layout filling bounds must satisfy 0 < minimum <= maximum <= 1")
        urls = tuple(self.source_urls)
        if not urls or any(not isinstance(url, str) or not url.startswith("https://") for url in urls):
            raise ValueError("source_urls must contain authoritative HTTPS URLs")
        object.__setattr__(self, "source_urls", urls)

    def to_dict(self) -> dict[str, object]:
        return {
            "rydberg_level": self.rydberg_level,
            "c6_over_hbar_rad_per_us_um6": self.c6_over_hbar_rad_per_us_um6,
            "minimum_atom_spacing_um": self.minimum_atom_spacing_um,
            "maximum_atoms": self.maximum_atoms,
            "maximum_radial_distance_um": self.maximum_radial_distance_um,
            "maximum_duration_us": self.maximum_duration_us,
            "maximum_runs": self.maximum_runs,
            "maximum_omega_rad_per_us": self.maximum_omega_rad_per_us,
            "maximum_abs_detuning_rad_per_us": self.maximum_abs_detuning_rad_per_us,
            "minimum_average_omega_rad_per_us": self.minimum_average_omega_rad_per_us,
            "dmm_bottom_detuning_rad_per_us": self.dmm_bottom_detuning_rad_per_us,
            "requires_layout": self.requires_layout,
            "minimum_layout_traps": self.minimum_layout_traps,
            "maximum_layout_traps": self.maximum_layout_traps,
            "minimum_layout_filling": self.minimum_layout_filling,
            "maximum_layout_filling": self.maximum_layout_filling,
            "source_urls": list(self.source_urls),
        }


@dataclass(frozen=True, slots=True)
class AdiabaticPulse:
    """A smooth four-microsecond sweep within WeightedAnalogDevice caps."""

    duration_us: float = 4.0
    ramp_duration_us: float = 0.4
    omega_peak_rad_per_us: float = 2.0 * pi
    # The final reward is twice Omega.  This leaves an interaction-safe
    # constraint radius compatible with the 5 um tweezer spacing.
    detuning_span_rad_per_us: float = 4.0 * pi
    time_steps: int = 101

    def __post_init__(self) -> None:
        for field_name in (
            "duration_us",
            "ramp_duration_us",
            "omega_peak_rad_per_us",
            "detuning_span_rad_per_us",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_real(getattr(self, field_name), field_name),
            )
        if 2.0 * self.ramp_duration_us > self.duration_us:
            raise ValueError("two pulse ramps must fit inside duration_us")
        object.__setattr__(
            self, "time_steps", _positive_integer(self.time_steps, "time_steps", minimum=3)
        )

    def validate_against(self, parameters: PasqalParameters) -> None:
        """Raise when any waveform or duration exceeds the reference device."""

        if self.duration_us > parameters.maximum_duration_us:
            raise ValueError("pulse duration exceeds the Pasqal reference limit")
        if self.omega_peak_rad_per_us > parameters.maximum_omega_rad_per_us:
            raise ValueError("pulse Omega exceeds the Pasqal reference limit")
        if self.average_omega_rad_per_us < parameters.minimum_average_omega_rad_per_us:
            raise ValueError("pulse average Omega is below the Pasqal reference minimum")
        if self.detuning_span_rad_per_us > parameters.maximum_abs_detuning_rad_per_us:
            raise ValueError("global detuning exceeds the Pasqal reference limit")
        if -self.detuning_span_rad_per_us < parameters.dmm_bottom_detuning_rad_per_us:
            raise ValueError("DMM detuning exceeds the Pasqal reference bottom limit")

    @property
    def average_omega_rad_per_us(self) -> float:
        """Time-average of the sin-squared ramps and constant plateau."""

        return self.omega_peak_rad_per_us * (
            1.0 - self.ramp_duration_us / self.duration_us
        )

    def _time(self, time_us: float) -> float:
        if (
            isinstance(time_us, bool)
            or not isinstance(time_us, Real)
            or not isfinite(float(time_us))
            or not 0.0 <= float(time_us) <= self.duration_us
        ):
            raise ValueError("time_us must lie within the pulse duration")
        return float(time_us)

    def omega(self, time_us: float) -> float:
        """Smooth turn-on, constant drive, and smooth turn-off."""

        time_us = self._time(time_us)
        if time_us < self.ramp_duration_us:
            phase = 0.5 * pi * time_us / self.ramp_duration_us
            return self.omega_peak_rad_per_us * sin(phase) ** 2
        if time_us > self.duration_us - self.ramp_duration_us:
            remaining = self.duration_us - time_us
            phase = 0.5 * pi * remaining / self.ramp_duration_us
            return self.omega_peak_rad_per_us * sin(phase) ** 2
        return self.omega_peak_rad_per_us

    def detuning(self, time_us: float) -> float:
        """Linear global detuning sweep from negative to positive."""

        time_us = self._time(time_us)
        fraction = time_us / self.duration_us
        return self.detuning_span_rad_per_us * (2.0 * fraction - 1.0)

    def dmm_detuning(self, time_us: float) -> float:
        """Negative local detuning used with ``epsilon_i = 1 - w_i/w_max``."""

        self._time(time_us)
        return -self.detuning_span_rad_per_us

    @property
    def times_us(self) -> np.ndarray:
        return np.linspace(0.0, self.duration_us, self.time_steps, dtype=float)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "duration_us": self.duration_us,
            "ramp_duration_us": self.ramp_duration_us,
            "omega_peak_rad_per_us": self.omega_peak_rad_per_us,
            "average_omega_rad_per_us": self.average_omega_rad_per_us,
            "detuning_span_rad_per_us": self.detuning_span_rad_per_us,
            "time_steps": self.time_steps,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingDiagnostics:
    """Physical register with topology and weighted-energy fidelity audits."""

    node_ids: tuple[int, ...]
    positions_um: tuple[tuple[float, float], ...]
    constraint_radius_um: float
    drive_blockade_radius_um: float
    expected_edges: tuple[tuple[int, int], ...]
    realized_edges: tuple[tuple[int, int], ...]
    missing_edges: tuple[tuple[int, int], ...]
    spurious_edges: tuple[tuple[int, int], ...]
    minimum_spacing_um: float | None
    maximum_radius_um: float
    spacing_valid: bool
    radius_valid: bool
    topology_fidelity: bool
    energy_audit_complete: bool
    weighted_objective_fidelity: bool | None
    exact_fidelity: bool
    optimization_cost: float
    minimum_edge_interaction_rad_per_us: float | None
    maximum_nonedge_interaction_rad_per_us: float | None
    maximum_nonedge_to_minimum_reward_ratio: float | None
    detuning_penalty_rad_per_us: float
    minimum_interaction_to_detuning_ratio: float | None
    abstract_optimal_node_ids: tuple[int, ...] | None
    physical_ground_node_ids: tuple[int, ...] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "node_ids": list(self.node_ids),
            "positions_um": [list(position) for position in self.positions_um],
            "constraint_radius_um": self.constraint_radius_um,
            "drive_blockade_radius_um": self.drive_blockade_radius_um,
            "expected_edges": [list(edge) for edge in self.expected_edges],
            "realized_edges": [list(edge) for edge in self.realized_edges],
            "missing_edges": [list(edge) for edge in self.missing_edges],
            "spurious_edges": [list(edge) for edge in self.spurious_edges],
            "minimum_spacing_um": self.minimum_spacing_um,
            "maximum_radius_um": self.maximum_radius_um,
            "spacing_valid": self.spacing_valid,
            "radius_valid": self.radius_valid,
            "topology_fidelity": self.topology_fidelity,
            "energy_audit_complete": self.energy_audit_complete,
            "weighted_objective_fidelity": self.weighted_objective_fidelity,
            "exact_fidelity": self.exact_fidelity,
            "optimization_cost": self.optimization_cost,
            "minimum_edge_interaction_rad_per_us": self.minimum_edge_interaction_rad_per_us,
            "maximum_nonedge_interaction_rad_per_us": self.maximum_nonedge_interaction_rad_per_us,
            "maximum_nonedge_to_minimum_reward_ratio": self.maximum_nonedge_to_minimum_reward_ratio,
            "detuning_penalty_rad_per_us": self.detuning_penalty_rad_per_us,
            "minimum_interaction_to_detuning_ratio": self.minimum_interaction_to_detuning_ratio,
            "abstract_optimal_node_ids": (
                list(self.abstract_optimal_node_ids)
                if self.abstract_optimal_node_ids is not None
                else None
            ),
            "physical_ground_node_ids": (
                list(self.physical_ground_node_ids)
                if self.physical_ground_node_ids is not None
                else None
            ),
        }


def _special_small_layout(
    node_ids: tuple[int, ...],
    edges: set[tuple[int, int]],
    edge_length: float,
    nonedge_length: float,
) -> np.ndarray | None:
    """Exact constructive layouts for every graph on at most three vertices."""

    count = len(node_ids)
    if count == 1:
        return np.zeros((1, 2), dtype=float)
    if count == 2:
        distance = edge_length if tuple(node_ids) in edges else nonedge_length
        return np.asarray(((-distance / 2.0, 0.0), (distance / 2.0, 0.0)))
    if count != 3:
        return None

    edge_count = len(edges)
    if edge_count in (0, 3):
        side = edge_length if edge_count == 3 else nonedge_length
        height = sqrt(3.0) * side / 2.0
        return np.asarray(((-side / 2.0, -height / 3.0),
                           (side / 2.0, -height / 3.0),
                           (0.0, 2.0 * height / 3.0)))
    if edge_count == 2:
        degrees = {
            node_id: sum(node_id in edge for edge in edges) for node_id in node_ids
        }
        center = max(node_ids, key=lambda node_id: (degrees[node_id], -node_id))
        leaves = sorted(set(node_ids) - {center})
        coordinates = {
            center: (0.0, 0.0),
            leaves[0]: (-edge_length, 0.0),
            leaves[1]: (edge_length, 0.0),
        }
        return np.asarray([coordinates[node_id] for node_id in node_ids], dtype=float)

    edge = next(iter(edges))
    isolated = next(node_id for node_id in node_ids if node_id not in edge)
    coordinates = {
        edge[0]: (-edge_length / 2.0, 0.0),
        edge[1]: (edge_length / 2.0, 0.0),
        isolated: (0.0, nonedge_length),
    }
    result = np.asarray([coordinates[node_id] for node_id in node_ids], dtype=float)
    return result - result.mean(axis=0)


def _initial_layouts(
    count: int,
    edge_length: float,
    nonedge_length: float,
    seed: int,
) -> list[np.ndarray]:
    columns = ceil(sqrt(count))
    grid = np.asarray(
        [
            ((index % columns) * edge_length, (index // columns) * edge_length)
            for index in range(count)
        ],
        dtype=float,
    )
    grid -= grid.mean(axis=0)
    radius = max(nonedge_length, edge_length * count / (2.0 * pi))
    circle = np.asarray(
        [
            (
                radius * cos(2.0 * pi * index / count),
                radius * sin(2.0 * pi * index / count),
            )
            for index in range(count)
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    layouts = [grid, circle]
    for scale in (edge_length, nonedge_length, 0.5 * (edge_length + nonedge_length)):
        for _ in range(3):
            layout = rng.normal(0.0, scale, size=(count, 2))
            layout -= layout.mean(axis=0)
            layouts.append(layout)
    return layouts


def _dmm_encoding(
    weights: np.ndarray, pulse: AdiabaticPulse
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized weights, DMM scales, and final diagonal rewards."""

    normalized_weights = weights / float(np.max(weights))
    epsilon = 1.0 - normalized_weights
    final_rewards = (
        pulse.detuning(pulse.duration_us)
        + epsilon * pulse.dmm_detuning(pulse.duration_us)
    )
    return normalized_weights, epsilon, np.asarray(final_rewards, dtype=float)


def embed_unit_disk(
    solver_input: SolverInput,
    parameters: PasqalParameters | None = None,
    pulse: AdiabaticPulse | None = None,
    *,
    maximum_energy_audit_atoms: int = MAXIMUM_EXHAUSTIVE_ENERGY_AUDIT_ATOMS,
) -> EmbeddingDiagnostics:
    """Deterministically embed one solver cluster into a physical 2-D register.

    The returned audit compares every abstract edge and non-edge with the
    threshold graph induced by the physical blockade radius.  It also
    enumerates the actual final diagonal Hamiltonian and verifies that every
    physical ground state is an abstract MWIS.  Backends must not simulate a
    graph unless ``exact_fidelity`` is true.
    """

    if not isinstance(solver_input, SolverInput):
        raise TypeError("solver_input must be a SolverInput")
    parameters = PasqalParameters() if parameters is None else parameters
    pulse = AdiabaticPulse() if pulse is None else pulse
    if not isinstance(parameters, PasqalParameters):
        raise TypeError("parameters must be PasqalParameters")
    if not isinstance(pulse, AdiabaticPulse):
        raise TypeError("pulse must be AdiabaticPulse")
    maximum_energy_audit_atoms = _positive_integer(
        maximum_energy_audit_atoms, "maximum_energy_audit_atoms"
    )
    if maximum_energy_audit_atoms > MAXIMUM_EXHAUSTIVE_ENERGY_AUDIT_ATOMS:
        raise ValueError(
            "maximum_energy_audit_atoms exceeds the bounded exhaustive-audit limit"
        )
    pulse.validate_against(parameters)

    node_ids = tuple(node.node_id for node in solver_input.nodes)
    index = {node_id: offset for offset, node_id in enumerate(node_ids)}
    expected_edges = tuple(sorted(solver_input.edges))
    expected_set = set(expected_edges)
    indexed_edges = {(index[left], index[right]) for left, right in expected_edges}
    drive_blockade_radius = (
        parameters.c6_over_hbar_rad_per_us_um6 / pulse.omega_peak_rad_per_us
    ) ** (1.0 / 6.0)
    # An edge is a hard MWIS constraint only when its interaction penalty is at
    # least the largest positive per-node detuning reward.  Omega controls the
    # dynamics, but using it as the graph threshold would be physically wrong
    # whenever the final detuning is larger than Omega.
    constraint_radius = (
        parameters.c6_over_hbar_rad_per_us_um6 / pulse.detuning_span_rad_per_us
    ) ** (1.0 / 6.0)
    minimum_spacing = parameters.minimum_atom_spacing_um
    edge_length = max(1.05 * minimum_spacing, 0.78 * constraint_radius)
    nonedge_length = 1.20 * constraint_radius

    special = _special_small_layout(
        node_ids,
        expected_set,
        edge_length=edge_length,
        nonedge_length=nonedge_length,
    )
    best_positions = special
    best_cost = 0.0

    if best_positions is None:
        pairs = tuple(itertools.combinations(range(len(node_ids)), 2))
        edge_upper = 0.92 * constraint_radius
        nonedge_lower = 1.08 * constraint_radius
        radius_limit = 0.98 * parameters.maximum_radial_distance_um

        def objective(flat: np.ndarray) -> float:
            positions = flat.reshape(len(node_ids), 2)
            center = positions.mean(axis=0)
            cost = 0.02 * float(center @ center)
            for left, right in pairs:
                distance = float(np.linalg.norm(positions[left] - positions[right]))
                if distance < minimum_spacing:
                    cost += 20.0 * (minimum_spacing - distance) ** 2
                pair = (left, right)
                if pair in indexed_edges and distance > edge_upper:
                    cost += 10.0 * (distance - edge_upper) ** 2
                elif pair not in indexed_edges and distance < nonedge_lower:
                    cost += 10.0 * (nonedge_lower - distance) ** 2
            for position in positions:
                radius = float(np.linalg.norm(position - center))
                if radius > radius_limit:
                    cost += 20.0 * (radius - radius_limit) ** 2
            return cost

        seed_bytes = sha256(
            f"{solver_input.fingerprint}:pasqal-unit-disk-v1".encode("ascii")
        ).digest()[:8]
        seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
        best_result = None
        for initial in _initial_layouts(
            len(node_ids), edge_length, nonedge_length, seed
        ):
            result = minimize(
                objective,
                initial.ravel(),
                method="L-BFGS-B",
                options={"maxiter": 800, "ftol": 1e-12},
            )
            if best_result is None or float(result.fun) < float(best_result.fun):
                best_result = result
        assert best_result is not None
        best_positions = best_result.x.reshape(len(node_ids), 2)
        best_positions -= best_positions.mean(axis=0)
        best_cost = float(best_result.fun)

    best_positions = np.asarray(best_positions, dtype=float)
    if len(best_positions):
        best_positions -= best_positions.mean(axis=0)

    realized: set[tuple[int, int]] = set()
    separations: list[float] = []
    edge_interactions: list[float] = []
    nonedge_interactions: list[float] = []
    indexed_interactions: dict[tuple[int, int], float] = {}
    for left_index, right_index in itertools.combinations(range(len(node_ids)), 2):
        distance = float(
            np.linalg.norm(best_positions[left_index] - best_positions[right_index])
        )
        separations.append(distance)
        node_pair = (node_ids[left_index], node_ids[right_index])
        interaction = (
            parameters.c6_over_hbar_rad_per_us_um6 / max(distance, 1e-12) ** 6
        )
        indexed_interactions[(left_index, right_index)] = interaction
        if node_pair in expected_set:
            edge_interactions.append(interaction)
        else:
            nonedge_interactions.append(interaction)
        if distance <= constraint_radius:
            realized.add((node_ids[left_index], node_ids[right_index]))
    realized_edges = tuple(sorted(realized))
    missing_edges = tuple(sorted(expected_set - realized))
    spurious_edges = tuple(sorted(realized - expected_set))
    min_spacing = min(separations) if separations else None
    max_radius = max(
        (float(np.linalg.norm(position)) for position in best_positions), default=0.0
    )
    spacing_valid = min_spacing is None or min_spacing + 1e-7 >= minimum_spacing
    radius_valid = max_radius <= parameters.maximum_radial_distance_um + 1e-7
    minimum_edge_interaction = min(edge_interactions) if edge_interactions else None
    maximum_nonedge_interaction = (
        max(nonedge_interactions) if nonedge_interactions else None
    )
    interaction_ratio = (
        minimum_edge_interaction / pulse.detuning_span_rad_per_us
        if minimum_edge_interaction is not None
        else None
    )
    interaction_valid = interaction_ratio is None or interaction_ratio >= 1.0
    topology_fidelity = not missing_edges and not spurious_edges

    weights = np.asarray([node.weight for node in solver_input.nodes], dtype=float)
    energy_audit_complete = (
        len(node_ids) <= maximum_energy_audit_atoms
        and bool(np.isfinite(weights).all())
        and bool(np.all(weights > 0.0))
    )
    weighted_objective_fidelity: bool | None = None
    abstract_optimal_node_ids: tuple[int, ...] | None = None
    physical_ground_node_ids: tuple[int, ...] | None = None
    maximum_nonedge_to_minimum_reward_ratio: float | None = None
    if energy_audit_complete:
        normalized_weights, _epsilon, final_rewards = _dmm_encoding(weights, pulse)
        # A positive abstract weight can be too small for binary64 to survive
        # the DMM normalization/cancellation. Such an input has no faithful
        # representation in this emulator and must never be certified exact.
        dmm_dynamic_range_valid = bool(np.all(normalized_weights > 0.0)) and bool(
            np.all(final_rewards > 0.0)
        )
        energy_audit_complete = energy_audit_complete and dmm_dynamic_range_valid
        minimum_reward = float(np.min(final_rewards))
        if maximum_nonedge_interaction is not None and minimum_reward > 0.0:
            maximum_nonedge_to_minimum_reward_ratio = (
                float(maximum_nonedge_interaction) / minimum_reward
            )

        if energy_audit_complete:
            abstract_best = -float("inf")
            abstract_optima: list[tuple[int, ...]] = []
            physical_best = float("inf")
            physical_optima: list[tuple[int, ...]] = []
            for state_index in range(1 << len(node_ids)):
                bits = format(state_index, f"0{len(node_ids)}b")
                selected_indices = tuple(
                    offset for offset, bit in enumerate(bits) if bit == "1"
                )
                selected_node_ids = tuple(
                    node_ids[offset] for offset in selected_indices
                )
                independent = not any(
                    bits[left] == "1" and bits[right] == "1"
                    for left, right in indexed_edges
                )
                if independent:
                    abstract_score = fsum(
                        float(weights[offset]) for offset in selected_indices
                    )
                    if abstract_score > abstract_best:
                        abstract_best = abstract_score
                        abstract_optima = [selected_node_ids]
                    elif abstract_score == abstract_best:
                        abstract_optima.append(selected_node_ids)

                energy_terms = [
                    -float(final_rewards[offset]) for offset in selected_indices
                ]
                energy_terms.extend(
                    indexed_interactions[(left, right)]
                    for left, right in itertools.combinations(selected_indices, 2)
                )
                physical_energy = fsum(energy_terms)
                if physical_energy < physical_best:
                    physical_best = physical_energy
                    physical_optima = [selected_node_ids]
                elif physical_energy == physical_best:
                    physical_optima.append(selected_node_ids)

            abstract_optima = sorted(set(abstract_optima))
            physical_optima = sorted(set(physical_optima))
            abstract_optimal_node_ids = abstract_optima[0]
            physical_ground_node_ids = physical_optima[0]
            weighted_objective_fidelity = set(physical_optima).issubset(
                set(abstract_optima)
            )
        else:
            weighted_objective_fidelity = False

    exact = (
        topology_fidelity
        and spacing_valid
        and radius_valid
        and interaction_valid
        and energy_audit_complete
        and weighted_objective_fidelity is True
    )

    return EmbeddingDiagnostics(
        node_ids=node_ids,
        positions_um=tuple(
            (float(position[0]), float(position[1])) for position in best_positions
        ),
        constraint_radius_um=float(constraint_radius),
        drive_blockade_radius_um=float(drive_blockade_radius),
        expected_edges=expected_edges,
        realized_edges=realized_edges,
        missing_edges=missing_edges,
        spurious_edges=spurious_edges,
        minimum_spacing_um=float(min_spacing) if min_spacing is not None else None,
        maximum_radius_um=float(max_radius),
        spacing_valid=bool(spacing_valid),
        radius_valid=bool(radius_valid),
        topology_fidelity=bool(topology_fidelity),
        energy_audit_complete=bool(energy_audit_complete),
        weighted_objective_fidelity=weighted_objective_fidelity,
        exact_fidelity=bool(exact),
        optimization_cost=float(best_cost),
        minimum_edge_interaction_rad_per_us=(
            float(minimum_edge_interaction)
            if minimum_edge_interaction is not None
            else None
        ),
        maximum_nonedge_interaction_rad_per_us=(
            float(maximum_nonedge_interaction)
            if maximum_nonedge_interaction is not None
            else None
        ),
        maximum_nonedge_to_minimum_reward_ratio=(
            float(maximum_nonedge_to_minimum_reward_ratio)
            if maximum_nonedge_to_minimum_reward_ratio is not None
            else None
        ),
        detuning_penalty_rad_per_us=float(pulse.detuning_span_rad_per_us),
        minimum_interaction_to_detuning_ratio=(
            float(interaction_ratio) if interaction_ratio is not None else None
        ),
        abstract_optimal_node_ids=abstract_optimal_node_ids,
        physical_ground_node_ids=physical_ground_node_ids,
    )


def _import_qutip() -> Any:
    """Import QuTiP only when the neutral-atom backend is actually invoked."""

    import qutip  # type: ignore[import-not-found]

    return qutip


def _operator_at(qutip: Any, operator: Any, site: int, count: int) -> Any:
    factors = [qutip.qeye(2) for _ in range(count)]
    factors[site] = operator
    return qutip.tensor(factors)


def _independent(index: int, count: int, indexed_edges: tuple[tuple[int, int], ...]) -> bool:
    bits = format(index, f"0{count}b")
    return not any(bits[left] == "1" and bits[right] == "1" for left, right in indexed_edges)


class NeutralAtomBackend:
    """Solve one weighted conflict-graph cluster by direct QuTiP evolution."""

    name = "neutral_atom_qutip"

    def __init__(
        self,
        *,
        parameters: PasqalParameters | None = None,
        pulse: AdiabaticPulse | None = None,
        maximum_simulation_atoms: int = 10,
    ) -> None:
        self.parameters = PasqalParameters() if parameters is None else parameters
        self.pulse = AdiabaticPulse() if pulse is None else pulse
        if not isinstance(self.parameters, PasqalParameters):
            raise TypeError("parameters must be PasqalParameters")
        if not isinstance(self.pulse, AdiabaticPulse):
            raise TypeError("pulse must be AdiabaticPulse")
        self.pulse.validate_against(self.parameters)
        maximum_simulation_atoms = _positive_integer(
            maximum_simulation_atoms, "maximum_simulation_atoms"
        )
        if maximum_simulation_atoms > self.parameters.maximum_atoms:
            raise ValueError("simulation atom limit cannot exceed the hardware atom limit")
        if maximum_simulation_atoms > MAXIMUM_EXHAUSTIVE_ENERGY_AUDIT_ATOMS:
            raise ValueError(
                "simulation atom limit cannot exceed the bounded exhaustive-energy-audit limit"
            )
        self.maximum_simulation_atoms = maximum_simulation_atoms

    def _result(
        self,
        solver_input: SolverInput,
        started: float,
        *,
        status: str,
        selected_ids: tuple[int, ...] = (),
        diagnostics: dict[str, object],
    ) -> SolverResult:
        selected = set(selected_ids)
        objective = fsum(
            node.weight for node in solver_input.nodes if node.node_id in selected
        )
        result = SolverResult(
            problem_id=solver_input.problem_id,
            input_fingerprint=solver_input.fingerprint,
            backend=self.name,
            selected_ids=tuple(sorted(selected_ids)),
            objective=float(objective),
            feasible=True,
            status=status,
            runtime_seconds=perf_counter() - started,
            diagnostics=diagnostics,
        )
        validate_result(solver_input, result)
        return result

    def solve(self, solver_input: SolverInput) -> SolverResult:
        started = perf_counter()
        if not isinstance(solver_input, SolverInput):
            raise TypeError("solver_input must be a SolverInput")
        nodes = tuple(solver_input.nodes)
        common_diagnostics: dict[str, object] = {
            "node_count": len(nodes),
            "edge_count": len(solver_input.edges),
            "parameters": self.parameters.to_dict(),
            "pulse": self.pulse.to_dict(),
            "qutip_reference_url": QUTIP_TIME_DEPENDENT_DYNAMICS_URL,
            "basis_convention": {"ground": "|0>", "rydberg": "|1>"},
            "physical_units": {
                "position": "um",
                "time": "us",
                "angular_frequency": "rad/us",
            },
            "distribution_retained": False,
            "hardware_execution_readiness": {
                "validated": False,
                "reason": (
                    "direct-coordinate QuTiP emulation does not validate the "
                    "required hardware register layout and filling constraints"
                ),
                "requires_layout": self.parameters.requires_layout,
            },
        }
        if len(nodes) > self.parameters.maximum_atoms:
            return self._result(
                solver_input,
                started,
                status="unsupported_size",
                diagnostics={
                    **common_diagnostics,
                    "reason": "hardware_atom_limit",
                    "maximum_atoms": self.parameters.maximum_atoms,
                    "solved": False,
                },
            )
        if len(nodes) > self.maximum_simulation_atoms:
            return self._result(
                solver_input,
                started,
                status="unsupported_size",
                diagnostics={
                    **common_diagnostics,
                    "reason": "state_vector_simulation_limit",
                    "maximum_simulation_atoms": self.maximum_simulation_atoms,
                    "solved": False,
                },
            )
        weights = np.asarray([node.weight for node in nodes], dtype=float)
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            return self._result(
                solver_input,
                started,
                status="invalid_weights",
                diagnostics={
                    **common_diagnostics,
                    "reason": "DMM encoding requires strictly positive weights",
                    "solved": False,
                },
            )
        normalized_weights, epsilon, final_rewards = _dmm_encoding(
            weights, self.pulse
        )
        if np.any(normalized_weights <= 0.0) or np.any(final_rewards <= 0.0):
            return self._result(
                solver_input,
                started,
                status="invalid_weights",
                diagnostics={
                    **common_diagnostics,
                    "reason": (
                        "DMM encoding cannot represent the supplied weight "
                        "dynamic range in binary64"
                    ),
                    "solved": False,
                },
            )

        embedding = embed_unit_disk(
            solver_input,
            self.parameters,
            self.pulse,
            maximum_energy_audit_atoms=self.maximum_simulation_atoms,
        )
        diagnostics = {**common_diagnostics, "embedding": embedding.to_dict()}
        if not embedding.exact_fidelity:
            return self._result(
                solver_input,
                started,
                status="embedding_error",
                diagnostics={**diagnostics, "solved": False},
            )

        try:
            qutip = _import_qutip()
        except (ImportError, ModuleNotFoundError):
            return self._result(
                solver_input,
                started,
                status="dependency_missing",
                diagnostics={
                    **diagnostics,
                    "dependency": "qutip",
                    "solved": False,
                },
            )

        diagnostics = {
            **diagnostics,
            "qutip_version": str(getattr(qutip, "__version__", "unknown")),
        }

        node_ids = tuple(node.node_id for node in nodes)
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        indexed_edges = tuple(
            (node_index[left], node_index[right]) for left, right in solver_input.edges
        )
        ground = qutip.basis(2, 0)
        rydberg = qutip.basis(2, 1)
        number_single = rydberg * rydberg.dag()
        x_single = ground * rydberg.dag() + rydberg * ground.dag()
        number_ops = [
            _operator_at(qutip, number_single, site, len(nodes))
            for site in range(len(nodes))
        ]
        x_ops = [
            _operator_at(qutip, x_single, site, len(nodes))
            for site in range(len(nodes))
        ]
        zero = 0.0 * number_ops[0]
        interaction = zero
        positions = np.asarray(embedding.positions_um, dtype=float)
        for left, right in itertools.combinations(range(len(nodes)), 2):
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            coupling = self.parameters.c6_over_hbar_rad_per_us_um6 / distance**6
            interaction = interaction + coupling * number_ops[left] * number_ops[right]
        drive = zero
        total_number = zero
        weighted_number = zero
        for index, (number_op, x_op) in enumerate(zip(number_ops, x_ops, strict=True)):
            drive = drive + 0.5 * x_op
            total_number = total_number + number_op
            weighted_number = weighted_number + float(epsilon[index]) * number_op

        def omega_coefficient(time_us: float, _args: object = None) -> float:
            # Adaptive ODE solvers may probe a few ulps outside the supplied
            # interval.  Clamp those internal probes while keeping the public
            # pulse methods strict about invalid user input.
            bounded = min(self.pulse.duration_us, max(0.0, float(time_us)))
            return self.pulse.omega(bounded)

        def detuning_coefficient(time_us: float, _args: object = None) -> float:
            bounded = min(self.pulse.duration_us, max(0.0, float(time_us)))
            return -self.pulse.detuning(bounded)

        def dmm_coefficient(time_us: float, _args: object = None) -> float:
            bounded = min(self.pulse.duration_us, max(0.0, float(time_us)))
            return -self.pulse.dmm_detuning(bounded)

        hamiltonian = [
            interaction,
            [drive, omega_coefficient],
            [total_number, detuning_coefficient],
            [weighted_number, dmm_coefficient],
        ]
        initial_state = qutip.tensor([ground for _ in nodes])
        try:
            evolution = qutip.sesolve(hamiltonian, initial_state, self.pulse.times_us)
            amplitudes = np.asarray(evolution.states[-1].full()).reshape(-1)
        except Exception as exc:  # pragma: no cover - depends on optional solver internals
            return self._result(
                solver_input,
                started,
                status="simulation_error",
                diagnostics={
                    **diagnostics,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "solved": False,
                },
            )
        probabilities = np.abs(amplitudes) ** 2
        probability_sum = float(np.sum(probabilities))
        if probability_sum <= 0.0 or not np.isfinite(probability_sum):
            return self._result(
                solver_input,
                started,
                status="simulation_error",
                diagnostics={
                    **diagnostics,
                    "error_type": "InvalidStateNorm",
                    "solved": False,
                },
            )
        probabilities /= probability_sum

        order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
        chosen_index = next(
            index
            for index in order
            if _independent(index, len(nodes), indexed_edges)
        )
        bits = format(chosen_index, f"0{len(nodes)}b")
        selected_ids = tuple(
            node_ids[index] for index, bit in enumerate(bits) if bit == "1"
        )
        infeasible_probability = float(
            sum(
                probabilities[index]
                for index in range(len(probabilities))
                if not _independent(index, len(nodes), indexed_edges)
            )
        )
        return self._result(
            solver_input,
            started,
            status="simulated",
            selected_ids=selected_ids,
            diagnostics={
                **diagnostics,
                "solved": True,
                "selected_bitstring": bits,
                "selected_probability": float(probabilities[chosen_index]),
                "discarded_infeasible_probability": infeasible_probability,
                "epsilon": [float(value) for value in epsilon],
                "state_dimension": int(len(probabilities)),
                "distribution_retained": False,
            },
        )
