"""Run the neutral-atom MWIS attempt behind the shared solver contract.

``QuantumSolver`` accepts one complete frame graph, factors disconnected
components internally, and maps every sampled bit back to the graph's original
node identifiers. The numerical embedding and pulse construction remain close
to the original Pulser experiment. Pulser itself is loaded only by the concrete
runner, so the package and its classical solver have no quantum requirement.

Plotting is deliberately absent. Immutable program and run artifacts can be
passed to :mod:`neutral_atom_visualization` when a caller
explicitly wants figures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import fsum, isfinite
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Protocol

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean, pdist, squareform

from graph import GraphCluster, cluster_graph
from solver import SUCCESS_STATUSES, Solver, SolverInput, SolverSelection


@dataclass(frozen=True, slots=True)
class NeutralAtomConfig:
    """Numerical settings inherited from the original quantum attempt."""

    random_seed: int = 0
    mapping_tolerance: float = 1e-6
    mapping_max_iterations: int = 200_000
    pulse_duration_ns: int = 40_000
    interaction_scale: float = 10.0
    maximum_component_nodes: int = 16
    qutip_cache_dir: Path = Path("data") / ".cache" / "qutip"

    def __post_init__(self) -> None:
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise ValueError("random_seed must be between 0 and 2**32 - 1")
        if not isfinite(self.mapping_tolerance) or self.mapping_tolerance <= 0.0:
            raise ValueError("mapping_tolerance must be finite and positive")
        if self.mapping_max_iterations < 1:
            raise ValueError("mapping_max_iterations must be positive")
        if self.pulse_duration_ns < 1:
            raise ValueError("pulse_duration_ns must be positive")
        if not isfinite(self.interaction_scale) or self.interaction_scale <= 0.0:
            raise ValueError("interaction_scale must be finite and positive")
        if self.maximum_component_nodes < 1:
            raise ValueError("maximum_component_nodes must be positive")


@dataclass(frozen=True, slots=True)
class NeutralAtomComponent:
    """One internally factored graph component in stable qubit order."""

    component_id: int
    node_ids: tuple[int, ...]
    weights: tuple[float, ...]
    edges: tuple[tuple[int, int], ...]
    matrix: tuple[tuple[float, ...], ...]

    @property
    def qubit_ids(self) -> tuple[str, ...]:
        """Pulser labels aligned position-for-position with ``node_ids``."""

        return tuple(f"q{index}" for index in range(len(self.node_ids)))


@dataclass(frozen=True, slots=True)
class NeutralAtomProgram:
    """Built Pulser objects and embedding metadata for optional inspection."""

    component: NeutralAtomComponent
    coordinates: tuple[tuple[float, float], ...]
    mapping_cost: float
    mapping_success: bool
    omega: float
    register: object = field(repr=False, compare=False)
    detuning_map: object = field(repr=False, compare=False)
    sequence: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class NeutralAtomRun:
    """Raw samples plus the graph selection decoded from one component."""

    component_id: int
    node_ids: tuple[int, ...]
    atom_order: tuple[str, ...]
    bitstring_counts: tuple[tuple[str, int], ...]
    coordinates: tuple[tuple[float, float], ...]
    mapping_cost: float
    mapping_success: bool
    program: NeutralAtomProgram | None = field(default=None, repr=False, compare=False)
    execution_mode: str = "runner"
    selected_bitstring: str = ""
    selected_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        counts = tuple(
            sorted(
                (str(bitstring), int(count))
                for bitstring, count in self.bitstring_counts
            )
        )
        if len({bitstring for bitstring, _ in counts}) != len(counts):
            raise ValueError("bitstring_counts must contain unique bitstrings")
        if any(count < 1 for _, count in counts):
            raise ValueError("bitstring counts must be positive")
        coordinates = tuple(
            (float(coordinate[0]), float(coordinate[1]))
            for coordinate in self.coordinates
        )
        if len(coordinates) != len(self.node_ids):
            raise ValueError("coordinates must align with component node IDs")
        mapping_cost = float(self.mapping_cost)
        if not isfinite(mapping_cost) or mapping_cost < 0.0:
            raise ValueError("mapping_cost must be finite and non-negative")
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(
            self,
            "atom_order",
            tuple(str(item) for item in self.atom_order),
        )
        object.__setattr__(self, "bitstring_counts", counts)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "mapping_cost", mapping_cost)
        object.__setattr__(self, "mapping_success", bool(self.mapping_success))
        object.__setattr__(self, "selected_ids", tuple(sorted(self.selected_ids)))


@dataclass(frozen=True, slots=True)
class NeutralAtomExecution:
    """One atomic full-frame execution, including inspectable component runs."""

    problem_id: str
    input_fingerprint: str
    selected_ids: tuple[int, ...]
    status: str
    runs: tuple[NeutralAtomRun, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_ids", tuple(sorted(self.selected_ids)))
        object.__setattr__(self, "runs", tuple(self.runs))
        frozen = SolverSelection(diagnostics=self.diagnostics).diagnostics
        object.__setattr__(self, "diagnostics", frozen)

    def to_selection(self) -> SolverSelection:
        """Discard vendor artifacts and enter the common solver output format."""

        return SolverSelection(
            selected_ids=self.selected_ids,
            status=self.status,
            diagnostics=self.diagnostics,
        )

    @property
    def successful(self) -> bool:
        """Whether this execution can safely enter the tracking update."""

        return self.status in SUCCESS_STATUSES


class NeutralAtomRunner(Protocol):
    """Structural interface for one component-level quantum executor."""

    def execute(self, component: NeutralAtomComponent) -> NeutralAtomRun:
        """Build, run, and return samples for one graph component."""


class NeutralAtomRunError(RuntimeError):
    """Base class for expected component-run failures with solver statuses."""

    status = "execution_error"


class NeutralAtomDependencyError(NeutralAtomRunError):
    """Raised when the optional Pulser simulation stack is unavailable."""

    status = "dependency_missing"


class NeutralAtomExecutionError(NeutralAtomRunError):
    """Raised when the physical program cannot be constructed safely."""


class NeutralAtomEmbeddingError(NeutralAtomExecutionError):
    """Raised when coordinate optimization does not converge."""

    status = "embedding_failed"


class NeutralAtomSampleError(NeutralAtomExecutionError):
    """Raised when a backend returns no valid feasible sample."""

    status = "no_feasible_sample"


class PulserQutipRunner:
    """Build and emulate the original weighted neutral-atom QAA program."""

    backend_name = "qutip_simulation"
    _random_lock = Lock()

    def __init__(
        self,
        config: NeutralAtomConfig | None = None,
        *,
        device: object | None = None,
        backend_factory: object | None = None,
    ) -> None:
        self.config = config if config is not None else NeutralAtomConfig()
        self.device = device
        self.backend_factory = backend_factory

    @staticmethod
    def evaluate_mapping(
        new_coordinates: np.ndarray,
        matrix: np.ndarray,
        device: object,
    ) -> float:
        """Measure how closely Rydberg interactions reproduce graph edges."""

        coordinates = np.reshape(new_coordinates, (len(matrix), 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            mapped = squareform(
                device.interaction_coeff / pdist(coordinates) ** 6
            ) / 4
        return float(np.linalg.norm(mapped - matrix))

    def execute(self, component: NeutralAtomComponent) -> NeutralAtomRun:
        """Embed, program, and emulate one non-trivial connected component."""

        with self._random_lock:
            # QuTiP chooses its coefficient cache while importing. Keep the
            # short working-directory change in the same process-wide critical
            # section as the backend execution.
            runtime = self._runtime()
            random_state = np.random.get_state()
            try:
                np.random.seed(self.config.random_seed)
                try:
                    return self._execute_seeded(component, runtime)
                except NeutralAtomRunError:
                    raise
                except Exception as exc:
                    raise NeutralAtomExecutionError(
                        f"component {component.component_id} vendor execution failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            finally:
                np.random.set_state(random_state)

    def _execute_seeded(
        self,
        component: NeutralAtomComponent,
        runtime: tuple[object, object, object],
    ) -> NeutralAtomRun:
        """Execute while NumPy's process RNG is scoped to the configured seed."""

        pulser, backend_factory, device = runtime
        matrix = np.asarray(component.matrix, dtype=float)
        initial_coordinates = np.random.random(len(matrix) * 2)

        mapping = minimize(
            self.evaluate_mapping,
            initial_coordinates,
            args=(~np.eye(matrix.shape[0], dtype=bool) * matrix, device),
            method="Nelder-Mead",
            tol=self.config.mapping_tolerance,
            options={
                "maxiter": self.config.mapping_max_iterations,
                "maxfev": None,
            },
        )
        coordinates_array = np.reshape(mapping.x, (len(matrix), 2))
        mapping_cost = float(mapping.fun)
        if not np.all(np.isfinite(coordinates_array)) or not isfinite(mapping_cost):
            raise NeutralAtomExecutionError(
                f"component {component.component_id} produced a non-finite embedding"
            )
        if not mapping.success:
            raise NeutralAtomEmbeddingError(
                f"component {component.component_id} embedding did not converge: "
                f"{mapping.message} (cost={mapping_cost:.6g})"
            )
        coordinates = tuple(
            (float(coordinate[0]), float(coordinate[1]))
            for coordinate in coordinates_array
        )

        qubits = dict(zip(component.qubit_ids, coordinates, strict=True))
        register = pulser.Register(qubits)
        sequence = pulser.Sequence(register, device)
        sequence.declare_channel("rydberg_global", "rydberg_global")

        node_weights = np.diag(matrix)
        maximum_weight = float(np.max(node_weights))
        if maximum_weight > 0.0:
            normalized_weights = np.clip(node_weights / maximum_weight, 0.0, 1.0)
        else:
            normalized_weights = np.zeros_like(node_weights)
        detuning_weights = 1.0 - normalized_weights
        detuning_map = register.define_detuning_map(
            {
                qubit_id: float(detuning_weights[index])
                for index, qubit_id in enumerate(component.qubit_ids)
            }
        )
        sequence.config_detuning_map(detuning_map, "dmm_0")

        nonedge_distances: list[float] = []
        for right in range(1, matrix.shape[0]):
            # Check every left < right. The notebook's range(right - 1)
            # accidentally skipped adjacent-index pairs.
            for left in range(right):
                distance = float(
                    euclidean(coordinates_array[right], coordinates_array[left])
                )
                if matrix[right, left] == 0.0:
                    nonedge_distances.append(distance)
        if not nonedge_distances or min(nonedge_distances) <= 0.0:
            raise NeutralAtomExecutionError(
                f"component {component.component_id} has no usable nonedge separation"
            )

        omega = float(
            device.interaction_coeff
            / min(nonedge_distances) ** 6
            * self.config.interaction_scale
        )
        if not isfinite(omega) or omega <= 0.0:
            raise NeutralAtomExecutionError(
                f"component {component.component_id} produced an invalid Rabi frequency"
            )
        delta_initial = -omega
        delta_final = -delta_initial
        duration = self.config.pulse_duration_ns

        adiabatic_pulse = pulser.Pulse(
            pulser.InterpolatedWaveform(duration, [1e-9, omega, 1e-9]),
            pulser.InterpolatedWaveform(
                duration,
                [delta_initial, 0.0, delta_final],
            ),
            0,
        )
        sequence.add(adiabatic_pulse, "rydberg_global")
        sequence.add_dmm_detuning(
            pulser.ConstantWaveform(duration, -delta_final),
            "dmm_0",
        )

        backend = backend_factory(sequence)
        results = backend.run()
        counts = tuple(
            sorted(
                (str(bitstring), int(count))
                for bitstring, count in results.final_bitstrings.items()
            )
        )
        program = NeutralAtomProgram(
            component=component,
            coordinates=coordinates,
            mapping_cost=mapping_cost,
            mapping_success=bool(mapping.success),
            omega=omega,
            register=register,
            detuning_map=detuning_map,
            sequence=sequence,
        )
        return NeutralAtomRun(
            component_id=component.component_id,
            node_ids=component.node_ids,
            atom_order=tuple(str(atom_id) for atom_id in results.atom_order),
            bitstring_counts=counts,
            coordinates=coordinates,
            mapping_cost=mapping_cost,
            mapping_success=bool(mapping.success),
            program=program,
            execution_mode="pulser_qutip",
        )

    def _runtime(self) -> tuple[object, object, object]:
        """Resolve optional vendor objects only when quantum execution starts."""

        try:
            import pulser
        except ModuleNotFoundError as exc:
            if exc.name != "pulser":
                raise
            raise NeutralAtomDependencyError(
                'Pulser is required for neutral-atom simulation; install ".[quantum]"'
            ) from exc

        if self.backend_factory is None:
            try:
                backend_factory = self._load_qutip_backend()
            except ModuleNotFoundError as exc:
                if exc.name != "pulser_simulation":
                    raise
                raise NeutralAtomDependencyError(
                    "pulser-simulation is required for neutral-atom simulation; "
                    'install ".[quantum]"'
                ) from exc
        else:
            backend_factory = self.backend_factory
        device = self.device if self.device is not None else pulser.MockDevice
        return pulser, backend_factory, device

    def _load_qutip_backend(self) -> object:
        """Import QuTiP through Pulser with its cache below ``data/.cache``."""

        original_directory = Path.cwd().resolve()
        cache_root = Path(self.config.qutip_cache_dir)
        if not cache_root.is_absolute():
            cache_root = original_directory / cache_root
        cache_root = cache_root.resolve()
        cache_root.mkdir(parents=True, exist_ok=True)

        previous_coefficient_root = self._active_coefficient_root(original_directory)
        os.chdir(cache_root)
        try:
            import pulser_simulation

            self._redirect_qutip_cache(cache_root)
        finally:
            os.chdir(original_directory)

        self._remove_empty_root_cache(previous_coefficient_root, original_directory)
        return pulser_simulation.QutipBackendV2

    @staticmethod
    def _active_coefficient_root(base_directory: Path) -> Path | None:
        """Resolve an already-imported QuTiP cache before redirecting it."""

        qutip = sys.modules.get("qutip")
        if qutip is None:
            return None
        coefficient_root = Path(str(qutip.settings.coeffroot))
        if not coefficient_root.is_absolute():
            coefficient_root = base_directory / coefficient_root
        return coefficient_root.resolve()

    @staticmethod
    def _redirect_qutip_cache(cache_root: Path) -> None:
        """Replace QuTiP's relative fallback with one stable absolute path."""

        qutip = sys.modules.get("qutip")
        if qutip is None:
            return
        previous_entry = str(qutip.settings.coeffroot)
        coefficient_name = Path(previous_entry).name
        if not coefficient_name.startswith("qutip_coeffs_"):
            coefficient_name = "qutip_coeffs"
        coefficient_root = cache_root / coefficient_name
        coefficient_root.mkdir(exist_ok=True)
        qutip.settings.tmproot = str(cache_root)
        qutip.settings.coeffroot = str(coefficient_root)
        if previous_entry != str(coefficient_root) and previous_entry in sys.path:
            sys.path.remove(previous_entry)

    @staticmethod
    def _remove_empty_root_cache(
        coefficient_root: Path | None,
        repository_root: Path,
    ) -> None:
        """Remove only QuTiP's empty, generated working-directory fallback."""

        if (
            coefficient_root is not None
            and coefficient_root.parent == repository_root
            and coefficient_root.name.startswith("qutip_coeffs_")
        ):
            try:
                coefficient_root.rmdir()
            except OSError:
                pass


class QuantumSolver(Solver):
    """Orchestrate component experiments and return one full-frame selection."""

    def __init__(
        self,
        config: NeutralAtomConfig | None = None,
        *,
        runner: NeutralAtomRunner | None = None,
    ) -> None:
        self.config = config if config is not None else NeutralAtomConfig()
        self.runner = runner if runner is not None else PulserQutipRunner(self.config)

    @property
    def solver_name(self) -> str:
        return "neutral_atom"

    def prepare(self, solver_input: SolverInput) -> tuple[NeutralAtomComponent, ...]:
        """Create stable component matrices without loading Pulser."""

        return tuple(
            self._component(solver_input, cluster)
            for cluster in cluster_graph(solver_input.graph)
        )

    def execute(self, solver_input: SolverInput) -> NeutralAtomExecution:
        """Execute every component atomically and retain optional artifacts."""

        components = self.prepare(solver_input)
        simulated_components = tuple(
            component for component in components if not self._is_clique(component)
        )
        oversized = tuple(
            component.component_id
            for component in simulated_components
            if len(component.node_ids) > self.config.maximum_component_nodes
        )
        negative_weight_components = tuple(
            component.component_id
            for component in simulated_components
            if any(weight < 0.0 for weight in component.weights)
        )
        base_diagnostics: dict[str, object] = {
            "backend": getattr(self.runner, "backend_name", "injected_runner"),
            "node_count": len(solver_input.nodes),
            "edge_count": len(solver_input.edges),
            "component_count": len(components),
            "component_sizes": tuple(len(component.node_ids) for component in components),
            "simulated_component_count": len(simulated_components),
            "analytical_clique_component_ids": tuple(
                component.component_id
                for component in components
                if self._is_clique(component)
            ),
            "maximum_component_nodes": self.config.maximum_component_nodes,
            "optimal": False,
        }
        if oversized:
            return NeutralAtomExecution(
                problem_id=solver_input.problem_id,
                input_fingerprint=solver_input.fingerprint,
                selected_ids=(),
                status="unsupported_size",
                runs=(),
                diagnostics={
                    **base_diagnostics,
                    "components": (),
                    "oversized_component_ids": oversized,
                },
            )
        if negative_weight_components:
            return NeutralAtomExecution(
                problem_id=solver_input.problem_id,
                input_fingerprint=solver_input.fingerprint,
                selected_ids=(),
                status="unsupported_weights",
                runs=(),
                diagnostics={
                    **base_diagnostics,
                    "components": (),
                    "negative_weight_component_ids": negative_weight_components,
                    "message": (
                        "the Pulser detuning map requires non-negative graph weights"
                    ),
                },
            )

        selected_ids: list[int] = []
        runs: list[NeutralAtomRun] = []
        component_diagnostics: list[dict[str, object]] = []
        for component in components:
            try:
                raw_run = (
                    self._clique_run(component)
                    if self._is_clique(component)
                    else self.runner.execute(component)
                )
                run, diagnostics = self._decode(component, raw_run)
            except NeutralAtomRunError as exc:
                return NeutralAtomExecution(
                    problem_id=solver_input.problem_id,
                    input_fingerprint=solver_input.fingerprint,
                    selected_ids=(),
                    status=exc.status,
                    runs=tuple(runs),
                    diagnostics={
                        **base_diagnostics,
                        "components": tuple(component_diagnostics),
                        "failed_component_id": component.component_id,
                        "message": str(exc),
                    },
                )
            selected_ids.extend(run.selected_ids)
            runs.append(run)
            component_diagnostics.append(diagnostics)

        return NeutralAtomExecution(
            problem_id=solver_input.problem_id,
            input_fingerprint=solver_input.fingerprint,
            selected_ids=tuple(sorted(selected_ids)),
            status="completed",
            runs=tuple(runs),
            diagnostics={
                **base_diagnostics,
                "components": tuple(component_diagnostics),
            },
        )

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        return self.execute(solver_input).to_selection()

    @staticmethod
    def _component(
        solver_input: SolverInput,
        cluster: GraphCluster,
    ) -> NeutralAtomComponent:
        node_ids = cluster.node_ids
        allowed = set(node_ids)
        weights = tuple(solver_input.graph.node(node_id).weight for node_id in node_ids)
        edges = tuple(
            edge
            for edge in solver_input.edges
            if edge[0] in allowed and edge[1] in allowed
        )
        positions = {node_id: index for index, node_id in enumerate(node_ids)}
        matrix = [[0.0 for _ in node_ids] for _ in node_ids]
        for index, weight in enumerate(weights):
            matrix[index][index] = weight
        for left, right in edges:
            left_index = positions[left]
            right_index = positions[right]
            matrix[left_index][right_index] = 1.0
            matrix[right_index][left_index] = 1.0
        return NeutralAtomComponent(
            component_id=cluster.cluster_id,
            node_ids=node_ids,
            weights=weights,
            edges=edges,
            matrix=tuple(tuple(row) for row in matrix),
        )

    @staticmethod
    def _is_clique(component: NeutralAtomComponent) -> bool:
        node_count = len(component.node_ids)
        return len(component.edges) == node_count * (node_count - 1) // 2

    @staticmethod
    def _clique_run(component: NeutralAtomComponent) -> NeutralAtomRun:
        """Resolve a clique exactly because the QAA nonedge scale is undefined."""

        best_index = max(
            range(len(component.node_ids)),
            key=lambda index: component.weights[index],
        )
        selected = component.weights[best_index] > 0.0
        bits = ["0"] * len(component.node_ids)
        if selected:
            bits[best_index] = "1"
        bitstring = "".join(bits)
        return NeutralAtomRun(
            component_id=component.component_id,
            node_ids=component.node_ids,
            atom_order=component.qubit_ids,
            bitstring_counts=((bitstring, 1),),
            coordinates=tuple(
                (float(index), 0.0) for index in range(len(component.node_ids))
            ),
            mapping_cost=0.0,
            mapping_success=True,
            execution_mode="analytical_clique",
        )

    @staticmethod
    def _decode(
        component: NeutralAtomComponent,
        run: NeutralAtomRun,
    ) -> tuple[NeutralAtomRun, dict[str, object]]:
        if run.component_id != component.component_id:
            raise ValueError("neutral-atom run component_id does not match request")
        if run.node_ids != component.node_ids:
            raise ValueError("neutral-atom run node IDs do not match request")
        if len(run.atom_order) != len(component.qubit_ids) or set(run.atom_order) != set(
            component.qubit_ids
        ):
            raise ValueError("neutral-atom run atom order does not match component qubits")

        node_for_qubit = dict(
            zip(component.qubit_ids, component.node_ids, strict=True)
        )
        atom_nodes = tuple(node_for_qubit[qubit_id] for qubit_id in run.atom_order)
        edge_set = {frozenset(edge) for edge in component.edges}
        weights = dict(zip(component.node_ids, component.weights, strict=True))
        candidates: list[tuple[float, int, tuple[int, ...], str]] = []
        invalid_samples = 0
        infeasible_samples = 0
        for bitstring, count in run.bitstring_counts:
            if len(bitstring) != len(atom_nodes) or set(bitstring) - {"0", "1"}:
                invalid_samples += count
                continue
            selected = tuple(
                sorted(
                    node_id
                    for node_id, bit in zip(atom_nodes, bitstring, strict=True)
                    if bit == "1"
                )
            )
            selected_set = set(selected)
            if any(edge <= selected_set for edge in edge_set):
                infeasible_samples += count
                continue
            objective = fsum(weights[node_id] for node_id in selected)
            candidates.append((objective, count, selected, bitstring))

        if not candidates:
            raise NeutralAtomSampleError(
                f"component {component.component_id} returned no valid feasible sample "
                f"({invalid_samples} malformed, {infeasible_samples} conflicting)"
            )
        empty_bitstring = "0" * len(atom_nodes)
        if not any(candidate[2] == () for candidate in candidates):
            candidates.append((0.0, 0, (), empty_bitstring))
        objective, chosen_count, selected_ids, selected_bitstring = min(
            candidates,
            key=lambda candidate: (
                -candidate[0],
                -candidate[1],
                candidate[2],
                candidate[3],
            ),
        )
        decoded = replace(
            run,
            selected_bitstring=selected_bitstring,
            selected_ids=selected_ids,
        )
        diagnostics: dict[str, object] = {
            "component_id": component.component_id,
            "node_ids": component.node_ids,
            "qubit_ids": component.qubit_ids,
            "atom_order": run.atom_order,
            "node_count": len(component.node_ids),
            "edge_count": len(component.edges),
            "coordinates": run.coordinates,
            "mapping_cost": run.mapping_cost,
            "mapping_success": run.mapping_success,
            "execution_mode": run.execution_mode,
            "bitstring_counts": run.bitstring_counts,
            "sample_count": sum(count for _, count in run.bitstring_counts),
            "invalid_sample_count": invalid_samples,
            "infeasible_sample_count": infeasible_samples,
            "selected_bitstring": selected_bitstring,
            "selected_sample_count": chosen_count,
            "selected_ids": selected_ids,
            "objective": objective,
        }
        return decoded, diagnostics


__all__ = [
    "NeutralAtomComponent",
    "NeutralAtomConfig",
    "NeutralAtomDependencyError",
    "NeutralAtomEmbeddingError",
    "NeutralAtomExecution",
    "NeutralAtomExecutionError",
    "NeutralAtomProgram",
    "NeutralAtomRun",
    "NeutralAtomRunError",
    "NeutralAtomRunner",
    "NeutralAtomSampleError",
    "PulserQutipRunner",
    "QuantumSolver",
]
