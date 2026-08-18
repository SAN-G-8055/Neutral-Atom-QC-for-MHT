"""Expose the project's small object-oriented API from one flat facade.

The facade exposes the objects needed to configure the HPC, its solvers, and
synthetic datasets. Detection, graph, and preprocessing details
remain in their focused modules instead of becoming compatibility aliases.
"""

from _version import __version__

from classical_solver import ClassicalSolver
from hpc import (
    HPC,
    HPCConfig,
    hpc,
)
from models import Observation, TrackState
from neutral_atom import (
    NeutralAtomComponent,
    NeutralAtomConfig,
    NeutralAtomError,
    NeutralAtomExecution,
    NeutralAtomProgram,
    NeutralAtomRun,
    NeutralAtomRunner,
    NeutralAtomSequenceFigures,
    NeutralAtomVisualizer,
    PulserQutipRunner,
    QuantumSolver,
)
from overnight_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    DEFAULT_AXES,
    DEFAULT_BENCHMARK_OUTPUT,
    DEFAULT_OBJECT_COUNTS,
    DEFAULT_SEEDS,
    DEFAULT_SEVERITY_LEVELS,
    BenchmarkResult,
    BenchmarkScenario,
    OvernightBenchmarkConfig,
    build_synthetic_scenarios,
    run_overnight_benchmark,
)
from solver import (
    ComponentSolver,
    Solver,
    SolverComparison,
    SolverInput,
    SolverResult,
    SolverSelection,
)
from synthetic_data import (
    DEFAULT_SYNTHETIC_DATA_ROOT,
    QUANTUM_DEMO_DATA_CONFIG,
    SyntheticDataConfig,
    SyntheticDataGenerator,
    SyntheticDataset,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkResult",
    "BenchmarkScenario",
    "ClassicalSolver",
    "ComponentSolver",
    "DEFAULT_AXES",
    "DEFAULT_BENCHMARK_OUTPUT",
    "DEFAULT_SYNTHETIC_DATA_ROOT",
    "DEFAULT_OBJECT_COUNTS",
    "DEFAULT_SEEDS",
    "DEFAULT_SEVERITY_LEVELS",
    "HPC",
    "HPCConfig",
    "NeutralAtomComponent",
    "NeutralAtomConfig",
    "NeutralAtomError",
    "NeutralAtomExecution",
    "NeutralAtomProgram",
    "NeutralAtomRun",
    "NeutralAtomRunner",
    "NeutralAtomSequenceFigures",
    "NeutralAtomVisualizer",
    "Observation",
    "OvernightBenchmarkConfig",
    "PulserQutipRunner",
    "QUANTUM_DEMO_DATA_CONFIG",
    "QuantumSolver",
    "Solver",
    "SolverComparison",
    "SolverInput",
    "SolverResult",
    "SolverSelection",
    "SyntheticDataConfig",
    "SyntheticDataGenerator",
    "SyntheticDataset",
    "TrackState",
    "build_synthetic_scenarios",
    "hpc",
    "run_overnight_benchmark",
]
