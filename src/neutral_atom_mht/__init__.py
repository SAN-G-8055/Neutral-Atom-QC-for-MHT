"""Expose the project's small object-oriented API from one import location.

Image detection and evaluation remain available as focused functions.  The
tracking workflow uses typed observations, an ``HPC`` controller, and
interchangeable ``Solver`` objects; internal helper modules are not intended
to be part of the normal user-facing workflow.
"""

from .classical_solver import ClassicalSolver
from .detection import (
    Detection,
    DetectionConfig,
    DetectionResult,
    detect_frame,
    detect_sequence,
)
from .evaluation import (
    FrameEvaluation,
    SequenceEvaluation,
    evaluate_frame,
    evaluate_sequence,
)
from .hpc import (
    FrameResult,
    HPC,
    HPCConfig,
    ObservedFrame,
    PreparedFrame,
    SequenceResult,
    hpc,
)
from .models import (
    AssociationHypothesis,
    GatedAssociation,
    Observation,
    TrackState,
    observations_from_detections,
)
from .neutral_atom import (
    NeutralAtomInput,
    NeutralAtomOutput,
    NeutralAtomSolver,
    QuantumSolver,
)
from .solver import (
    Solver,
    SolverComparison,
    SolverInput,
    SolverResult,
    SolverRun,
)

__all__ = [
    "AssociationHypothesis",
    "ClassicalSolver",
    "Detection",
    "DetectionConfig",
    "DetectionResult",
    "FrameEvaluation",
    "FrameResult",
    "GatedAssociation",
    "HPC",
    "HPCConfig",
    "NeutralAtomInput",
    "NeutralAtomOutput",
    "NeutralAtomSolver",
    "Observation",
    "ObservedFrame",
    "PreparedFrame",
    "QuantumSolver",
    "SequenceEvaluation",
    "SequenceResult",
    "Solver",
    "SolverComparison",
    "SolverInput",
    "SolverResult",
    "SolverRun",
    "TrackState",
    "detect_frame",
    "detect_sequence",
    "evaluate_frame",
    "evaluate_sequence",
    "hpc",
    "observations_from_detections",
]

__version__ = "0.1.0"
