"""Interpretable data association for neutral-atom multiple-hypothesis tracking."""

from .detection import Detection, DetectionConfig, DetectionResult, detect_frame, detect_sequence
from .evaluation import FrameEvaluation, SequenceEvaluation, evaluate_frame, evaluate_sequence
from .backends import ClassicalBackend, NeutralAtomBackend, SolverInput, SolverResult
from .tracking import Observation, TrackingConfig, TrackingInterface, TrackState

__all__ = [
    "Detection",
    "DetectionConfig",
    "DetectionResult",
    "ClassicalBackend",
    "FrameEvaluation",
    "NeutralAtomBackend",
    "Observation",
    "SequenceEvaluation",
    "SolverInput",
    "SolverResult",
    "TrackState",
    "TrackingConfig",
    "TrackingInterface",
    "detect_frame",
    "detect_sequence",
    "evaluate_frame",
    "evaluate_sequence",
]

__version__ = "0.1.0"
