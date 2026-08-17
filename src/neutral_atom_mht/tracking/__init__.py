"""Interpretable, solver-independent tracking primitives."""

from .filtering import (
    FilterConfig,
    filter_association_hypotheses,
    filter_tracks,
    predict_tracks,
)
from .gating import GateConfig, gate_observations
from .likelihood import (
    BayesianConfig,
    apply_bayesian_updates,
    calculate_association_hypotheses,
)
from .models import (
    AssociationHypothesis,
    GatedAssociation,
    Observation,
    TrackState,
    observations_from_detections,
)
from .interface import (
    BackendComparison,
    BackendRun,
    PreparedStep,
    TrackingConfig,
    TrackingInterface,
    TrackingStepResult,
)

__all__ = [
    "AssociationHypothesis",
    "BayesianConfig",
    "BackendComparison",
    "BackendRun",
    "FilterConfig",
    "GateConfig",
    "GatedAssociation",
    "Observation",
    "PreparedStep",
    "TrackState",
    "TrackingConfig",
    "TrackingInterface",
    "TrackingStepResult",
    "apply_bayesian_updates",
    "calculate_association_hypotheses",
    "filter_association_hypotheses",
    "filter_tracks",
    "gate_observations",
    "observations_from_detections",
    "predict_tracks",
]
