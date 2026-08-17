"""Calculate association weights and update track-existence probabilities.

A hit compares the predicted Gaussian measurement density with uniform clutter;
a miss uses the declared detection probability.  These calculations happen
before and after solving in the HPC, so every solver sees the same weights and
can never invent its own Bayesian update rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

import numpy as np

from .filtering import update_track_state
from .models import (
    AssociationHypothesis,
    GatedAssociation,
    Observation,
    TrackState,
    _sigmoid_probability,
)


@dataclass(frozen=True, slots=True)
class BayesianConfig:
    detection_probability: float = 0.90
    clutter_spatial_density: float = 1e-4
    initial_existence_probability: float = 0.80

    def __post_init__(self) -> None:
        for name in ("detection_probability", "initial_existence_probability"):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
            object.__setattr__(self, name, value)
        density = float(self.clutter_spatial_density)
        if not isfinite(density) or density <= 0.0:
            raise ValueError("clutter_spatial_density must be finite and positive")
        object.__setattr__(self, "clutter_spatial_density", density)


def probability_to_log_odds(probability: float) -> float:
    value = float(probability)
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")
    return log(value / (1.0 - value))


def log_odds_to_probability(log_odds: float) -> float:
    value = float(log_odds)
    if not isfinite(value):
        raise ValueError("log_odds must be finite")
    return _sigmoid_probability(value)


def hit_log_likelihood_ratio(
    gate: GatedAssociation,
    config: BayesianConfig,
) -> float:
    """Gaussian target likelihood divided by uniform clutter likelihood."""

    covariance = np.asarray(gate.innovation_covariance)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not isfinite(float(log_determinant)):
        raise ValueError("innovation covariance must be positive definite")
    return float(
        log(config.detection_probability)
        - log(config.clutter_spatial_density)
        - log(2.0 * np.pi)
        - 0.5 * log_determinant
        - 0.5 * gate.mahalanobis_sq
    )


def miss_log_likelihood_ratio(config: BayesianConfig) -> float:
    return log(1.0 - config.detection_probability)


def calculate_association_hypotheses(
    predicted_tracks: tuple[TrackState, ...],
    gated_associations: tuple[GatedAssociation, ...],
    config: BayesianConfig,
) -> tuple[AssociationHypothesis, ...]:
    """Calculate local candidate weights before either solver is called.

    Every unselected track receives the same missed-detection update after the
    solver returns.  That miss score is therefore the constant baseline.  A
    vertex weight is the improvement ``hit_increment - miss_increment``;
    non-positive improvements can never help the MWIS objective.
    """

    tracks = {track.track_id: track for track in predicted_tracks}
    if len(tracks) != len(predicted_tracks):
        raise ValueError("predicted track IDs must be unique")
    hypotheses: list[AssociationHypothesis] = []
    ordered = sorted(
        gated_associations,
        key=lambda item: (item.track_id, item.observation_id),
    )
    for hypothesis_id, gate in enumerate(ordered, start=1):
        if gate.track_id not in tracks:
            raise ValueError("gated association references an unknown track")
        track = tracks[gate.track_id]
        increment = hit_log_likelihood_ratio(gate, config)
        posterior_log_odds = track.log_odds + increment
        weight = increment - miss_log_likelihood_ratio(config)
        if weight <= 0.0:
            continue
        hypotheses.append(
            AssociationHypothesis(
                hypothesis_id=hypothesis_id,
                frame=gate.frame,
                track_id=gate.track_id,
                observation_id=gate.observation_id,
                log_likelihood_ratio=increment,
                posterior_log_odds=posterior_log_odds,
                posterior_probability=log_odds_to_probability(posterior_log_odds),
                weight=weight,
                gate=gate,
            )
        )
    return tuple(hypotheses)


def apply_bayesian_updates(
    predicted_tracks: tuple[TrackState, ...],
    observations: tuple[Observation, ...],
    hypotheses: tuple[AssociationHypothesis, ...],
    selected_hypothesis_ids: tuple[int, ...],
    config: BayesianConfig,
) -> tuple[tuple[TrackState, ...], frozenset[int]]:
    """Apply selected hits and the same miss rule after any solver returns."""

    tracks = {track.track_id: track for track in predicted_tracks}
    observations_by_id = {item.observation_id: item for item in observations}
    hypotheses_by_id = {item.hypothesis_id: item for item in hypotheses}
    if len(tracks) != len(predicted_tracks):
        raise ValueError("predicted track IDs must be unique")
    if len(observations_by_id) != len(observations):
        raise ValueError("observation IDs must be unique within the frame")
    if len(hypotheses_by_id) != len(hypotheses):
        raise ValueError("hypothesis IDs must be unique")
    for hypothesis in hypotheses:
        if hypothesis.track_id not in tracks:
            raise ValueError("hypothesis references an unknown predicted track")
        if hypothesis.observation_id not in observations_by_id:
            raise ValueError("hypothesis references an unknown observation")
    if len(selected_hypothesis_ids) != len(set(selected_hypothesis_ids)):
        raise ValueError("selected hypothesis IDs must be unique")
    selected: list[AssociationHypothesis] = []
    for hypothesis_id in selected_hypothesis_ids:
        if hypothesis_id not in hypotheses_by_id:
            raise ValueError("selected result references an unknown hypothesis")
        selected.append(hypotheses_by_id[hypothesis_id])
    if len({item.track_id for item in selected}) != len(selected):
        raise ValueError("selected hypotheses reuse a track")
    if len({item.observation_id for item in selected}) != len(selected):
        raise ValueError("selected hypotheses reuse an observation")
    selected_by_track = {item.track_id: item for item in selected}
    updated: list[TrackState] = []
    assigned_observations: set[int] = set()
    for track_id, track in sorted(tracks.items()):
        hypothesis = selected_by_track.get(track_id)
        if hypothesis is None:
            log_odds = track.log_odds + miss_log_likelihood_ratio(config)
            updated.append(
                TrackState(
                    track_id=track.track_id,
                    frame=track.frame,
                    state=track.state,
                    covariance=track.covariance,
                    log_odds=log_odds,
                    posterior_probability=log_odds_to_probability(log_odds),
                    hits=track.hits,
                    misses=track.misses + 1,
                    observation_history=track.observation_history,
                )
            )
            continue
        if hypothesis.observation_id not in observations_by_id:
            raise ValueError("selected hypothesis references an unknown observation")
        state, covariance = update_track_state(track, hypothesis.gate)
        assigned_observations.add(hypothesis.observation_id)
        updated.append(
            TrackState(
                track_id=track.track_id,
                frame=track.frame,
                state=state,
                covariance=covariance,
                log_odds=hypothesis.posterior_log_odds,
                posterior_probability=hypothesis.posterior_probability,
                hits=track.hits + 1,
                misses=0,
                observation_history=track.observation_history
                + ((track.frame, hypothesis.observation_id),),
            )
        )
    return tuple(updated), frozenset(assigned_observations)
