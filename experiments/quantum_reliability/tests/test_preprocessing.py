"""Check each mathematical preprocessing stage independently from all solvers."""

from __future__ import annotations

from math import log

import numpy as np
import pytest

from filtering import (
    FilterConfig,
    filter_association_hypotheses,
    filter_tracks,
    predict_tracks,
)
from gating import GateConfig, gate_observations
from likelihood import (
    BayesianConfig,
    apply_bayesian_updates,
    calculate_association_hypotheses,
    hit_log_likelihood_ratio,
    miss_log_likelihood_ratio,
)
from models import Observation, TrackState


def track(
    track_id: int = 1,
    *,
    frame: int = 0,
    state: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 2.0),
    log_odds: float = 1.0,
    misses: int = 0,
) -> TrackState:
    return TrackState(
        track_id=track_id,
        frame=frame,
        state=state,
        covariance=tuple(map(tuple, np.eye(4))),
        log_odds=log_odds,
        hits=2,
        misses=misses,
        observation_history=((frame, track_id),),
    )


def test_prediction_is_a_separate_deterministic_constant_velocity_step() -> None:
    predicted = predict_tracks(
        (track(),),
        frame=2,
        seconds_per_frame=0.5,
        config=FilterConfig(acceleration_std=0.0),
    )[0]

    assert predicted.frame == 2
    assert predicted.state == pytest.approx((1.0, 2.0, 1.0, 2.0))
    assert predicted.log_odds == 1.0


def test_gate_is_inclusive_at_the_declared_mahalanobis_boundary() -> None:
    predicted = predict_tracks(
        (track(state=(0.0, 0.0, 0.0, 0.0)),),
        frame=1,
        seconds_per_frame=1.0,
        config=FilterConfig(acceleration_std=0.0),
    )
    observation = Observation(
        1, 4, np.sqrt(6.0), 0.0, ((1.0, 0.0), (0.0, 1.0))
    )
    candidate = gate_observations(predicted, (observation,), GateConfig(2.0))[0]

    assert candidate.mahalanobis_sq == pytest.approx(2.0)


def test_innovation_distance_gate_is_independent_of_statistical_gate() -> None:
    predicted = predict_tracks(
        (track(state=(0.0, 0.0, 0.0, 0.0)),),
        frame=1,
        seconds_per_frame=1.0,
        config=FilterConfig(acceleration_std=10.0),
    )
    observation = Observation(1, 1, 3.0, 0.0)

    assert gate_observations(
        predicted,
        (observation,),
        GateConfig(mahalanobis_sq=100.0, maximum_innovation_distance=2.0),
    ) == ()


def test_hit_and_miss_likelihoods_match_the_declared_equations() -> None:
    predicted = predict_tracks(
        (track(state=(0.0, 0.0, 0.0, 0.0)),),
        frame=1,
        seconds_per_frame=1.0,
        config=FilterConfig(acceleration_std=0.0),
    )
    gate = gate_observations(
        predicted,
        (Observation(1, 1, 0.0, 0.0),),
        GateConfig(),
    )[0]
    config = BayesianConfig(detection_probability=0.8, clutter_spatial_density=0.01)
    determinant = np.linalg.det(np.asarray(gate.innovation_covariance))
    expected = (
        log(0.8)
        - log(0.01)
        - log(2.0 * np.pi)
        - 0.5 * log(determinant)
    )

    assert hit_log_likelihood_ratio(gate, config) == pytest.approx(expected)
    assert miss_log_likelihood_ratio(config) == pytest.approx(log(0.2))


def test_finite_extreme_log_odds_stay_inside_probability_interval() -> None:
    assert 0.0 < track(log_odds=-1_000.0).posterior_probability < 1.0
    assert 0.0 < track(log_odds=1_000.0).posterior_probability < 1.0


def test_hypothesis_weights_use_hit_benefit_over_the_miss_baseline() -> None:
    predicted = predict_tracks(
        (track(),),
        frame=1,
        seconds_per_frame=1.0,
        config=FilterConfig(acceleration_std=0.0),
    )
    observations = (Observation(1, 1, 1.0, 2.0), Observation(1, 2, 1.5, 2.0))
    gated = gate_observations(predicted, observations, GateConfig())
    hypotheses = calculate_association_hypotheses(
        predicted,
        gated,
        BayesianConfig(clutter_spatial_density=1e-4),
    )

    assert [(item.track_id, item.observation_id) for item in hypotheses] == [
        (1, 1),
        (1, 2),
    ]
    miss_increment = miss_log_likelihood_ratio(BayesianConfig(clutter_spatial_density=1e-4))
    assert all(
        item.weight == pytest.approx(item.log_likelihood_ratio - miss_increment)
        for item in hypotheses
    )
    assert filter_association_hypotheses(hypotheses) == hypotheses


def test_bayesian_update_applies_selected_hits_and_unselected_misses() -> None:
    config = BayesianConfig(clutter_spatial_density=1e-4)
    predicted = predict_tracks(
        (track(), track(2, state=(10.0, 0.0, 0.0, 0.0))),
        frame=1,
        seconds_per_frame=1.0,
        config=FilterConfig(acceleration_std=0.0),
    )
    observations = (Observation(1, 5, 1.0, 2.0),)
    hypotheses = calculate_association_hypotheses(
        predicted,
        gate_observations(predicted, observations, GateConfig(mahalanobis_sq=20.0)),
        config,
    )
    chosen = next(item for item in hypotheses if item.track_id == 1)
    updated, assigned = apply_bayesian_updates(
        predicted,
        observations,
        hypotheses,
        (chosen.hypothesis_id,),
        config,
    )

    by_id = {item.track_id: item for item in updated}
    assert assigned == frozenset({5})
    assert by_id[1].hits == 3 and by_id[1].misses == 0
    assert by_id[2].misses == 1
    assert by_id[2].log_odds == pytest.approx(1.0 + log(0.1))


def test_bayesian_update_rejects_a_hypothesis_for_an_unknown_track() -> None:
    config = BayesianConfig(clutter_spatial_density=1e-4)
    predicted = predict_tracks(
        (track(), track(2, state=(10.0, 0.0, 0.0, 0.0))),
        frame=1,
        seconds_per_frame=1.0,
        config=FilterConfig(acceleration_std=0.0),
    )
    observations = (Observation(1, 5, 10.0, 0.0),)
    hypotheses = calculate_association_hypotheses(
        predicted,
        gate_observations(predicted, observations, GateConfig(mahalanobis_sq=20.0)),
        config,
    )
    track_two = next(item for item in hypotheses if item.track_id == 2)

    with pytest.raises(ValueError, match="unknown predicted track"):
        apply_bayesian_updates(
            predicted[:1],
            observations,
            (track_two,),
            (track_two.hypothesis_id,),
            config,
        )


def test_track_filtering_has_deterministic_probability_then_id_order() -> None:
    low = track(1, log_odds=-5.0)
    tied_later = track(3, log_odds=2.0)
    tied_first = track(2, log_odds=2.0)
    config = FilterConfig(minimum_posterior=0.1, maximum_tracks=1)

    assert filter_tracks((low, tied_later, tied_first), config) == (tied_first,)


def test_candidate_filter_includes_the_declared_weight_boundary() -> None:
    predicted = predict_tracks(
        (track(),),
        frame=1,
        seconds_per_frame=1.0,
        config=FilterConfig(acceleration_std=0.0),
    )
    observations = (Observation(1, 1, 1.0, 2.0),)
    hypotheses = calculate_association_hypotheses(
        predicted,
        gate_observations(predicted, observations, GateConfig()),
        BayesianConfig(clutter_spatial_density=1e-4),
    )

    assert filter_association_hypotheses(
        hypotheses, minimum_weight=hypotheses[0].weight
    ) == hypotheses


def test_candidate_filter_owns_removal_of_negative_assignment_benefits() -> None:
    predicted = predict_tracks(
        (track(),),
        frame=1,
        seconds_per_frame=1.0,
        config=FilterConfig(acceleration_std=0.0),
    )
    observations = (Observation(1, 1, 1.0, 2.0),)
    hypotheses = calculate_association_hypotheses(
        predicted,
        gate_observations(predicted, observations, GateConfig()),
        BayesianConfig(
            detection_probability=0.5,
            clutter_spatial_density=1.0,
        ),
    )

    assert len(hypotheses) == 1 and hypotheses[0].weight < 0.0
    assert filter_association_hypotheses(hypotheses) == ()
