import sys
from pathlib import Path

# Make the checkout importable when tests run without the package installed.
project_root = Path(__file__).resolve().parents[1]
for source_root in (project_root / "src", project_root):
    if source_root.is_dir():
        sys.path.insert(0, str(source_root))

import math

import numpy as np
import pytest

from orbitforge.recommendations.simulation_recommender import (
    ParameterRecommendation,
    SimulationRecommendations,
    SimulationRecommender,
)
from orbitforge.state import SystemState


@pytest.fixture
def two_body_state():
    return SystemState(
        names=("Body A", "Body B"),
        masses=np.array([1.0, 1.0]),
        positions=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        velocities=np.zeros((2, 2)),
    )


def test_recommender_returns_parameter_recommendation(two_body_state):
    recommender = SimulationRecommender()

    recommendation = recommender.recommend_dt(
        two_body_state,
        gravitational_constant=1.0,
    )

    assert isinstance(recommendation, ParameterRecommendation)
    assert recommendation.value == pytest.approx(0.02 / math.sqrt(2.0))
    assert "convergence" in recommendation.explanation.lower()


def test_recommender_uses_configured_safety_factor(two_body_state):
    recommender = SimulationRecommender(safety_factor=0.01)

    recommendation = recommender.recommend_dt(
        two_body_state,
        gravitational_constant=1.0,
    )

    assert recommendation.value == pytest.approx(0.01 / math.sqrt(2.0))


def test_recommender_rejects_invalid_safety_factor():
    with pytest.raises(ValueError):
        SimulationRecommender(safety_factor=0.0)


def test_recommender_rejects_invalid_gravitational_constant(two_body_state):
    recommender = SimulationRecommender()

    with pytest.raises(ValueError):
        recommender.recommend_dt(two_body_state, gravitational_constant=0.0)


def test_recommender_rejects_invalid_state():
    recommender = SimulationRecommender()

    with pytest.raises(TypeError):
        recommender.recommend_dt([], gravitational_constant=1.0)


def test_recommender_rejects_single_body_system():
    state = SystemState(
        names=("Body A",),
        masses=np.array([1.0]),
        positions=np.array([[0.0, 0.0]]),
        velocities=np.zeros((1, 2)),
    )

    recommender = SimulationRecommender()

    with pytest.raises(ValueError, match="body pair"):
        recommender.recommend_dt(state, gravitational_constant=1.0)


def test_recommender_rejects_non_finite_timescale():
    state = SystemState(
        names=("Body A", "Body B"),
        masses=np.array([1e200, 1e200]),
        positions=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        velocities=np.zeros((2, 2)),
    )

    recommender = SimulationRecommender()

    with pytest.raises(ValueError, match="gravitational timescale"):
        recommender.recommend_dt(state, gravitational_constant=1e200)


def test_recommender_rejects_coincident_bodies():
    state = SystemState(
        names=("Body A", "Body B"),
        masses=np.array([1.0, 1.0]),
        positions=np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        velocities=np.zeros((2, 2)),
    )

    recommender = SimulationRecommender()

    with pytest.raises(ValueError, match="Coincident bodies"):
        recommender.recommend_dt(state, gravitational_constant=1.0)


@pytest.mark.parametrize("duration_factor", [0.0, -1.0, float("inf"), float("nan")])
def test_recommender_rejects_invalid_duration_factor(duration_factor):
    with pytest.raises(ValueError, match="duration_factor"):
        SimulationRecommender(duration_factor=duration_factor)


@pytest.mark.parametrize("target_saved_states", [0, 1, -1, 2.5, True])
def test_recommender_rejects_invalid_target_saved_states(target_saved_states):
    with pytest.raises(ValueError, match="target_saved_states"):
        SimulationRecommender(target_saved_states=target_saved_states)


def test_recommender_recommends_total_time(two_body_state):
    recommender = SimulationRecommender()

    recommendation = recommender.recommend_total_time(
        two_body_state,
        gravitational_constant=1.0,
    )

    assert isinstance(recommendation, ParameterRecommendation)
    assert recommendation.value == pytest.approx(10.0 / math.sqrt(2.0))
    assert "heuristic" in recommendation.explanation.lower()


def test_recommender_uses_configured_duration_factor(two_body_state):
    recommender = SimulationRecommender(duration_factor=5.0)

    recommendation = recommender.recommend_total_time(
        two_body_state,
        gravitational_constant=1.0,
    )

    assert recommendation.value == pytest.approx(5.0 / math.sqrt(2.0))


def test_recommended_total_time_is_multiple_of_recommended_dt(two_body_state):
    recommender = SimulationRecommender(
        safety_factor=0.03,
        duration_factor=7.0,
    )

    dt = recommender.recommend_dt(
        two_body_state,
        gravitational_constant=1.0,
    ).value

    total_time = recommender.recommend_total_time(
        two_body_state,
        gravitational_constant=1.0,
    ).value

    assert total_time / dt == pytest.approx(round(total_time / dt))


def test_recommender_recommends_save_every():
    recommender = SimulationRecommender(target_saved_states=101)

    recommendation = recommender.recommend_save_every(total_steps=1000)

    assert isinstance(recommendation, ParameterRecommendation)
    assert recommendation.value == 10
    assert "storage frequency" in recommendation.explanation.lower()


def test_recommender_respects_target_saved_states():
    recommender = SimulationRecommender(target_saved_states=101)

    recommendation = recommender.recommend_save_every(total_steps=1000)

    estimated_saved_states = math.ceil(1000 / recommendation.value) + 1

    assert estimated_saved_states <= 101


@pytest.mark.parametrize("total_steps", [0, -1, 1.5, True])
def test_recommender_rejects_invalid_total_steps(total_steps):
    recommender = SimulationRecommender()

    with pytest.raises(ValueError, match="total_steps"):
        recommender.recommend_save_every(total_steps)


def test_recommender_returns_complete_recommendations(two_body_state):
    recommender = SimulationRecommender()

    recommendations = recommender.recommend(
        two_body_state,
        gravitational_constant=1.0,
    )

    assert isinstance(recommendations, SimulationRecommendations)
    assert recommendations.dt.value > 0
    assert recommendations.total_time.value > 0
    assert recommendations.save_every.value >= 1
    assert recommendations.integrator.value == "velocity_verlet"
    assert len(recommendations.warnings) >= 1


def test_recommended_configuration_has_integer_number_of_steps(two_body_state):
    recommender = SimulationRecommender()

    recommendations = recommender.recommend(
        two_body_state,
        gravitational_constant=1.0,
    )

    steps = recommendations.total_time.value / recommendations.dt.value

    assert steps == pytest.approx(round(steps))
