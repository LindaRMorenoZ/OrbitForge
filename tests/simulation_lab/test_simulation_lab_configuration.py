
"""Tests for Simulation Lab configuration utilities."""

import pytest

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.experiments import ExperimentConfiguration
from orbitforge.recommendations import (
    ParameterRecommendation,
    SimulationRecommendations,
)
from orbitforge.simulation_lab.configuration import apply_recommendations


@pytest.fixture
def configuration():
    builder = SystemBuilder()
    builder.add_body(
        name="Body",
        mass=1.0,
        position=[0.0, 0.0],
        velocity=[1.0, 0.0],
    )

    return ExperimentConfiguration.from_builder(
        name="Test configuration",
        builder=builder,
        config=SimulationConfig(total_time=0.1, dt=0.01),
    )


@pytest.fixture
def recommendations():
    return SimulationRecommendations(
        dt=ParameterRecommendation(0.005, "Recommended timestep"),
        total_time=ParameterRecommendation(0.2, "Recommended duration"),
        save_every=ParameterRecommendation(2, "Recommended save interval"),
        integrator=ParameterRecommendation(
            "velocity_verlet",
            "Recommended integrator",
        ),
    )


def test_apply_all_recommendations(configuration, recommendations):
    updated = apply_recommendations(configuration, recommendations)

    assert updated.config.dt == 0.005
    assert updated.config.total_time == 0.2
    assert updated.config.save_every == 2
    assert updated.config.integrator == "velocity_verlet"


def test_apply_selected_recommendations(configuration, recommendations):
    updated = apply_recommendations(
        configuration,
        recommendations,
        parameters=("dt", "total_time"),
    )

    assert updated.config.dt == 0.005
    assert updated.config.total_time == 0.2
    assert updated.config.save_every == configuration.config.save_every
    assert updated.config.integrator == configuration.config.integrator


def test_original_configuration_remains_unchanged(
    configuration,
    recommendations,
):
    apply_recommendations(configuration, recommendations)

    assert configuration.config.dt == 0.01
    assert configuration.config.total_time == 0.1


def test_rejects_invalid_configuration(recommendations):
    with pytest.raises(TypeError):
        apply_recommendations("invalid", recommendations)


def test_rejects_invalid_recommendations(configuration):
    with pytest.raises(TypeError):
        apply_recommendations(configuration, "invalid")


def test_rejects_unsupported_parameter(configuration, recommendations):
    with pytest.raises(ValueError):
        apply_recommendations(
            configuration,
            recommendations,
            parameters=("dt", "softening"),
        )


def test_rejects_duplicate_parameters(configuration, recommendations):
    with pytest.raises(ValueError):
        apply_recommendations(
            configuration,
            recommendations,
            parameters=("dt", "dt"),
        )


def test_rejects_non_tuple_parameters(configuration, recommendations):
    with pytest.raises(TypeError):
        apply_recommendations(
            configuration,
            recommendations,
            parameters=["dt"],
        )
