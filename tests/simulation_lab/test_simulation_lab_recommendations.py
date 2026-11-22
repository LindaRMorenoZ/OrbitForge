"""Tests for Simulation Lab recommendation exports."""

from orbitforge.recommendations import (
    ParameterRecommendation,
    SimulationRecommendations,
    SimulationRecommender,
)
from orbitforge.simulation_lab import (
    ParameterRecommendation as LabParameterRecommendation,
)
from orbitforge.simulation_lab import (
    SimulationRecommendations as LabSimulationRecommendations,
)
from orbitforge.simulation_lab import (
    SimulationRecommender as LabSimulationRecommender,
)


def test_recommendation_classes_are_exported_from_simulation_lab():
    assert LabParameterRecommendation is ParameterRecommendation
    assert LabSimulationRecommendations is SimulationRecommendations
    assert LabSimulationRecommender is SimulationRecommender