
import numpy as np
import pytest

from orbitforge.analysis import SimulationAnalysis
from orbitforge.metrics import ConservationMetrics
from orbitforge.simulation import Simulation
from orbitforge.state import SystemState


def make_simulation():
    state = SystemState(
        names=("A", "B"),
        masses=np.array([1.0, 1.0]),
        positions=np.array([
            [-0.5, 0.0],
            [0.5, 0.0],
        ]),
        velocities=np.array([
            [0.0, -0.5],
            [0.0, 0.5],
        ]),
    )
    return Simulation(state)


def test_metrics_calculate_energy_drift():
    result = make_simulation().run(steps=100, dt=0.001)
    analysis = SimulationAnalysis.from_result(result)

    metrics = ConservationMetrics.from_analysis(analysis)

    assert metrics.max_absolute_energy_drift >= 0.0
    assert metrics.max_relative_energy_drift >= 0.0


def test_metrics_detect_conserved_linear_momentum():
    result = make_simulation().run(steps=100, dt=0.001)
    analysis = SimulationAnalysis.from_result(result)

    metrics = ConservationMetrics.from_analysis(analysis)

    assert metrics.max_absolute_linear_momentum_drift < 1e-12


def test_metrics_detect_conserved_angular_momentum():
    result = make_simulation().run(steps=100, dt=0.001)
    analysis = SimulationAnalysis.from_result(result)

    metrics = ConservationMetrics.from_analysis(analysis)

    assert metrics.max_absolute_angular_momentum_drift < 1e-10


def test_metrics_reject_invalid_analysis():
    with pytest.raises(TypeError):
        ConservationMetrics.from_analysis("invalid")


def test_metrics_relative_momentum_is_undefined_when_initial_is_zero():
    result = make_simulation().run(steps=100, dt=0.001)
    analysis = SimulationAnalysis.from_result(result)

    metrics = ConservationMetrics.from_analysis(analysis)

    assert metrics.max_relative_linear_momentum_drift is None
