
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
def make_synthetic_analysis():
    """Create deterministic diagnostics with known conservation deviations."""

    return SimulationAnalysis(
        times=np.array([0.0, 1.0, 2.0]),
        kinetic_energies=np.array([15.0, 15.0, 15.0]),
        potential_energies=np.array([-5.0, -6.0, -4.0]),
        total_energies=np.array([10.0, 9.0, 11.0]),
        linear_momenta=np.array([
            [3.0, 4.0],
            [6.0, 8.0],
            [0.0, 0.0],
        ]),
        angular_momenta=np.array([
            [2.0],
            [3.0],
            [0.0],
        ]),
        relative_energy_drift=np.array([0.0, -0.1, 0.1]),
    )


def test_metrics_match_known_energy_and_momentum_deviations():
    analysis = make_synthetic_analysis()

    metrics = ConservationMetrics.from_analysis(analysis)

    assert metrics.max_absolute_energy_drift == pytest.approx(1.0)
    assert metrics.max_relative_energy_drift == pytest.approx(0.1)

    assert metrics.max_absolute_linear_momentum_drift == pytest.approx(5.0)
    assert metrics.max_relative_linear_momentum_drift == pytest.approx(1.0)

    assert metrics.max_absolute_angular_momentum_drift == pytest.approx(2.0)
    assert metrics.max_relative_angular_momentum_drift == pytest.approx(1.0)


def test_metrics_relative_angular_momentum_is_undefined_when_initial_is_zero():
    analysis = make_synthetic_analysis()

    zero_angular = np.zeros((3, 1))

    analysis = SimulationAnalysis(
        times=analysis.times,
        kinetic_energies=analysis.kinetic_energies,
        potential_energies=analysis.potential_energies,
        total_energies=analysis.total_energies,
        linear_momenta=analysis.linear_momenta,
        angular_momenta=zero_angular,
        relative_energy_drift=analysis.relative_energy_drift,
    )

    metrics = ConservationMetrics.from_analysis(analysis)

    assert metrics.max_relative_angular_momentum_drift is None
    assert metrics.max_absolute_angular_momentum_drift == 0.0

def test_metrics_support_three_dimensional_momenta():
    analysis = SimulationAnalysis(
        times=np.array([0.0, 1.0]),
        kinetic_energies=np.array([10.0, 10.0]),
        potential_energies=np.array([-2.0, -3.0]),
        total_energies=np.array([8.0, 7.0]),
        linear_momenta=np.array([
            [1.0, 2.0, 2.0],
            [2.0, 4.0, 5.0],
        ]),
        angular_momenta=np.array([
            [1.0, 2.0, 2.0],
            [2.0, 4.0, 5.0],
        ]),
        relative_energy_drift=np.array([0.0, -0.125]),
    )

    metrics = ConservationMetrics.from_analysis(analysis)

    assert metrics.max_absolute_energy_drift == pytest.approx(1.0)
    assert metrics.max_relative_energy_drift == pytest.approx(0.125)

    assert metrics.max_absolute_linear_momentum_drift == pytest.approx(
        np.sqrt(14.0)
    )
    assert metrics.max_absolute_angular_momentum_drift == pytest.approx(
        np.sqrt(14.0)
    )

    assert metrics.max_relative_linear_momentum_drift == pytest.approx(
        np.sqrt(14.0) / 3.0
    )
    assert metrics.max_relative_angular_momentum_drift == pytest.approx(
        np.sqrt(14.0) / 3.0
    )

def test_metrics_do_not_modify_original_analysis():
    analysis = make_synthetic_analysis()

    original_energy = analysis.total_energies.copy()
    original_linear = analysis.linear_momenta.copy()
    original_angular = analysis.angular_momenta.copy()

    ConservationMetrics.from_analysis(analysis)

    np.testing.assert_array_equal(
        analysis.total_energies, original_energy
    )
    np.testing.assert_array_equal(
        analysis.linear_momenta, original_linear
    )
    np.testing.assert_array_equal(
        analysis.angular_momenta, original_angular
    )

@pytest.mark.parametrize(
    "field_name,value,error",
    [
        ("max_absolute_energy_drift", -1.0, ValueError),
        ("max_relative_energy_drift", float("nan"), ValueError),
        ("max_absolute_linear_momentum_drift", float("inf"), ValueError),
        ("max_absolute_angular_momentum_drift", -float("inf"), ValueError),
        ("max_absolute_energy_drift", True, TypeError),
        ("max_absolute_energy_drift", "1.0", TypeError),
        ("max_relative_energy_drift", None, TypeError),
    ],
)
def test_metrics_reject_invalid_values(field_name, value, error):
    values = {
        "max_absolute_energy_drift": 0.1,
        "max_relative_energy_drift": 0.01,
        "max_absolute_linear_momentum_drift": 0.2,
        "max_relative_linear_momentum_drift": 0.02,
        "max_absolute_angular_momentum_drift": 0.3,
        "max_relative_angular_momentum_drift": 0.03,
    }

    values[field_name] = value

    with pytest.raises(error):
        ConservationMetrics(**values)