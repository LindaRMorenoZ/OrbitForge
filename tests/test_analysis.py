
import numpy as np
import pytest

from orbitforge.analysis import SimulationAnalysis
from orbitforge.simulation import Simulation


def make_two_body_simulation():
    """Create a simple two-body system for analysis tests."""

    from orbitforge.state import SystemState

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

    return Simulation(state, gravitational_constant=1.0)


def test_analysis_calculates_energy_components():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(result)

    assert analysis.kinetic_energies.shape == result.times.shape
    assert analysis.potential_energies.shape == result.times.shape
    assert analysis.total_energies.shape == result.times.shape

    np.testing.assert_allclose(
        analysis.kinetic_energies[0],
        0.25,
    )

    np.testing.assert_allclose(
        analysis.potential_energies[0],
        -1.0,
    )

    np.testing.assert_allclose(
        analysis.total_energies[0],
        -0.75,
    )


def test_analysis_calculates_momenta():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(result)

    np.testing.assert_allclose(
        analysis.linear_momenta,
        0.0,
        atol=1e-12,
    )

    assert analysis.angular_momenta.shape == (len(result.times), 1)


def test_analysis_initial_energy_drift_is_zero():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(result)

    assert analysis.relative_energy_drift[0] == pytest.approx(0.0)


def test_analysis_preserves_saved_times():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01, save_every=2)

    analysis = SimulationAnalysis.from_result(result)

    np.testing.assert_array_equal(
        analysis.times,
        result.times,
    )

def test_energy_drift_remains_small_for_two_body_system():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=1000, dt=0.001)

    analysis = SimulationAnalysis.from_result(result)

    assert np.max(np.abs(analysis.relative_energy_drift)) < 1e-5



def test_angular_momentum_is_conserved():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=1000, dt=0.001)

    analysis = SimulationAnalysis.from_result(result)

    initial = analysis.angular_momenta[0]

    np.testing.assert_allclose(
        analysis.angular_momenta,
        np.broadcast_to(initial, analysis.angular_momenta.shape),
        rtol=1e-9,
        atol=1e-10,
    )


def test_analysis_rejects_invalid_result():
    with pytest.raises(TypeError):
        SimulationAnalysis.from_result("invalid")


def test_analysis_rejects_invalid_gravitational_constant():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    with pytest.raises(ValueError):
        SimulationAnalysis.from_result(
            result,
            gravitational_constant=-1.0,
        )


def test_analysis_rejects_invalid_softening():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    with pytest.raises(ValueError):
        SimulationAnalysis.from_result(
            result,
            softening=-0.1,
        )


def test_analysis_rejects_zero_initial_energy():
    from orbitforge.state import SystemState

    state = SystemState(
        names=("A", "B"),
        masses=np.array([1.0, 1.0]),
        positions=np.array([
            [-0.5, 0.0],
            [0.5, 0.0],
        ]),
        velocities=np.array([
            [0.0, -1.0],
            [0.0, 1.0],
        ]),
    )

    simulation = Simulation(state)
    result = simulation.run(steps=10, dt=0.01)

    with pytest.raises(ValueError, match="initial energy is zero"):
        SimulationAnalysis.from_result(result)


def test_analysis_supports_three_dimensional_systems():
    from orbitforge.state import SystemState

    state = SystemState(
        names=("A", "B"),
        masses=np.array([1.0, 1.0]),
        positions=np.array([
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]),
        velocities=np.array([
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]),
    )

    simulation = Simulation(state)
    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(result)

    assert analysis.linear_momenta.shape == (11, 3)
    assert analysis.angular_momenta.shape == (11, 3)


def test_analysis_supports_two_dimensional_angular_momentum():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(result)

    assert analysis.angular_momenta.shape == (11, 1)



from dataclasses import FrozenInstanceError


def test_analysis_arrays_are_read_only():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(result)

    with pytest.raises(ValueError):
        analysis.total_energies[0] = 999.0

    with pytest.raises(ValueError):
        analysis.angular_momenta[0, 0] = 999.0


def test_analysis_attributes_are_frozen():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(result)

    with pytest.raises(FrozenInstanceError):
        analysis.times = np.array([0.0])


def test_analysis_does_not_modify_original_result():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    original_times = result.times.copy()
    original_positions = result.positions.copy()
    original_velocities = result.velocities.copy()
    original_energies = result.energies.copy()

    SimulationAnalysis.from_result(result)

    np.testing.assert_array_equal(result.times, original_times)
    np.testing.assert_array_equal(result.positions, original_positions)
    np.testing.assert_array_equal(result.velocities, original_velocities)
    np.testing.assert_array_equal(result.energies, original_energies)


def test_analysis_total_energy_matches_simulation_result():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=100, dt=0.001)

    analysis = SimulationAnalysis.from_result(result)

    np.testing.assert_allclose(
        analysis.total_energies,
        result.energies,
        rtol=1e-12,
        atol=1e-12,
    )

def test_analysis_uses_simulation_physical_parameters():
    simulation = Simulation(
        make_two_body_simulation().initial_state,
        gravitational_constant=2.0,
        softening=0.05,
    )

    result = simulation.run(steps=10, dt=0.01)
    analysis = SimulationAnalysis.from_result(result)

    assert result.gravitational_constant == 2.0
    assert result.softening == 0.05

    np.testing.assert_allclose(
        analysis.total_energies,
        result.energies,
        rtol=1e-12,
        atol=1e-12,
    )


def test_analysis_allows_explicit_physical_parameters():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(
        result,
        gravitational_constant=2.0,
        softening=0.05,
    )

    assert analysis.total_energies[0] != pytest.approx(
        result.energies[0]
    )


def test_analysis_uses_original_parameters_when_only_one_is_overridden():
    simulation = Simulation(
        make_two_body_simulation().initial_state,
        gravitational_constant=2.0,
        softening=0.05,
    )

    result = simulation.run(steps=10, dt=0.01)

    analysis = SimulationAnalysis.from_result(
        result,
        gravitational_constant=3.0,
    )

    expected = SimulationAnalysis.from_result(
        result,
        gravitational_constant=3.0,
        softening=0.05,
    )

    np.testing.assert_allclose(
        analysis.total_energies,
        expected.total_energies,
    )

def test_analysis_rejects_inconsistent_energy_shapes():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)
    analysis = SimulationAnalysis.from_result(result)

    with pytest.raises(ValueError, match="kinetic_energies"):
        SimulationAnalysis(
            times=analysis.times,
            kinetic_energies=analysis.kinetic_energies[:-1],
            potential_energies=analysis.potential_energies,
            total_energies=analysis.total_energies,
            linear_momenta=analysis.linear_momenta,
            angular_momenta=analysis.angular_momenta,
            relative_energy_drift=analysis.relative_energy_drift,
        )


def test_analysis_rejects_incompatible_momentum_dimensions():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)
    analysis = SimulationAnalysis.from_result(result)

    with pytest.raises(ValueError, match="angular_momenta"):
        SimulationAnalysis(
            times=analysis.times,
            kinetic_energies=analysis.kinetic_energies,
            potential_energies=analysis.potential_energies,
            total_energies=analysis.total_energies,
            linear_momenta=analysis.linear_momenta,
            angular_momenta=np.zeros((11, 3)),
            relative_energy_drift=analysis.relative_energy_drift,
        )


def test_analysis_rejects_non_finite_values():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)
    analysis = SimulationAnalysis.from_result(result)

    invalid_energies = analysis.total_energies.copy()
    invalid_energies[0] = np.nan

    with pytest.raises(ValueError, match="total_energies"):
        SimulationAnalysis(
            times=analysis.times,
            kinetic_energies=analysis.kinetic_energies,
            potential_energies=analysis.potential_energies,
            total_energies=invalid_energies,
            linear_momenta=analysis.linear_momenta,
            angular_momenta=analysis.angular_momenta,
            relative_energy_drift=analysis.relative_energy_drift,
        )

def test_analysis_rejects_unsorted_times():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)
    analysis = SimulationAnalysis.from_result(result)

    invalid_times = analysis.times.copy()
    invalid_times[5] = invalid_times[4] - 0.001

    with pytest.raises(ValueError, match="strictly increasing"):
        SimulationAnalysis(
            times=invalid_times,
            kinetic_energies=analysis.kinetic_energies,
            potential_energies=analysis.potential_energies,
            total_energies=analysis.total_energies,
            linear_momenta=analysis.linear_momenta,
            angular_momenta=analysis.angular_momenta,
            relative_energy_drift=analysis.relative_energy_drift,
        )


def test_analysis_rejects_duplicate_times():
    simulation = make_two_body_simulation()
    result = simulation.run(steps=10, dt=0.01)
    analysis = SimulationAnalysis.from_result(result)

    invalid_times = analysis.times.copy()
    invalid_times[5] = invalid_times[4]

    with pytest.raises(ValueError, match="strictly increasing"):
        SimulationAnalysis(
            times=invalid_times,
            kinetic_energies=analysis.kinetic_energies,
            potential_energies=analysis.potential_energies,
            total_energies=analysis.total_energies,
            linear_momenta=analysis.linear_momenta,
            angular_momenta=analysis.angular_momenta,
            relative_energy_drift=analysis.relative_energy_drift,
        )
