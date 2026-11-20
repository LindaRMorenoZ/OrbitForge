
import numpy as np
import pytest

from orbitforge.builder.system_builder import SystemBuilder



def test_build_two_body_system():
    builder = SystemBuilder(dimensions=2)

    builder.add_body("Star", 10.0, [0, 0], [0, 0])
    builder.add_body("Planet", 1.0, [1, 0], [0, 2])

    state = builder.build()

    assert state.n_bodies == 2
    assert state.dimensions == 2
    assert state.names == ("Star", "Planet")
    np.testing.assert_allclose(state.masses, [10.0, 1.0])
    np.testing.assert_allclose(state.positions, [[0, 0], [1, 0]])
    np.testing.assert_allclose(state.velocities, [[0, 0], [0, 2]])


def test_build_three_dimensional_system():
    builder = SystemBuilder(dimensions=3)

    builder.add_body("Star", 10.0, [0, 0, 0], [0, 0, 0])
    builder.add_body("Planet", 1.0, [1, 0, 0], [0, 2, 0])

    state = builder.build()

    assert state.dimensions == 3
    assert state.n_bodies == 2


def test_remove_body():
    builder = SystemBuilder()

    builder.add_body("Star", 10.0, [0, 0], [0, 0])
    builder.add_body("Planet", 1.0, [1, 0], [0, 2])

    builder.remove_body("Planet")

    assert builder.n_bodies == 1
    assert builder.bodies[0].name == "Star"


def test_duplicate_names_are_rejected():
    builder = SystemBuilder()

    builder.add_body("Star", 10.0, [0, 0], [0, 0])

    with pytest.raises(ValueError, match="already exists"):
        builder.add_body("Star", 1.0, [1, 0], [0, 2])


def test_empty_system_is_rejected():
    builder = SystemBuilder()

    with pytest.raises(ValueError, match="empty system"):
        builder.build()


def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        SystemBuilder(dimensions=4)


def test_invalid_position_dimensions_are_rejected():
    builder = SystemBuilder(dimensions=2)

    with pytest.raises(ValueError, match="position"):
        builder.add_body("Planet", 1.0, [1, 0, 0], [0, 2])


def test_invalid_velocity_dimensions_are_rejected():
    builder = SystemBuilder(dimensions=2)

    with pytest.raises(ValueError, match="velocity"):
        builder.add_body("Planet", 1.0, [1, 0], [0, 2, 0])


def test_non_finite_velocity_is_rejected():
    builder = SystemBuilder()

    with pytest.raises(ValueError, match="finite"):
        builder.add_body("Planet", 1.0, [1, 0], [np.nan, 2])


def test_removing_unknown_body_is_rejected():
    builder = SystemBuilder()

    with pytest.raises(ValueError, match="not found"):
        builder.remove_body("Unknown")

def test_update_body_properties():
    builder = SystemBuilder()

    builder.add_body("Planet", 1.0, [1, 0], [0, 2])

    builder.update_body(
        "Planet",
        new_name="Earth",
        mass=2.0,
        position=[2, 0],
        velocity=[0, 3],
    )

    state = builder.build()

    assert state.names == ("Earth",)
    np.testing.assert_allclose(state.masses, [2.0])
    np.testing.assert_allclose(state.positions, [[2, 0]])
    np.testing.assert_allclose(state.velocities, [[0, 3]])


def test_update_unknown_body():
    builder = SystemBuilder()

    with pytest.raises(ValueError, match="not found"):
        builder.update_body("Unknown", mass=2.0)


def test_update_with_duplicate_name():
    builder = SystemBuilder()

    builder.add_body("Star", 10.0, [0, 0], [0, 0])
    builder.add_body("Planet", 1.0, [1, 0], [0, 2])

    with pytest.raises(ValueError, match="already exists"):
        builder.update_body("Planet", new_name="Star")


def test_update_with_invalid_mass():
    builder = SystemBuilder()
    builder.add_body("Planet", 1.0, [1, 0], [0, 2])

    with pytest.raises(ValueError, match="mass"):
        builder.update_body("Planet", mass=-1.0)

def test_builder_creates_system_compatible_with_simulation():
    from orbitforge import Simulation

    builder = SystemBuilder(dimensions=2)

    builder.add_body("Star", 1.0, [-0.5, 0], [0, -0.5])
    builder.add_body("Planet A", 1.0, [0.5, 0], [0, 0.5])
    builder.add_body("Planet B", 0.5, [0, 2], [-1, 0])

    initial_state = builder.build()

    simulation = Simulation(
        initial_state,
        gravitational_constant=1.0,
    )

    result = simulation.run(
        steps=1000,
        dt=0.001,
    )

    assert result.final_state.n_bodies == 3
    assert result.positions.shape[1] == 3
    assert result.times[-1] == pytest.approx(1.0)
    assert np.all(np.isfinite(result.positions))
    assert np.all(np.isfinite(result.energies))

from orbitforge.units import ASTRONOMICAL_UNITS, SI_UNITS


def test_builder_uses_astronomical_units():
    builder = SystemBuilder(
        dimensions=3,
        unit_system=ASTRONOMICAL_UNITS,
    )

    assert builder.unit_system is ASTRONOMICAL_UNITS
    assert builder.gravitational_constant == pytest.approx(
        4.0 * np.pi**2, rel=1e-4
    )


def test_builder_uses_si_units():
    builder = SystemBuilder(unit_system=SI_UNITS)

    assert builder.gravitational_constant == pytest.approx(6.67430e-11)


def test_builder_defaults_to_arbitrary_units():
    builder = SystemBuilder()

    assert builder.unit_system is None
    assert builder.gravitational_constant == 1.0


def test_builder_rejects_invalid_unit_system():
    with pytest.raises(TypeError):
        SystemBuilder(unit_system="AU")

def test_create_simulation_uses_builder_units():
    builder = SystemBuilder(
        dimensions=2,
        unit_system=ASTRONOMICAL_UNITS,
    )

    builder.add_body("Sun", 1.0, [0, 0], [0, 0])
    builder.add_body("Earth", 3.003e-6, [1, 0], [0, 2 * np.pi])

    simulation = builder.create_simulation()

    assert simulation.gravitational_constant == pytest.approx(
        ASTRONOMICAL_UNITS.gravitational_constant
    )


def test_create_simulation_uses_softening():
    builder = SystemBuilder()

    builder.add_body("A", 1.0, [-1, 0], [0, 0])
    builder.add_body("B", 1.0, [1, 0], [0, 0])

    simulation = builder.create_simulation(softening=0.01)

    assert simulation.softening == pytest.approx(0.01)


def test_create_simulation_uses_initial_time():
    builder = SystemBuilder()

    builder.add_body("A", 1.0, [-1, 0], [0, 0])
    builder.add_body("B", 1.0, [1, 0], [0, 0])

    simulation = builder.create_simulation(time=5.0)

    assert simulation.initial_state.time == pytest.approx(5.0)