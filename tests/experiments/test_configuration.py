import json

import numpy as np
import pytest

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.experiments import (
    Experiment,
    ExperimentConfiguration,
)
from orbitforge.simulation import SimulationResult
from orbitforge.units import ASTRONOMICAL_UNITS, SI_UNITS


def make_builder(unit_system=None):
    builder = SystemBuilder(
        dimensions=2,
        unit_system=unit_system,
    )
    builder.add_body("Sun", 1.0, [0.0, 0.0], [0.0, 0.0])
    builder.add_body("Earth", 3.003e-6, [1.0, 0.0], [0.0, 6.283])

    return builder


def make_config():
    return SimulationConfig(
        total_time=1.0,
        dt=0.001,
        save_every=10,
    )


def make_configuration(unit_system=None):
    return ExperimentConfiguration.from_builder(
        name="Earth Orbit",
        builder=make_builder(unit_system),
        config=make_config(),
        description="A simple planetary orbit.",
    )


def test_configuration_creation():
    configuration = make_configuration()

    assert configuration.name == "Earth Orbit"
    assert configuration.description == "A simple planetary orbit."
    assert configuration.initial_state.n_bodies == 2
    assert configuration.initial_state.dimensions == 2
    assert configuration.config.total_time == 1.0


def test_configuration_preserves_initial_state():
    builder = make_builder()

    configuration = ExperimentConfiguration.from_builder(
        "Earth Orbit",
        builder,
        make_config(),
    )

    original_positions = configuration.initial_state.positions.copy()

    builder.update_body("Earth", position=[2.0, 0.0])

    np.testing.assert_allclose(
        configuration.initial_state.positions,
        original_positions,
    )


def test_returned_initial_state_is_independent():
    configuration = make_configuration()

    state = configuration.initial_state
    state.positions[0, 0] = 100.0

    assert configuration.initial_state.positions[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "unit_system",
    [ASTRONOMICAL_UNITS, SI_UNITS],
)
def test_configuration_preserves_unit_system(unit_system):
    configuration = make_configuration(unit_system)

    assert configuration.unit_system is unit_system
    assert configuration.gravitational_constant == pytest.approx(unit_system.gravitational_constant)


def test_configuration_uses_arbitrary_units_by_default():
    configuration = make_configuration()

    assert configuration.unit_system is None
    assert configuration.gravitational_constant == pytest.approx(1.0)


def test_configuration_preserves_softening():
    builder = make_builder()

    configuration = ExperimentConfiguration.from_builder(
        "Softened orbit",
        builder,
        make_config(),
        softening=0.01,
    )

    assert configuration.softening == pytest.approx(0.01)


def test_create_experiment_returns_experiment():
    configuration = make_configuration()

    experiment = configuration.create_experiment()

    assert isinstance(experiment, Experiment)
    assert experiment.name == configuration.name
    assert experiment.config is configuration.config


def test_create_experiment_accepts_custom_name():
    configuration = make_configuration()

    experiment = configuration.create_experiment(name="Modified orbit")

    assert experiment.name == "Modified orbit"
    assert configuration.name == "Earth Orbit"


def test_created_experiment_preserves_initial_state():
    configuration = make_configuration()

    experiment = configuration.create_experiment()

    np.testing.assert_allclose(
        experiment.initial_state.positions,
        configuration.initial_state.positions,
    )


def test_created_experiment_uses_independent_initial_state():
    configuration = make_configuration()

    experiment = configuration.create_experiment()
    state = experiment.initial_state
    state.positions[0, 0] = 100.0

    assert configuration.initial_state.positions[0, 0] == pytest.approx(0.0)


def test_created_experiment_runs():
    configuration = make_configuration()

    experiment = configuration.create_experiment()
    result = experiment.run()

    assert isinstance(result, SimulationResult)
    assert result.final_state.n_bodies == 2


def test_configuration_rejects_invalid_builder():
    with pytest.raises(TypeError):
        ExperimentConfiguration.from_builder(
            "Invalid",
            "not a builder",
            make_config(),
        )


def test_configuration_rejects_empty_name():
    with pytest.raises(ValueError):
        ExperimentConfiguration.from_builder(
            "",
            make_builder(),
            make_config(),
        )


@pytest.mark.parametrize(
    "softening",
    [-0.01, float("inf"), float("nan")],
)
def test_configuration_rejects_invalid_softening(softening):
    with pytest.raises(ValueError):
        ExperimentConfiguration.from_builder(
            "Invalid",
            make_builder(),
            make_config(),
            softening=softening,
        )


def test_with_updates_changes_selected_parameters():
    configuration = make_configuration()

    variant = configuration.with_updates(
        name="Earth Orbit - Fine timestep",
        description="Higher numerical resolution.",
        total_time=2.0,
        dt=0.0005,
        save_every=20,
        softening=0.01,
    )

    assert variant.name == "Earth Orbit - Fine timestep"
    assert variant.description == "Higher numerical resolution."
    assert variant.config.total_time == pytest.approx(2.0)
    assert variant.config.dt == pytest.approx(0.0005)
    assert variant.config.save_every == 20
    assert variant.softening == pytest.approx(0.01)


def test_with_updates_preserves_unmodified_parameters():
    configuration = make_configuration()

    variant = configuration.with_updates(dt=0.0005)

    assert variant.config.total_time == configuration.config.total_time
    assert variant.config.save_every == configuration.config.save_every
    assert variant.config.integrator == configuration.config.integrator
    assert variant.gravitational_constant == configuration.gravitational_constant
    assert variant.softening == configuration.softening
    assert variant.unit_system is configuration.unit_system


def test_with_updates_does_not_modify_original_configuration():
    configuration = make_configuration()

    variant = configuration.with_updates(
        name="Modified orbit",
        dt=0.0005,
        softening=0.01,
    )

    assert configuration.name == "Earth Orbit"
    assert configuration.config.dt == pytest.approx(0.001)
    assert configuration.softening == pytest.approx(0.0)

    assert variant.name == "Modified orbit"
    assert variant.config.dt == pytest.approx(0.0005)
    assert variant.softening == pytest.approx(0.01)


def test_with_updates_preserves_initial_state_independence():
    configuration = make_configuration()

    variant = configuration.with_updates()
    state = variant.initial_state
    state.positions[0, 0] = 100.0

    assert configuration.initial_state.positions[0, 0] == pytest.approx(0.0)
    assert variant.initial_state.positions[0, 0] == pytest.approx(0.0)


def test_with_updates_rejects_invalid_simulation_parameters():
    configuration = make_configuration()

    with pytest.raises(ValueError):
        configuration.with_updates(dt=-0.001)


def test_with_updates_rejects_invalid_name():
    configuration = make_configuration()

    with pytest.raises(ValueError):
        configuration.with_updates(name="")


def test_variant_creates_independent_experiment():
    configuration = make_configuration()

    variant = configuration.with_updates(
        name="Fine timestep",
        dt=0.0005,
    )

    experiment = variant.create_experiment()

    assert isinstance(experiment, Experiment)
    assert experiment.name == "Fine timestep"
    assert experiment.config.dt == pytest.approx(0.0005)


def test_variant_experiment_runs():
    configuration = make_configuration()

    variant = configuration.with_updates(
        name="Longer simulation",
        total_time=2.0,
    )

    experiment = variant.create_experiment()
    result = experiment.run()

    assert isinstance(result, SimulationResult)
    assert result.final_state.time == pytest.approx(2.0)


def test_configuration_to_dict():
    configuration = make_configuration(ASTRONOMICAL_UNITS)

    data = configuration.to_dict()

    assert data["format_version"] == 1
    assert data["name"] == "Earth Orbit"
    assert data["initial_state"]["names"] == ["Sun", "Earth"]
    assert data["simulation"]["dt"] == pytest.approx(0.001)
    assert data["unit_system"]["name"] == "Astronomical"


def test_configuration_from_dict():
    configuration = make_configuration(ASTRONOMICAL_UNITS)

    restored = ExperimentConfiguration.from_dict(configuration.to_dict())

    assert restored.name == configuration.name
    assert restored.description == configuration.description
    assert restored.config.dt == pytest.approx(configuration.config.dt)
    assert restored.gravitational_constant == pytest.approx(configuration.gravitational_constant)
    assert restored.unit_system == configuration.unit_system

    np.testing.assert_allclose(
        restored.initial_state.positions,
        configuration.initial_state.positions,
    )


def test_configuration_round_trip_without_unit_system():
    configuration = make_configuration()

    restored = ExperimentConfiguration.from_dict(configuration.to_dict())

    assert restored.unit_system is None
    assert restored.gravitational_constant == pytest.approx(1.0)


def test_from_dict_rejects_unsupported_version():
    configuration = make_configuration()
    data = configuration.to_dict()
    data["format_version"] = 99

    with pytest.raises(ValueError):
        ExperimentConfiguration.from_dict(data)


def test_from_dict_rejects_non_dictionary():
    with pytest.raises(TypeError, match="dictionary"):
        ExperimentConfiguration.from_dict([])


def test_from_dict_rejects_boolean_format_version():
    configuration = make_configuration()
    data = configuration.to_dict()
    data["format_version"] = True

    with pytest.raises(ValueError, match="format version"):
        ExperimentConfiguration.from_dict(data)


def test_from_dict_rejects_missing_top_level_field():
    configuration = make_configuration()
    data = configuration.to_dict()
    del data["gravitational_constant"]

    with pytest.raises(ValueError, match="Missing configuration fields"):
        ExperimentConfiguration.from_dict(data)


def test_from_dict_rejects_invalid_initial_state_section():
    configuration = make_configuration()
    data = configuration.to_dict()
    data["initial_state"] = []

    with pytest.raises(TypeError, match="initial_state"):
        ExperimentConfiguration.from_dict(data)


def test_from_dict_rejects_missing_initial_state_field():
    configuration = make_configuration()
    data = configuration.to_dict()
    del data["initial_state"]["masses"]

    with pytest.raises(ValueError, match="Missing initial_state fields"):
        ExperimentConfiguration.from_dict(data)


def test_from_dict_rejects_invalid_simulation_section():
    configuration = make_configuration()
    data = configuration.to_dict()
    data["simulation"] = []

    with pytest.raises(TypeError, match="simulation"):
        ExperimentConfiguration.from_dict(data)


def test_from_dict_rejects_missing_simulation_field():
    configuration = make_configuration()
    data = configuration.to_dict()
    del data["simulation"]["dt"]

    with pytest.raises(ValueError, match="Missing simulation fields"):
        ExperimentConfiguration.from_dict(data)


def test_from_dict_rejects_invalid_unit_system_section():
    configuration = make_configuration()
    data = configuration.to_dict()
    data["unit_system"] = "Astronomical"

    with pytest.raises(TypeError, match="unit_system"):
        ExperimentConfiguration.from_dict(data)


def test_configuration_save_and_load(tmp_path):
    configuration = make_configuration(ASTRONOMICAL_UNITS)
    file_path = tmp_path / "earth_orbit.json"

    configuration.save(file_path)
    restored = ExperimentConfiguration.load(file_path)

    assert restored.name == configuration.name
    assert restored.description == configuration.description
    assert restored.config == configuration.config
    assert restored.gravitational_constant == pytest.approx(configuration.gravitational_constant)
    assert restored.unit_system == configuration.unit_system

    np.testing.assert_allclose(
        restored.initial_state.positions,
        configuration.initial_state.positions,
    )
    np.testing.assert_allclose(
        restored.initial_state.velocities,
        configuration.initial_state.velocities,
    )


def test_configuration_save_creates_valid_json(tmp_path):
    import json

    configuration = make_configuration()
    file_path = tmp_path / "configuration.json"

    configuration.save(file_path)

    with file_path.open(encoding="utf-8") as file:
        data = json.load(file)

    assert data["format_version"] == 1
    assert data["name"] == configuration.name


def test_configuration_load_rejects_invalid_json(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        ExperimentConfiguration.load(file_path)


def test_loaded_configuration_can_create_and_run_experiment(tmp_path):
    configuration = make_configuration(ASTRONOMICAL_UNITS)

    variant = configuration.with_updates(
        name="Loaded Earth Orbit",
        dt=0.0005,
        total_time=1.0,
    )

    file_path = tmp_path / "loaded_orbit.json"
    variant.save(file_path)

    restored = ExperimentConfiguration.load(file_path)
    experiment = restored.create_experiment()
    result = experiment.run()

    assert experiment.name == "Loaded Earth Orbit"
    assert experiment.config.dt == pytest.approx(0.0005)
    assert isinstance(result, SimulationResult)
    assert result.final_state.time == pytest.approx(1.0)
