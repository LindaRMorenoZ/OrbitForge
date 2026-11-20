import numpy as np
import pytest

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.experiments import Experiment
from orbitforge.simulation import SimulationResult
from orbitforge.analysis import SimulationAnalysis


def make_builder():
    builder = SystemBuilder(dimensions=2)

    builder.add_body(
        "Sun", 1.0, [0.0, 0.0], [0.0, 0.0]
    )
    builder.add_body(
        "Earth", 3.003e-6, [1.0, 0.0], [0.0, 6.283]
    )

    return builder

def make_config():
    return SimulationConfig(
        total_time=1.0,
        dt=0.001,
        save_every=10,
    )

def test_experiment_creation():
    builder = make_builder()
    config = make_config()

    experiment = Experiment(
        name="Earth Orbit",
        builder=builder,
        config=config,
    )

    assert experiment.name == "Earth Orbit"
    assert experiment.config is config
    assert experiment.initial_state.n_bodies == 2
    assert experiment.initial_state.dimensions == 2

def test_experiment_preserves_initial_state():
    builder = make_builder()

    experiment = Experiment(
        name="Earth Orbit",
        builder=builder,
        config=make_config(),
    )

    original_positions = experiment.initial_state.positions.copy()

    builder.update_body(
        "Earth",
        position=[2.0, 0.0],
    )

    np.testing.assert_allclose(
        experiment.initial_state.positions,
        original_positions,
    )

def test_experiment_rejects_empty_name():
    with pytest.raises(ValueError):
        Experiment(
            name="",
            builder=make_builder(),
            config=make_config(),
        )

def test_experiment_rejects_invalid_builder():
    with pytest.raises(TypeError):
        Experiment(
            name="Invalid",
            builder="not a builder",
            config=make_config(),
        )

def test_experiment_rejects_invalid_config():
    with pytest.raises(TypeError):
        Experiment(
            name="Invalid",
            builder=make_builder(),
            config="not a config",
        )

def test_experiment_run_returns_simulation_result():
    experiment = Experiment(
        name="Earth Orbit",
        builder=make_builder(),
        config=make_config(),
    )

    result = experiment.run()

    assert isinstance(result, SimulationResult)
    assert result.final_state.n_bodies == 2

def test_experiment_run_respects_configuration():
    experiment = Experiment(
        name="Earth Orbit",
        builder=make_builder(),
        config=make_config(),
    )

    result = experiment.run()

    assert result.times[0] == pytest.approx(0.0)
    assert result.times[-1] == pytest.approx(1.0)
    assert len(result.times) == 101

def test_experiment_stores_simulation_result():
    experiment = Experiment(
        name="Earth Orbit",
        builder=make_builder(),
        config=make_config(),
    )

    result = experiment.run()

    assert experiment.result is result

def test_experiment_result_is_initially_none():
    experiment = Experiment(
        name="Earth Orbit",
        builder=make_builder(),
        config=make_config(),
    )

    assert experiment.result is None

def test_experiment_analyze_requires_completed_run():
    experiment = Experiment(
        name="Earth Orbit",
        builder=make_builder(),
        config=make_config(),
    )

    with pytest.raises(RuntimeError):
        experiment.analyze()

def test_experiment_analyze_returns_simulation_analysis():
    experiment = Experiment(
        name="Earth Orbit",
        builder=make_builder(),
        config=make_config(),
    )

    experiment.run()
    analysis = experiment.analyze()

    assert isinstance(analysis, SimulationAnalysis)

def test_experiment_stores_analysis():
    experiment = Experiment(
        name="Earth Orbit",
        builder=make_builder(),
        config=make_config(),
    )

    experiment.run()
    analysis = experiment.analyze()

    assert experiment.analysis is analysis

def test_experiment_analysis_is_initially_none():
    experiment = Experiment(
        name="Earth Orbit",
        builder=make_builder(),
        config=make_config(),
    )

    assert experiment.analysis is None

def test_evaluate_requires_prior_analysis():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    experiment.run()

    with pytest.raises(RuntimeError):
        experiment.evaluate()

def test_evaluate_returns_conservation_metrics():
    from orbitforge.metrics import ConservationMetrics

    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    experiment.run()
    experiment.analyze()

    metrics = experiment.evaluate()

    assert isinstance(metrics, ConservationMetrics)

def test_evaluate_stores_metrics():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    experiment.run()
    experiment.analyze()

    metrics = experiment.evaluate()

    assert experiment.metrics is metrics

def test_metrics_initially_none():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    assert experiment.metrics is None
from orbitforge.units import ASTRONOMICAL_UNITS, SI_UNITS

@pytest.mark.parametrize(
    "unit_system",
    [ASTRONOMICAL_UNITS, SI_UNITS],
)
def test_run_preserves_builder_gravitational_constant(unit_system):
    builder = SystemBuilder(
        dimensions=2,
        unit_system=unit_system,
    )
    builder.add_body("Sun", 1.0, [0.0, 0.0], [0.0, 0.0])
    builder.add_body("Earth", 3.003e-6, [1.0, 0.0], [0.0, 6.283])

    experiment = Experiment(
        "Unit system test",
        builder,
        make_config(),
    )

    result = experiment.run()

    assert result.gravitational_constant == pytest.approx(
        builder.gravitational_constant
    )
def test_rerun_invalidates_previous_analysis_and_metrics():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    experiment.run()
    experiment.analyze()
    experiment.evaluate()

    experiment.run()

    assert experiment.analysis is None
    assert experiment.metrics is None

def test_experiment_uses_default_softening():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    result = experiment.run()

    assert result.softening == 0.0

def test_experiment_preserves_custom_softening():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
        softening=0.01,
    )

    result = experiment.run()

    assert result.softening == pytest.approx(0.01)


@pytest.mark.parametrize("softening", [-0.01, float("inf"), float("nan")])
def test_experiment_rejects_invalid_softening(softening):
    with pytest.raises(ValueError):
        Experiment(
            "Earth orbit",
            make_builder(),
            make_config(),
            softening=softening,
        )
from orbitforge.controller import (
    ControllerStatus,
    SimulationController,
)

def test_create_controller_returns_simulation_controller():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    controller = experiment.create_controller()

    assert isinstance(controller, SimulationController)

def test_create_controller_starts_in_ready_state():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    controller = experiment.create_controller()

    assert controller.status == ControllerStatus.READY

def test_create_controller_preserves_simulation_parameters():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
        softening=0.01,
    )

    controller = experiment.create_controller()

    assert (
        controller.simulation.gravitational_constant
        == experiment._gravitational_constant
    )
    assert controller.simulation.softening == pytest.approx(0.01)
    assert controller.config is experiment.config

def test_create_controller_uses_independent_initial_state():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    controller = experiment.create_controller()

    controller.current_state.positions[0, 0] = 100.0

    assert experiment.initial_state.positions[0, 0] == pytest.approx(0.0)

def test_created_controller_completes_experiment():
    experiment = Experiment(
        "Earth orbit",
        make_builder(),
        make_config(),
    )

    controller = experiment.create_controller()
    controller.start()

    while controller.status == ControllerStatus.RUNNING:
        controller.step()

    assert controller.status == ControllerStatus.COMPLETED
    assert controller.result is not None
    assert controller.result.final_state.time == pytest.approx(
        experiment.config.total_time
    )
