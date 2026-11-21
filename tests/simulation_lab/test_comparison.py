import pytest

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.experiments import Experiment
from orbitforge.simulation_lab.comparison import ExperimentComparison
from orbitforge.simulation_lab.manager import ExperimentManager


@pytest.fixture
def evaluated_manager():
    manager = ExperimentManager()
    config = SimulationConfig(total_time=0.1, dt=0.01)

    for name, mass, velocity in [
        ("Experiment A", 1.0, [1.0, 0.0]),
        ("Experiment B", 2.0, [0.0, 1.0]),
    ]:
        builder = SystemBuilder()
        builder.add_body(
            name="Body",
            mass=mass,
            position=[0.0, 0.0],
            velocity=velocity,
        )

        experiment = Experiment(name, builder, config)
        session = manager.create_session(experiment)

        session.start()

        while session.status.name == "RUNNING":
            session.step()

        session.analyze()
        session.evaluate()

    return manager


def test_comparison_requires_manager():
    with pytest.raises(TypeError):
        ExperimentComparison("not a manager")


def test_comparison_requires_at_least_two_sessions():
    manager = ExperimentManager()

    with pytest.raises(ValueError):
        ExperimentComparison(manager).compare()


def test_comparison_requires_evaluated_sessions():
    manager = ExperimentManager()

    builder = SystemBuilder()
    builder.add_body(
        name="Body",
        mass=1.0,
        position=[0.0, 0.0],
        velocity=[1.0, 0.0],
    )

    experiment = Experiment(
        "Experiment A",
        builder,
        SimulationConfig(total_time=0.1, dt=0.01),
    )
    manager.create_session(experiment)

    with pytest.raises(ValueError):
        ExperimentComparison(manager).compare()

def test_comparison_returns_pairwise_differences(evaluated_manager):
    comparison = ExperimentComparison(evaluated_manager)
    result = comparison.compare()

    assert list(result) == [("Experiment A", "Experiment B")]

    differences = result[("Experiment A", "Experiment B")]

    assert differences["max_relative_energy_drift"] == pytest.approx(0.0)
    assert differences["max_relative_linear_momentum_drift"] == pytest.approx(0.0)
    assert differences["max_relative_angular_momentum_drift"] is None        