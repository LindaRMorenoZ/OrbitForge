"""Interactive three-body gravitational simulation in 3D."""

import math

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.experiments.experiment import Experiment
from orbitforge.simulation_lab import (
    ExperimentManager,
    visualize_interactive_session,
)


def main():
    builder = SystemBuilder(dimensions=3)

    position_a = [1.0, 0.0, 0.0]
    velocity_a = [0.0, 1.0, 0.0]

    position_b = [0.0, 1.5, 0.8]
    radius_b = math.sqrt(sum(x * x for x in position_b))
    speed_b = math.sqrt(1.0 / radius_b)
    velocity_b = [-speed_b, 0.0, 0.0]

    central_velocity = [
        -0.001 * (velocity_a[i] + velocity_b[i])
        for i in range(3)
    ]

    builder.add_body(
        "Central mass", 1.0, [0.0, 0.0, 0.0], central_velocity
    )
    builder.add_body("Body A", 0.001, position_a, velocity_a)
    builder.add_body("Body B", 0.001, position_b, velocity_b)

    config = SimulationConfig(
        total_time=40.0,
        dt=0.002,
        save_every=1,
        integrator="velocity_verlet",
    )

    experiment = Experiment(
        name="Three-body 3D demonstration",
        builder=builder,
        config=config,
        softening=0.0,
    )

    manager = ExperimentManager()
    session = manager.create_session(experiment)

    session.start()

    visualize_interactive_session(
        session,
        interval=30,
        steps_per_frame=10,
    )


if __name__ == "__main__":
    main()
