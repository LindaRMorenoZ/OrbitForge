
"""Interactive ten-body gravitational simulation in 3D."""

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

    central_mass = 1.0
    builder.add_body(
        "Central mass",
        central_mass,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )

    orbital_data = [
        ("Body 1", 1.0, 0.0, 0.0),
        ("Body 2", 1.5, 0.4, 0.0),
        ("Body 3", 2.0, 0.8, 0.5),
        ("Body 4", 2.5, 1.2, 0.8),
        ("Body 5", 3.0, 1.6, 1.0),
        ("Body 6", 3.5, 2.0, 1.2),
        ("Body 7", 4.0, 2.4, 1.5),
        ("Body 8", 4.5, 2.8, 1.8),
        ("Body 9", 5.0, 3.2, 2.0),
    ]

    for name, radius, angle, inclination in orbital_data:
        position = [
            radius * math.cos(angle),
            radius * math.sin(angle) * math.cos(inclination),
            radius * math.sin(angle) * math.sin(inclination),
        ]

        distance = math.sqrt(sum(x * x for x in position))
        speed = math.sqrt(central_mass / distance)

        velocity = [
            -speed * math.sin(angle),
            speed * math.cos(angle) * math.cos(inclination),
            speed * math.cos(angle) * math.sin(inclination),
        ]

        builder.add_body(name, 0.001, position, velocity)

    config = SimulationConfig(
        total_time=40.0,
        dt=0.002,
        save_every=1,
        integrator="velocity_verlet",
    )

    experiment = Experiment(
        name="Ten-body 3D diagnostics test",
        builder=builder,
        config=config,
        softening=0.01,
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
