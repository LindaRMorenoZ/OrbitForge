
"""Visualize N-body simulations with Matplotlib."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .simulation import Simulation
from .state import Body, SystemState


def create_demo_simulation(steps: int, dt: float):
    """Create and run a reproducible two-body simulation."""

    bodies = [
        Body("primary", 1.0, np.array([-0.5, 0.0])),
        Body("secondary", 1.0, np.array([0.5, 0.0])),
    ]

    state = SystemState.from_bodies(
        bodies,
        velocities=np.array([[0.0, -0.5], [0.0, 0.5]]),
    )

    simulation = Simulation(
        state,
        gravitational_constant=1.0,
        softening=0.01,
    )

    save_every = max(1, steps // 500)

    return simulation.run(
        steps,
        dt,
        save_every=save_every,
    ), state.names


def visualize(result, names, interval: int = 30):
    """Display orbital trajectories and energy evolution."""

    positions = result.positions
    times = result.times
    energies = result.energies

    if positions.shape[2] != 2:
        raise ValueError("This visualization currently supports 2D systems only.")

    fig, (ax_orbit, ax_energy) = plt.subplots(
        1, 2, figsize=(13, 6)
    )

    fig.suptitle("OrbitForge - N-body Simulation", fontsize=15)

    # Orbital panel
    ax_orbit.set_title("Orbital trajectories")
    ax_orbit.set_xlabel("X position")
    ax_orbit.set_ylabel("Y position")
    ax_orbit.grid(True, alpha=0.3)
    ax_orbit.set_aspect("equal", adjustable="datalim")

    all_x = positions[:, :, 0]
    all_y = positions[:, :, 1]

    margin = 0.15
    x_range = np.ptp(all_x)
    y_range = np.ptp(all_y)

    ax_orbit.set_xlim(
        all_x.min() - margin * max(x_range, 1),
        all_x.max() + margin * max(x_range, 1),
    )
    ax_orbit.set_ylim(
        all_y.min() - margin * max(y_range, 1),
        all_y.max() + margin * max(y_range, 1),
    )

    lines = []
    points = []

    for name in names:
        line, = ax_orbit.plot([], [], label=name)
        point, = ax_orbit.plot([], [], "o", markersize=8)

        lines.append(line)
        points.append(point)

    ax_orbit.legend()

    # Energy panel
    ax_energy.set_title("Relative energy change")
    ax_energy.set_xlabel("Time")
    ax_energy.set_ylabel("Relative energy change")
    ax_energy.grid(True, alpha=0.3)

    relative_energy = (energies - energies[0]) / abs(energies[0])

    ax_energy.plot(
        times,
        relative_energy,
        color="black",
        alpha=0.25,
        label="Complete simulation",
    )

    energy_line, = ax_energy.plot([], [], color="red", label="Current time")
    ax_energy.legend()

    time_label = fig.text(0.5, 0.02, "", ha="center")

    def update(frame):
        for index in range(len(names)):
            x = positions[:frame + 1, index, 0]
            y = positions[:frame + 1, index, 1]

            lines[index].set_data(x, y)
            points[index].set_data(
                [positions[frame, index, 0]],
                [positions[frame, index, 1]],
            )

        energy_line.set_data(
            times[:frame + 1],
            relative_energy[:frame + 1],
        )

        time_label.set_text(f"Simulation time: {times[frame]:.4f}")

        return (*lines, *points, energy_line, time_label)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval,
        blit=False,
        repeat=True,
    )

    # Keep a reference to the animation while the window is open.
    fig._orbitforge_animation = animation

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize an OrbitForge two-body simulation."
    )

    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--interval", type=int, default=30)

    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")

    if args.interval < 1:
        parser.error("--interval must be positive")

    result, names = create_demo_simulation(args.steps, args.dt)

    visualize(result, names, args.interval)


if __name__ == "__main__":
    main()
