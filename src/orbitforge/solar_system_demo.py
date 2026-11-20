
"""Run and visualize the fixed Solar System reference scenario."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .scenarios.solar_system import G_AU, build_solar_system
from .simulation import Simulation


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the fixed OrbitForge Solar System."
    )

    parser.add_argument("--years", type=float, default=165.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--interval", type=int, default=30)

    args = parser.parse_args()

    if args.years <= 0 or args.dt <= 0 or args.interval <= 0:
        parser.error("years, dt and interval must be positive")

    # Load the predefined, immutable reference scenario.
    initial_state = build_solar_system()

    steps = int(round(args.years / args.dt))
    save_every = 1

    print("Running the fixed Solar System simulation...")
    print(f"Bodies: {initial_state.n_bodies}")
    print(f"Duration: {args.years:g} years")
    print(f"Time step: {args.dt:g} years")
    print(f"Integration steps: {steps}")

    simulation = Simulation(
        initial_state,
        gravitational_constant=G_AU,
        softening=0.0,
    )

    result = simulation.run(
        steps,
        args.dt,
        save_every=save_every,
    )

    relative_drift = (
        result.energies - result.energies[0]
    ) / abs(result.energies[0])

    print(f"Final time: {result.final_state.time:.4f} years")
    print(
        "Maximum relative energy drift: "
        f"{np.max(np.abs(relative_drift)):.3e}"
    )

    
    
    positions = result.positions
    times = result.times

    frame_indices = np.linspace(
    0,
    len(times) - 1,
    min(3000, len(times)),
    dtype=int,
)


    names = initial_state.names

    fig, ax_orbit = plt.subplots( figsize=(11, 11))

    fig.suptitle(
        "OrbitForge - Solar System Reference Simulation",
        fontsize=15,
    )

    # Orbital trajectories.
    ax_orbit.set_title("Planetary orbits (XY projection)")
    ax_orbit.set_xlabel("X positions (AU)")
    ax_orbit.set_ylabel("Y positions (AU)")
    ax_orbit.set_aspect("equal")
    ax_orbit.grid(True, alpha=0.3)
    ax_orbit.set_xlim(-34, 34)
    ax_orbit.set_ylim(-34, 34)

    lines = []
    points = []

    for index, name in enumerate(names):
        line, = ax_orbit.plot(
             [],
             [],
	        linewidth=0.8,
            label=name,
        )

        size = 10 if name == "Sun" else 5

        point, = ax_orbit.plot(
            [], [],
            marker="o",
            linestyle="None",
            markersize=size,
        )

        lines.append(line)
        points.append(point)

    ax_orbit.legend(
        loc="upper left",
        fontsize=8,
        ncol=2,
    )

    time_label = fig.text(
        0.5, 0.02, "",
        ha="center",
    )


    def update(frame):
        current = frame_indices[frame]

        for index in range(len(names)):
            lines[index].set_data(
                positions[:current +1, index, 0],
                positions[:current +1, index, 1],
            )

            points[index].set_data(
               [positions[current, index, 0]],
               [positions[current, index, 1]],
            )
       
        time_label.set_text(
            f"Simulation time: {times[current]:.2f} years"
        )

        return (*lines, *points, time_label)


    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    fig._orbitforge_animation = animation

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
