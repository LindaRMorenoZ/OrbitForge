"""Animate a hierarchical three-body simulation with OrbitForge."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from orbitforge.simulation import Simulation
from orbitforge.state import Body, SystemState


def main():
    outer_mass = 0.2
    inner_mass = 1.0
    outer_radius = 4.0
    total_inner_mass = 2.0
    total_mass = total_inner_mass + outer_mass

    outer_speed = np.sqrt(total_mass / outer_radius)
    inner_speed = np.sqrt(0.5)

    inner_com_x = -outer_mass / total_mass * outer_radius
    outer_x = total_inner_mass / total_mass * outer_radius

    inner_com_vy = -outer_mass / total_mass * outer_speed
    outer_vy = total_inner_mass / total_mass * outer_speed

    bodies = [
        Body("Body A", inner_mass, np.array([inner_com_x - 0.5, 0.0])),
        Body("Body B", inner_mass, np.array([inner_com_x + 0.5, 0.0])),
        Body("Body C", outer_mass, np.array([outer_x, 0.0])),
    ]

    velocities = np.array([
        [0.0, inner_com_vy - inner_speed],
        [0.0, inner_com_vy + inner_speed],
        [0.0, outer_vy],
    ])

    state = SystemState.from_bodies(bodies, velocities)

    dt = 0.001
    steps = 40000

    simulation = Simulation(
        state,
        gravitational_constant=1.0,
        softening=0.0,
    )

    result = simulation.run(steps=steps, dt=dt, save_every=10)

    positions = result.positions
    times = result.times
    relative_energy_error = (
        result.energies - result.energies[0]
    ) / abs(result.energies[0])

    colors = ["#54D6FF", "#FFB45C", "#D69BFF"]

    plt.style.use("dark_background")
    fig, (ax_orbit, ax_energy) = plt.subplots(
        1, 2, figsize=(14, 7),
        facecolor="#080D18",
        gridspec_kw={"width_ratios": [1.5, 1]},
    )

    for ax in (ax_orbit, ax_energy):
        ax.set_facecolor("#0D1524")
        ax.grid(color="#40516A", alpha=0.35)

    ax_orbit.set_title("Hierarchical three-body system")
    ax_orbit.set_xlabel("X position")
    ax_orbit.set_ylabel("Y position")
    ax_orbit.set_aspect("equal", adjustable="box")

    # Show the complete computed paths faintly as a reference.
    for index, color in enumerate(colors):
        ax_orbit.plot(
            positions[:, index, 0],
            positions[:, index, 1],
            color=color,
            alpha=0.18,
            linewidth=0.8,
        )

    all_x = positions[:, :, 0]
    all_y = positions[:, :, 1]
    x_margin = 0.08 * np.ptp(all_x)
    y_margin = 0.08 * np.ptp(all_y)

    ax_orbit.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax_orbit.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    trails = []
    markers = []

    for index, color in enumerate(colors):
        trail, = ax_orbit.plot([], [], color=color, linewidth=1.8)
        marker, = ax_orbit.plot(
            [], [], "o",
            color=color,
            markersize=10 if index < 2 else 7,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=f"Body {chr(65 + index)}",
        )
        trails.append(trail)
        markers.append(marker)

    ax_orbit.legend(loc="upper right")

    ax_energy.set_title("Relative energy drift")
    ax_energy.set_xlabel("Simulation time")
    ax_energy.set_ylabel(r"$(E(t)-E_0)/|E_0|$")
    ax_energy.plot(
        times,
        relative_energy_error,
        color="#66E0B5",
        alpha=0.25,
    )
    energy_line, = ax_energy.plot([], [], color="#66E0B5", linewidth=1.8)
    ax_energy.axhline(0, color="white", alpha=0.3, linewidth=0.7)

    fig.suptitle(
        "ORBITFORGE | Hierarchical three-body simulation",
        color="white",
    )
    status_label = fig.text(
        0.5, 0.02, "",
        ha="center",
        color="#AAB7C8",
    )

    def update(frame):
        for index in range(3):
            trails[index].set_data(
                positions[:frame + 1, index, 0],
                positions[:frame + 1, index, 1],
            )
            markers[index].set_data(
                [positions[frame, index, 0]],
                [positions[frame, index, 1]],
            )

        energy_line.set_data(
            times[:frame + 1],
            relative_energy_error[:frame + 1],
        )

        status_label.set_text(
            f"Time: {times[frame]:.3f}  |  "
            f"Relative energy drift: {relative_energy_error[frame]:.3e}"
        )

        return (*trails, *markers, energy_line, status_label)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=20,
        blit=False,
        repeat=True,
    )

    # Keep the animation alive while the window is open.
    fig._orbitforge_animation = animation

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.show()


if __name__ == "__main__":
    main()
