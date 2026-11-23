"""Generate a three-body figure-eight simulation with OrbitForge."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from orbitforge.simulation import Simulation
from orbitforge.state import Body, SystemState


def main():
    bodies = [
        Body("Body A", 1.0, np.array([-0.97000436, 0.24308753])),
        Body("Body B", 1.0, np.array([0.97000436, -0.24308753])),
        Body("Body C", 1.0, np.array([0.0, 0.0])),
    ]

    velocities = np.array([
        [0.466203685, 0.43236573],
        [0.466203685, 0.43236573],
        [-0.93240737, -0.86473146],
    ])

    state = SystemState.from_bodies(bodies, velocities)

    dt = 0.001
    steps = 6326

    simulation = Simulation(
        state,
        gravitational_constant=1.0,
        softening=0.0,
    )

    result = simulation.run(steps=steps, dt=dt, save_every=2)

    positions = result.positions
    times = result.times
    relative_energy_error = (
        result.energies - result.energies[0]
    ) / abs(result.energies[0])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": "#E8EDF5",
        "axes.labelcolor": "#AAB7C8",
        "xtick.color": "#AAB7C8",
        "ytick.color": "#AAB7C8",
    })

    fig = plt.figure(figsize=(16, 9), facecolor="#080D18")
    grid = fig.add_gridspec(
        2, 2,
        width_ratios=[1.65, 1],
        height_ratios=[1, 0.12],
        hspace=0.12,
        wspace=0.16,
    )

    ax_orbit = fig.add_subplot(grid[0, 0])
    ax_energy = fig.add_subplot(grid[0, 1])
    ax_footer = fig.add_subplot(grid[1, :])
    ax_footer.axis("off")

    for ax in (ax_orbit, ax_energy):
        ax.set_facecolor("#0D1524")
        for spine in ax.spines.values():
            spine.set_color("#26354B")
        ax.grid(color="#26354B", linewidth=0.7, alpha=0.55)

    colors = ["#54D6FF", "#FFB45C", "#D69BFF"]

    for index, color in enumerate(colors):
        ax_orbit.plot(
            positions[:, index, 0],
            positions[:, index, 1],
            color=color,
            linewidth=1.8,
            alpha=0.9,
            label=f"Body {chr(65 + index)}",
        )
        ax_orbit.scatter(
            positions[0, index, 0],
            positions[0, index, 1],
            color=color,
            s=65,
            edgecolor="#FFFFFF",
            linewidth=0.8,
            zorder=5,
        )

    ax_orbit.set_title(
        "THREE-BODY TRAJECTORIES",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#F2F6FC",
        pad=14,
    )
    ax_orbit.set_xlabel("X position")
    ax_orbit.set_ylabel("Y position")
    ax_orbit.set_aspect("equal", adjustable="box")
    ax_orbit.legend(
        frameon=False,
        labelcolor="#E8EDF5",
        loc="upper right",
    )

    ax_energy.plot(
        times,
        relative_energy_error,
        color="#66E0B5",
        linewidth=1.5,
    )
    ax_energy.axhline(0, color="#AAB7C8", linewidth=0.7, alpha=0.5)
    ax_energy.set_title(
        "RELATIVE ENERGY DRIFT",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#F2F6FC",
        pad=14,
    )
    ax_energy.set_xlabel("Simulation time")
    ax_energy.set_ylabel(r"$(E(t)-E_0)/|E_0|$")

    max_drift = np.max(np.abs(relative_energy_error))

    fig.suptitle(
        "ORBITFORGE  /  N-BODY DYNAMICS",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#FFFFFF",
    )
    fig.text(
        0.08,
        0.915,
        "THE FIGURE-EIGHT ORBIT  •  NEWTONIAN GRAVITY",
        fontsize=10,
        color="#8FA4BD",
    )

    ax_footer.text(
        0.0, 0.55,
        f"3 equal masses     |     G = 1     |     Velocity Verlet"
        f"     |     dt = {dt}     |     duration = {times[-1]:.3f}",
        fontsize=10,
        color="#AAB7C8",
        transform=ax_footer.transAxes,
    )
    ax_footer.text(
        1.0, 0.55,
        f"MAX RELATIVE ENERGY DRIFT: {max_drift:.3e}",
        fontsize=10,
        color="#66E0B5",
        ha="right",
        transform=ax_footer.transAxes,
    )

    fig.savefig(
        "outputs/orbitforge_figure_eight.png",
        dpi=220,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    print("Saved: outputs/orbitforge_figure_eight.png")
    print(f"Snapshots: {len(times)}")
    print(f"Simulated duration: {times[-1]:.6f}")
    print(f"Maximum relative energy drift: {max_drift:.6e}")


if __name__ == "__main__":
    main()
