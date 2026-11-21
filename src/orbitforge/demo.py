"""Run a reproducible two-body orbit demo."""
import argparse

import numpy as np

from .simulation import Simulation
from .state import Body, SystemState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()

    bodies = [
        Body("primary", 1.0, np.array([-0.5, 0.0])),
        Body("secondary", 1.0, np.array([0.5, 0.0])),
    ]
    state = SystemState.from_bodies(
        bodies, velocities=np.array([[0.0, -0.5], [0.0, 0.5]])
    )
    result = Simulation(state, gravitational_constant=1.0, softening=0.01).run(
        args.steps, args.dt, save_every=max(1, args.steps // 10)
    )
    relative_drift = (result.energies - result.energies[0]) / abs(result.energies[0])
    print(f"Final time: {result.final_state.time:.6f}")
    print(f"Final positions:\n{result.final_state.positions}")
    print(f"Maximum relative energy drift: {np.max(np.abs(relative_drift)):.3e}")


if __name__ == "__main__":
    main()
