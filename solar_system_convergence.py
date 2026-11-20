
import numpy as np

from orbitforge.scenarios.solar_system import build_solar_system
from orbitforge.validation.convergence import run_convergence_test


def main():
    initial_state = build_solar_system()

    result = run_convergence_test(
        initial_state=initial_state,
        duration=2.0,
        time_steps=[0.002, 0.001, 0.0002],
        gravitational_constant=4.0 * np.pi**2,
    )

    print("\nSOLAR SYSTEM CONVERGENCE TEST")
    print("-" * 75)
    print(
        f"{'dt (years)':>12} "
        f"{'Position error':>18} "
        f"{'Energy drift':>18} "
        f"{'Angular momentum drift':>24}"
    )
    print("-" * 75)

    for i, dt in enumerate(result.dt):
        print(
            f"{dt:12.6f} "
            f"{result.position_error[i]:18.6e} "
            f"{result.energy_drift[i]:18.6e} "
            f"{result.angular_momentum_drift[i]:24.6e}"
        )

    print("\nOBSERVED CONVERGENCE ORDER")
    for i in range(1, len(result.dt)):
        print(
            f"{result.dt[i - 1]:.6f} -> {result.dt[i]:.6f}: "
            f"{result.observed_order[i]:.4f}"
        )


if __name__ == "__main__":
    main()
