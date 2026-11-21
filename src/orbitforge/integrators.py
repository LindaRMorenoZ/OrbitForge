"""Time integration schemes."""
from collections.abc import Callable

import numpy as np

from .state import SystemState

AccelerationFunction = Callable[[SystemState], np.ndarray]


def velocity_verlet(
    state: SystemState, dt: float, acceleration: AccelerationFunction
) -> SystemState:
    """Advance one step using the symplectic velocity-Verlet method."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    current_a = acceleration(state)
    new_positions = state.positions + state.velocities * dt + 0.5 * current_a * dt**2

    midpoint = SystemState(
        state.names, state.masses, new_positions, state.velocities, state.time + dt
    )
    new_a = acceleration(midpoint)
    new_velocities = state.velocities + 0.5 * (current_a + new_a) * dt
    return SystemState(
        state.names, state.masses, new_positions, new_velocities, state.time + dt
    )
