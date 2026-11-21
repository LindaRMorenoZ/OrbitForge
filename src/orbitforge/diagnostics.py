"""Conserved-quantity diagnostics for Newtonian systems."""
import numpy as np

from .state import SystemState


def kinetic_energy(state: SystemState) -> float:
    return float(0.5 * np.sum(state.masses[:, None] * state.velocities**2))


def potential_energy(
    state: SystemState, gravitational_constant: float = 1.0, softening: float = 0.0
) -> float:
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("gravitational_constant must be finite and positive")
    if not np.isfinite(softening) or softening < 0:
        raise ValueError("softening must be finite and non-negative")
    total = 0.0
    eps2 = softening**2
    for i in range(state.n_bodies - 1):
        for j in range(i + 1, state.n_bodies):
            delta = state.positions[j] - state.positions[i]
            distance = float(np.sqrt(np.dot(delta, delta) + eps2))
            if distance == 0:
                raise ValueError("coincident bodies require positive softening")
            total -= gravitational_constant * state.masses[i] * state.masses[j] / distance
    return float(total)


def total_energy(
    state: SystemState, gravitational_constant: float = 1.0, softening: float = 0.0
) -> float:
    return kinetic_energy(state) + potential_energy(state, gravitational_constant, softening)


def linear_momentum(state: SystemState) -> np.ndarray:
    return np.sum(state.masses[:, None] * state.velocities, axis=0)


def angular_momentum(state: SystemState) -> np.ndarray:
    """Return scalar z-angular momentum in 2D, vector angular momentum in 3D."""
    weighted = state.masses[:, None] * state.positions
    if state.dimensions == 2:
        return np.array([np.sum(weighted[:, 0] * state.velocities[:, 1]
                                - weighted[:, 1] * state.velocities[:, 0])])
    return np.sum(np.cross(state.positions, state.masses[:, None] * state.velocities), axis=0)
