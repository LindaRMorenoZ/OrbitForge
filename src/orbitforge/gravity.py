"""Pairwise Newtonian gravity with optional Plummer softening."""
import numpy as np
from numpy.typing import NDArray

from .state import SystemState

FloatArray = NDArray[np.float64]


def accelerations(
    state: SystemState, gravitational_constant: float = 1.0, softening: float = 0.0
) -> FloatArray:
    """Return accelerations with shape (N, D), accumulating equal-and-opposite pairs.

    `softening` has units of length and must be non-negative. Coincident bodies
    with zero softening are rejected because the point-mass force is singular.
    """
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("gravitational_constant must be finite and positive")
    if not np.isfinite(softening) or softening < 0:
        raise ValueError("softening must be finite and non-negative")

    n, dimensions = state.positions.shape
    result = np.zeros((n, dimensions), dtype=float)
    eps2 = softening * softening
    for i in range(n - 1):
        for j in range(i + 1, n):
            displacement = state.positions[j] - state.positions[i]
            radius2 = float(np.dot(displacement, displacement) + eps2)
            if radius2 == 0.0:
                raise ValueError("coincident bodies require positive softening")
            scale = gravitational_constant / (radius2 * np.sqrt(radius2))
            result[i] += state.masses[j] * scale * displacement
            result[j] -= state.masses[i] * scale * displacement
    return result
