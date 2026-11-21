"""Validated data structures for N-body systems."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Body:
    """A point mass with a stable identifier and positive mass."""

    name: str
    mass: float
    position: FloatArray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        if not self.name.strip():
            raise ValueError("body name must not be empty")
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("body mass must be finite and positive")
        if position.ndim != 1 or position.size not in (2, 3):
            raise ValueError("position must be a one-dimensional 2D or 3D vector")
        if not np.all(np.isfinite(position)):
            raise ValueError("position must contain only finite values")
        position = position.copy()
        position.setflags(write=False)
        object.__setattr__(self, "position", position)


@dataclass
class SystemState:
    """Snapshot of positions, velocities, masses, and simulation time."""

    names: tuple[str, ...]
    masses: FloatArray
    positions: FloatArray
    velocities: FloatArray
    time: float = 0.0

    def __post_init__(self) -> None:
        self.masses = np.asarray(self.masses, dtype=float).copy()
        self.positions = np.asarray(self.positions, dtype=float).copy()
        self.velocities = np.asarray(self.velocities, dtype=float).copy()
        self.validate()

    @classmethod
    def from_bodies(
        cls, bodies: list[Body] | tuple[Body, ...], velocities: FloatArray,
        time: float = 0.0,
    ) -> SystemState:
        if not bodies:
            raise ValueError("at least one body is required")
        dimensions = {body.position.size for body in bodies}
        if len(dimensions) != 1:
            raise ValueError("all bodies must have the same spatial dimension")
        return cls(
            names=tuple(body.name for body in bodies),
            masses=np.array([body.mass for body in bodies], dtype=float),
            positions=np.stack([body.position for body in bodies]),
            velocities=np.asarray(velocities, dtype=float),
            time=time,
        )

    @property
    def n_bodies(self) -> int:
        return int(self.masses.size)

    @property
    def dimensions(self) -> int:
        return int(self.positions.shape[1])

    def validate(self) -> None:
        if self.masses.ndim != 1 or self.masses.size == 0:
            raise ValueError("masses must be a non-empty one-dimensional array")
        if len(self.names) != self.masses.size:
            raise ValueError("names and masses must have the same length")
        if np.any(~np.isfinite(self.masses)) or np.any(self.masses <= 0):
            raise ValueError("all masses must be finite and positive")
        if self.positions.ndim != 2 or self.positions.shape[0] != self.masses.size:
            raise ValueError("positions must have shape (N, D)")
        if self.positions.shape[1] not in (2, 3):
            raise ValueError("spatial dimension must be 2 or 3")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities and positions must have identical shapes")
        if not np.all(np.isfinite(self.positions)) or not np.all(np.isfinite(self.velocities)):
            raise ValueError("positions and velocities must be finite")
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")

    def copy(self) -> SystemState:
        return SystemState(
            self.names, self.masses.copy(), self.positions.copy(),
            self.velocities.copy(), self.time,
        )
