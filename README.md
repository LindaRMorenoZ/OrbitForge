# OrbitForge

OrbitForge is an original, educational computational-physics codebase for deterministic
Newtonian N-body experiments. It provides a modular Python implementation and a small
dependency-free C++20 implementation of the same core ideas.

The repository is intentionally self-contained: examples and tests generate their own
initial conditions. No external datasets, model weights, credentials, network services,
or private information are required.

## Features

- Point-mass Newtonian gravity in 2D or 3D, with optional Plummer-style softening.
- Velocity-Verlet time integration (kick-drift-kick), suitable for long-term orbital tests.
- Immutable body definitions and validated mutable simulation state.
- Energy, linear momentum, and angular momentum diagnostics.
- Deterministic simulation histories and a small command-line demo.
- C++20 core library with matching force and integration routines.
- Unit and integration tests for physical invariants and validation behavior.

## Scope and limitations

This is a classical Newtonian simulator, not a general-relativistic solver. It does not
model radiation, relativistic corrections, collisions, extended bodies, or adaptive
time stepping. Softening changes the short-distance force law and should be chosen
deliberately. The code is intended for education, numerical-method experiments, and
software-engineering evaluation, not safety-critical or scientific production use.

## Python quick start

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
pytest
python -m orbitforge.demo --steps 1000 --dt 0.001
```

Python 3.10+ and NumPy 1.23+ are required.

## C++ quick start

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/orbitforge_demo
```

C++20 and CMake 3.16+ are required. The C++ core has no third-party dependencies.

## Repository layout

```text
src/orbitforge/       Python package
  state.py            Body and system-state validation
  gravity.py          Newtonian acceleration calculation
  integrators.py      Time integrators
  diagnostics.py      Conserved-quantity diagnostics
  simulation.py       Simulation orchestration and history
  demo.py             Reproducible two-body example
cpp/                  C++20 implementation and demo
tests/                Python tests
docs/                 Design notes and numerical conventions
```

## Units and conventions

The library is unit-agnostic. Supply a consistent system of units for mass, length,
and time. The gravitational constant `G` must use the corresponding derived units.
Positions and velocities are arrays with shape `(N, D)`, where `N` is the number
of bodies and `D` is 2 or 3. Masses are positive scalars.

The softened acceleration on body *i* is

`a_i = G sum_(j != i) m_j (r_j - r_i) / (|r_j-r_i|^2 + eps^2)^(3/2)`.

With `eps=0`, the point-mass Newtonian force is recovered away from collisions.
The implementation evaluates pair interactions symmetrically to reduce momentum drift.

## Example

```python
import numpy as np
from orbitforge import Body, SystemState, Simulation

bodies = [
    Body("primary", 1.0, np.array([-0.5, 0.0])),
    Body("secondary", 1.0, np.array([0.5, 0.0])),
]
state = SystemState.from_bodies(
    bodies,
    velocities=np.array([[0.0, -0.5], [0.0, 0.5]]),
)
result = Simulation(state, gravitational_constant=1.0, softening=0.01).run(
    steps=100, dt=0.001
)
print(result.final_state.time)
```

## Development

Run the Python test suite with `pytest`. Run the C++ test executable with
`ctest --test-dir build --output-on-failure`.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/design.md](docs/design.md).

## License

MIT. See [LICENSE](LICENSE).
