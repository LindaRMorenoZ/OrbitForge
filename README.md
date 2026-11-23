# OrbitForge

**A modular computational laboratory for deterministic Newtonian N-body simulations.**

OrbitForge is a Python-based scientific computing project for constructing, running, analyzing, and comparing gravitational N-body experiments. It combines a modular simulation engine, numerical diagnostics, experiment management tools, and interactive 2D/3D visualization.

The project also includes a dependency-free C++20 implementation of the core numerical simulation.

## Features

### N-body simulation engine
- Newtonian gravitational interactions in two and three dimensions.
- Plummer gravitational softening.
- Velocity-Verlet integration.
- Configurable simulation parameters and time steps.
- Reproducible simulation histories.
- Immutable body definitions and independently managed simulation states.

### System construction and experiment management
- Build custom gravitational systems using `SystemBuilder`.
- Define reusable experiments with `ExperimentConfiguration`.
- Run and manage experiments through `Experiment`.
- Persist experiment configurations in JSON format.
- Create and manage multiple interactive simulation sessions with `ExperimentManager`.

### Simulation Lab
- Interactive 2D and 3D simulation viewer.
- Start, pause, resume, stop, and reset controls.
- Real-time trajectory visualization.
- Reusable simulation sessions.
- Experiment comparison.
- Heuristic parameter recommendations for simulation configuration.

### Scientific analysis
- Kinetic and potential energy.
- Total energy.
- Linear and angular momentum.
- Conservation diagnostics.
- Simulation analysis and numerical evaluation.

### C++ implementation
- Dependency-free C++20 core.
- CMake-based build system.
- Independent implementation of the fundamental simulation components.

## Scientific scope and limitations

OrbitForge currently implements classical Newtonian gravity for systems represented by point masses.

The project does not currently include:
- General relativity or relativistic corrections.
- Radiation or gravitational-wave emission.
- Physical collisions or collision resolution.
- Extended-body models.
- Adaptive time-stepping.

Gravitational softening modifies the force law at short distances and should be selected according to the physical and numerical context of an experiment.

Parameter recommendations are based on heuristic dynamical timescales. They are not guarantees of numerical accuracy or stability. Convergence studies and conservation diagnostics remain necessary when assessing simulation results.

## Installation

Clone the repository:

```bash
git clone https://github.com/LindaRMorenoZ/OrbitForge.git
cd OrbitForge
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package and development dependencies:

```bash
pip install -e ".[dev]"
```

For interactive visualization, install the optional Matplotlib dependency:

```bash
pip install -e ".[dev,visualization]"
```

## Quick start

The following example constructs and runs a simple two-body system:

```python
import numpy as np

from orbitforge import Body, Simulation, SystemState

bodies = [
    Body("primary", 1.0, np.array([-0.5, 0.0])),
    Body("secondary", 1.0, np.array([0.5, 0.0])),
]

state = SystemState.from_bodies(
    bodies,
    velocities=np.array([
        [0.0, -0.5],
        [0.0, 0.5],
    ]),
)

simulation = Simulation(
    state,
    gravitational_constant=1.0,
    softening=0.01,
)

result = simulation.run(
    steps=1000,
    dt=0.001,
)

print(result.final_state.time)
```

This example uses arbitrary units with \(G=1\).

## Simulation Lab

The Simulation Lab provides a higher-level workflow for defining and managing computational experiments.

### 1. Construct a system

```python
from orbitforge.builder import SystemBuilder

builder = SystemBuilder(dimensions=2)

builder.add_body(
    name="primary",
    mass=1.0,
    position=[-0.5, 0.0],
    velocity=[0.0, -0.5],
)

builder.add_body(
    name="secondary",
    mass=1.0,
    position=[0.5, 0.0],
    velocity=[0.0, 0.5],
)
```

### 2. Define an experiment

```python
from orbitforge.config import SimulationConfig
from orbitforge.experiments import ExperimentConfiguration

config = SimulationConfig(
    total_time=10.0,
    dt=0.001,
    save_every=10,
    integrator="velocity_verlet",
)

configuration = ExperimentConfiguration.from_builder(
    name="Two-body experiment",
    builder=builder,
    config=config,
    softening=0.01,
)
```

### 3. Create an interactive session

```python
from orbitforge.simulation_lab import ExperimentManager

manager = ExperimentManager()

session = manager.create_session_from_configuration(
    configuration
)
```

### 4. Open the interactive viewer

```python
from orbitforge.simulation_lab import visualize_interactive_session

viewer = visualize_interactive_session(session)
```

The viewer supports two- and three-dimensional systems and provides controls to start, pause, resume, stop, and reset the simulation.

Interactive visualization requires the optional Matplotlib dependency.

### 5. Run and analyze an experiment

For a non-interactive workflow, an experiment can also be executed and analyzed directly:

```python
experiment = configuration.create_experiment()

result = experiment.run()
analysis = experiment.analyze()
metrics = experiment.evaluate()
```

The experiment workflow separates simulation execution, scientific analysis, and conservation evaluation.

## Units and conventions

OrbitForge supports arbitrary units as well as configurable physical unit systems.

When no unit system is supplied, the gravitational constant defaults to:

\[
G=1.
\]

For a softened Newtonian gravitational interaction, the acceleration of body \(i\) is calculated as:

\[
\mathbf{a}_i =
G \sum_{j\ne i}
m_j
\frac{\mathbf{r}_j-\mathbf{r}_i}
{\left(
|\mathbf{r}_j-\mathbf{r}_i|^2+\epsilon^2
\right)^{3/2}},
\]

where:

- \(G\) is the gravitational constant.
- \(m_j\) is the mass of the interacting body.
- \(\mathbf{r}_i\) and \(\mathbf{r}_j\) are position vectors.
- \(\epsilon\) is the Plummer softening parameter.

When \(\epsilon=0\), the expression reduces to the standard Newtonian point-mass interaction away from singular configurations.

## C++ implementation

The C++ implementation can be built independently using CMake.

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

Run the C++ tests using the configured CMake test setup:

```bash
ctest --test-dir cpp/build --output-on-failure
```

## Development and testing

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the Python test suite:

```bash
pytest
```

Run Ruff:

```bash
ruff check .
```

## Project structure

```text
OrbitForge/
├── cpp/                 # C++20 implementation
├── src/
│   └── orbitforge/
│       ├── builder/     # System construction
│       ├── experiments/ # Experiment definitions
│       ├── simulation_lab/ # Session management and visualization
│       ├── controller/  # Simulation control
│       ├── diagnostics/ # Physical diagnostics
│       └── ...
├── tests/               # Python test suite
├── README.md
├── pyproject.toml
└── LICENSE
```

## Project direction

OrbitForge is being developed as a modular computational environment for exploring gravitational N-body dynamics.

The main priorities are numerical reliability, reproducibility, clear separation of scientific components, and extensibility for future computational experiments.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
