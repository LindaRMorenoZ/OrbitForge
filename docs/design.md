# Design notes

## Architecture

- `state`: owns validated system snapshots and body metadata.
- `gravity`: computes pairwise accelerations. Each pair is evaluated once and
  contributes equal-and-opposite forces, preserving total force to roundoff.
- `integrators`: contains time-stepping algorithms and depends only on the state
  abstraction plus an acceleration callback.
- `diagnostics`: computes conserved quantities independently of the integrator.
- `simulation`: coordinates integration, saving, and diagnostics.

This separation makes it possible to test the force law, integrator, diagnostics,
and orchestration independently.

## Numerical method

Velocity-Verlet is a second-order, time-reversible, symplectic method for
separable Hamiltonian systems. For a step `h`:

1. `r_(n+1) = r_n + v_n h + a(r_n) h^2 / 2`
2. `v_(n+1) = v_n + [a(r_n) + a(r_(n+1))] h / 2`

The method is not unconditionally accurate: choose a time step small enough to
resolve the shortest dynamical timescale. Energy error generally oscillates for
well-resolved bound systems rather than drifting monotonically.

## Complexity

The direct force calculation is O(N^2) in body count and O(ND) in storage.
This deliberate baseline is simple enough to audit and provides a reference
implementation for future tree or mesh methods.

## C++ parity

The C++ implementation mirrors the pairwise acceleration and velocity-Verlet
steps. It is intentionally dependency-free and uses `std::vector` for a compact
reference implementation. Python and C++ tests should be extended together when
the numerical contract changes.
