# Contributing

Thanks for considering a contribution.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
cmake -S cpp -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Contribution expectations

- Keep the Python and C++ numerical conventions aligned.
- Add tests for new behavior and regression tests for bug fixes.
- Validate public inputs and raise informative exceptions.
- Avoid hidden global state and nondeterministic behavior.
- Document numerical assumptions, units, and limitations.
- Do not add external datasets, credentials, personal information, or network
  dependencies to examples or tests.

## Suggested task areas

Potential engineering tasks include improving input validation, adding convergence
tests, reducing allocations in pair-force evaluation, extending diagnostics, improving
C++/Python parity, and strengthening error handling.
