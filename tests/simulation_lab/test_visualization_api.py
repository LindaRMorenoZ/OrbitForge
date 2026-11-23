"""Tests for the public simulation visualization API."""

import subprocess
import sys


def test_public_visualization_api_does_not_load_matplotlib():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from orbitforge.simulation_lab import "
                "visualize_interactive_session; "
                "assert callable(visualize_interactive_session); "
                "assert 'matplotlib' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
