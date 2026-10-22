"""The suite runs where CI does not: SIGINT inherited as ignored."""

from __future__ import annotations

import pathlib
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_ctrl_c_tests_pass_when_pytest_starts_with_sigint_ignored(tmp_path: pathlib.Path) -> None:
    """A background job of a non-interactive shell (``cmd &`` in a script) starts with SIGINT
    ignored, and ``uv run`` passes that on; the suite still interrupts the pivot it starts."""
    nodeid = (
        "packages/pivot/tests/cli/test_worker_termination.py"
        + "::test_ctrl_c_in_watch_mode_lets_the_running_stage_clean_up"
    )
    proc = subprocess.run(
        [
            "sh",
            "-c",
            'trap "" INT; exec "$0" "$@"',
            sys.executable,
            "-m",
            "pytest",
            nodeid,
            "-p",
            "no:cacheprovider",
            f"--basetemp={tmp_path / 'basetemp'}",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"{nodeid} failed with SIGINT ignored:\n{proc.stdout[-3000:]}"
