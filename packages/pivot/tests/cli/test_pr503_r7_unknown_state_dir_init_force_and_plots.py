"""R7 defect G1: with an unreadable local config, `init --force` and `plots show` now refuse.

With `.pivot/config.yaml` that is invalid YAML or unreadable (mode 000), 0.5.2 runs `init --force`
(exit 0, config left byte-identical) and `plots show` on a pipeline with no plots (exit 0, "No plots
found."): neither reads the config there. At ce74ad41 both exit 1 with the config's error, because
`cli/init.py` calls `project.state_dir_is_configured` and `cli/plots.py:43` resolves the default
output under the state directory before it knows there is anything to render. The 06:42Z ruling
and the census name only `list` and `dag`; these two refusals are new and not in the census.

`status` is the control: it fails on 0.5.2 and on the head with the same `Error:` line.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


@pytest.fixture(params=["invalid-yaml", "unreadable"])
def broken_local(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> Iterator[tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]]:
    kind = str(request.param)
    if kind == "unreadable" and os.geteuid() == 0:
        pytest.skip("root ignores file permission bits")
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    config = root / ".pivot" / "config.yaml"
    if kind == "invalid-yaml":
        config.write_text("core: [unclosed\n")
    else:
        config.write_text("core:\n  max_workers: 2\n")
        config.chmod(0o000)
    try:
        yield kind, home, root, config
    finally:
        config.chmod(0o644)


def test_init_force_runs_with_an_unreadable_local_config(
    broken_local: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Like 0.5.2, `init --force` exits 0 and leaves the config file as it was."""
    _, home, root, config = broken_local
    config.chmod(0o644)
    before = config.read_bytes()
    if broken_local[0] == "unreadable":
        config.chmod(0o000)

    result = _run(root, home, _PIVOT, "init", "--force")

    config.chmod(0o644)
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    assert config.read_bytes() == before


def test_plots_show_with_no_plots_runs_with_an_unreadable_local_config(
    broken_local: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Like 0.5.2, a pipeline with no plots prints "No plots found." and exits 0."""
    _, home, root, _ = broken_local

    result = _run(root, home, _PIVOT, "plots", "show")

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    assert "No plots found." in result.stdout, output


def test_status_still_fails_with_the_config_error(
    broken_local: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """Control, passes on 0.5.2 and at ce74ad41: a state command fails with one `Error:` line."""
    kind, home, root, config = broken_local

    result = _run(root, home, _PIVOT, "status")

    assert result.returncode == 1, result.stdout + result.stderr
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
    expected = (
        f"Error: Invalid YAML in {config}:"
        if kind == "invalid-yaml"
        else f"Error: Permission denied reading {config}"
    )
    assert errors and errors[0].startswith(expected), result.stderr
