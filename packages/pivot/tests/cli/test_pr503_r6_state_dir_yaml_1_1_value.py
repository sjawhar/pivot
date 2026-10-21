"""R6 defect 3: the resolver reads `.pivot/config.yaml` as YAML 1.1, everything else as YAML 1.2.

`config set` writes with ruamel (YAML 1.2), where `on`, `off`, `yes`, `no` and `1:20` are plain
strings, so it stores `state_dir: on` unquoted, exactly as 0.5.2 does. `project._read_core_state_dir`
reads the same file with PyYAML (YAML 1.1), where `on` is True and `1:20` is the integer 80. So the
value is written, `config set` exits 1 with a traceback, and every later command refuses the
config. `config get core.state_dir` still prints the string. 0.5.2 uses the directory `on/`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _run(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    return root


@pytest.mark.parametrize("how", ["config-set", "hand-edited"])
@pytest.mark.parametrize("value", ["y", "on", "off", "yes", "no", "1:20"])
def test_a_state_dir_named_like_a_yaml_1_1_scalar_is_a_directory(
    tmp_path: pathlib.Path, value: str, how: str
) -> None:
    """As in 0.5.2, the value names the directory `<root>/<value>` and the pipeline runs there.

    `y` is the control: it is a string in both YAML versions and passes at 3c3cceb8.
    """
    root = _project(tmp_path)
    config_path = root / ".pivot" / "config.yaml"
    if how == "config-set":
        result = _run(root, _PIVOT, "config", "set", "core.state_dir", value)
        output = result.stdout + result.stderr
        assert "Traceback" not in output, output
        assert result.returncode == 0, output
        assert output.strip().splitlines()[-1] == f"Set core.state_dir = {value} in local config", (
            output
        )
    else:
        config_path.write_text(f"core:\n  state_dir: {value}\n")

    got = _run(root, _PIVOT, "config", "get", "core.state_dir")
    status = _run(root, _PIVOT, "status")
    repro = _run(root, _PIVOT, "repro")

    assert got.stdout == f"core.state_dir = {value} (local)\n", got.stdout + got.stderr
    assert status.returncode == 0, status.stdout + status.stderr
    assert repro.returncode == 0, repro.stdout + repro.stderr
    assert (root / value / "state.lmdb").is_dir(), sorted(p.name for p in root.iterdir())
