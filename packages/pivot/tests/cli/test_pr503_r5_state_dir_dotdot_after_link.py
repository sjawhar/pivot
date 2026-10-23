"""A `core.state_dir` reached through a link and `..` is checked where the kernel resolves it."""

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


@pytest.mark.parametrize(
    "case",
    [
        # `nested` is missing, so `nested/..` is the root.
        "control-missing-component",
        # `nested` -> outside/dir, so the kernel reads `nested/../state` as outside/state, a file.
        "symlink-then-parent-segment",
        # `link` dangles, so the kernel cannot resolve `link/..` at all.
        "dangling-link-then-parent-segment",
    ],
)
def test_config_set_refuses_an_unusable_target_reached_through_a_link(
    tmp_path: pathlib.Path, case: str
) -> None:
    """The path the kernel resolves is checked, and an unusable one is refused before writing."""
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    assert _run(root, _PIVOT, "config", "set", "core.max_workers", "2").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    config_path = root / ".pivot" / "config.yaml"
    before = config_path.read_bytes()
    if case == "control-missing-component":
        (root / "state").write_text("not a directory\n")
        value = "nested/../state"
    elif case == "symlink-then-parent-segment":
        outside = tmp_path / "outside"
        (outside / "dir").mkdir(parents=True)
        (outside / "state").write_text("not a directory\n")
        (root / "nested").symlink_to(outside / "dir", target_is_directory=True)
        value = "nested/../state"
    else:
        (root / "link").symlink_to(root / "missing-parent")
        value = "link/../state"

    result = _run(root, _PIVOT, "config", "set", "core.state_dir", value)
    status = _run(root, _PIVOT, "status")

    output = result.stdout + result.stderr
    assert config_path.read_bytes() == before, config_path.read_text()
    assert status.returncode == 0, status.stdout + status.stderr
    assert result.returncode == 1, output
    assert "Traceback" not in output, output
    lines = output.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("Error: core.state_dir"), output


_SECRET_PIPELINE = """\
import pydantic

from pivot import Pipeline, StageParams, stage


class SecretParams(StageParams):
    token: pydantic.SecretStr = pydantic.SecretStr("acceptance-secret")


@stage
def secret_length(params: SecretParams) -> dict:
    return {"length": len(params.token.get_secret_value())}


with Pipeline("test") as pipeline:
    secret_length(params=SecretParams())
"""


def test_a_state_dir_accepted_through_a_link_works(tmp_path: pathlib.Path) -> None:
    """`nested -> outside/dir` and `outside/state` is an empty directory.

    Pivot 0.5.2 accepts `nested/../state` and `repro` succeeds. Refusing the value in one line
    is also acceptable; accepting it and then failing `repro` is not.
    """
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_SECRET_PIPELINE)
    (root / "params.yaml").write_text("test/secret_length:\n  token: end-to-end-secret\n")
    outside = tmp_path / "outside"
    (outside / "dir").mkdir(parents=True)
    (outside / "state").mkdir()
    (root / "nested").symlink_to(outside / "dir", target_is_directory=True)
    config_path = root / ".pivot" / "config.yaml"
    before = config_path.read_bytes() if config_path.exists() else None

    result = _run(root, _PIVOT, "config", "set", "core.state_dir", "nested/../state")

    output = result.stdout + result.stderr
    if result.returncode != 0:
        assert (config_path.read_bytes() if config_path.exists() else None) == before
        lines = output.strip().splitlines()
        assert len(lines) == 1 and lines[0].startswith("Error: core.state_dir"), output
        return
    repro = _run(root, _PIVOT, "repro")
    assert repro.returncode == 0, repro.stdout + repro.stderr
    status = _run(root, _PIVOT, "status")
    assert status.returncode == 0, status.stdout + status.stderr
