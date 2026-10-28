"""`core.state_dir` names the directory the kernel resolves, and `config set` refuses one Pivot
cannot use."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest

from tests import helpers

if TYPE_CHECKING:
    import pathlib

# A `core.state_dir` reached through a link and `..` is checked where the kernel resolves it.

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


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
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    assert helpers.run_pivot(root, home, "config", "set", "core.max_workers", "2").returncode == 0
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

    result = helpers.run_pivot(root, home, "config", "set", "core.state_dir", value)
    status = helpers.run_pivot(root, home, "status")

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
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
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

    result = helpers.run_pivot(root, home, "config", "set", "core.state_dir", "nested/../state")

    output = result.stdout + result.stderr
    if result.returncode != 0:
        assert (config_path.read_bytes() if config_path.exists() else None) == before
        lines = output.strip().splitlines()
        assert len(lines) == 1 and lines[0].startswith("Error: core.state_dir"), output
        return
    repro = helpers.run_pivot(root, home, "repro")
    assert repro.returncode == 0, repro.stdout + repro.stderr
    status = helpers.run_pivot(root, home, "status")
    assert status.returncode == 0, status.stdout + status.stderr


# `config set core.state_dir` refuses a directory that is not writable and searchable.


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize(
    ("mode", "value"),
    [
        # Read-only parent.
        (0o555, "dir/state"),
        # Writable but not searchable (no x bit): nothing can be created or opened inside.
        (0o600, "dir"),
        (0o600, "dir/state"),
    ],
)
def test_config_set_refuses_a_directory_pivot_cannot_use(
    tmp_path: pathlib.Path, mode: int, value: str
) -> None:
    """A target or nearest parent pivot cannot create entries in is refused before writing."""
    home = tmp_path / "home"
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    assert helpers.run_pivot(root, home, "config", "set", "core.max_workers", "2").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    config_path = root / ".pivot" / "config.yaml"
    before = config_path.read_bytes()
    restricted = root / "dir"
    restricted.mkdir()
    restricted.chmod(mode)
    try:
        result = helpers.run_pivot(root, home, "config", "set", "core.state_dir", value)
        status = helpers.run_pivot(root, home, "status")
    finally:
        restricted.chmod(0o755)

    output = result.stdout + result.stderr
    assert config_path.read_bytes() == before, config_path.read_text()
    assert status.returncode == 0, status.stdout + status.stderr
    assert result.returncode == 1, output
    assert "Traceback" not in output, output
    lines = output.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("Error: core.state_dir"), output


# A `core.state_dir` value that YAML 1.1 reads as a bool or a number names a directory.
#
# `config set` writes with ruamel (YAML 1.2), where `on`, `off`, `yes`, `no` and `1:20` are plain
# strings, so it stores `state_dir: on` unquoted, exactly as 0.5.2 does. The resolver reads the file
# with the same YAML 1.2 loader, so the value names the directory `on/`, as on 0.5.2.


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
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

    `y` is a string in both YAML versions.
    """
    home = tmp_path / "home"
    root = _project(tmp_path)
    config_path = root / ".pivot" / "config.yaml"
    if how == "config-set":
        result = helpers.run_pivot(root, home, "config", "set", "core.state_dir", value)
        output = result.stdout + result.stderr
        assert "Traceback" not in output, output
        assert result.returncode == 0, output
        assert output.strip().splitlines()[-1] == f"Set core.state_dir = {value} in local config", (
            output
        )
    else:
        config_path.write_text(f"core:\n  state_dir: {value}\n")

    got = helpers.run_pivot(root, home, "config", "get", "core.state_dir")
    status = helpers.run_pivot(root, home, "status")
    repro = helpers.run_pivot(root, home, "repro")

    assert got.stdout == f"core.state_dir = {value} (local)\n", got.stdout + got.stderr
    assert status.returncode == 0, status.stdout + status.stderr
    assert repro.returncode == 0, repro.stdout + repro.stderr
    assert (root / value / "state.lmdb").is_dir(), sorted(p.name for p in root.iterdir())
