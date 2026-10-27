"""A global `core.state_dir` is ignored: commands that use the state directory warn once, `config`
labels the value `global, ignored`, and a broken global config keeps 0.5.2's behaviour."""

from __future__ import annotations

import json
import os
import typing
from typing import TYPE_CHECKING

import pytest

from tests import helpers

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator

# `config get --global` and `config list --global` label a global `core.state_dir` ignored.


def test_config_get_global_marks_the_state_dir_ignored(tmp_path: pathlib.Path) -> None:
    """Both global reporting surfaces say the value is ignored, and warn once."""
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("core:\n  state_dir: globalstate\n")
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    warning = (
        f"core.state_dir in {global_config} is ignored; set it in {root / '.pivot' / 'config.yaml'}"
    )

    listed = helpers.run_pivot(root, home, "config", "list", "--global")
    got = helpers.run_pivot(root, home, "config", "get", "--global", "core.state_dir")

    assert "core.state_dir = globalstate (global, ignored)" in listed.stdout, listed.stdout
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout == "core.state_dir = globalstate (global, ignored)\n", got.stdout
    assert got.stderr.splitlines().count(warning) == 1, got.stderr


# Every command that uses the state directory warns once that a global `core.state_dir` is
# ignored; commands that never use it stay silent.

if typing.TYPE_CHECKING:
    import pathlib


_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _project(tmp_path: pathlib.Path, value: str) -> tuple[pathlib.Path, pathlib.Path, str]:
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text(f"core:\n  state_dir: {value}\n")
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    repro = helpers.run_pivot(root, home, "repro")
    assert repro.returncode == 0, repro.stdout + repro.stderr
    warning = (
        f"core.state_dir in {global_config} is ignored; set it in {root / '.pivot' / 'config.yaml'}"
    )
    return home, root, warning


def _case_id(value: object) -> str | None:
    """Name a case by its command words; the boolean column takes pytest's default id."""
    if isinstance(value, tuple):
        return "-".join(str(word) for word in typing.cast("tuple[object, ...]", value))
    return None


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
@pytest.mark.parametrize(
    ("args", "uses_state"),
    [
        # Commands that use the state directory.
        (("status",), True),
        # Read the state database whose location changed for global-value users.
        (("history",), True),
        (("show",), True),
        (("fingerprint", "reset"), True),
        # Commands that resolve the state directory while loading the pipeline.
        (("init", "--force"), True),
        (("list",), True),
        (("dag",), True),
        (("plots", "show"), True),
        # These read remotes, imports or another key, never state, so they stay quiet.
        (("remote", "list"), False),
        (("update", "--dry-run"), False),
        (("config", "set", "core.max_workers", "2"), False),
        (("config", "unset", "core.max_workers"), False),
    ],
    ids=_case_id,
)
def test_every_command_warns_once_about_an_ignored_global_state_dir(
    tmp_path: pathlib.Path, spelling: str, args: tuple[str, ...], uses_state: bool
) -> None:
    """Every command that uses the state directory prints the one-line warning exactly once."""
    value = "globalstate" if spelling == "relative" else str(tmp_path / "shared-state")
    home, root, warning = _project(tmp_path, value)

    result = helpers.run_pivot(root, home, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.stderr.splitlines().count(warning) == (1 if uses_state else 0), result.stderr


# A broken `~/.config/pivot/config.yaml` keeps 0.5.2's behaviour in a project without one.
#
# Pivot reads the global file only to decide whether to warn about an ignored global
# `core.state_dir`, and stays silent when it cannot. With an invalid or unreadable global file,
# `init`, `list`, `dag`, `plots show` and `init --force` exit 0 as on 0.5.2, and `status` and
# `history` keep 0.5.2's one-line error naming the file.


def _setup(tmp_path: pathlib.Path, kind: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    clean_home = tmp_path / "clean-home"
    clean_home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, clean_home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, clean_home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert helpers.run_pivot(root, clean_home, "repro").returncode == 0
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    if kind == "invalid-yaml":
        global_config.write_text("core: [unclosed\n")
    else:
        global_config.write_text("display:\n  precision: 3\n")
        global_config.chmod(0o000)
    return home, root, global_config


@pytest.fixture(params=["invalid-yaml", "unreadable"])
def broken(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> Iterator[tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]]:
    kind = str(request.param)
    if kind == "unreadable" and os.geteuid() == 0:
        pytest.skip("root ignores file permission bits")
    home, root, global_config = _setup(tmp_path, kind)
    try:
        yield kind, home, root, global_config
    finally:
        global_config.chmod(0o644)


@pytest.mark.parametrize(
    "args",
    [("init",), ("list",), ("dag",), ("plots", "show"), ("init", "--force")],
    ids=lambda args: "-".join(args),
)
def test_commands_that_never_read_global_config_still_work(
    broken: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path],
    tmp_path: pathlib.Path,
    args: tuple[str, ...],
) -> None:
    """0.5.2 runs these with a broken global config: none of them read it there."""
    _, home, root, _ = broken
    cwd = root
    if args == ("init",):
        cwd = tmp_path / "fresh"
        cwd.mkdir()
        assert helpers.run_cli(cwd, home, "git", "init", "-q").returncode == 0

    result = helpers.run_pivot(cwd, home, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output


def test_second_init_after_a_broken_global_config_behaves_like_052(
    broken: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """The first init leaves a whole project, so a second init refuses as 0.5.2 does."""
    _, home, _, _ = broken
    cwd = tmp_path / "fresh"
    cwd.mkdir()
    assert helpers.run_cli(cwd, home, "git", "init", "-q").returncode == 0

    first = helpers.run_pivot(cwd, home, "init")
    second = helpers.run_pivot(cwd, home, "init")

    assert first.returncode == 0, first.stdout + first.stderr
    assert sorted(p.name for p in (cwd / ".pivot").iterdir()) == [".gitignore", "stages"]
    assert second.returncode == 1, second.stdout + second.stderr
    assert second.stdout == "", second.stdout
    assert second.stderr == (
        f"Error: Pivot already initialized in {cwd}\n\nTip: Use --force to reinitialize\n"
    ), second.stderr


@pytest.mark.parametrize("args", [("status",), ("history",)], ids=lambda args: "-".join(args))
def test_commands_that_read_global_config_keep_the_052_message(
    broken: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path], args: tuple[str, ...]
) -> None:
    """0.5.2 exits 1 here with one `Error:` line naming the file."""
    kind, home, root, global_config = broken

    result = helpers.run_pivot(root, home, *args)

    assert result.returncode == 1, result.stdout + result.stderr
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
    expected = (
        f"Error: Invalid YAML in {global_config}:"
        if kind == "invalid-yaml"
        else f"Error: Permission denied reading {global_config}"
    )
    assert errors and errors[0].startswith(expected), result.stderr


# `config get --global core.state_dir` and `config unset --global` work outside a project.
#
# Outside a project nothing ignores the global value, so it is labeled `(global)` as on 0.5.2.


def _outside_any_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("core:\n  state_dir: globalstate\n")
    cwd = tmp_path / "not-a-project"
    cwd.mkdir()
    assert not any((parent / ".pivot").exists() for parent in (cwd, *cwd.parents)), (
        "test needs no .pivot above"
    )
    return home, cwd


@pytest.mark.parametrize("output_json", [False, True], ids=["text", "json"])
def test_config_get_global_state_dir_works_outside_a_project(
    tmp_path: pathlib.Path, output_json: bool
) -> None:
    """Like 0.5.2, the global value is printed and the command exits 0; no project ignores it."""
    home, cwd = _outside_any_project(tmp_path)
    args = ["config", "get", "--global", "core.state_dir", *(["--json"] if output_json else [])]

    result = helpers.run_pivot(cwd, home, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if output_json:
        assert json.loads(result.stdout) == {
            "key": "core.state_dir",
            "value": "globalstate",
            "source": "global",
        }, result.stdout
    else:
        assert result.stdout == "core.state_dir = globalstate (global)\n", result.stdout


@pytest.mark.parametrize("output_json", [False, True], ids=["text", "json"])
def test_config_get_global_state_dir_ignores_an_invalid_local_config(
    tmp_path: pathlib.Path, output_json: bool
) -> None:
    """Like 0.5.2, labeling a global value never parses the project's own config."""
    home, cwd = _outside_any_project(tmp_path)
    (cwd / ".pivot").mkdir()
    (cwd / ".pivot" / "config.yaml").write_text("core: [unclosed\n")
    args = ["config", "get", "--global", "core.state_dir", *(["--json"] if output_json else [])]

    result = helpers.run_pivot(cwd, home, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if output_json:
        assert json.loads(result.stdout) == {
            "key": "core.state_dir",
            "value": "globalstate",
            "source": "global, ignored",
        }, result.stdout
    else:
        assert result.stdout == "core.state_dir = globalstate (global, ignored)\n", result.stdout


def test_config_unset_global_state_dir_works_outside_a_project(tmp_path: pathlib.Path) -> None:
    """The fix the ignored-value warning names works outside a project."""
    home, cwd = _outside_any_project(tmp_path)

    result = helpers.run_pivot(cwd, home, "config", "unset", "--global", "core.state_dir")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "Removed core.state_dir from global config\n", result.stdout


# `config list --global` labels a global `core.state_dir` `(global, ignored)`.
#
# A global `core.state_dir` is ignored, so the label holds whether or not the project sets its own
# value. 0.5.2 printed `(local)` when the project set one.


@pytest.mark.parametrize("shadowed", [True, False], ids=["shadowed", "unshadowed-control"])
def test_config_list_global_marks_the_global_state_dir_ignored(
    tmp_path: pathlib.Path, shadowed: bool
) -> None:
    """The global value is shown with the `(global, ignored)` label, shadowed or not."""
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("core:\n  state_dir: globalstate\n")
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    if shadowed:
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: localstate\n")

    result = helpers.run_pivot(root, home, "config", "list", "--global")

    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line for line in result.stdout.splitlines() if line.startswith("core.state_dir")]
    assert rows == ["core.state_dir = globalstate (global, ignored)"], result.stdout


# An unusable local state directory owns the one warning: a value that cannot be used, or a path
# that cannot become a directory (`afile` is a regular file).

_CASES = {
    "empty": "''",
    "integer": "5",
    "list": "[a, b]",
    "mapping": "{x: 1}",
    "regular-file": "afile",
}


@pytest.mark.parametrize("value", _CASES.values(), ids=_CASES)
def test_unusable_local_state_dir_suppresses_the_global_state_dir_notice(
    tmp_path: pathlib.Path, value: str
) -> None:
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("core:\n  state_dir: global-state\n")
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
    (root / "afile").write_text("not a directory\n")
    (root / "pipeline.py").write_text("from pivot import Pipeline\n\npipeline = Pipeline('test')\n")

    listed = helpers.run_pivot(root, home, "list")

    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert str(global_config) not in listed.stderr
    assert listed.stderr.count("Pivot uses the default .pivot.") == 1, listed.stderr


# Every command that uses the state directory reports a path that cannot become one, once, with
# or without a global value; the global notice is left out. `history`, `show` and `fingerprint
# reset` reach the state directory only through the lookup of a command that needs it, and fail
# as on 0.5.2.

_PATH_CASES = {"regular-file": "afile", "under-a-regular-file": "afile/state"}

# name -> (arguments, 0.5.2's exit code with such a path)
_STATE_COMMANDS: dict[str, tuple[tuple[str, ...], int]] = {
    "list": (("list",), 0),
    "history": (("history",), 1),
    "history-json": (("history", "--json"), 1),
    "show": (("show",), 1),
    "fingerprint-reset": (("fingerprint", "reset"), 1),
}


@pytest.mark.parametrize("command", _STATE_COMMANDS.values(), ids=_STATE_COMMANDS)
@pytest.mark.parametrize("value", _PATH_CASES.values(), ids=_PATH_CASES)
@pytest.mark.parametrize("with_global", [True, False], ids=["global", "no-global"])
def test_every_state_command_reports_an_unusable_path_once(
    tmp_path: pathlib.Path, with_global: bool, value: str, command: tuple[tuple[str, ...], int]
) -> None:
    args, exit_code = command
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    if with_global:
        global_config.parent.mkdir(parents=True)
        global_config.write_text("core:\n  state_dir: global-state\n")
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
    (root / "afile").write_text("not a directory\n")
    (root / "pipeline.py").write_text("from pivot import Pipeline\n\npipeline = Pipeline('test')\n")

    result = helpers.run_pivot(root, home, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == exit_code, output
    assert str(global_config) not in result.stderr
    assert result.stderr.count("Pivot uses the default .pivot.") == 1, result.stderr
    # The warning names the file in the way, which the command's `Error:` line does not.
    assert f"must be a directory: {os.path.realpath(root / 'afile')} (" in result.stderr
    if exit_code:
        errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
        assert len(errors) == 1, result.stderr
