"""A local config Pivot cannot read, or a local `core.state_dir` it cannot use, keeps 0.5.2's exit
codes and output."""

from __future__ import annotations

import json
import os
import pathlib
from typing import TYPE_CHECKING

import pytest
import yaml

from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Iterator

# `config set` and `config unset` replace a hand-edited `core.state_dir` that names no usable
# directory, and exit 0 as on 0.5.2.


@pytest.mark.parametrize("action", ["set", "unset"])
@pytest.mark.parametrize("old", ["empty-string", "regular-file", "unreadable-directory"])
def test_config_set_and_unset_replace_an_unusable_hand_edited_state_dir(
    tmp_path: pathlib.Path, old: str, action: str
) -> None:
    """Like 0.5.2, the command succeeds with its confirmation and the config holds the new value."""
    home = tmp_path / "home"
    if old == "unreadable-directory" and os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    config_path = root / ".pivot" / "config.yaml"
    locked = root / "locked"
    if old == "empty-string":
        config_path.write_text("core:\n  state_dir: ''\n")
    elif old == "regular-file":
        (root / "afile").write_text("not a directory\n")
        config_path.write_text("core:\n  state_dir: afile\n")
    else:
        (locked / "stages").mkdir(parents=True)
        locked.chmod(0o000)
        config_path.write_text("core:\n  state_dir: locked\n")

    try:
        if action == "set":
            result = helpers.run_pivot(root, home, "config", "set", "core.state_dir", "altstate")
        else:
            result = helpers.run_pivot(root, home, "config", "unset", "core.state_dir")
    finally:
        if locked.exists():
            locked.chmod(0o755)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if action == "set":
        assert "state_dir: altstate" in config_path.read_text()
        assert output.strip().splitlines()[-1] == "Set core.state_dir = altstate in local config"
    else:
        assert "state_dir" not in config_path.read_text()
        assert output.strip().splitlines()[-1] == "Removed core.state_dir from local config"


# A local config that is not UTF-8 keeps 0.5.2's listing behaviour.
#
# `.pivot/config.yaml` holding one non-UTF-8 byte (here a Latin-1 `é` in a comment) cannot be read.
# 0.5.2 runs `list`, `list --json`, `list --all`, `dag`, `plots show` and `init --force` (exit 0) and
# lists the stage in shell completion, because none of them read the config. Pivot does the same and
# `list` and `dag` print one warning on stderr naming the unknown state directory. `history`, which
# uses state, still fails in one `Error:` line.

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


_UNKNOWN = "The state directory is unknown"


def _non_utf8_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    (root / ".pivot" / "config.yaml").write_bytes(b"core:\n  max_workers: 2\n# caf\xe9\n")
    return home, root


@pytest.mark.parametrize(
    "args",
    [("list",), ("list", "--all"), ("dag",), ("plots", "show"), ("init", "--force")],
    ids=lambda args: "-".join(args),
)
def test_commands_that_never_read_state_still_run_with_a_non_utf8_config(
    tmp_path: pathlib.Path, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, these exit 0 without a traceback; `list` and `dag` still show the stage."""
    home, root = _non_utf8_project(tmp_path)

    result = helpers.run_pivot(root, home, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if args[0] in {"list", "dag"}:
        assert "test/process" in result.stdout, output


@pytest.mark.parametrize("args", [("list",), ("dag",)], ids=lambda args: "-".join(args))
def test_list_and_dag_warn_once_on_stderr(tmp_path: pathlib.Path, args: tuple[str, ...]) -> None:
    """One warning on stderr naming the unknown state directory."""
    home, root = _non_utf8_project(tmp_path)

    result = helpers.run_pivot(root, home, *args)

    assert result.stderr.count(_UNKNOWN) == 1, result.stderr
    assert _UNKNOWN not in result.stdout, result.stdout


def test_list_json_still_parses_with_a_non_utf8_config(tmp_path: pathlib.Path) -> None:
    """Like 0.5.2, `list --json` exits 0 and stdout is the stage list."""
    home, root = _non_utf8_project(tmp_path)

    result = helpers.run_pivot(root, home, "list", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert [stage["name"] for stage in json.loads(result.stdout)["stages"]] == ["test/process"]


def test_tab_completion_still_lists_stages_with_a_non_utf8_config(tmp_path: pathlib.Path) -> None:
    """Like 0.5.2, `pivot run <TAB>` offers the stage."""
    home, root = _non_utf8_project(tmp_path)

    result = helpers.run_pivot(
        root,
        home,
        extra_env={
            "_PIVOT_COMPLETE": "bash_complete",
            "COMP_WORDS": "pivot run ",
            "COMP_CWORD": "2",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "plain,test/process" in result.stdout.splitlines(), result.stdout + result.stderr


def test_history_still_fails(tmp_path: pathlib.Path) -> None:
    """A state command exits 1 with one `Error:` line, as on 0.5.2."""
    home, root = _non_utf8_project(tmp_path)

    result = helpers.run_pivot(root, home, "history")

    assert result.returncode == 1, result.stdout + result.stderr
    assert any(line.startswith("Error") for line in result.stderr.splitlines()), result.stderr


# With an unreadable local config, `init --force` and `plots show` run as on 0.5.2.
#
# With `.pivot/config.yaml` that is invalid YAML or unreadable (mode 000), 0.5.2 runs `init --force`
# (exit 0, config left byte-identical) and `plots show` on a pipeline with no plots (exit 0, "No plots
# found."): neither reads the config there. `status` still fails with the same `Error:` line.


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
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
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

    result = helpers.run_pivot(root, home, "init", "--force")

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

    result = helpers.run_pivot(root, home, "plots", "show")

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    assert "No plots found." in result.stdout, output


def test_status_still_fails_with_the_config_error(
    broken_local: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    """A state command fails with one `Error:` line, as on 0.5.2."""
    kind, home, root, config = broken_local

    result = helpers.run_pivot(root, home, "status")

    assert result.returncode == 1, result.stdout + result.stderr
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
    expected = (
        f"Error: Invalid YAML in {config}:"
        if kind == "invalid-yaml"
        else f"Error: Permission denied reading {config}"
    )
    assert errors and errors[0].startswith(expected), result.stderr


# A readable local config whose `core.state_dir` is not a path keeps 0.5.2's listing behaviour.
#
# With a hand-edited `core.state_dir` of `''`, `5`, `[a, b]` or `{x: 1}`, the file itself is valid YAML.
# 0.5.2 runs `list`, `list --json`, `dag`, `plots show` and `init --force` (exit 0) and lists the stage
# in shell completion, because none of them read `core.state_dir`. Pivot does the same with one
# warning, since listing never needs the state directory. `status` keeps 0.5.2's exit code.

_VALUES = {"empty": "''", "integer": "5", "list": "[a, b]", "mapping": "{x: 1}"}


def _non_path_project(tmp_path: pathlib.Path, literal: str) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {literal}\n")
    return home, root


@pytest.mark.parametrize("value", list(_VALUES), ids=list(_VALUES))
@pytest.mark.parametrize(
    "args",
    [("list",), ("dag",), ("plots", "show"), ("init", "--force")],
    ids=lambda args: "-".join(args),
)
def test_commands_that_never_read_state_still_run_with_a_non_path_value(
    tmp_path: pathlib.Path, value: str, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, these exit 0 without a traceback."""
    home, root = _non_path_project(tmp_path, _VALUES[value])

    result = helpers.run_pivot(root, home, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if args in {("list",), ("dag",)}:
        assert "test/process" in result.stdout, output


@pytest.mark.parametrize("value", list(_VALUES), ids=list(_VALUES))
def test_list_json_still_parses_with_a_non_path_value(tmp_path: pathlib.Path, value: str) -> None:
    """Like 0.5.2, `list --json` exits 0 and stdout is the stage list."""
    home, root = _non_path_project(tmp_path, _VALUES[value])

    result = helpers.run_pivot(root, home, "list", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert [stage["name"] for stage in json.loads(result.stdout)["stages"]] == ["test/process"]


@pytest.mark.parametrize("value", list(_VALUES), ids=list(_VALUES))
def test_tab_completion_still_lists_stages_with_a_non_path_value(
    tmp_path: pathlib.Path, value: str
) -> None:
    """Like 0.5.2, `pivot run <TAB>` offers the stage."""
    home, root = _non_path_project(tmp_path, _VALUES[value])

    result = helpers.run_pivot(
        root,
        home,
        extra_env={
            "_PIVOT_COMPLETE": "bash_complete",
            "COMP_WORDS": "pivot run ",
            "COMP_CWORD": "2",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "plain,test/process" in result.stdout.splitlines(), result.stdout + result.stderr


@pytest.mark.parametrize("value", list(_VALUES), ids=list(_VALUES))
def test_status_still_refuses_in_one_line(tmp_path: pathlib.Path, value: str) -> None:
    """A state command keeps the 0.5.2 exit code without a traceback."""
    home, root = _non_path_project(tmp_path, _VALUES[value])

    result = helpers.run_pivot(root, home, "status")

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    expected = 0 if value == "empty" else 1
    assert result.returncode == expected, output
    if expected != 0:
        assert any(line.startswith("Error") for line in result.stderr.splitlines()), output


# The warning and the docs state the rule for an unusable local `core.state_dir`.
#
# The rule: every command 0.5.2 ran keeps 0.5.2's exit code and uses the default `.pivot`; every
# command 0.5.2 failed still fails, in one line. Under `core.state_dir: afile` (a regular file),
# `pivot track` writes state, exits 0 as on 0.5.2, and writes its cache under `.pivot`, so neither
# the warning nor `docs/concepts/pipelines.md` may say that commands which use state fail.
# Run from `packages/pivot/tests/cli/` of a source tree; the docs check reads that tree's docs.

_DOCS = pathlib.Path(__file__).resolve().parents[4] / "docs" / "concepts" / "pipelines.md"


def test_warning_does_not_say_only_commands_without_state_use_the_default(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "p"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "afile").write_text("not a directory\n")
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: afile\n")

    result = helpers.run_pivot(root, home, "track", "data/raw/input.txt")

    assert result.returncode == 0, result.stdout + result.stderr
    assert any((root / ".pivot" / "cache").rglob("*")), "track wrote its cache elsewhere"
    assert "commands that do not need it use the default .pivot" not in result.stderr, result.stderr


def test_docs_do_not_say_state_commands_fail() -> None:
    text = " ".join(_DOCS.read_text().split())
    assert "fail with a one-line error until the file is fixed" not in text


# Configuration displays do not resolve an unusable state directory.

_CASES = {
    "empty": "core:\n  state_dir: ''\n",
    "integer": "core:\n  state_dir: 5\n",
    "list": "core:\n  state_dir: [a, b]\n",
    "file": "core:\n  state_dir: afile\n",
}


@pytest.mark.parametrize("config", _CASES.values(), ids=_CASES)
@pytest.mark.parametrize(
    "command",
    [("config", "get", "core.state_dir"), ("config", "get", "cache.dir"), ("config", "list")],
)
def test_config_display_never_warns_about_a_stored_unusable_state_dir(
    tmp_path: pathlib.Path, config: str, command: tuple[str, ...]
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    if config == _CASES["file"]:
        (root / "afile").write_text("not a directory\n")
    (root / ".pivot" / "config.yaml").write_text(config)

    result = helpers.run_pivot(root, home, *command)

    assert "Pivot uses the default .pivot." not in result.stderr, result.stderr


@pytest.mark.parametrize(
    ("config", "stored_value"),
    [
        ("core:\n  state_dir: 5\n", "5"),
        ("core:\n  state_dir: [a, b]\n", "a, b"),
        ("core:\n  state_dir: {a: 1}\n", "{'a': 1}"),
    ],
    ids=["integer", "list", "mapping"],
)
def test_non_string_state_dir_matches_release_display_contract(
    tmp_path: pathlib.Path, config: str, stored_value: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(config)

    listed = helpers.run_pivot(root, home, "config", "list")
    shown = helpers.run_pivot(root, home, "config", "get", "core.state_dir")

    assert listed.returncode == 1, listed.stdout + listed.stderr
    errors = [line for line in listed.stderr.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1, listed.stderr
    assert "core.state_dir" in errors[0]
    assert "string" in errors[0].lower()
    assert shown.returncode == 0, shown.stdout + shown.stderr
    assert shown.stdout == f"core.state_dir = {stored_value} (local)\n"
    assert "Pivot uses the default .pivot." not in shown.stderr


@pytest.mark.parametrize(
    ("value", "stored_value"), [("5", "5"), ("[a, b]", "a, b")], ids=["integer", "list"]
)
@pytest.mark.parametrize(
    "scope",
    [("--local",), ("--local", "--json"), ("--global",)],
    ids=["local", "local-json", "global"],
)
def test_scoped_config_list_shows_a_non_string_state_dir_as_052_does(
    tmp_path: pathlib.Path, value: str, stored_value: str, scope: tuple[str, ...]
) -> None:
    """Scoped `config list` exits 0 and prints 0.5.2's stdout for a non-string value: no note.

    `--global` inside a project labels the value `(global, ignored)`, as `config get --global`
    does; 0.5.2 labelled it with the merged source. Plain `config list` still exits 1.
    """
    home = tmp_path / "home"
    (home / ".config" / "pivot").mkdir(parents=True)
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    (root / ".pivot").mkdir()
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
    if scope[0] == "--global":
        (home / ".config" / "pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")

    listed = helpers.run_pivot(root, home, "config", "list", *scope)

    assert listed.returncode == 0, listed.stdout + listed.stderr
    expected: dict[tuple[str, ...], str] = {
        ("--local",): f"core.state_dir = {stored_value} (local)\n",
        ("--local", "--json"): json.dumps({"core": {"state_dir": yaml.safe_load(value)}}, indent=2)
        + "\n",
        ("--global",): f"core.state_dir = {stored_value} (global, ignored)\n",
    }
    assert listed.stdout == expected[scope]
    assert helpers.run_pivot(root, home, "config", "list").returncode == 1
