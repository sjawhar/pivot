"""The `.gitignore` files Pivot writes for the `.pivot` marker and the state directory: when, with
which bytes, and with which mode."""

from __future__ import annotations

import os
import stat
import subprocess
from typing import TYPE_CHECKING

import pytest

from pivot import state_dir as state_dir_mod
from tests import helpers

if TYPE_CHECKING:
    import pathlib

# In the default layout, only `init` writes `.pivot/.gitignore`, as on 0.5.2.
#
# A state directory gets its ignore file only when Pivot creates it, and `.pivot` already exists.
# So `list`, `dag` and `status` leave a deleted `.pivot/.gitignore` deleted, and run in a `.pivot`
# the user cannot write (mode 0555), with or without the file.

_DEFAULT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_DEFAULT_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    return home, root


@pytest.mark.parametrize("args", [("list",), ("dag",), ("status",)], ids=lambda a: "-".join(a))
def test_read_commands_do_not_recreate_a_deleted_ignore_file(
    tmp_path: pathlib.Path, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, a command other than `init` leaves a deleted `.pivot/.gitignore` deleted."""
    home, root = _project(tmp_path)
    (root / ".pivot" / ".gitignore").unlink()

    result = helpers.run_pivot(root, home, *args)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / ".pivot" / ".gitignore").exists()


@pytest.mark.parametrize("layout", ["absent", "present"])
@pytest.mark.parametrize("args", [("list",), ("dag",), ("status",)], ids=lambda a: "-".join(a))
def test_read_commands_run_in_a_read_only_pivot_directory(
    tmp_path: pathlib.Path, layout: str, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, these exit 0 when `.pivot` is mode 0555, with or without the ignore file."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    home, root = _project(tmp_path)
    if layout == "absent":
        (root / ".pivot" / ".gitignore").unlink()
    (root / ".pivot").chmod(0o555)
    try:
        result = helpers.run_pivot(root, home, *args)
    finally:
        (root / ".pivot").chmod(0o755)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output


# `.gitignore` creation, bytes and mode, and config locking, for each state layout.
#
# Under umask 0002 and 0022, every command exits 0, as on 0.5.2. The one intended difference from
# 0.5.2: a configured state directory that Pivot creates gets the default template (0.5.2 wrote
# none there, because it kept its state in `.pivot`).

_DRIVER_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


_UMASKS = [0o002, 0o022]


class _Project:
    umask: int
    home: pathlib.Path
    root: pathlib.Path

    def __init__(self, base: pathlib.Path, umask: int) -> None:
        self.umask = umask
        self.home = base / "home"
        self.root = base / "proj"
        self.home.mkdir(parents=True)
        self.root.mkdir()
        for args in (
            ("init", "-q", "-b", "main"),
            ("config", "user.email", "t@e.com"),
            ("config", "user.name", "t"),
        ):
            assert self.run("git", *args).returncode == 0
        (self.root / "data" / "raw").mkdir(parents=True)
        (self.root / "data" / "raw" / "input.txt").write_text("input\n")
        (self.root / "pipeline.py").write_text(_DRIVER_PIPELINE)
        assert self.pivot("init").returncode == 0

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = helpers.cli_env(self.home)
        umask = self.umask
        return subprocess.run(
            args,
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            preexec_fn=lambda: os.umask(umask),  # noqa: PLW0108
        )

    def pivot(self, *args: str) -> subprocess.CompletedProcess[str]:
        result = self.run(helpers.PIVOT, *args)
        assert result.returncode == 0, (args, result.stdout + result.stderr)
        return result

    def file_mode(self) -> int:
        return 0o666 & ~self.umask


def _mode(path: pathlib.Path) -> int | None:
    return stat.S_IMODE(path.stat().st_mode) if os.path.lexists(path) else None


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_a_default_init_writes_the_template(tmp_path: pathlib.Path, umask: int) -> None:
    proj = _Project(tmp_path, umask)

    gitignore = proj.root / ".pivot" / ".gitignore"
    assert gitignore.read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT
    assert _mode(gitignore) == proj.file_mode()


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_b_repro_creates_a_configured_state_dir_with_the_template(
    tmp_path: pathlib.Path, umask: int
) -> None:
    proj = _Project(tmp_path, umask)
    (proj.root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")

    proj.pivot("repro")

    gitignore = proj.root / "altstate" / ".gitignore"
    assert gitignore.read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT
    assert _mode(gitignore) == proj.file_mode()


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_c_config_set_creates_the_state_dir_with_only_the_template(
    tmp_path: pathlib.Path, umask: int
) -> None:
    proj = _Project(tmp_path, umask)

    proj.pivot("config", "set", "core.state_dir", "st2")

    assert sorted(os.listdir(proj.root / "st2")) == [".gitignore"]
    assert (proj.root / "st2" / ".gitignore").read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT
    assert _mode(proj.root / "st2" / ".gitignore") == proj.file_mode()
    assert sorted(os.listdir(proj.root / ".pivot")) == [
        ".gitignore",
        "config.yaml",
        "config.yaml.lock",
        "stages",
    ]


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_d_a_pre_existing_state_dir_gets_a_template_only_from_init_force(
    tmp_path: pathlib.Path, umask: int
) -> None:
    proj = _Project(tmp_path, umask)
    (proj.root / "pre").mkdir()
    (proj.root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: pre\n")
    gitignore = proj.root / "pre" / ".gitignore"

    for command in ("list", "status", "repro", "commit"):
        proj.pivot(command)
        assert not os.path.lexists(gitignore), command

    result = proj.pivot("init", "--force")

    assert gitignore.read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT
    assert _mode(gitignore) == proj.file_mode()
    assert "  pre/.gitignore" in result.stdout.splitlines(), result.stdout


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_e_a_custom_state_dir_gitignore_is_never_replaced(
    tmp_path: pathlib.Path, umask: int
) -> None:
    proj = _Project(tmp_path, umask)
    (proj.root / "cus").mkdir()
    gitignore = proj.root / "cus" / ".gitignore"
    gitignore.write_text("user-rule\n")
    (proj.root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: cus\n")

    for command in (("repro",), ("init", "--force"), ("status",)):
        proj.pivot(*command)
        assert gitignore.read_text() == "user-rule\n", command


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_f_a_deleted_default_gitignore_returns_only_with_init_force(
    tmp_path: pathlib.Path, umask: int
) -> None:
    proj = _Project(tmp_path, umask)
    gitignore = proj.root / ".pivot" / ".gitignore"
    gitignore.unlink()

    for command in ("list", "status", "repro", "commit"):
        proj.pivot(command)
        assert not os.path.lexists(gitignore), command

    proj.pivot("init", "--force")

    assert gitignore.read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT
    assert _mode(gitignore) == proj.file_mode()


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_g_config_writes_hold_a_lock_and_keep_the_config_private(
    tmp_path: pathlib.Path, umask: int
) -> None:
    proj = _Project(tmp_path, umask)
    pivot_dir = proj.root / ".pivot"
    assert sorted(os.listdir(pivot_dir)) == [".gitignore", "stages"]

    proj.pivot("config", "set", "core.max_workers", "3")

    assert sorted(os.listdir(pivot_dir)) == [
        ".gitignore",
        "config.yaml",
        "config.yaml.lock",
        "stages",
    ]
    assert _mode(pivot_dir / "config.yaml") == 0o600
    assert _mode(pivot_dir / "config.yaml.lock") == proj.file_mode()

    proj.pivot("config", "set", "--global", "core.max_workers", "4")

    global_dir = proj.home / ".config" / "pivot"
    assert sorted(os.listdir(global_dir)) == ["config.yaml", "config.yaml.lock"]
    assert _mode(global_dir / "config.yaml") == 0o600

    proj.pivot("config", "unset", "core.max_workers")

    assert sorted(os.listdir(pivot_dir)) == [
        ".gitignore",
        "config.yaml",
        "config.yaml.lock",
        "stages",
    ]
    assert (pivot_dir / "config.yaml").read_text() == "{}\n"


# A custom cache below core.state_dir retains its state ignore template.


def test_track_ignores_a_custom_cache_inside_a_new_state_directory(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert helpers.run_cli(root, home, "git", *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data").mkdir()
    (root / "data" / "item.txt").write_text("item\n")
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "base").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(
        "core:\n  state_dir: fresh\ncache:\n  dir: fresh/custom-cache\n"
    )

    tracked = helpers.run_pivot(root, home, "track", "data/item.txt")

    assert tracked.returncode == 0, tracked.stdout + tracked.stderr
    assert (root / "fresh" / ".gitignore").read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT
    status = helpers.run_cli(root, home, "git", "status", "--porcelain", "-uall")
    assert status.returncode == 0, status.stderr
    assert not any(
        line.endswith("custom-cache") or "custom-cache/" in line
        for line in status.stdout.splitlines()
    )


# `track` keeps what it writes under a new configured state directory out of git.
#
# `core.state_dir: fresh` is set in `.pivot/config.yaml` while `fresh/` does not exist yet (a hand
# edit, or a state directory the user deleted). `pivot track` writes its cache under `fresh/cache/`,
# and Pivot creates `fresh/` with its `.gitignore` before writing there, with or without a pipeline.

_TRACK_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("one.txt", t=str))
"""


def _exposed_state_files(root: pathlib.Path, home: pathlib.Path) -> list[str]:
    """Untracked, non-ignored files Pivot writes: anything under `fresh/` or `.pivot/` except
    the ignore file, the lock files meant to be committed, and the config the user edited."""
    status = helpers.run_cli(root, home, "git", "status", "--porcelain", "-uall")
    assert status.returncode == 0, status.stderr
    paths = [line[3:] for line in status.stdout.splitlines() if line.startswith("?? ")]
    return [
        path
        for path in paths
        if path.startswith(("fresh/", ".pivot/"))
        and path not in {"fresh/.gitignore", ".pivot/config.yaml"}
        and not path.startswith("fresh/stages/")
    ]


@pytest.mark.parametrize("with_pipeline", [True, False], ids=["pipeline", "no-pipeline"])
def test_track_keeps_what_it_writes_out_of_git(tmp_path: pathlib.Path, with_pipeline: bool) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for git_args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "test"),
    ):
        assert helpers.run_cli(root, home, "git", *git_args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "one.txt").write_text("one\n")
    if with_pipeline:
        (root / "pipeline.py").write_text(_TRACK_PIPELINE)
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "base").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: fresh\n")
    (root / "data" / "two.txt").write_text("two\n")

    result = helpers.run_pivot(root, home, "track", "data/two.txt")

    assert result.returncode == 0, result.stdout + result.stderr
    if (root / "fresh").exists():
        assert (root / "fresh" / ".gitignore").is_file(), sorted(
            str(p.relative_to(root)) for p in (root / "fresh").rglob("*")
        )
    assert _exposed_state_files(root, home) == []

    # A later state command keeps the runtime files ignored too.
    assert helpers.run_pivot(root, home, "status").returncode in {0, 1}
    assert _exposed_state_files(root, home) == []
