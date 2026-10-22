"""Round 14 B8: the deep audit's `gitignore.sh`, case for case, as a test.

Under umask 0002 and 0022, each case checks `.gitignore` creation, bytes and mode, and config
locking, as the driver printed them (`out/gi_head.out`). Every command exits 0, as on 0.5.2
(`out/gi_rel.out`). The one intended difference from 0.5.2: a configured state directory that
Pivot creates gets the default template (0.5.2 wrote none there, because it kept its state in
`.pivot`).
"""

from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import sys

import pytest

from pivot import project

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_PIPELINE = """\
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
        (self.root / "pipeline.py").write_text(_PIPELINE)
        assert self.pivot("init").returncode == 0

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("PIVOT")}
        env["HOME"] = str(self.home)
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
        result = self.run(_PIVOT, *args)
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
    assert gitignore.read_text() == project.STATE_GITIGNORE_CONTENT
    assert _mode(gitignore) == proj.file_mode()


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_b_repro_creates_a_configured_state_dir_with_the_template(
    tmp_path: pathlib.Path, umask: int
) -> None:
    proj = _Project(tmp_path, umask)
    (proj.root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")

    proj.pivot("repro")

    gitignore = proj.root / "altstate" / ".gitignore"
    assert gitignore.read_text() == project.STATE_GITIGNORE_CONTENT
    assert _mode(gitignore) == proj.file_mode()


@pytest.mark.parametrize("umask", _UMASKS, ids=["umask0002", "umask0022"])
def test_c_config_set_creates_the_state_dir_with_only_the_template(
    tmp_path: pathlib.Path, umask: int
) -> None:
    proj = _Project(tmp_path, umask)

    proj.pivot("config", "set", "core.state_dir", "st2")

    assert sorted(os.listdir(proj.root / "st2")) == [".gitignore"]
    assert (proj.root / "st2" / ".gitignore").read_text() == project.STATE_GITIGNORE_CONTENT
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

    assert gitignore.read_text() == project.STATE_GITIGNORE_CONTENT
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

    assert gitignore.read_text() == project.STATE_GITIGNORE_CONTENT
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
