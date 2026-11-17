"""Helpers for the CLI tests of the `dist:` keys of installed distributions: a venv that uv makes
over this test environment, setuptools projects installed into it, and what the `pivot` that the
venv runs reports.

The venv's site-packages holds a `.pth` file that adds this test environment's purelib directory,
so pivot, its dependencies and setuptools are importable there and every install of a fixture,
by uv or by the venv's own pip, runs offline without build isolation. Pivot runs as
`<venv>/bin/python -P -c "<_console_main>"` through `helpers.run_cli`: `-P` keeps the working
directory off `sys.path`, as the `pivot` console script does. This module imports only the
standard library, pytest, PyYAML, `tests.helpers` and the `pivot` package, so the tests that use
it still collect when `packages/pivot/src` is pivot 0.10.4's.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import sysconfig
from typing import TYPE_CHECKING, Any

import pytest
import yaml

import pivot
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The code `python -c` runs to start pivot's CLI, as the `pivot` console script does.
_PIVOT_MAIN = "from pivot.cli import _console_main; _console_main()"


@dataclasses.dataclass(frozen=True)
class Venv:
    """A venv `make_venv` made: its interpreter and its site-packages directory."""

    python: pathlib.Path
    site_packages: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Project:
    """A pivot project at `root`, run by the `pivot` of `venv`, with `home` as `HOME`."""

    root: pathlib.Path
    home: pathlib.Path
    venv: Venv


def uv() -> str:
    """The uv executable: the one `PIVOT_TEST_UV` names, else the `uv` on PATH. The test fails
    when neither names an executable file."""
    found = os.environ.get("PIVOT_TEST_UV") or shutil.which("uv")
    if found is None or not (pathlib.Path(found).is_file() and os.access(found, os.X_OK)):
        pytest.fail(
            "these tests need uv: set PIVOT_TEST_UV to the uv executable or put uv on PATH "
            + f"(PIVOT_TEST_UV is {os.environ.get('PIVOT_TEST_UV')!r}, uv on PATH is "
            + f"{shutil.which('uv')!r})",
            pytrace=False,
        )
    return found


def run_uv(cwd: pathlib.Path, *args: str) -> str:
    """Run `uv <args>` in `cwd` with `helpers.clean_env()`; the test fails unless it exits 0.
    Returns what it printed."""
    return _run_checked(cwd, [uv(), *args])


def make_venv(venv: pathlib.Path, *, seed: bool = False) -> Venv:
    """Make a venv at `venv` with `uv venv --python <this interpreter>` (`seed`: with pip, `uv
    venv --seed`), whose site-packages holds a `.pth` file that adds this test environment's
    purelib directory to `sys.path` after it."""
    run_uv(
        venv.parent, "venv", *(["--seed"] if seed else []), "--python", sys.executable, str(venv)
    )
    python = venv / "bin" / "python"
    ran = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        env=helpers.clean_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert ran.returncode == 0, f"{python} could not name its purelib:\n{ran.stdout}{ran.stderr}"
    site_packages = pathlib.Path(ran.stdout.strip())
    (site_packages / "_parent.pth").write_text(
        f"import site; site.addsitedir({sysconfig.get_path('purelib')!r})\n"
    )
    return Venv(python=python, site_packages=site_packages)


def write_setuptools_project(
    root: pathlib.Path,
    name: str,
    version: str,
    *,
    packages: Sequence[str],
    files: Mapping[str, str],
    dependencies: Sequence[str] = (),
    scripts: Mapping[str, str] | None = None,
) -> pathlib.Path:
    """Write at `root` a setuptools project: `files` (each path relative to `root`, with its text)
    and a `pyproject.toml` for distribution `name` at `version` that names its import `packages`,
    its `dependencies` and its console `scripts` (command, then `module:function`). Returns
    `root`."""
    # A JSON string, or array of strings, that `json.dumps` writes is also valid TOML.
    lines = [
        "[build-system]",
        'requires = ["setuptools"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f"name = {json.dumps(name)}",
        f"version = {json.dumps(version)}",
        f"dependencies = {json.dumps(list(dependencies))}",
        "",
    ]
    if scripts:
        lines.append("[project.scripts]")
        lines.extend(
            f"{json.dumps(command)} = {json.dumps(target)}" for command, target in scripts.items()
        )
        lines.append("")
    lines.extend(["[tool.setuptools]", f"packages = {json.dumps(list(packages))}", ""])
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("\n".join(lines))
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def write_library(
    root: pathlib.Path,
    name: str,
    module: str,
    source: str,
    *,
    version: str = "0.1",
    dependencies: Sequence[str] = (),
    scripts: Mapping[str, str] | None = None,
    files: Mapping[str, str] | None = None,
) -> pathlib.Path:
    """Write at `root/<module>` the setuptools project (`write_setuptools_project`) of
    distribution `name`, whose package `module` has `source` as its `__init__.py`, with `files`
    beside it. Returns the project's directory."""
    return write_setuptools_project(
        root / module,
        name,
        version,
        packages=[module],
        files={f"{module}/__init__.py": source, **(files or {})},
        dependencies=dependencies,
        scripts=scripts,
    )


def remove_build_outputs(source: pathlib.Path) -> None:
    """Delete what a setuptools build leaves in the project at `source` (`build/` and each
    `*.egg-info` directory), so that its next build starts from a clean tree and no file of an
    earlier build reaches a RECORD."""
    for leftover in [source / "build", *source.glob("*.egg-info")]:
        if leftover.exists():
            shutil.rmtree(leftover)


def replace_once(path: pathlib.Path, old: str, new: str) -> None:
    """Replace the one occurrence of `old` in the file at `path` with `new`; the test fails when
    the file does not hold `old` exactly once."""
    text = path.read_text()
    assert text.count(old) == 1, f"{path} must hold {old!r} once:\n{text}"
    path.write_text(text.replace(old, new))


def install(venv: Venv, *spec: str, reinstall: bool = False) -> str:
    """`uv pip install --offline --python <venv>/bin/python --no-build-isolation --no-deps
    [--reinstall] <spec>`; the test fails unless it succeeds. Returns what uv printed."""
    return run_uv(
        venv.python.parent,
        "pip",
        "install",
        "--offline",
        "--python",
        str(venv.python),
        "--no-build-isolation",
        "--no-deps",
        *(["--reinstall"] if reinstall else []),
        *spec,
    )


def install_directory(
    venv: Venv, source: pathlib.Path, *, reinstall: bool = False, editable: bool = False
) -> str:
    """Install the project at `source` from its path (`editable`: with `-e`), after deleting what
    an earlier build left in it (`remove_build_outputs`)."""
    remove_build_outputs(source)
    return install(venv, *(["-e"] if editable else []), str(source), reinstall=reinstall)


def build_wheel(venv: Venv, source: pathlib.Path, wheels: pathlib.Path) -> str:
    """Build a wheel of the project at `source` into the directory `wheels` with `uv build
    --offline --no-build-isolation --wheel` on the venv's interpreter, after deleting what an
    earlier build left in it."""
    remove_build_outputs(source)
    return run_uv(
        source.parent,
        "build",
        "--offline",
        "--python",
        str(venv.python),
        "--no-build-isolation",
        "--wheel",
        "--out-dir",
        str(wheels),
        str(source),
    )


def install_from_wheels(venv: Venv, wheels: pathlib.Path, requirement: str) -> str:
    """Install `requirement` from the wheels in `wheels` as from a package index
    (`--no-index --find-links`), which writes no `direct_url.json`."""
    return install(venv, "--no-index", "--find-links", str(wheels), requirement)


def pip_install_directory(venv: Venv, source: pathlib.Path) -> str:
    """Install the project at `source` from its path with the venv's own pip (`python -m pip
    install --no-index --no-build-isolation --no-deps`; the venv needs pip, `make_venv(...,
    seed=True)`), after deleting what an earlier build left in it; the test fails unless it
    succeeds. Returns what pip printed."""
    remove_build_outputs(source)
    return _run_checked(
        venv.python.parent,
        [
            str(venv.python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-index",
            "--no-build-isolation",
            "--no-deps",
            str(source),
        ],
    )


def git(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> str:
    """Run `git <args>` in `cwd` as `helpers.run_cli` runs a command, with a fixed author and no
    signing; the test fails unless it exits 0. Returns its standard output."""
    ran = helpers.run_cli(
        cwd,
        home,
        "git",
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "-c",
        "commit.gpgsign=false",
        *args,
    )
    assert ran.returncode == 0, f"git {shlex.join(args)} exited {ran.returncode}:\n{shown(ran)}"
    return ran.stdout


def commit_all(repo: pathlib.Path, home: pathlib.Path, message: str) -> str:
    """Commit every file of the git repository `repo`; returns the new commit's id."""
    git(repo, home, "add", "-A")
    git(repo, home, "commit", "-q", "-m", message)
    return git(repo, home, "rev-parse", "HEAD").strip()


def git_requirement(name: str, repo: pathlib.Path, commit: str) -> str:
    """The requirement that installs distribution `name` from `commit` of the local git
    repository `repo`."""
    return f"{name} @ git+{repo.as_uri()}@{commit}"


def init_git_requirement(source: pathlib.Path, home: pathlib.Path, name: str) -> str:
    """Make the project at `source` a git repository of one commit, `C1`; the requirement that
    installs distribution `name` from that commit (`git_requirement`)."""
    git(source, home, "init", "-q")
    commit = commit_all(source, home, "C1")
    return git_requirement(name, source, commit)


def installed(venv: Venv, name: str) -> importlib.metadata.Distribution:
    """The one distribution called `name` (as `importlib.metadata` matches names) in the venv's
    site-packages; the test fails when it holds none or more than one."""
    found = list(
        importlib.metadata.Distribution.discover(name=name, path=[str(venv.site_packages)])
    )
    assert len(found) == 1, (
        f"{venv.site_packages} must hold one distribution {name!r}; it holds {len(found)}: "
        + f"{sorted(os.listdir(venv.site_packages))}"
    )
    return found[0]


def direct_url(venv: Venv, name: str) -> dict[str, Any] | None:
    """The parsed `direct_url.json` (PEP 610) of distribution `name` in the venv, None when its
    metadata directory holds none."""
    text = installed(venv, name).read_text("direct_url.json")
    return None if text is None else json.loads(text)


def make_project(root: pathlib.Path, home: pathlib.Path, venv: Venv, pipeline: str) -> Project:
    """`git init -q` and `pivot init` a new project at `root`, then write `pipeline` as its
    `pipeline.py`."""
    root.mkdir(parents=True)
    project = Project(root=root, home=home, venv=venv)
    git(root, home, "init", "-q")
    pivot_ok(project, "init")
    (root / "pipeline.py").write_text(pipeline)
    return project


def run_pivot(project: Project, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the venv's `pivot <args>` in the project; the exit code is not checked."""
    return helpers.run_cli(
        project.root, project.home, str(project.venv.python), "-P", "-c", _PIVOT_MAIN, *args
    )


def run_pivot_holding(
    project: Project, module: str, *args: str
) -> subprocess.CompletedProcess[str]:
    """Run the venv's `pivot <args>` in the project from a program that imports `module` before
    it starts pivot's CLI (`python -P -c "import <module>; <_console_main>"`), so the program
    holds the module `-P` lets it find, the installed one, when pivot's import session starts;
    the exit code is not checked."""
    return helpers.run_cli(
        project.root,
        project.home,
        str(project.venv.python),
        "-P",
        "-c",
        f"import {module}; {_PIVOT_MAIN}",
        *args,
    )


def pivot_ok(project: Project, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the venv's `pivot <args>` in the project; the test fails unless it exits 0."""
    ran = run_pivot(project, *args)
    assert ran.returncode == 0, f"pivot {shlex.join(args)} exited {ran.returncode}:\n{shown(ran)}"
    return ran


def module_file(project: Project, module: str) -> pathlib.Path:
    """The resolved `__file__` of `module` as the venv's interpreter imports it in the project,
    started as `run_pivot` starts pivot."""
    ran = helpers.run_cli(
        project.root,
        project.home,
        str(project.venv.python),
        "-P",
        "-c",
        f"import {module}; print({module}.__file__)",
    )
    assert ran.returncode == 0, f"the venv cannot import {module}:\n{shown(ran)}"
    return pathlib.Path(ran.stdout.strip()).resolve()


def repro_status(project: Project, stage: str) -> tuple[str, str]:
    """What `pivot repro --json` did to `stage` (`ran`, `cached`, ...), from its one
    `stage_complete` event, and the command's output."""
    ran = pivot_ok(project, "repro", "--json")
    statuses = [
        event["status"]
        for line in ran.stdout.splitlines()
        if line.startswith("{")
        and (event := _parsed(line, ran))["type"] == "stage_complete"
        and event["stage"] == stage
    ]
    assert len(statuses) == 1, f"pivot repro must complete {stage} once:\n{shown(ran)}"
    return statuses[0], shown(ran)


def stage_status(project: Project, stage: str) -> tuple[str, str]:
    """`stage`'s status in `pivot status --json`, and the command's output."""
    ran = pivot_ok(project, "status", "--json")
    return _stage_entry(ran, stage)["status"], shown(ran)


def code_changes(project: Project, stage: str) -> tuple[list[str], str]:
    """The key of each of `stage`'s `code_changes` in `pivot status --explain --json`, in the
    order it reports them, and the command's output."""
    ran = pivot_ok(project, "status", "--explain", "--json")
    keys = [change["key"] for change in _stage_entry(ran, stage)["code_changes"]]
    return keys, shown(ran)


def ran_then_cached(project: Project, stage: str, output_path: pathlib.Path, output: str) -> None:
    """`pivot repro` runs `stage`, which writes `output` to `output_path` (relative to the
    project's root); `pivot status --json` then reports it `cached`."""
    status, shown = repro_status(project, stage)
    assert status == "ran", f"the first `pivot repro` must run {stage}:\n{shown}"
    written = (project.root / output_path).read_text()
    assert written == output, f"{stage} must write {output!r}, not {written!r}"
    status, shown = stage_status(project, stage)
    assert status == "cached", f"`pivot status` must report {stage} cached after a run:\n{shown}"


def assert_stale_on(project: Project, stage: str, key: str) -> None:
    """`pivot status --json` reports `stage` stale, and `pivot status --explain --json` names
    exactly one code change of it, `key`."""
    status, shown = stage_status(project, stage)
    assert status == "stale", f"`pivot status` must report {stage} stale on {key}:\n{shown}"
    keys, shown = code_changes(project, stage)
    assert keys == [key], f"`pivot status --explain` must name exactly {key} changed:\n{shown}"


def assert_marks_nothing(project: Project, stage: str) -> None:
    """`pivot status --json` reports `stage` cached, and `pivot status --explain --json` names no
    code change of it."""
    status, shown = stage_status(project, stage)
    assert status == "cached", f"`pivot status` must report {stage} cached:\n{shown}"
    keys, shown = code_changes(project, stage)
    assert keys == [], f"`pivot status --explain` must name no code change:\n{shown}"


def lock_path(project: Project, stage: str) -> pathlib.Path:
    """The lock of `stage`, named `<pipeline>/<stage>`: `.pivot/stages/<pipeline>/<stage>.lock`."""
    return project.root / ".pivot" / "stages" / f"{stage}.lock"


def lock_manifest(project: Project, stage: str) -> dict[str, str]:
    """The `code_manifest` of `stage`'s lock (`lock_path`)."""
    path = lock_path(project, stage)
    document = yaml.safe_load(path.read_text())
    assert isinstance(document, dict) and "code_manifest" in document, f"{path}: {document!r}"
    manifest = document["code_manifest"]
    assert isinstance(manifest, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in manifest.items()
    ), f"{path}: code_manifest {manifest!r} must map strings to strings"
    return manifest


def dist_keys(manifest: Mapping[str, str]) -> list[str]:
    """The `dist:` keys of a code manifest, sorted."""
    return sorted(key for key in manifest if key.startswith("dist:"))


def pivot_package_dir() -> pathlib.Path:
    """The directory of the `pivot` package that this test process imported."""
    file = pivot.__file__
    assert file is not None, "the pivot package this process imported has no __file__"
    return pathlib.Path(file).resolve().parent


def shown(ran: subprocess.CompletedProcess[str]) -> str:
    """A command's standard output and standard error, for an assertion message."""
    return f"stdout:\n{ran.stdout}\nstderr:\n{ran.stderr}"


def _stage_entry(ran: subprocess.CompletedProcess[str], stage: str) -> dict[str, Any]:
    """`stage`'s one entry in the `stages` of the JSON document a `pivot status --json` printed."""
    document = _parsed(ran.stdout, ran)
    entries = [entry for entry in document["stages"] if entry["name"] == stage]
    assert len(entries) == 1, f"pivot status must report {stage} once:\n{shown(ran)}"
    return entries[0]


def _parsed(text: str, ran: subprocess.CompletedProcess[str]) -> Any:
    """The JSON value `text`, which command `ran` printed; the test fails, showing the command's
    output, when `text` is not JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        pytest.fail(f"pivot printed {text!r}, which is not JSON ({error}):\n{shown(ran)}")


def _run_checked(cwd: pathlib.Path, command: Sequence[str]) -> str:
    """Run `command` in `cwd` with `helpers.clean_env()`; the test fails unless it exits 0.
    Returns what it printed."""
    ran = subprocess.run(
        command,
        cwd=cwd,
        env=helpers.clean_env(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    output = ran.stdout + ran.stderr
    assert ran.returncode == 0, f"`{shlex.join(command)}` exited {ran.returncode}:\n{output}"
    return output
