"""What the CLI tests of the project's part of a package it shares with an installed library share:
that part, `company/local.py` (`LOCAL`), of the namespace package `company`, which the namespace
portion `company.client` of `company-client` (`CLIENT`, `CLIENT_FILES`) makes such a package,
`company-client` and `Fix_Lib` installed into a venv from their paths
(`install_client_and_fixlib`), the files of a `company-client` that shares `company.sub` too
(`CLIENT_AND_SUB_FILES`), the project's `company/local.py` that also holds a lock
(`LOCAL_WITH_LOCK`), or that runs code of a test's when a second import runs it (`SECOND_IMPORT`),
a project module that reads it through the package's name (`READS_LOCAL`), a `PYTHONPATH` entry
from which a worker imports by name (`pythonpath_entry`), a project with those installs and files
in its root and in such an entry (`project_with`), what the stage of such a project writes
(`SHARED_OUTPUT`, the project's value then `LIBRARY_VALUES`), the `pipeline.py` whose stage reads
`company.local` through the package's name (`TOP_NAME_PIPELINE`), what one `pivot repro --json`
gives of that stage (`repro_outcome`) and a run that writes it and reruns on an edit
(`assert_runs_then_reruns_on_an_edit`).

It imports only the standard library and `tests.installed_dists` (pytest for type checking alone),
so the tests that use it still collect when `packages/pivot/src` is pivot 0.10.4's.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

from tests import installed_dists

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest

# The project's part of the namespace package `company` it shares with `company-client`, the module
# `company.local`, and the module of `company-client`'s namespace portion `company.client`, which
# imports its package, as a library module that reads a sibling through it does, so
# `company.client` holds `company`. `CLIENT_FILES` are the files of a `company-client` whose one
# package is that portion, each path relative to its project (`install_client_and_fixlib`).
LOCAL = 'def value():\n    return "local-1"\n'
CLIENT = 'import company\n\n\ndef value():\n    return "client-1"\n'
CLIENT_FILES: Mapping[str, str] = {"company/client/__init__.py": CLIENT}
# The files of a `company-client` whose `company.client` lies beside the portion
# `company.sub.shared`, so that `company.sub` is a namespace package it shares with the project too.
CLIENT_AND_SUB_FILES: Mapping[str, str] = {
    **CLIENT_FILES,
    "company/sub/shared.py": 'def value():\n    return "shared-1"\n',
}
# The project's `company/local.py` of `LOCAL` that also holds a lock no stage reads and that
# cloudpickle cannot pickle, and a project module that imports `company.local` and reads it through
# the package's name.
LOCAL_WITH_LOCK = "import threading\n\n_LOCK = threading.Lock()\n\n\n" + LOCAL
READS_LOCAL = "import company.local\n\n\ndef value():\n    return company.local.value()\n"
# The project's `company/local.py`, with the `value()` of `LOCAL`, that runs `{second}` when a
# second import in the project runs it: the first, in pivot's own process, leaves a file beside it.
SECOND_IMPORT = """\
import pathlib

if pathlib.Path(__file__).with_name("imported").exists():
    {second}
pathlib.Path(__file__).with_name("imported").touch()


def value():
    return "local-1"
"""
# What the stage of a test of the project's part of `company` writes (`SHARED_OUTPUT`): the
# project's `value()`, then `company.client.value()` and `fixlib.value()`, joined, the expression of
# those two (`LIBRARY_VALUES`) following the project's in the stage (`installed_dists.pipeline_text`).
SHARED_OUTPUT = "local-1:client-1:value-1"
LIBRARY_VALUES = ' + ":" + company.client.value() + ":" + fixlib.value()'
# The `pipeline.py` whose stage reads `company.local` through the package's name, which its module
# imported, and writes `SHARED_OUTPUT`.
TOP_NAME_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport company.local\nimport fixlib",
    "company.local.value()" + LIBRARY_VALUES,
)


def install_client_and_fixlib(
    venv: installed_dists.Venv, src: pathlib.Path, client: Mapping[str, str]
) -> None:
    """Install into the venv from their paths, under `src`, `Fix_Lib` (module `fixlib`,
    `installed_dists.FIXLIB_ALONE`) and `company-client`, whose files are `client` (each path
    relative to its project, with its text, as `CLIENT_FILES`), in the import packages of the
    directories that hold them."""
    fixlib = installed_dists.write_library(src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE)
    packages = sorted(
        {str(pathlib.PurePosixPath(relative).parent).replace("/", ".") for relative in client}
    )
    source = installed_dists.write_setuptools_project(
        src / "company-client", "company-client", "0.1", packages=packages, files=client
    )
    installed_dists.install(venv, str(fixlib), str(source))


def pythonpath_entry(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """`tmp_path/extra`, made here, outside the project root, which `PYTHONPATH`, set through
    `monkeypatch`, puts on the path of every pivot the test runs: a worker imports a module there
    by name."""
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(extra), prepend=os.pathsep)
    return extra


def project_with(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    client: Mapping[str, str],
    files: Mapping[str, str],
    extra_files: Mapping[str, str],
    pipeline_text: str,
) -> installed_dists.Project:
    """A project whose venv has `company-client`, whose files are `client`, and `Fix_Lib`
    installed into site-packages from their paths (`install_client_and_fixlib`), with
    `pipeline_text` as its `pipeline.py`, `files` in its root and `extra_files` in a `PYTHONPATH`
    entry (`pythonpath_entry`), each path relative to the directory that holds it."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    install_client_and_fixlib(venv, tmp_path / "src", client)
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    extra = pythonpath_entry(tmp_path, monkeypatch)
    for directory, held in ((project.root, files), (extra, extra_files)):
        for relative, text in held.items():
            (directory / relative).parent.mkdir(parents=True, exist_ok=True)
            (directory / relative).write_text(text)
    return project


def repro_outcome(project: installed_dists.Project) -> tuple[tuple[int, str, str], str]:
    """What one `pivot repro --json`, which must refuse and warn of nothing and complete
    `installed_dists.STAGE` once, gives of that stage: its exit code, the status of the stage's
    `stage_complete` event, and what the stage wrote when it ran, else the last line of the reason
    the event gives; and the command's output."""
    ran = installed_dists.run_pivot(project, "repro", "--json")
    output = installed_dists.shown(ran)
    events = [
        event
        for event in installed_dists.stage_complete_events(ran)
        if event["stage"] == installed_dists.STAGE
    ]
    assert len(events) == 1, f"`pivot repro` must complete {installed_dists.STAGE} once:\n{output}"
    assert installed_dists.REFUSAL_FIX not in output, (
        f"`pivot repro` must refuse nothing:\n{output}"
    )
    status = events[0]["status"]
    detail = (
        (project.root / installed_dists.OUTPUT).read_text()
        if status == "ran"
        else events[0]["reason"].splitlines()[-1]
    )
    return (ran.returncode, status, detail), output


def assert_runs_then_reruns_on_an_edit(
    project: installed_dists.Project, edited: pathlib.Path, key: str
) -> None:
    """`pivot repro` runs `installed_dists.STAGE`, which writes `SHARED_OUTPUT`, refusing nothing,
    and its lock holds the keys of `company-client` and `Fix_Lib` alone and no `submodule:` entry;
    after one byte of the project's file `edited` changes (`local-1` to `local-2`), the stage is
    stale on its code entry `key` alone, and `pivot repro` writes the new value."""
    manifest = installed_dists.assert_runs_with_no_refusal(project, SHARED_OUTPUT)
    assert installed_dists.dist_keys(manifest) == ["dist:company-client", "dist:fix-lib"], (
        f"the lock must hold the keys of company-client and Fix_Lib alone: {manifest}"
    )
    assert [entry for entry in manifest if "submodule:" in entry] == [], (
        f"the lock must hold no submodule: entry, which only the coordinator reads: {manifest}"
    )
    installed_dists.replace_once(edited, '"local-1"', '"local-2"')

    installed_dists.assert_stale_on(project, installed_dists.STAGE, key)
    installed_dists.assert_runs_with_no_refusal(
        project, SHARED_OUTPUT.replace("local-1", "local-2")
    )
