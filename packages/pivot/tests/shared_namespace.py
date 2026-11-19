"""What the CLI tests of the project's part of a package it shares with an installed library share:
`company-client`, whose namespace portion `company.client` makes `company` such a package, and
`Fix_Lib`, installed into a venv from their paths (`install_client_and_fixlib`), the files of a
`company-client` that shares `company.sub` too (`CLIENT_AND_SUB_FILES`), the project's
`company/local.py` that also holds a lock (`LOCAL_WITH_LOCK`), a project module that reads it
through the package's name (`READS_LOCAL`), a `PYTHONPATH` entry from which a worker imports by
name (`pythonpath_entry`), what the stage of such a project writes (`SHARED_OUTPUT`, the project's
value then `LIBRARY_VALUES`), the `pipeline.py` whose stage reads `company.local` through the
package's name (`TOP_NAME_PIPELINE`) and a run that writes it and reruns on an edit
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

# The files of a `company-client` whose `company.client` lies beside the portion
# `company.sub.shared`, so that `company.sub` is a namespace package it shares with the project too.
CLIENT_AND_SUB_FILES: Mapping[str, str] = {
    **installed_dists.CLIENT_FILES,
    "company/sub/shared.py": 'def value():\n    return "shared-1"\n',
}
# The project's `company/local.py` of `installed_dists.LOCAL` that also holds a lock no stage reads
# and that cloudpickle cannot pickle, and a project module that imports `company.local` and reads
# it through the package's name.
LOCAL_WITH_LOCK = "import threading\n\n_LOCK = threading.Lock()\n\n\n" + installed_dists.LOCAL
READS_LOCAL = "import company.local\n\n\ndef value():\n    return company.local.value()\n"
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
    relative to its project, with its text, as `installed_dists.CLIENT_FILES`), in the import
    packages of the directories that hold them."""
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
