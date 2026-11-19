"""The tests of spec requirement N1 (dispatch://BET-1194/spec#n1): a namespace package shared with
an installed library works, and stays keyed, wherever it is installed, checked through the `pivot`
CLI on real installs.

A stage that reads the project's part of a package it shares with an installed library through
the package's name (`import company.local` in its module, `company.local.value()` in its code)
runs in the worker as a plain `python` run of its module does, and one byte changed in that part
reruns it: the walk follows a user submodule read through a library package and records it
(`submodule:` entries), and the coordinator sends each such submodule with the stage
(`loaded_source.submodules_to_send`), which the worker binds on the package it imports by name
when it is sent by value, and imports by name, which binds it, when it is sent by reference.

A namespace package's directory under the interpreter's prefix but in no `site-packages`
directory, as a `uv pip install --target` directory that a `.pth` file in `site-packages` puts on
`sys.path`, is keyed as the same install in `site-packages` is: Pivot gives the `dist:` keys each
of a namespace package's directories that is a library path (`fingerprint.is_library_path`), the
rule by which the files in it are library code, which the walk does not hash.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`) and its
own pivot project. The file runs pivot only through its CLI and imports nothing that pivot 0.10.4
lacks, so it collects with `packages/pivot/src` at that release too.
"""

import collections.abc
import importlib.metadata
import json
import os
import pathlib

import pytest
from tests import installed_dists

# The modules of `acme-core` and `acme-b`, the two portions `acme.core` and `acme.b` of the
# namespace package `acme`; the stage writes what `acme.b.value()` returns.
_ACME_CORE = 'def value():\n    return "core"\n'
_ACME_B = 'def value():\n    return "b1"\n'

# The module `company.sub.shared` of `company-client`'s part of the namespace package `company.sub`;
# the project's part of `company` and `company-client`'s `company.client` are
# `installed_dists.LOCAL` and `installed_dists.CLIENT`, whose `value()`s each stage of the
# shared-package test writes (`LOCAL` also as `company.sub.local`).
_SHARED = 'def value():\n    return "shared-1"\n'
# `company/__init__.py` of a `company-client` that makes `company` a regular package, which extends
# its `__path__` over each `company` directory on `sys.path`, the project root's included.
_EXTENDING_INIT = "import pkgutil\n\n__path__ = pkgutil.extend_path(__path__, __name__)\n"
# The files of `company-client`, each path relative to its project: `company.client` alone, with
# `company.sub.shared` too, or in a regular package `company` that extends its `__path__`.
_CLIENT_ALONE = {"company/client/__init__.py": installed_dists.CLIENT}
_CLIENT_AND_SUB = {
    "company/client/__init__.py": installed_dists.CLIENT,
    "company/sub/shared.py": _SHARED,
}
_CLIENT_EXTENDING = {
    "company/__init__.py": _EXTENDING_INIT,
    "company/client/__init__.py": installed_dists.CLIENT,
}
# What each stage of the shared-package test writes: the project's `value()`, then
# `company.client.value()` and `fixlib.value()`, joined.
_SHARED_OUTPUT = "local-1:client-1:value-1"

# The stage reads `company.local` through the package's name, which its module imported.
_TOP_NAME_PIPELINE = """\
import company.client
import company.local
import fixlib

from pivot import Pipeline, stage


@stage
def make() -> str:
    return company.local.value() + ":" + company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
# The stage reads `company.local` through the package's name inside a lambda, whose names
# cloudpickle does not follow to import a module sent by reference.
_LAMBDA_PIPELINE = """\
import company.client
import company.local
import fixlib

from pivot import Pipeline, stage


@stage
def make() -> str:
    return (lambda: company.local.value())() + ":" + company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
# The stage calls `helper.value()` of the project's module `helper` (`_HELPER`), which reads
# `company.local` through the package's name.
_HELPER = "import company.local\n\n\ndef value():\n    return company.local.value()\n"
_HELPER_PIPELINE = """\
import company.client
import fixlib
import helper

from pivot import Pipeline, stage


@stage
def make() -> str:
    return helper.value() + ":" + company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
# The stage's output writer, a `loaders.Text` whose `save` the worker runs after the stage,
# prefixes what it writes with `company.local.value()`, read through the package's name.
_WRITER_PIPELINE = """\
import dataclasses
import pathlib
from typing import Annotated

import company.client
import company.local
import fixlib

from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class Prefixed(loaders.Text):
    def save(self, data: str, path: pathlib.Path) -> None:
        super().save(company.local.value() + ":" + data, path)


@stage
def make() -> Annotated[str, Prefixed()]:
    return company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
# The stage reads `company.sub.local` through `company` and `company.sub`, a namespace package the
# project shares with `company-client` too.
_DEEPER_PIPELINE = """\
import company.client
import company.sub.local
import fixlib

from pivot import Pipeline, stage


@stage
def make() -> str:
    return company.sub.local.value() + ":" + company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
# The stage reads the module itself, which `from company import local` bound in its module.
_FROM_IMPORT_PIPELINE = """\
import company.client
import fixlib
from company import local

from pivot import Pipeline, stage


@stage
def make() -> str:
    return local.value() + ":" + company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
# The stage calls the function `from company.local import value` bound in its module, whose module
# (`_LOCAL_WITH_LOCK`) also holds a lock that no stage reads and that cloudpickle cannot pickle.
_FUNCTION_IMPORT_PIPELINE = """\
import company.client
import fixlib
from company.local import value

from pivot import Pipeline, stage


@stage
def make() -> str:
    return value() + ":" + company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
_LOCAL_WITH_LOCK = "import threading\n\n_LOCK = threading.Lock()\n\n\n" + installed_dists.LOCAL


def _helper_in_site_packages(venv: installed_dists.Venv) -> tuple[pathlib.Path, list[str]]:
    """The venv's site-packages, where `uv pip install` installs with no option, and no option."""
    return venv.site_packages, []


def _helper_in_a_target_directory_under_the_prefix(
    venv: installed_dists.Venv,
) -> tuple[pathlib.Path, list[str]]:
    """`<venv>/vendor`, under the interpreter's prefix and in no `site-packages` directory, which
    the `.pth` file `vendor.pth` in site-packages puts on `sys.path` after site-packages, and the
    options that install there (`--target <venv>/vendor`)."""
    vendor = venv.python.parent.parent / "vendor"
    (venv.site_packages / "vendor.pth").write_text(f"{vendor}\n")
    return vendor, ["--target", str(vendor)]


def _helper_installed_commit(directory: pathlib.Path) -> str:
    """The commit that the `direct_url.json` of `acme-b` in `directory` names; the test fails
    unless `directory` holds one `acme-b`, installed from git."""
    found = list(importlib.metadata.Distribution.discover(name="acme-b", path=[str(directory)]))
    assert len(found) == 1, f"{directory} must hold one acme-b; it holds {len(found)}"
    text = found[0].read_text("direct_url.json")
    assert text is not None, f"acme-b in {directory} must have a direct_url.json"
    return json.loads(text)["vcs_info"]["commit_id"]


@pytest.mark.parametrize(
    "place",
    [
        pytest.param(_helper_in_a_target_directory_under_the_prefix, id="prefix-target-dir"),
        pytest.param(_helper_in_site_packages, id="site-packages"),
    ],
)
def test_a_git_install_in_a_shared_namespace_reruns_its_stage_wherever_it_lies(
    tmp_path: pathlib.Path,
    place: collections.abc.Callable[[installed_dists.Venv], tuple[pathlib.Path, list[str]]],
) -> None:
    """`acme-core`, whose one package is the namespace portion `acme.core`, is installed into
    site-packages from its path. `acme-b`, whose one package is the portion `acme.b`, is installed
    from commit C1 of its git repository, its RECORD kept, into `<venv>/vendor` with `uv pip
    install --target`, a directory under the interpreter's prefix that a `.pth` file in
    site-packages puts on `sys.path` (`prefix-target-dir`), or into site-packages
    (`site-packages`). The stage imports `acme.b` and writes `acme.b.value()`, so it reaches the
    namespace package `acme` by its top name, whose directories are `acme` in site-packages and,
    for `prefix-target-dir`, `acme` in `<venv>/vendor`. `pivot repro` runs the stage, which writes
    `b1`, refusing nothing, and its lock holds the keys of `acme-b` and `acme-core`. After one byte
    of `acme.b` changes at a new commit C2 and `acme-b` is reinstalled from it, `pivot status`
    reports the stage stale on `dist:acme-b` alone, and `pivot repro` writes `b2`. Were the
    directory under the prefix taken as the user's part of `acme`, as `is_user_code` takes a
    namespace directory outside site-packages, `acme-b` would get no key there: the stage would
    stay cached after the change, and `pivot repro` would keep `b1`."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    core = installed_dists.write_setuptools_project(
        tmp_path / "src" / "acme-core",
        "acme-core",
        "0.1",
        packages=["acme.core"],
        files={"acme/core/__init__.py": _ACME_CORE},
    )
    acme_b = installed_dists.write_setuptools_project(
        tmp_path / "src" / "acme-b",
        "acme-b",
        "0.1",
        packages=["acme.b"],
        files={"acme/b/__init__.py": _ACME_B},
    )
    installed_dists.install_directory(venv, core)
    project = installed_dists.make_project_in(tmp_path, venv, installed_dists.pipeline("acme.b"))
    directory, options = place(venv)
    installed_dists.install(
        venv, *options, installed_dists.init_git_requirement(acme_b, project.home, "acme-b")
    )
    imported = installed_dists.module_file(project, "acme.b")
    assert imported == (directory / "acme" / "b" / "__init__.py").resolve(), (
        f"the venv must import acme.b from {directory}, not {imported}"
    )

    manifest = installed_dists.assert_runs_with_no_refusal(project, "b1")
    assert installed_dists.dist_keys(manifest) == ["dist:acme-b", "dist:acme-core"], (
        f"the lock must hold the keys of acme-b and acme-core: {manifest}"
    )
    installed_dists.replace_once(acme_b / "acme" / "b" / "__init__.py", '"b1"', '"b2"')
    commit = installed_dists.commit_all(acme_b, project.home, "C2")
    installed_dists.install(
        venv,
        *options,
        installed_dists.git_requirement("acme-b", acme_b, commit),
        reinstall=True,
    )
    assert _helper_installed_commit(directory) == commit, (
        f"acme-b in {directory} must now be installed from C2 {commit}"
    )

    installed_dists.assert_stale_on(project, installed_dists.STAGE, "dist:acme-b")
    installed_dists.assert_runs_and_writes(project, "b2")


def _helper_install_client_and_fixlib(
    venv: installed_dists.Venv, src: pathlib.Path, client: collections.abc.Mapping[str, str]
) -> None:
    """Install from their paths, under `src`, `Fix_Lib` (module `fixlib`,
    `installed_dists.FIXLIB_ALONE`) and `company-client`, whose files are `client` (each path
    relative to its project, with its text), in the import packages of the directories that hold
    them."""
    fixlib = installed_dists.write_library(src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE)
    packages = sorted(
        {str(pathlib.PurePosixPath(relative).parent).replace("/", ".") for relative in client}
    )
    source = installed_dists.write_setuptools_project(
        src / "company-client", "company-client", "0.1", packages=packages, files=client
    )
    installed_dists.install(venv, str(fixlib), str(source))


def _helper_in_the_project_root(
    tmp_path: pathlib.Path, project: installed_dists.Project, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """The project root: discovery sends each module under it by value."""
    return project.root


def _helper_in_a_pythonpath_entry(
    tmp_path: pathlib.Path, project: installed_dists.Project, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """`extra`, outside the project root, which `PYTHONPATH` puts on the path of every pivot this
    test runs: a worker imports a module there by name."""
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(extra), prepend=os.pathsep)
    return extra


@pytest.mark.parametrize(
    ("pipeline_text", "files", "client", "place", "edited", "key"),
    [
        pytest.param(
            _TOP_NAME_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            _CLIENT_ALONE,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="top-name",
        ),
        pytest.param(
            _HELPER_PIPELINE,
            {"helper.py": _HELPER, "company/local.py": installed_dists.LOCAL},
            _CLIENT_ALONE,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="helper-module",
        ),
        pytest.param(
            _WRITER_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            _CLIENT_ALONE,
            _helper_in_the_project_root,
            "company/local.py",
            f"out:{installed_dists.STAGE}:func:company.local.value",
            id="writer-class",
        ),
        pytest.param(
            _TOP_NAME_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            _CLIENT_EXTENDING,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="pkgutil-package",
        ),
        pytest.param(
            _DEEPER_PIPELINE,
            {"company/sub/local.py": installed_dists.LOCAL},
            _CLIENT_AND_SUB,
            _helper_in_the_project_root,
            "company/sub/local.py",
            "func:company.sub.local.value",
            id="deeper-chain",
        ),
        pytest.param(
            _FROM_IMPORT_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            _CLIENT_ALONE,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="from-import",
        ),
        pytest.param(
            _FUNCTION_IMPORT_PIPELINE,
            {"company/local.py": _LOCAL_WITH_LOCK},
            _CLIENT_ALONE,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="function-import",
        ),
        pytest.param(
            _TOP_NAME_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            _CLIENT_ALONE,
            _helper_in_a_pythonpath_entry,
            "company/local.py",
            "func:company.local.value",
            id="pythonpath-by-reference",
        ),
        pytest.param(
            _LAMBDA_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            _CLIENT_ALONE,
            _helper_in_a_pythonpath_entry,
            "company/local.py",
            "func:company.local.value",
            id="lambda-by-reference",
        ),
    ],
)
def test_the_projects_part_of_a_shared_package_read_through_its_name_runs_and_reruns_on_an_edit(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_text: str,
    files: dict[str, str],
    client: dict[str, str],
    place: collections.abc.Callable[
        [pathlib.Path, installed_dists.Project, pytest.MonkeyPatch], pathlib.Path
    ],
    edited: str,
    key: str,
) -> None:
    """`company-client` is installed into site-packages from its path: the namespace portion
    `company.client`, which imports `company`, also with the portion `company.sub.shared`
    (`deeper-chain`), or a regular package `company` whose `__init__` extends its `__path__`
    (`pkgutil-package`). The project holds its part of `company`, `company/local.py`
    (`company/sub/local.py`), in its root, which discovery sends by value while the worker imports
    `company` by name, or in a `PYTHONPATH` entry, which the worker imports by name
    (`pythonpath-by-reference`, `lambda-by-reference`). The stage's module imports that module,
    and the stage reads it through the package's name (`company.local.value()`), in a helper
    module of the project that the stage calls (`helper-module`), in the output writer's `save`,
    which the worker runs after the stage (`writer-class`), through `company.sub`
    (`deeper-chain`), inside a lambda, whose names cloudpickle does not follow
    (`lambda-by-reference`), as the module that `from company import local` bound
    (`from-import`), or only as the function `from company.local import value` bound, in a module
    that also holds a lock no stage reads (`function-import`). `pivot repro` runs the stage, which
    writes `local-1:client-1:value-1`, as a plain `python` run of its module would, refusing
    nothing, and its lock holds the keys of `company-client` and `Fix_Lib` alone. After one byte
    of the project's module changes, the stage is stale on that module's `func:` entry alone, and
    `pivot repro` writes the new value. Were the module read through the package neither bound on
    the package the worker imported nor imported there by name, the stage would fail with
    `AttributeError: module 'company' has no attribute 'local'`; were it not walked, the stage
    would stay cached after the change, and `pivot repro` would keep `local-1`; were the module
    the stage reaches only by other names sent with it too, the stage could not be pickled, for
    its lock. The lock holds no `submodule:` entry, as it holds no `byname:` entry: the stage
    runs although the entries that name the submodules it reads stay outside its lock."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    _helper_install_client_and_fixlib(venv, tmp_path / "src", client)
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    directory = place(tmp_path, project, monkeypatch)
    for relative, text in files.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_text(text)

    manifest = installed_dists.assert_runs_with_no_refusal(project, _SHARED_OUTPUT)
    assert installed_dists.dist_keys(manifest) == ["dist:company-client", "dist:fix-lib"], (
        f"the lock must hold the keys of company-client and Fix_Lib alone: {manifest}"
    )
    assert [key for key in manifest if "submodule:" in key] == [], (
        f"the lock must hold no submodule: entry, which only the coordinator reads: {manifest}"
    )
    installed_dists.replace_once(directory / edited, '"local-1"', '"local-2"')

    installed_dists.assert_stale_on(project, installed_dists.STAGE, key)
    installed_dists.assert_runs_with_no_refusal(
        project, _SHARED_OUTPUT.replace("local-1", "local-2")
    )
