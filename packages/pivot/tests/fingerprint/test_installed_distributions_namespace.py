"""The tests of spec requirement N1 (dispatch://BET-1194/spec#n1): a namespace package shared with
an installed library works, and stays keyed, wherever it is installed, checked through the `pivot`
CLI on real installs.

A stage that reads the project's part of a package it shares with an installed library through
the package's name (`import company.local`, then `company.local.value()` in the stage's code, a
helper's, a lambda's or an output writer's) runs in the worker as a plain `python` run of its
module does, and one byte changed in that part reruns it. The walk follows such a submodule and
records it (`submodule:` entries). The coordinator pickles the worker's call once more, with a
pickler built on loky's, to see what cloudpickle carries of the submodule
(`submodule_delivery.submodule_delivery`): when it carries the module itself by value, the module
is sent with the stage and bound on its package, so the stage's values and the module the code
reads are one copy; when it carries only a function or class of it, the module is sent so where
it can be pickled whole, and imported by name in the worker otherwise, as pivot 0.10.4 does; when
it carries nothing of it, the worker imports it by name before the stage runs, as the code that
reads it would, and nothing of its values is pickled. A sent module joins what the call carries,
so another submodule of which it holds a class is decided by the same rules. The worker binds the
sent modules, imports the others by name, a package before its submodules, and binds the sent
ones again, so neither a package it imports by name nor an import by name that loads a sent
module from its file leaves the stage's code another copy. An import there that fails fails the
stage with its traceback and a note naming the module, and one that exits fails it as an exit of
the stage does. A stage that reaches the module by another name (`from company import local`, or
only `from company.local import value`) runs too. A `@no_fingerprint` stage's own code is scanned
for the same reads (`fingerprint.submodule_entries`), but not a helper module it calls: a stated
limit, which one case pins.

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
import subprocess
from typing import Any

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
# The files of `company-client`, each path relative to its project: `company.client`
# (`installed_dists.CLIENT_FILES`) with `company.sub.shared` too, or in a regular package `company`
# that extends its `__path__`.
_CLIENT_AND_SUB = {**installed_dists.CLIENT_FILES, "company/sub/shared.py": _SHARED}
_CLIENT_EXTENDING = {"company/__init__.py": _EXTENDING_INIT, **installed_dists.CLIENT_FILES}
# What each stage of the shared-package tests writes: the project's `value()`, then
# `company.client.value()` and `fixlib.value()`, joined, the expression of those two
# (`_LIBRARY_VALUES`) following the project's in each stage (`installed_dists.pipeline_text`).
_SHARED_OUTPUT = "local-1:client-1:value-1"
_LIBRARY_VALUES = ' + ":" + company.client.value() + ":" + fixlib.value()'

# The stage reads `company.local` through the package's name, which its module imported.
_TOP_NAME_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport company.local\nimport fixlib",
    "company.local.value()" + _LIBRARY_VALUES,
)
# The stage reads `company.local` through the package's name inside a lambda, whose names
# cloudpickle does not follow to import a module sent by reference.
_LAMBDA_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport company.local\nimport fixlib",
    "(lambda: company.local.value())()" + _LIBRARY_VALUES,
)
# The stage calls `helper.value()` of the project's module `helper` (`_HELPER`), which reads
# `company.local` through the package's name.
_HELPER = "import company.local\n\n\ndef value():\n    return company.local.value()\n"
_HELPER_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport fixlib\nimport helper", "helper.value()" + _LIBRARY_VALUES
)
# The module `helper`'s `read`, a lambda, which cloudpickle sends by value whatever its module,
# reads `company.local` through the package's name; the stage calls the lambda itself, which
# `from helper import read` bound in its module.
_LAMBDA_HELPER = "import company.local\n\nread = lambda: company.local.value()\n"
_LAMBDA_HELPER_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport fixlib\nfrom helper import read", "read()" + _LIBRARY_VALUES
)
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
_DEEPER_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport company.sub.local\nimport fixlib",
    "company.sub.local.value()" + _LIBRARY_VALUES,
)
# The stage reads the module itself, which `from company import local` bound in its module.
_FROM_IMPORT_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport fixlib\nfrom company import local",
    "local.value()" + _LIBRARY_VALUES,
)
# The stage calls the function `from company.local import value` bound in its module, whose module
# (`_LOCAL_WITH_LOCK`) also holds a lock that no stage reads and that cloudpickle cannot pickle.
_FUNCTION_IMPORT_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport fixlib\nfrom company.local import value",
    "value()" + _LIBRARY_VALUES,
)
_LOCAL_WITH_LOCK = "import threading\n\n_LOCK = threading.Lock()\n\n\n" + installed_dists.LOCAL
# The module `helper`, in a `PYTHONPATH` entry, which the worker imports by name, with `import
# company` alone (but `_HELPER`, which imports `company.local` itself), and code of it that reads
# `company.local` through the package's name: a function, a static method, a class method, a
# function under a `functools.wraps` decorator, the wrapper of a decorator without it, a lambda, a
# function under `functools.cache`, a property's getter. The stage calls that code;
# `_BY_REFERENCE_IMPORTS` are the imports of its module.
_IMPORTS_THE_PACKAGE = "import company\n\n\ndef value():\n    return company.local.value()\n"
_STATIC_METHOD = """\
import company


class Tools:
    @staticmethod
    def value():
        return company.local.value()
"""
_CLASS_METHOD = """\
import company


class Tools:
    @classmethod
    def value(cls):
        return company.local.value()
"""
_UNDER_WRAPS = """\
import functools

import company


def traced(function):
    @functools.wraps(function)
    def wrapper():
        return function()

    return wrapper


@traced
def value():
    return company.local.value()
"""
_UNDER_A_DECORATOR = """\
import company


def prefixed(function):
    def wrapper():
        return company.local.value() + function()

    return wrapper


@prefixed
def value():
    return ""
"""
_LAMBDA = "import company\n\nvalue = lambda: company.local.value()\n"
_UNDER_CACHE = """\
import functools

import company


@functools.cache
def value():
    return company.local.value()
"""
_PROPERTY = """\
import company


class Tools:
    @property
    def value(self):
        return company.local.value()
"""
_BY_REFERENCE_IMPORTS = "import company.client\nimport company.local\nimport fixlib\nimport helper"
# The project's `company/local.py` that runs `{second}` when a project's second import runs it: the
# first, in pivot's own process, leaves a file beside it. It raises (`_LOCAL_IMPORTED_ONCE`) or
# exits with status 3 (`_LOCAL_EXITING_ON_A_SECOND_IMPORT`).
_SECOND_IMPORT = """\
import pathlib

if pathlib.Path(__file__).with_name("imported").exists():
    {second}
pathlib.Path(__file__).with_name("imported").touch()


def value():
    return "local-1"
"""
_LOCAL_IMPORTED_ONCE = _SECOND_IMPORT.format(second='raise RuntimeError("a second import")')
_LOCAL_EXITING_ON_A_SECOND_IMPORT = _SECOND_IMPORT.format(second="raise SystemExit(3)")
# The project's `company/local.py` of the `Mode` tests: an enum `Mode`, and `run`, which tells its
# members apart by identity, as code that compares enum members does.
_MODES = """\
import enum


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


def run(mode):
    return "fast" if mode is Mode.FAST else "slow"
"""
# The module `stages`, in a `PYTHONPATH` entry, which the worker imports by name, defines the stage
# `make` that `_STAGES_PIPELINE` imports and calls, whose body is `{body}`. Its module holds `Mode`,
# which `from company.local import Mode` bound, and the stage reads `company.local` through the
# package's name or calls `tools.run` (`_TOOLS`, in the same entry), which reads it so.
_STAGES = """\
import company
import tools
from company.local import Mode

from pivot import stage


@stage
def make() -> str:
{body}
"""
_TOOLS = "import company\n\n\ndef run(mode):\n    return company.local.run(mode)\n"
_STAGES_PIPELINE = """\
from stages import make

from pivot import Pipeline

with Pipeline("p") as pipeline:
    make()
"""
# The stage calls `company.sub.local.run` (`_MODES`, in `company/sub/local.py`) inside a lambda,
# whose names cloudpickle does not follow, and holds its `Mode`: of the module, the stage's pickle
# carries that class alone, and nothing of `company.sub`, a namespace package the project shares
# with `company-client` (`_CLIENT_AND_SUB`).
_SUB_MODES_PIPELINE = installed_dists.pipeline_text(
    "import company.sub.local\nfrom company.sub.local import Mode",
    "(lambda: company.sub.local.run(Mode.FAST))()",
)
# The project's `company/other.py`, which imports `company.local` itself (`_IMPORTS_LOCAL`), or
# reads it through the package's name as it is imported (`_READS_LOCAL_AT_IMPORT`: `run` with
# `Mode.FAST`); only `tools.g` (`_READS_OTHER`, in a `PYTHONPATH` entry) reads it through the
# package's name, and the stage reads `company.local` (`_MODES`) so and holds its `Mode`.
_IMPORTS_LOCAL = 'import company.local\n\n\ndef g():\n    return "g"\n'
_READS_LOCAL_AT_IMPORT = """\
import company

FIRST = company.local.run(company.local.Mode.FAST)


def g():
    return FIRST
"""
_READS_OTHER = "import company\n\n\ndef g():\n    return company.other.g()\n"
_READS_OTHER_PIPELINE = installed_dists.pipeline_text(
    "import company.local\nimport company.other\nimport tools\nfrom company.local import Mode",
    'tools.g() + ":" + company.local.run(Mode.FAST)',
)
# The project's `company/local.py` of `_MODES` with `kind`, which returns `Kind.A` of
# `company/other.py` (`_KINDS`), bound by `from company.other import Kind`. `tools.check`
# (`_CHECKS_KIND`, in a `PYTHONPATH` entry), which the stage calls with its `Mode.FAST`, reads both
# modules through the package's name: what `run` returns, and whether `kind()` returns the member
# of the `Kind` that `company.other` holds.
_KINDS = 'import enum\n\n\nclass Kind(enum.Enum):\n    A = "a"\n'
_MODES_AND_KIND = (
    "from company.other import Kind\n" + _MODES + "\n\ndef kind():\n    return Kind.A\n"
)
_CHECKS_KIND = """\
import company


def check(mode):
    same = company.local.kind() is company.other.Kind.A
    return company.local.run(mode) + ":" + ("a" if same else "not-a")
"""
_CHECKS_KIND_PIPELINE = installed_dists.pipeline_text(
    "import tools\nfrom company.local import Mode", "tools.check(Mode.FAST)"
)
# The project's `company/local.py` of `_LOCAL_WITH_LOCK` with `name`; the stage calls `name`, which
# `from company.local import name` bound, and `helper.value()` (`_HELPER`, in a `PYTHONPATH`
# entry), which reads the module through the package's name.
_LOCAL_WITH_LOCK_AND_NAME = _LOCAL_WITH_LOCK + '\n\ndef name():\n    return "n"\n'
_NAME_PIPELINE = installed_dists.pipeline_text(
    "import helper\nfrom company.local import name", 'name() + ":" + helper.value()'
)


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


def _helper_runs_then_reruns_on_an_edit(
    project: installed_dists.Project, edited: pathlib.Path, key: str
) -> None:
    """`pivot repro` runs the stage, which writes `_SHARED_OUTPUT`, refusing nothing, and its
    lock holds the keys of `company-client` and `Fix_Lib` alone and no `submodule:` entry; after
    one byte of the project's file `edited` changes (`local-1` to `local-2`), the stage is stale on
    its code entry `key` alone, and `pivot repro` writes the new value."""
    manifest = installed_dists.assert_runs_with_no_refusal(project, _SHARED_OUTPUT)
    assert installed_dists.dist_keys(manifest) == ["dist:company-client", "dist:fix-lib"], (
        f"the lock must hold the keys of company-client and Fix_Lib alone: {manifest}"
    )
    assert [entry for entry in manifest if "submodule:" in entry] == [], (
        f"the lock must hold no submodule: entry, which only the coordinator reads: {manifest}"
    )
    installed_dists.replace_once(edited, '"local-1"', '"local-2"')

    installed_dists.assert_stale_on(project, installed_dists.STAGE, key)
    installed_dists.assert_runs_with_no_refusal(
        project, _SHARED_OUTPUT.replace("local-1", "local-2")
    )


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
            installed_dists.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="top-name",
        ),
        pytest.param(
            _HELPER_PIPELINE,
            {"helper.py": _HELPER, "company/local.py": installed_dists.LOCAL},
            installed_dists.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="helper-module",
        ),
        pytest.param(
            _WRITER_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            installed_dists.CLIENT_FILES,
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
            installed_dists.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="from-import",
        ),
        pytest.param(
            _FUNCTION_IMPORT_PIPELINE,
            {"company/local.py": _LOCAL_WITH_LOCK},
            installed_dists.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="function-import",
        ),
        pytest.param(
            _TOP_NAME_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            installed_dists.CLIENT_FILES,
            _helper_in_a_pythonpath_entry,
            "company/local.py",
            "func:company.local.value",
            id="pythonpath-by-reference",
        ),
        pytest.param(
            _LAMBDA_PIPELINE,
            {"company/local.py": installed_dists.LOCAL},
            installed_dists.CLIENT_FILES,
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
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", client)
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    directory = place(tmp_path, project, monkeypatch)
    for relative, text in files.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_text(text)

    _helper_runs_then_reruns_on_an_edit(project, directory / edited, key)


@pytest.mark.parametrize(
    ("helper", "call"),
    [
        pytest.param(_IMPORTS_THE_PACKAGE, "helper.value()", id="function"),
        pytest.param(_HELPER, "helper.value()", id="function-importing-it"),
        pytest.param(_STATIC_METHOD, "helper.Tools.value()", id="staticmethod"),
        pytest.param(_CLASS_METHOD, "helper.Tools.value()", id="classmethod"),
        pytest.param(_UNDER_WRAPS, "helper.value()", id="wraps-decorator"),
        pytest.param(_UNDER_A_DECORATOR, "helper.value()", id="decorator-without-wraps"),
        pytest.param(_LAMBDA, "helper.value()", id="lambda"),
        pytest.param(_UNDER_CACHE, "helper.value()", id="functools-cache"),
        pytest.param(_PROPERTY, "helper.Tools().value", id="property"),
    ],
)
def test_a_submodule_only_code_sent_by_reference_reads_through_its_package_is_imported_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, helper: str, call: str
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, which discovery sends by value, and that module
    holds a lock no stage reads and that cloudpickle cannot pickle. `pipeline.py` imports it, and
    the stage calls code of the module `helper`, in a `PYTHONPATH` entry, which the worker imports
    by name; only that code reads `company.local` through the package's name: a function, whose
    module imports `company` alone or `company.local` itself (`function-importing-it`), a static
    method, a class method, a function under a `functools.wraps` decorator, the wrapper of a
    decorator without it, a lambda, a function under `functools.cache`, or a property's getter.
    The stage's pickle holds the module `helper` by reference and nothing of `company.local`.
    `pivot repro` runs the stage, which writes `local-1:client-1:value-1`, refusing nothing: the
    worker imports `company.local` by name before the stage runs, as the import of that code
    would, and nothing of the module is pickled. After one byte of the project's module changes,
    the stage is stale on that module's `func:` entry alone, and `pivot repro` writes the new
    value. Were the module neither sent nor imported, the stage would fail in the worker with
    `AttributeError: module 'company' has no attribute 'local'` where `helper` imports `company`
    alone; were it sent, the stage could not be pickled, for its lock."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", installed_dists.CLIENT_FILES)
    pipeline_text = installed_dists.pipeline_text(_BY_REFERENCE_IMPORTS, call + _LIBRARY_VALUES)
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    extra = _helper_in_a_pythonpath_entry(tmp_path, project, monkeypatch)
    (extra / "helper.py").write_text(helper)
    local = project.root / "company" / "local.py"
    local.parent.mkdir()
    local.write_text(_LOCAL_WITH_LOCK)

    _helper_runs_then_reruns_on_an_edit(project, local, "func:company.local.value")


def test_a_lambda_of_a_module_sent_by_reference_that_reads_the_projects_part_gets_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, which discovery sends by value. The module
    `helper`, in a `PYTHONPATH` entry, which the worker would import by name, binds a lambda as
    `read` that reads `company.local` through the package's name, and the stage calls that lambda
    (`from helper import read`). cloudpickle sends the lambda by value, as it sends every lambda,
    and with it the module it reads through the package, so no import of `helper` binds the
    module in the worker: the coordinator sends it with the stage, and `pivot repro` runs the
    stage, which writes `local-1:client-1:value-1`, refusing nothing. After one byte of the
    project's module changes, the stage is stale on that module's `func:` entry alone, and `pivot
    repro` writes the new value. Were the module neither sent nor imported by name, the stage
    would fail in the worker with `AttributeError: module 'company' has no attribute 'local'`."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", installed_dists.CLIENT_FILES)
    project = installed_dists.make_project_in(tmp_path, venv, _LAMBDA_HELPER_PIPELINE)
    extra = _helper_in_a_pythonpath_entry(tmp_path, project, monkeypatch)
    (extra / "helper.py").write_text(_LAMBDA_HELPER)
    local = project.root / "company" / "local.py"
    local.parent.mkdir()
    local.write_text(installed_dists.LOCAL)

    _helper_runs_then_reruns_on_an_edit(project, local, "func:company.local.value")


def _helper_repro_event(
    project: installed_dists.Project,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], str]:
    """One `pivot repro --json`, which must refuse and warn of nothing: the command, the one
    `stage_complete` event of `STAGE` (`installed_dists.stage_complete_events`), and the command's
    output."""
    ran = installed_dists.run_pivot(project, "repro", "--json")
    shown = installed_dists.shown(ran)
    events = [
        event
        for event in installed_dists.stage_complete_events(ran)
        if event["stage"] == installed_dists.STAGE
    ]
    assert len(events) == 1, f"`pivot repro` must complete {installed_dists.STAGE} once:\n{shown}"
    assert installed_dists.REFUSAL_FIX not in shown, f"`pivot repro` must refuse nothing:\n{shown}"
    return ran, events[0], shown


def _helper_repro_outcome(project: installed_dists.Project) -> tuple[tuple[int, str, str], str]:
    """What one `pivot repro --json` gives of `STAGE` (`_helper_repro_event`): its exit code, the
    status of the stage's `stage_complete` event, and what the stage wrote when it ran, else the
    last line of the reason the event gives; and the command's output."""
    ran, event, shown = _helper_repro_event(project)
    status = event["status"]
    detail = (
        (project.root / installed_dists.OUTPUT).read_text()
        if status == "ran"
        else event["reason"].splitlines()[-1]
    )
    return (ran.returncode, status, detail), shown


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("    return company.local.run(Mode.FAST)", id="own-code"),
        pytest.param(
            '    assert company.local.Mode is Mode, "the stage holds another copy of Mode"\n'
            + "    return company.local.run(Mode.FAST)",
            id="one-module",
        ),
        pytest.param("    return tools.run(Mode.FAST)", id="through-a-helper"),
    ],
)
def test_a_stage_and_the_submodule_it_reads_through_its_package_hold_one_copy_of_its_values(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, which discovery sends by value: an enum `Mode`
    and `run`, which returns `fast` for `Mode.FAST` alone, compared by identity. The stage is
    defined in the module `stages`, in a `PYTHONPATH` entry, which the worker imports by name, and
    `pipeline.py` imports and calls it. Its module binds `Mode` (`from company.local import
    Mode`), and the stage passes `Mode.FAST` to `company.local.run`, read through the package's
    name (`own-code`), also after asserting that the module it reads so holds the stage's `Mode`
    (`one-module`), or to `tools.run` of the module `tools` in the same entry, which reads
    `company.local` so (`through-a-helper`). cloudpickle pickles the stage function by value, and
    with it `Mode`, a class of a module it sends by value, and, where the stage's own code reads
    `company.local` through the package, the module itself. `pivot repro` runs the stage, which
    writes `fast`, refusing nothing, as a plain `python` run of `stages` does: the coordinator
    sends the module with the stage, so the worker binds on the package the module of which the
    stage's `Mode` is the class. Were the worker to import `company.local` by name instead, `run`
    would compare the stage's `Mode.FAST` with another copy of it and the stage would write
    `slow`, or fail on its assertion."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", installed_dists.CLIENT_FILES)
    project = installed_dists.make_project_in(tmp_path, venv, _STAGES_PIPELINE)
    extra = _helper_in_a_pythonpath_entry(tmp_path, project, monkeypatch)
    (extra / "stages.py").write_text(_STAGES.format(body=body))
    (extra / "tools.py").write_text(_TOOLS)
    local = project.root / "company" / "local.py"
    local.parent.mkdir()
    local.write_text(_MODES)

    installed_dists.assert_runs_with_no_refusal(project, "fast")


def _helper_project_with(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    client: collections.abc.Mapping[str, str],
    files: collections.abc.Mapping[str, str],
    extra_files: collections.abc.Mapping[str, str],
    pipeline_text: str,
) -> installed_dists.Project:
    """A project whose venv has `company-client`, whose files are `client`, and `Fix_Lib`
    installed into site-packages from their paths (`installed_dists.install_client_and_fixlib`),
    with `pipeline_text` as its `pipeline.py`, `files` in its root and `extra_files` in a
    `PYTHONPATH` entry (`_helper_in_a_pythonpath_entry`), each path relative to the directory
    that holds it."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", client)
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    extra = _helper_in_a_pythonpath_entry(tmp_path, project, monkeypatch)
    for directory, held in ((project.root, files), (extra, extra_files)):
        for relative, text in held.items():
            (directory / relative).parent.mkdir(parents=True, exist_ok=True)
            (directory / relative).write_text(text)
    return project


@pytest.mark.parametrize(
    ("client", "files", "extra_files", "pipeline_text", "output"),
    [
        pytest.param(
            _CLIENT_AND_SUB,
            {"company/sub/local.py": _MODES},
            {},
            _SUB_MODES_PIPELINE,
            "fast",
            id="package-imported-by-name",
        ),
        pytest.param(
            installed_dists.CLIENT_FILES,
            {"company/local.py": _MODES, "company/other.py": _READS_LOCAL_AT_IMPORT},
            {"tools.py": _READS_OTHER},
            _READS_OTHER_PIPELINE,
            "fast:fast",
            id="read-by-an-import-by-name",
        ),
        pytest.param(
            installed_dists.CLIENT_FILES,
            {"company/local.py": _MODES, "company/other.py": _IMPORTS_LOCAL},
            {"tools.py": _READS_OTHER},
            _READS_OTHER_PIPELINE,
            "g:fast",
            id="loaded-again-by-an-import-by-name",
        ),
    ],
)
def test_a_submodule_sent_with_the_stage_is_bound_before_and_after_the_imports_by_name(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    client: dict[str, str],
    files: dict[str, str],
    extra_files: dict[str, str],
    pipeline_text: str,
    output: str,
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds `Mode`
    and `run` (`_MODES`) in its part of a namespace package it shares with that library, and the
    coordinator sends that module with the stage, whose pickle carries `Mode`: as
    `company/sub/local.py`, which the stage reads through `company` and `company.sub` inside a
    lambda, so that the pickle carries nothing of `company.sub`, which the worker imports by name
    (`package-imported-by-name`); or as `company/local.py`, which the stage reads through the
    package's name, beside `company/other.py`, which the worker imports by name, as only `tools.g`,
    in a `PYTHONPATH` entry, reads it through the package, and which reads `company.local` through
    the package's name as it is imported (`read-by-an-import-by-name`) or imports it itself
    (`loaded-again-by-an-import-by-name`). `pivot repro` runs the stage, which writes what a plain
    `python` run of `pipeline.py` writes (`fast`, `fast:fast`, `g:fast`), refusing nothing: the
    worker binds the sent module on its package before its imports by name and again after them.
    Were it bound only after them, the import of `company.other` would fail with
    `AttributeError: module 'company' has no attribute 'local'`. Were it bound only before them,
    the stage would fail with `AttributeError: module 'company.sub' has no attribute 'local'`, as
    the package was not there to bind it on, or write `g:slow`, as the import of `company.other`
    loads `company.local` from its file and binds that copy on the package, whose `Mode` is not the
    stage's."""
    project = _helper_project_with(tmp_path, monkeypatch, client, files, extra_files, pipeline_text)

    found, shown = _helper_repro_outcome(project)

    assert found == (0, "ran", output), shown


def test_a_submodule_of_which_a_sent_module_holds_a_class_is_sent_with_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project's part of
    `company` holds `company/local.py`, with `Mode`, `run` and `kind`, which returns `Kind.A` of
    `company/other.py` (`from company.other import Kind`). The stage holds `Mode` and returns
    `tools.check(Mode.FAST)` of the module `tools`, in a `PYTHONPATH` entry, which reads both
    modules through the package's name and checks that `company.local.kind()` is
    `company.other.Kind.A`. The stage's pickle carries `Mode` of `company.local` and nothing of
    `company.other`; the coordinator sends `company.local` with the stage, and its namespace, which
    the call then carries too, holds `Kind`, so the coordinator sends `company.other` with it.
    `pivot repro` runs the stage, which writes `fast:a`, as a plain `python` run does, refusing
    nothing. Were `company.other` decided by what the stage alone carries, the worker would import
    it by name from its file, whose `Kind` is not the one the sent `company.local` holds, and the
    stage would write `fast:not-a`."""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": _MODES_AND_KIND, "company/other.py": _KINDS},
        {"tools.py": _CHECKS_KIND},
        _CHECKS_KIND_PIPELINE,
    )

    found, shown = _helper_repro_outcome(project)

    assert found == (0, "ran", "fast:a"), shown


def test_a_submodule_whose_function_the_stage_holds_and_that_cannot_be_pickled_is_imported_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project's
    `company/local.py`, in its root, which discovery sends by value, holds a lock no stage reads
    and that cloudpickle cannot pickle, `value` and `name`. The stage calls `name`, which `from
    company.local import name` bound in its module, and `helper.value()` of the module `helper`,
    in a `PYTHONPATH` entry, which reads `company.local` through the package's name. The stage's
    pickle carries `name` by value, and the module cannot be pickled whole, so the worker imports
    it by name, as pivot 0.10.4 does: `pivot repro` runs the stage, which writes `n:local-1`,
    refusing nothing, the stage's `name` and the worker's module being two copies (a stated
    limit). Were the module sent because the stage carries a function of it, the stage would fail
    as the pickling of its worker's call fails: `Could not pickle the task to send it to the
    workers.`"""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": _LOCAL_WITH_LOCK_AND_NAME},
        {"helper.py": _HELPER},
        _NAME_PIPELINE,
    )

    found, shown = _helper_repro_outcome(project)

    assert found == (0, "ran", "n:local-1"), shown


def test_a_submodule_its_worker_fails_to_import_by_name_fails_the_stage_with_its_traceback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, which raises `RuntimeError: a second import`
    when a second import in the project runs it. `pipeline.py` imports it, and the stage calls
    `helper.value()` of the module `helper`, in a `PYTHONPATH` entry, which reads
    `company.local` through the package's name, so the worker imports `company.local` by name
    before the stage runs, a second import. `pivot repro` exits 1, and the stage fails: its reason
    is the traceback of that import, whose last frame is the module's raising line, and whose
    last lines are the error and the note that Pivot imported `company.local` by name before the
    stage ran. Were the import made outside the stage's handling of a failure, the run would
    report `Unexpected RuntimeError: a second import`, with no traceback and no module's name."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", installed_dists.CLIENT_FILES)
    pipeline_text = installed_dists.pipeline_text(
        _BY_REFERENCE_IMPORTS, "helper.value()" + _LIBRARY_VALUES
    )
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    extra = _helper_in_a_pythonpath_entry(tmp_path, project, monkeypatch)
    (extra / "helper.py").write_text(_IMPORTS_THE_PACKAGE)
    local = project.root / "company" / "local.py"
    local.parent.mkdir()
    local.write_text(_LOCAL_IMPORTED_ONCE)

    ran, event, shown = _helper_repro_event(project)

    assert (ran.returncode, event["status"]) == (1, "failed"), shown
    lines = event["reason"].splitlines()
    assert lines[0] == "Traceback (most recent call last):", shown
    assert lines[-4].endswith('company/local.py", line 4, in <module>'), shown
    assert lines[-2:] == [
        "RuntimeError: a second import",
        "Pivot imported company.local by name before the stage ran: the stage reads it through "
        + "its package.",
    ], shown


def test_a_submodule_whose_import_by_name_exits_fails_the_stage_as_an_exit_of_the_stage_does(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, which raises `SystemExit(3)` when a second
    import in the project runs it. `pipeline.py` imports it, and the stage calls `helper.value()`
    of the module `helper`, in a `PYTHONPATH` entry, which reads `company.local` through the
    package's name, so the worker imports `company.local` by name before the stage runs, a second
    import. `pivot repro` exits 1, and the stage fails with the reason a `sys.exit(3)` in the
    stage gives it: `Stage called sys.exit(3)`. Were the exit not handled there, the run would end
    with `Error: Unexpected SystemExit: 3`, and the stage would complete with no event."""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": _LOCAL_EXITING_ON_A_SECOND_IMPORT},
        {"helper.py": _IMPORTS_THE_PACKAGE},
        installed_dists.pipeline_text(_BY_REFERENCE_IMPORTS, "helper.value()" + _LIBRARY_VALUES),
    )

    ran, event, shown = _helper_repro_event(project)

    assert (ran.returncode, event["status"], event["reason"]) == (
        1,
        "failed",
        "Stage called sys.exit(3)",
    ), shown


def test_a_stage_whose_submodule_cloudpickle_cannot_pickle_fails_as_its_pickling_fails(
    tmp_path: pathlib.Path,
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, which discovery sends by value, and which holds
    a lock that cloudpickle cannot pickle, and the stage reads it through the package's name
    (`company.local.value()`): cloudpickle pickles the stage function by value, and with it the
    module, so the coordinator's pickling of the call to see what it carries raises. `pivot repro`
    exits 1, and the stage fails as the pickling of its worker's call fails, with loky's `Could
    not pickle the task to send it to the workers.`. Were the error of the first pickling raised in
    the coordinator, the run would end with it, and the stage would complete with no event."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", installed_dists.CLIENT_FILES)
    project = installed_dists.make_project_in(tmp_path, venv, _TOP_NAME_PIPELINE)
    local = project.root / "company" / "local.py"
    local.parent.mkdir()
    local.write_text(_LOCAL_WITH_LOCK)

    found, shown = _helper_repro_outcome(project)

    assert found == (1, "failed", "Could not pickle the task to send it to the workers."), shown


@pytest.mark.parametrize(
    ("imports", "expression", "files", "outcome"),
    [
        pytest.param(
            "import company.client\nimport company.local\nimport fixlib",
            "company.local.value()" + _LIBRARY_VALUES,
            {"company/local.py": installed_dists.LOCAL},
            (0, "ran", _SHARED_OUTPUT),
            id="own-code",
        ),
        pytest.param(
            "import company.client\nimport fixlib\nimport helper",
            "helper.value()" + _LIBRARY_VALUES,
            {"helper.py": _HELPER, "company/local.py": installed_dists.LOCAL},
            (1, "failed", "AttributeError: module 'company' has no attribute 'local'"),
            id="limit-helper-under-the-root",
        ),
    ],
)
def test_a_no_fingerprint_stage_gets_the_projects_part_its_own_code_reads_not_a_helpers(
    tmp_path: pathlib.Path,
    imports: str,
    expression: str,
    files: dict[str, str],
    outcome: tuple[int, str, str],
) -> None:
    """`company-client` is installed into site-packages from its path, and the project holds its
    part of `company`, `company/local.py`, in its root, which discovery sends by value. A
    `@no_fingerprint` stage, whose code the walk does not read, reads that module through the
    package's name in its own code (`own-code`): `pivot repro` runs the stage, which writes
    `local-1:client-1:value-1`, refusing nothing, as a scan of the stage function's code finds the
    module (`fingerprint.submodule_entries`), which the coordinator sends with the stage; were it
    not sent, the stage would fail in the worker with
    `AttributeError: module 'company' has no attribute 'local'`. The case
    `limit-helper-under-the-root` pins a stated limit (`docs/architecture/fingerprinting.md`,
    "Imports a Stage Makes While It Runs"): the scan reads the stage function's own code, not a
    helper module it calls. Only the project's module `helper`, in its root, which discovery
    sends by value, reads `company.local` so (`import company.local`, `company.local.value()`),
    and the stage calls `helper.value()`: `pivot repro` exits 1, the stage failing in its worker
    with that `AttributeError`, as no code the scan read reads the module through the package."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", installed_dists.CLIENT_FILES)
    pipeline_text = installed_dists.pipeline_text(
        imports + "\nfrom pivot import no_fingerprint", expression, "@no_fingerprint()"
    )
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    for relative, text in files.items():
        (project.root / relative).parent.mkdir(parents=True, exist_ok=True)
        (project.root / relative).write_text(text)

    found, shown = _helper_repro_outcome(project)

    assert found == outcome, shown
