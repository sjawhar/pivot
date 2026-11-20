"""The tests that a namespace package shared with an installed library works, and stays keyed,
wherever it is installed, checked through the `pivot` CLI on real installs.

A stage that reads the project's part of a package it shares with an installed library through
the package's name (`import company.local`, then `company.local.value()` in the stage's code, a
helper's, a lambda's or an output writer's) runs in the worker as a plain `python` run of its
module does, and one byte changed in that part reruns it. The walk follows such a submodule and
records it (`submodule:` entries); the coordinator sends with the stage the name of each such
module that cloudpickle pickles by reference, which the worker imports, and each it pickles by
value that pickles alone, and the worker binds it on its package while the stage runs. Discovery
pickles the project's part by value, as it pickles every module under the project root, so the
stage and the project's code it holds carry one copy of each class and the values the coordinator
holds. A stage that reaches the module by another name (`from company import local`, or only
`from company.local import value`) runs too. Where the worker's import of a module sent by name
fails, the output of each stage it is sent with holds a warning that names the module, the worker
binds nothing for it, and the stage runs: a stage that reads the module fails with
`AttributeError`, as in pivot 0.10.4, one that does not read it in that run runs, and a stage
running beside them finishes. How the worker gets the module, case by case, against a plain run,
and the stated limits: `test_installed_distributions_namespace_worker.py`.

A namespace package's directory under the interpreter's prefix but in no `site-packages`
directory, as a `uv pip install --target` directory that a `.pth` file in `site-packages` puts on
`sys.path`, is keyed as the same install in `site-packages` is: Pivot gives the `dist:` keys each
of a namespace package's directories that is a library path (`fingerprint.is_library_path`), the
rule by which the files in it are library code, which the walk does not hash.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`,
`tests.shared_namespace`) and its own pivot project. The file runs pivot only through its CLI and
imports nothing that pivot 0.10.4 lacks, so it collects with `packages/pivot/src` at that release
too.
"""

import collections.abc
import importlib.metadata
import json
import pathlib

import pytest
from tests import installed_dists, shared_namespace

# The modules of `acme-core` and `acme-b`, the two portions `acme.core` and `acme.b` of the
# namespace package `acme`; the stage writes what `acme.b.value()` returns.
_ACME_CORE = 'def value():\n    return "core"\n'
_ACME_B = 'def value():\n    return "b1"\n'

# `company/__init__.py` of a `company-client` that makes `company` a regular package, which extends
# its `__path__` over each `company` directory on `sys.path`, the project root's included; the files
# of such a `company-client`, each path relative to its project. The project's part of `company`
# and `company-client`'s `company.client` are `shared_namespace.LOCAL` and
# `shared_namespace.CLIENT`, whose `value()`s each stage of the shared-package test writes (`LOCAL`
# also as `company.sub.local`), with `fixlib.value()` (`shared_namespace.LIBRARY_VALUES`).
_EXTENDING_INIT = "import pkgutil\n\n__path__ = pkgutil.extend_path(__path__, __name__)\n"
_CLIENT_EXTENDING = {"company/__init__.py": _EXTENDING_INIT, **shared_namespace.CLIENT_FILES}

# The stage of `shared_namespace.TOP_NAME_PIPELINE` reads `company.local` through the package's
# name; this one reads it so inside a lambda, whose names cloudpickle does not follow to import a
# module it pickles by reference.
_LAMBDA_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport company.local\nimport fixlib",
    "(lambda: company.local.value())()" + shared_namespace.LIBRARY_VALUES,
)
# The stage calls `helper.value()` of the project's module `helper`
# (`shared_namespace.READS_LOCAL`), which reads `company.local` through the package's name.
_HELPER_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport fixlib\nimport helper",
    "helper.value()" + shared_namespace.LIBRARY_VALUES,
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
    "company.sub.local.value()" + shared_namespace.LIBRARY_VALUES,
)
# The stage reads the module itself, which `from company import local` bound in its module.
_FROM_IMPORT_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport fixlib\nfrom company import local",
    "local.value()" + shared_namespace.LIBRARY_VALUES,
)
# The stage calls the function `from company.local import value` bound in its module, whose module
# (`shared_namespace.LOCAL_WITH_LOCK`) also holds a lock that no stage reads and that cloudpickle
# cannot pickle.
_FUNCTION_IMPORT_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport fixlib\nfrom company.local import value",
    "value()" + shared_namespace.LIBRARY_VALUES,
)
# The project's top-level module `common` (`_COMMON`) defines the enum `Kind`; its
# `company/local.py` (`_RUNS_KIND`) binds it (`from common import Kind`), and `run` returns `a` for
# `Kind.A` alone, compared by identity, as code that compares enum members does. The stage passes
# its own `Kind.A` to `run`.
_COMMON = 'import enum\n\n\nclass Kind(enum.Enum):\n    A = "a"\n'
_RUNS_KIND = (
    'from common import Kind\n\n\ndef run(k):\n    return "a" if k is Kind.A else "other"\n'
)
_KIND_PIPELINE = installed_dists.pipeline_text(
    "from common import Kind\nfrom company.local import run", "run(Kind.A)"
)
# The project's `company/config.py` (`_CONFIG`) holds `MODE = "default"`, which `mode` returns;
# `pipeline.py` sets `company.config.MODE` to `prod`, and the stage returns `mode()`.
_CONFIG = 'MODE = "default"\n\n\ndef mode():\n    return MODE\n'
_MODE_PIPELINE = installed_dists.pipeline_text(
    'import company.config\nfrom company.config import mode\n\ncompany.config.MODE = "prod"',
    "mode()",
)
# Three stages of `Pipeline("p")`, after `import company.local`: `reads` reads `company.local`
# through the package's name inside a lambda alone, whose names cloudpickle does not follow, so
# nothing of the module is pickled with it; `skips` reads it so only where the variable
# `_READ_LOCAL` names is set, and otherwise returns `skipped`; `other` reads nothing of `company`,
# and returns `other` one second after the file `{marker}` appears, so that it runs while a worker
# imports the module.
_READ_LOCAL = "READ_COMPANY_LOCAL"
_OVERLAPPING_STAGES = """\
import os
import pathlib
import time

import company.local

from pivot import Pipeline, stage

MARKER = {marker!r}


@stage
def reads() -> str:
    return (lambda: company.local.value())()


@stage
def skips() -> str:
    if os.environ.get({read_local!r}):
        return (lambda: company.local.value())()
    return "skipped"


@stage
def other() -> str:
    deadline = time.monotonic() + 60
    while not pathlib.Path(MARKER).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("no worker imported company.local a second time")
        time.sleep(0.05)
    time.sleep(1)
    return "other"


with Pipeline("p") as pipeline:
    reads()
    skips()
    other()
"""
# The warning, word for word, that a stage's output holds where its worker's import of
# `company.local`, sent by name, raised what `{raised}` names: the last line of what it raised.
_IMPORT_FAILED = (
    "WARNING: Could not import module 'company.local', which the stage reads through its package: "
    + "{raised}. The module is not bound on its package, so code that reads it through the package "
    + "fails."
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


def _helper_in_the_project_root(
    tmp_path: pathlib.Path, project: installed_dists.Project, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """The project root."""
    return project.root


def _helper_in_a_pythonpath_entry(
    tmp_path: pathlib.Path, project: installed_dists.Project, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """A `PYTHONPATH` entry outside the project root (`shared_namespace.pythonpath_entry`)."""
    return shared_namespace.pythonpath_entry(tmp_path, monkeypatch)


# Each case writes what a plain run writes, `shared_namespace.SHARED_OUTPUT`, then its edit.
@pytest.mark.parametrize(
    ("pipeline_text", "files", "client", "place", "edited", "key"),
    [
        pytest.param(
            shared_namespace.TOP_NAME_PIPELINE,
            {"company/local.py": shared_namespace.LOCAL},
            shared_namespace.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="top-name",
        ),
        pytest.param(
            _HELPER_PIPELINE,
            {"helper.py": shared_namespace.READS_LOCAL, "company/local.py": shared_namespace.LOCAL},
            shared_namespace.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="helper-module",
        ),
        pytest.param(
            _WRITER_PIPELINE,
            {"company/local.py": shared_namespace.LOCAL},
            shared_namespace.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            f"out:{installed_dists.STAGE}:func:company.local.value",
            id="writer-class",
        ),
        pytest.param(
            shared_namespace.TOP_NAME_PIPELINE,
            {"company/local.py": shared_namespace.LOCAL},
            _CLIENT_EXTENDING,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="pkgutil-package",
        ),
        pytest.param(
            _DEEPER_PIPELINE,
            {"company/sub/local.py": shared_namespace.LOCAL},
            shared_namespace.CLIENT_AND_SUB_FILES,
            _helper_in_the_project_root,
            "company/sub/local.py",
            "func:company.sub.local.value",
            id="deeper-chain",
        ),
        pytest.param(
            _FROM_IMPORT_PIPELINE,
            {"company/local.py": shared_namespace.LOCAL},
            shared_namespace.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="from-import",
        ),
        pytest.param(
            _FUNCTION_IMPORT_PIPELINE,
            {"company/local.py": shared_namespace.LOCAL_WITH_LOCK},
            shared_namespace.CLIENT_FILES,
            _helper_in_the_project_root,
            "company/local.py",
            "func:company.local.value",
            id="function-import",
        ),
        pytest.param(
            shared_namespace.TOP_NAME_PIPELINE,
            {"company/local.py": shared_namespace.LOCAL},
            shared_namespace.CLIENT_FILES,
            _helper_in_a_pythonpath_entry,
            "company/local.py",
            "func:company.local.value",
            id="pythonpath-by-reference",
        ),
        pytest.param(
            _LAMBDA_PIPELINE,
            {"company/local.py": shared_namespace.LOCAL},
            shared_namespace.CLIENT_FILES,
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
    (`company/sub/local.py`), in its root, which discovery pickles by value, or in a `PYTHONPATH`
    entry, which cloudpickle pickles by reference, so the worker imports it by name
    (`pythonpath-by-reference`, `lambda-by-reference`). The stage's module imports that module, and
    the stage reads it through the package's name (`company.local.value()`), in a helper module of
    the project that the stage calls (`helper-module`), in the output writer's `save`, which the
    worker runs after the stage (`writer-class`), through `company.sub` (`deeper-chain`), inside a
    lambda, whose names cloudpickle does not follow (`lambda-by-reference`), as the module that
    `from company import local` bound (`from-import`), or only as the function `from company.local
    import value` bound, whose module also holds a lock no stage reads and that cloudpickle cannot
    pickle (`function-import`). `pivot repro` runs the stage, which writes
    `local-1:client-1:value-1`, as a plain `python` run of its module would, refusing nothing, and
    its lock holds the keys of `company-client` and `Fix_Lib` alone. After one byte of the project's
    module changes, the stage is stale on that module's `func:` entry alone, and `pivot repro`
    writes the new value. Were the module read through the package neither sent with the stage nor
    bound on the package in the worker, the stage would fail with
    `AttributeError: module 'company' has no attribute 'local'`; were it not walked, the stage would
    stay cached after the change, and `pivot repro` would keep `local-1`. The lock holds no
    `submodule:` entry, as it holds no `byname:` entry: the stage runs although the entries that
    name the submodules it reads stay outside its lock."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    shared_namespace.install_client_and_fixlib(venv, tmp_path / "src", client)
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    directory = place(tmp_path, project, monkeypatch)
    for relative, text in files.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_text(text)

    shared_namespace.assert_runs_then_reruns_on_an_edit(project, directory / edited, key)


@pytest.mark.parametrize(
    ("files", "pipeline_text", "output"),
    [
        # A plain run writes `a`: `run` compares with the stage's own `Kind`.
        pytest.param(
            {"common.py": _COMMON, "company/local.py": _RUNS_KIND},
            _KIND_PIPELINE,
            "a",
            id="a-class-of-a-top-level-module",
        ),
        # A plain run writes `prod`: `mode` reads the value `pipeline.py` set.
        pytest.param(
            {"company/config.py": _CONFIG}, _MODE_PIPELINE, "prod", id="a-value-pipeline-py-sets"
        ),
    ],
)
def test_the_projects_part_of_a_shared_package_reaches_the_worker_as_the_coordinator_holds_it(
    tmp_path: pathlib.Path, files: dict[str, str], pipeline_text: str, output: str
) -> None:
    """`company-client` is installed into site-packages from its path, so `company` is a namespace
    package the project shares with it. The project's `company/local.py` binds `Kind` of the
    project's top-level module `common` (`from common import Kind`), and its `run` returns `a` for
    `Kind.A` alone, compared by identity; the stage holds `Kind` and `run` and returns
    `run(Kind.A)` (`a-class-of-a-top-level-module`). Or the project's `company/config.py` holds
    `MODE = "default"` and `mode`, which returns it, `pipeline.py` sets `company.config.MODE` to
    `prod`, and the stage returns `mode()` (`a-value-pipeline-py-sets`). `pivot repro` runs the
    stage, which writes `a` or `prod`, refusing nothing, as a plain `python` run of the project
    does: discovery pickles the project's part of `company` by value, as it pickles `common` and
    `pipeline.py`, so `run` compares with the stage's own `Kind`, and `mode` reads the value the
    coordinator holds. Were the project's part pickled by reference, the worker would import
    `company.local` from its file, and with it a second `common`, whose `Kind` is not the
    stage's (`other`), and `company.config` from its file (`default`)."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    shared_namespace.install_client_and_fixlib(
        venv, tmp_path / "src", shared_namespace.CLIENT_FILES
    )
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    for relative, text in files.items():
        (project.root / relative).parent.mkdir(parents=True, exist_ok=True)
        (project.root / relative).write_text(text)

    installed_dists.assert_runs_with_no_refusal(project, output)


@pytest.mark.parametrize(
    ("second", "raised"),
    [
        pytest.param(
            'raise RuntimeError("a second import")', "RuntimeError: a second import", id="raises"
        ),
        pytest.param("raise SystemExit(3)", "SystemExit: 3", id="exits"),
        pytest.param("raise KeyboardInterrupt", "KeyboardInterrupt", id="is-interrupted"),
        # `traceback.format_exception_only` gives a line naming where it was raised first.
        pytest.param('exec("return")', "SyntaxError: 'return' outside function", id="syntax-error"),
    ],
)
def test_a_module_sent_by_name_whose_import_fails_warns_and_fails_only_a_stage_that_reads_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, second: str, raised: str
) -> None:
    """`company-client` is installed into site-packages from its path. The project's part of
    `company`, `company/local.py`, lies in a `PYTHONPATH` entry, so cloudpickle pickles it by
    reference, and a second import of it, after `pipeline.py`'s, leaves a file `second` beside it
    and raises `RuntimeError` (`raises`), `SystemExit(3)` (`exits`), `KeyboardInterrupt`
    (`is-interrupted`) or a `SyntaxError`, whose description starts with a line naming where it
    was raised (`syntax-error`). Two workers (`core.max_workers 2`) run three stages: `reads`,
    which reads the module through the package's name inside a lambda alone, so the coordinator
    sends its name; `skips`, which reads it so only where `READ_COMPANY_LOCAL` is set, which it is
    not in this run, so the coordinator sends its name too; and `other`, which reads nothing of
    `company` and runs until that file appears. `pivot repro --keep-going --show-output` exits 1.
    The worker's import of the module fails before each of `reads` and `skips`, and that stage's
    output holds one warning, on one line, that names the module and the last line of what the
    import raised and says the module is not bound: `reads` then fails with
    `AttributeError: module 'company' has no attribute 'local'`, as in Pivot 0.10.4, which binds
    nothing on the package; `skips` runs and writes `skipped`, as in 0.10.4; `other` runs and
    writes `other`, and its output holds no warning. Were a failed import to fail the stage,
    `skips` would fail too; were the module sent as a module object, the worker would import it
    as it unpickles the call, which breaks the pool, and each stage running would fail with
    loky's `A task has failed to un-serialize`."""
    monkeypatch.delenv(_READ_LOCAL, raising=False)
    # So wide a console prints each line a stage logs on one line.
    monkeypatch.setenv("COLUMNS", "1000")
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {},
        {"company/local.py": shared_namespace.SECOND_IMPORT.format(second=second)},
        _OVERLAPPING_STAGES.format(
            marker=str(tmp_path / "extra" / "company" / "second"), read_local=_READ_LOCAL
        ),
    )
    installed_dists.pivot_ok(project, "config", "set", "core.max_workers", "2")

    ran = installed_dists.run_pivot(project, "repro", "--keep-going", "--show-output")

    lines = ran.stdout.splitlines()
    data = project.root / "data" / "p"
    outputs = {stage: data / f"{stage}.txt" for stage in ("reads", "skips", "other")}
    written = {
        stage: path.read_text() if path.exists() else None for stage, path in outputs.items()
    }
    found = (
        ran.returncode,
        written,
        sorted(line for line in lines if line.startswith("[p/")),
        [line.strip() for line in lines].count(shared_namespace.MISSING),
        "Summary: 2 ran, 0 cached, 0 blocked, 1 failed" in ran.stderr,
        "Failed stages:\n  p/reads\n" in ran.stderr,
    )
    warning = _IMPORT_FAILED.format(raised=raised)
    assert found == (
        1,
        {"reads": None, "skips": "skipped", "other": "other"},
        [f"[p/reads] {warning}", f"[p/skips] {warning}"],
        1,
        True,
        True,
    ), installed_dists.shown(ran)
