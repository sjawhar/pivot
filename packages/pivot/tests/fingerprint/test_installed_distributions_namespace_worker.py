"""The tests of how a stage's worker gets the project's part of a package it shares with an
installed library (spec requirement N1, dispatch://BET-1194/spec#n1), through the `pivot` CLI on
real installs, each against what a plain `python` run of the project writes.

Discovery registers a module under the project root for pickling by value only where its package
goes by value too, or where it has no package (`discovery._track_root_modules`). The project's
`company/local.py`, in a package `company` that a library shares, is pickled by reference, so the
worker imports it by name, as a plain run does: it holds one copy of the module, whose
import-time code runs there. The walk records each submodule a stage reads through such a package
(`submodule:` entries), and the worker imports each by name before the stage runs, which binds it
on the package as the import of the code that reads it would: code that cloudpickle sends by
reference can import the package alone. So the stage writes what a plain run writes when only
such code reads the module, when the stage holds a class of the module (`from company.local import
Mode`) while other code reads or imports it, when a module the project's part imports registers a
class as it is imported, when a later stage in the same worker reads the module, and when the
module holds a value cloudpickle cannot pickle. An edit of the module after discovery and before
the worker imports it is reported after the run, and the next run reruns the stage. An import
there that fails fails the stage with its traceback and a note naming the module, one that exits
fails it as an exit of the stage does, with that note, and an interrupt fails it as one of the
stage does. A `@no_fingerprint` stage's own code is scanned for the same reads
(`fingerprint.submodule_entries`), but not a helper module it calls: a helper under the root that
reads the module only inside a lambda, and a helper the worker imports by name that imports the
package alone (`import company`), are a stated limit, which two cases pin.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`) and its
own pivot project. The file runs pivot only through its CLI and imports nothing that pivot 0.10.4
lacks, so it collects with `packages/pivot/src` at that release too.
"""

import collections.abc
import pathlib

import pytest
from tests import installed_dists

# The module `helper`, in a `PYTHONPATH` entry, which the worker imports by name, with `import
# company` alone (but `installed_dists.READS_LOCAL`, which imports `company.local` itself), and code
# of it that reads `company.local` through the package's name: a function, a static method, a class
# method, a function under a `functools.wraps` decorator, the wrapper of a decorator without it, a
# lambda, a function under `functools.cache`, a property's getter. The stage calls that code;
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
# The module `helper`'s `read`, a lambda, which cloudpickle pickles by value whatever its module,
# reads `company.local` through the package's name; the stage calls the lambda itself, which
# `from helper import read` bound in its module.
_LAMBDA_HELPER = "import company.local\n\nread = lambda: company.local.value()\n"
_LAMBDA_HELPER_PIPELINE = installed_dists.pipeline_text(
    "import company.client\nimport fixlib\nfrom helper import read",
    "read()" + installed_dists.LIBRARY_VALUES,
)
# The project's `company/local.py` that runs `{second}` when a project's second import runs it: the
# first, in pivot's own process, leaves a file beside it. It raises (`_LOCAL_IMPORTED_ONCE`), exits
# with status 3 (`_LOCAL_EXITING_ON_A_SECOND_IMPORT`) or raises `KeyboardInterrupt`
# (`_LOCAL_INTERRUPTED_ON_A_SECOND_IMPORT`). `_IMPORTED_BY_NAME_NOTE` is the note Pivot adds to the
# failure of the worker's import of it before the stage runs.
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
_LOCAL_INTERRUPTED_ON_A_SECOND_IMPORT = _SECOND_IMPORT.format(second="raise KeyboardInterrupt")
_IMPORTED_BY_NAME_NOTE = (
    "Pivot imported company.local by name before the stage ran: the stage reads it through its "
    + "package."
)
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
# whose names cloudpickle does not follow, and holds its `Mode`: nothing of `company.sub`, a
# namespace package the project shares with `company-client` (`installed_dists.
# CLIENT_AND_SUB_FILES`), is pickled with the stage.
_SUB_MODES_PIPELINE = installed_dists.pipeline_text(
    "import company.sub.local\nfrom company.sub.local import Mode",
    "(lambda: company.sub.local.run(Mode.FAST))()",
)
# The project's `company/other.py`, which imports `company.local` itself (`_IMPORTS_LOCAL`), reads
# it through the package's name as it is imported (`_READS_LOCAL_AT_IMPORT`: `run` with
# `Mode.FAST`), or binds `Mode` itself and passes it to `company.local.run`, read through the
# package's name (`_OTHER_HOLDS_MODE`); only `tools.g` (`_READS_OTHER`, in a `PYTHONPATH` entry)
# reads it through the package's name, and the stage reads `company.local` (`_MODES`) so and holds
# its `Mode`.
_IMPORTS_LOCAL = 'import company.local\n\n\ndef g():\n    return "g"\n'
_READS_LOCAL_AT_IMPORT = """\
import company

FIRST = company.local.run(company.local.Mode.FAST)


def g():
    return FIRST
"""
_OTHER_HOLDS_MODE = """\
import company
from company.local import Mode as OtherMode


def g():
    return company.local.run(OtherMode.FAST)
"""
_READS_OTHER = "import company\n\n\ndef g():\n    return company.other.g()\n"
_READS_OTHER_PIPELINE = installed_dists.pipeline_text(
    "import company.local\nimport company.other\nimport tools\nfrom company.local import Mode",
    'tools.g() + ":" + company.local.run(Mode.FAST)',
)
# The module `tools2`, in a `PYTHONPATH` entry, binds `Mode` itself and passes it to
# `company.local.run`, read through the package's name; the stage holds `Mode` too and writes
# `tools2.h()` and the value of its own `Mode.FAST`.
_HOLDS_MODE = (
    "import company\nfrom company.local import Mode\n\n\n"
    + "def h():\n    return company.local.run(Mode.FAST)\n"
)
_HOLDS_MODE_PIPELINE = installed_dists.pipeline_text(
    "import tools2\nfrom company.local import Mode", 'tools2.h() + ":" + Mode.FAST.value'
)
# The project's `company/local.py` of `_MODES` with `kind`, which returns `Kind.A` of
# `company/other.py` (`_KINDS`), bound by `from company.other import Kind`. `tools.check`
# (`_CHECKS_KIND`, in a `PYTHONPATH` entry), which the stage calls with its `Mode.FAST`
# (`_CHECKS_PIPELINE`), reads both modules through the package's name: what `run` returns, and
# whether `kind()` returns the member of the `Kind` that `company.other` holds.
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
_CHECKS_PIPELINE = installed_dists.pipeline_text(
    "import tools\nfrom company.local import Mode", "tools.check(Mode.FAST)"
)
# A `company-client` whose `company.client` also holds a registry, `REGISTRY`; the project's
# `company/other.py` (`_REGISTERS_KIND`) registers its `Kind` there as it is imported, and
# `company/local.py` of `_MODES` imports it (`_MODES_IMPORTING_KIND`). `tools.check`
# (`_CHECKS_REGISTRY`, in a `PYTHONPATH` entry), which the stage calls with its `Mode.FAST`
# (`_CHECKS_PIPELINE`), reads both modules through the package's name: what `run` returns, and
# whether the registry holds the `Kind` that `company.other` holds.
_CLIENT_WITH_REGISTRY = {
    "company/client/__init__.py": installed_dists.CLIENT + "\n\nREGISTRY = {}\n"
}
_MODES_IMPORTING_KIND = "from company.other import Kind\n" + _MODES
_REGISTERS_KIND = """\
import enum

import company.client


class Kind(enum.Enum):
    A = "a"


company.client.REGISTRY["kind"] = Kind
"""
_CHECKS_REGISTRY = """\
import company
import company.client


def check(mode):
    registered = company.client.REGISTRY.get("kind") is company.other.Kind
    return company.local.run(mode) + ":" + ("registered" if registered else "missing")
"""
# The project's `company/local.py` of `installed_dists.LOCAL_WITH_LOCK` with `name`; the stage calls
# `name`, which `from company.local import name` bound, and `helper.value()`
# (`installed_dists.READS_LOCAL`, in a `PYTHONPATH` entry), which reads the module through the
# package's name.
_LOCAL_WITH_LOCK_AND_NAME = installed_dists.LOCAL_WITH_LOCK + '\n\ndef name():\n    return "n"\n'
_NAME_PIPELINE = installed_dists.pipeline_text(
    "import helper\nfrom company.local import name", 'name() + ":" + helper.value()'
)
# Two stages of `Pipeline("p")`: after the lines `{imports}`, `first` returns `{first}`, and
# `second`, which reads what `first` writes, so that it runs after `first`, returns `tools2.h()`
# (`_HOLDS_MODE`). The module `tools3`, in a `PYTHONPATH` entry, imports `company.local` itself.
_TWO_STAGES = """\
{imports}

from pivot import Pipeline, stage


@stage
def first() -> str:
    return {first}


@stage
def second(ready: str) -> str:
    return tools2.h()


with Pipeline("p") as pipeline:
    second(first())
"""
_IMPORTS_LOCAL_ITSELF = 'import company.local\n\n\ndef t():\n    return "t3"\n'
# The project's module `helper`, which imports `company.local` and reads it through the package's
# name only inside a lambda.
_READS_LOCAL_IN_A_LAMBDA = (
    "import company.local\n\n\ndef value():\n    return (lambda: company.local.value())()\n"
)
# The stage `edit` rewrites the project's `company/local.py`, at `{local}`, after discovery loaded
# it (`local-1` to `local-2`); the stage `make`, which reads what `edit` writes, so that it runs
# after `edit`, then reads `company.local` through the package's name.
_EDIT_THEN_READ = """\
import pathlib

import company.client
import company.local
import fixlib

from pivot import Pipeline, stage

LOCAL = {local!r}


@stage
def edit() -> str:
    path = pathlib.Path(LOCAL)
    path.write_text(path.read_text().replace("local-1", "local-2"))
    return "edited"


@stage
def make(ready: str) -> str:
    return company.local.value(){library_values}


with Pipeline("p") as pipeline:
    make(edit())
"""


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
    `PYTHONPATH` entry (`installed_dists.pythonpath_entry`), each path relative to the directory
    that holds it."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    installed_dists.install_client_and_fixlib(venv, tmp_path / "src", client)
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    extra = installed_dists.pythonpath_entry(tmp_path, monkeypatch)
    for directory, held in ((project.root, files), (extra, extra_files)):
        for relative, text in held.items():
            (directory / relative).parent.mkdir(parents=True, exist_ok=True)
            (directory / relative).write_text(text)
    return project


@pytest.mark.parametrize(
    ("helper", "call"),
    [
        pytest.param(_IMPORTS_THE_PACKAGE, "helper.value()", id="function"),
        pytest.param(installed_dists.READS_LOCAL, "helper.value()", id="function-importing-it"),
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
    of `company`, `company/local.py`, in its root, and that module holds a lock no stage reads and
    that cloudpickle cannot pickle. `pipeline.py` imports it, and the stage calls code of the
    module `helper`, in a `PYTHONPATH` entry, which the worker imports by name; only that code
    reads `company.local` through the package's name: a function, whose module imports `company`
    alone or `company.local` itself (`function-importing-it`), a static method, a class method, a
    function under a `functools.wraps` decorator, the wrapper of a decorator without it, a lambda,
    a function under `functools.cache`, or a property's getter. The stage's pickle holds the
    module `helper` by reference and nothing of `company.local`. `pivot repro` runs the stage,
    which writes `local-1:client-1:value-1`, refusing nothing: the worker imports `company.local`
    by name before the stage runs, as the import of that code would, and nothing of the module is
    pickled. After one byte of the project's module changes, the stage is stale on that module's
    `func:` entry alone, and `pivot repro` writes the new value. Were the module not imported
    before the stage, the stage would fail in the worker with
    `AttributeError: module 'company' has no attribute 'local'` where `helper` imports `company`
    alone."""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": installed_dists.LOCAL_WITH_LOCK},
        {"helper.py": helper},
        installed_dists.pipeline_text(_BY_REFERENCE_IMPORTS, call + installed_dists.LIBRARY_VALUES),
    )

    installed_dists.assert_runs_then_reruns_on_an_edit(
        project, project.root / "company" / "local.py", "func:company.local.value"
    )


def test_a_lambda_of_a_module_sent_by_reference_that_reads_the_projects_part_gets_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root. The module `helper`, in a `PYTHONPATH` entry,
    binds a lambda as `read` that reads `company.local` through the package's name, and the stage
    calls that lambda (`from helper import read`). cloudpickle pickles the lambda by value, as it
    pickles every lambda, so no import of `helper` runs in the worker for it, and imports there the
    module it reads through the package, which it pickles by reference; the worker imports that
    module by name before the stage runs too. `pivot repro` runs the stage, which writes
    `local-1:client-1:value-1`, refusing nothing. After one byte of the project's module changes,
    the stage is stale on that module's `func:` entry alone, and `pivot repro` writes the new
    value. Were the module imported in the worker neither way, the stage would fail there with
    `AttributeError: module 'company' has no attribute 'local'`."""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": installed_dists.LOCAL},
        {"helper.py": _LAMBDA_HELPER},
        _LAMBDA_HELPER_PIPELINE,
    )

    installed_dists.assert_runs_then_reruns_on_an_edit(
        project, project.root / "company" / "local.py", "func:company.local.value"
    )


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
    of `company`, `company/local.py`, in its root: an enum `Mode` and `run`, which returns `fast`
    for `Mode.FAST` alone, compared by identity. The stage is defined in the module `stages`, in a
    `PYTHONPATH` entry, which the worker imports by name, and `pipeline.py` imports and calls it.
    Its module binds `Mode` (`from company.local import Mode`), and the stage passes `Mode.FAST` to
    `company.local.run`, read through the package's name (`own-code`), also after asserting that
    the module it reads so holds the stage's `Mode` (`one-module`), or to `tools.run` of the module
    `tools` in the same entry, which reads `company.local` so (`through-a-helper`). cloudpickle
    pickles the stage function by value, and with it `Mode` by reference, as discovery pickles the
    project's part of a library's package by reference. `pivot repro` runs the stage, which writes
    `fast`, refusing nothing, as a plain `python` run of `stages` does: the worker imports
    `company.local` by name, and the stage's `Mode` is that module's class. Were the stage's `Mode`
    a copy apart from the module the code reads through the package, `run` would compare the
    stage's `Mode.FAST` with another copy of it and the stage would write `slow`, or fail on its
    assertion."""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": _MODES},
        {"stages.py": _STAGES.format(body=body), "tools.py": _TOOLS},
        _STAGES_PIPELINE,
    )

    installed_dists.assert_runs_with_no_refusal(project, "fast")


@pytest.mark.parametrize(
    ("client", "files", "extra_files", "pipeline_text", "output"),
    [
        pytest.param(
            installed_dists.CLIENT_AND_SUB_FILES,
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
        pytest.param(
            installed_dists.CLIENT_FILES,
            {"company/local.py": _MODES, "company/other.py": _OTHER_HOLDS_MODE},
            {"tools.py": _READS_OTHER},
            _READS_OTHER_PIPELINE,
            "fast:fast",
            id="an-import-by-name-holds-the-class",
        ),
        pytest.param(
            installed_dists.CLIENT_FILES,
            {"company/local.py": _MODES},
            {"tools2.py": _HOLDS_MODE},
            _HOLDS_MODE_PIPELINE,
            "fast:fast",
            id="a-helper-holds-the-class",
        ),
        pytest.param(
            installed_dists.CLIENT_FILES,
            {"company/local.py": _MODES_AND_KIND, "company/other.py": _KINDS},
            {"tools.py": _CHECKS_KIND},
            _CHECKS_PIPELINE,
            "fast:a",
            id="a-class-another-part-holds",
        ),
        pytest.param(
            _CLIENT_WITH_REGISTRY,
            {"company/local.py": _MODES_IMPORTING_KIND, "company/other.py": _REGISTERS_KIND},
            {"tools.py": _CHECKS_REGISTRY},
            _CHECKS_PIPELINE,
            "fast:registered",
            id="import-time-code",
        ),
    ],
)
def test_a_stage_holding_a_class_of_the_projects_part_writes_what_a_plain_run_writes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    client: dict[str, str],
    files: dict[str, str],
    extra_files: dict[str, str],
    pipeline_text: str,
    output: str,
) -> None:
    """`company-client` is installed into site-packages from its path, and the project holds `Mode`
    and `run` (`_MODES`) in its part of a namespace package it shares with that library, which
    discovery pickles by reference. The stage holds `Mode` (`from company.local import Mode`), so
    its pickle holds the class by reference, and other code reads or imports the module: the stage
    reads `company/sub/local.py` through `company` and `company.sub` inside a lambda, so that its
    pickle holds nothing of `company.sub`, which the worker imports by name
    (`package-imported-by-name`); or it reads `company/local.py` through the package's name beside
    `company/other.py`, which only `tools.g`, in a `PYTHONPATH` entry, reads through the package,
    and which reads `company.local` through the package's name as it is imported
    (`read-by-an-import-by-name`), imports it itself (`loaded-again-by-an-import-by-name`), or
    binds `Mode` itself and passes it to `company.local.run` (`an-import-by-name-holds-the-class`);
    or it writes `tools2.h()` and its own `Mode.FAST.value`, `tools2`, in a `PYTHONPATH` entry,
    binding `Mode` itself and passing it to `company.local.run` (`a-helper-holds-the-class`); or
    `company/local.py` holds `Kind` of `company/other.py` (`from company.other import Kind`), and
    `tools.check`, in a `PYTHONPATH` entry, reads both modules through the package's name and
    checks that `company.local.kind()` is `company.other.Kind.A` (`a-class-another-part-holds`),
    or that `company.client.REGISTRY`, where `company/other.py` registers its `Kind` as it is
    imported, holds `company.other.Kind` (`import-time-code`). `pivot repro` runs the stage, which
    writes what a plain `python` run of `pipeline.py` writes (`fast`, `fast:fast`, `g:fast`,
    `fast:fast`, `fast:fast`, `fast:a`, `fast:registered`), refusing nothing: the worker imports
    each module of the project's part by name, once, as a plain run does, so the stage's `Mode`,
    the modules other code reads through the package and every name an import of them binds are of
    one copy of each module, whose import-time code ran in the worker. Were a module of the
    project's part given to the worker as a copy apart from the one it imports, `run` would compare
    one copy's `Mode.FAST` with the other's (`slow`), `kind()` would return another copy's member
    (`not-a`), or `company.other` would not run its import-time code there (`missing`)."""
    project = _helper_project_with(tmp_path, monkeypatch, client, files, extra_files, pipeline_text)

    found, shown = installed_dists.repro_stage_outcome(project)

    assert found == (0, "ran", output), shown


def test_a_submodule_whose_function_the_stage_holds_and_that_cannot_be_pickled_is_imported_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project's
    `company/local.py`, in its root, holds a lock no stage reads and that cloudpickle cannot
    pickle, `value` and `name`. The stage calls `name`, which `from company.local import name`
    bound in its module, and `helper.value()` of the module `helper`, in a `PYTHONPATH` entry,
    which reads `company.local` through the package's name. The stage's pickle holds `name` by
    reference, as discovery pickles the project's part of a library's package by reference, so
    nothing of the module is pickled and the worker imports it by name: `pivot repro` runs the
    stage, which writes `n:local-1`, refusing nothing, as a plain `python` run does. Were the
    module pickled whole with the stage, the stage would fail as the pickling of its worker's call
    fails: `Could not pickle the task to send it to the workers.`"""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": _LOCAL_WITH_LOCK_AND_NAME},
        {"helper.py": installed_dists.READS_LOCAL},
        _NAME_PIPELINE,
    )

    found, shown = installed_dists.repro_stage_outcome(project)

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
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": _LOCAL_IMPORTED_ONCE},
        {"helper.py": _IMPORTS_THE_PACKAGE},
        installed_dists.pipeline_text(
            _BY_REFERENCE_IMPORTS, "helper.value()" + installed_dists.LIBRARY_VALUES
        ),
    )

    ran, event, shown = installed_dists.repro_stage_event(project)

    assert (ran.returncode, event["status"]) == (1, "failed"), shown
    lines = event["reason"].splitlines()
    assert lines[0] == "Traceback (most recent call last):", shown
    assert lines[-4].endswith('company/local.py", line 4, in <module>'), shown
    assert lines[-2:] == ["RuntimeError: a second import", _IMPORTED_BY_NAME_NOTE], shown


@pytest.mark.parametrize(
    ("local", "reason"),
    [
        pytest.param(
            _LOCAL_EXITING_ON_A_SECOND_IMPORT,
            f"Stage called sys.exit(3)\n{_IMPORTED_BY_NAME_NOTE}",
            id="exit",
        ),
        pytest.param(
            _LOCAL_INTERRUPTED_ON_A_SECOND_IMPORT, "KeyboardInterrupt", id="keyboard-interrupt"
        ),
    ],
)
def test_a_submodule_whose_import_by_name_exits_or_is_interrupted_fails_the_stage_as_the_stage_would(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, local: str, reason: str
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, which raises `SystemExit(3)` (`exit`) or
    `KeyboardInterrupt` (`keyboard-interrupt`) when a second import in the project runs it.
    `pipeline.py` imports it, and the stage calls `helper.value()` of the module `helper`, in a
    `PYTHONPATH` entry, which reads `company.local` through the package's name, so the worker
    imports `company.local` by name before the stage runs, a second import. `pivot repro` exits 1,
    and the stage fails with the reason the stage's own handling gives the same exception: for the
    exit, `Stage called sys.exit(3)`, then the note that Pivot imported `company.local` by name
    before the stage ran, and `KeyboardInterrupt` for the interrupt. Were the exit or the interrupt
    not handled there, the run would end with `Error: Unexpected SystemExit: 3` or the interrupt,
    and the stage would complete with no event; were the note left out, the exit's reason would
    not name the module."""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": local},
        {"helper.py": _IMPORTS_THE_PACKAGE},
        installed_dists.pipeline_text(
            _BY_REFERENCE_IMPORTS, "helper.value()" + installed_dists.LIBRARY_VALUES
        ),
    )

    ran, event, shown = installed_dists.repro_stage_event(project)

    assert (ran.returncode, event["status"], event["reason"]) == (1, "failed", reason), shown


@pytest.mark.parametrize(
    ("files", "extra_files", "imports", "first", "outputs"),
    [
        pytest.param(
            {"company/local.py": _MODES, "company/other.py": _IMPORTS_LOCAL},
            {"tools.py": _READS_OTHER, "tools2.py": _HOLDS_MODE},
            "import company.local\nimport company.other\nimport tools\nimport tools2\n"
            + "from company.local import Mode",
            'tools.g() + ":" + company.local.run(Mode.FAST)',
            ("g:fast", "fast"),
            id="after-an-import-by-name-imports-it",
        ),
        pytest.param(
            {"company/local.py": _MODES},
            {"tools3.py": _IMPORTS_LOCAL_ITSELF, "tools2.py": _HOLDS_MODE},
            "import company.local\nimport tools2\nimport tools3\nfrom company.local import Mode",
            'tools3.t() + ":" + company.local.run(Mode.FAST)',
            ("t3:fast", "fast"),
            id="after-a-helper-imports-it",
        ),
    ],
)
def test_a_later_stage_in_the_same_worker_reads_the_one_module_a_plain_run_holds(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    extra_files: dict[str, str],
    imports: str,
    first: str,
    outputs: tuple[str, str],
) -> None:
    """`company-client` is installed into site-packages from its path, and the project holds
    `Mode` and `run` (`_MODES`) in its part of a namespace package it shares with that library,
    `company/local.py`. One worker (`core.max_workers 1`) runs the stage `first`, which holds
    `Mode` and reads `company.local` through the package's name, then `second`, which reads what
    `first` writes and returns `tools2.h()`: `tools2`, in a `PYTHONPATH` entry, binds `Mode`
    itself and passes it to `company.local.run`, read through the package's name. In `first`,
    `company/other.py`, which only `tools.g` reads through the package, imports `company.local`
    itself (`after-an-import-by-name-imports-it`), or the module `tools3`, in a `PYTHONPATH`
    entry, which the worker imports by name, does (`after-a-helper-imports-it`). `pivot repro` runs
    both stages, which write what a plain `python` run of their functions writes (`g:fast` or
    `t3:fast`, then `fast`): the worker holds one copy of `company.local` for both stages, the one
    it imported by name. Were a copy given to `first` left bound on the package while an import
    loaded another into `sys.modules`, `second` would pass the second copy's `Mode.FAST` to the
    first copy's `run` and write `slow`."""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        files,
        extra_files,
        _TWO_STAGES.format(imports=imports, first=first),
    )
    installed_dists.pivot_ok(project, "config", "set", "core.max_workers", "1")

    ran = installed_dists.run_pivot(project, "repro", "--json")

    statuses = {
        event["stage"]: event["status"] for event in installed_dists.stage_complete_events(ran)
    }
    written = [project.root / "data" / "p" / f"{stage}.txt" for stage in ("first", "second")]
    found = tuple(path.read_text() if path.exists() else None for path in written)
    assert (ran.returncode, statuses, found) == (
        0,
        {"p/first": "ran", "p/second": "ran"},
        outputs,
    ), installed_dists.shown(ran)


def test_an_edit_of_the_projects_part_after_discovery_is_reported_and_reruns_the_stage(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path, and the project holds its
    part of `company`, `company/local.py`, in its root. The stage `edit` changes one byte of that
    file after discovery loaded it (`local-1` to `local-2`), and the stage `make` then reads
    `company.local` through the package's name, so the worker imports the edited file by name.
    `pivot repro` runs `make`, which writes `local-2:client-1:value-1`, as a plain `python` run
    after the edit does, warns that the stage imported user modules that changed after pivot
    loaded them, naming `company.local`, and marks its lock (`edited-source:company.local`); the
    next `pivot repro` reruns it for that, and the one after finds it cached. Were the module
    pickled by value with the stage, `make` would write `local-1:client-1:value-1`, the code pivot
    loaded, with no warning."""
    local = tmp_path / "project" / "company" / "local.py"
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        {"company/local.py": installed_dists.LOCAL},
        {},
        _EDIT_THEN_READ.format(local=str(local), library_values=installed_dists.LIBRARY_VALUES),
    )

    ran = installed_dists.run_pivot(project, "repro")

    first = installed_dists.shown(ran)
    written = (project.root / installed_dists.OUTPUT).read_text()
    edited = installed_dists.SHARED_OUTPUT.replace("local-1", "local-2")
    assert (ran.returncode, written) == (0, edited), first
    assert (
        "Stage 'p/make' imported user modules that changed after pivot loaded them: company.local."
    ) in " ".join(first.split()), first
    manifest = installed_dists.lock_manifest(project, installed_dists.STAGE)
    assert "edited-source:company.local" in manifest, manifest
    _, event, second = installed_dists.repro_stage_event(project)
    assert (event["status"], event["reason"]) == (
        "ran",
        "Previous run imported modules edited after pivot loaded them: company.local",
    ), second
    assert installed_dists.repro_status(project, installed_dists.STAGE)[0] == "cached"


@pytest.mark.parametrize(
    ("imports", "expression", "files", "extra_files", "outcome"),
    [
        pytest.param(
            "import company.client\nimport company.local\nimport fixlib",
            "company.local.value()" + installed_dists.LIBRARY_VALUES,
            {"company/local.py": installed_dists.LOCAL},
            {},
            (0, "ran", installed_dists.SHARED_OUTPUT),
            id="own-code",
        ),
        pytest.param(
            "import company.client\nimport fixlib\nimport helper",
            "helper.value()" + installed_dists.LIBRARY_VALUES,
            {"helper.py": installed_dists.READS_LOCAL, "company/local.py": installed_dists.LOCAL},
            {},
            (0, "ran", installed_dists.SHARED_OUTPUT),
            id="helper-under-the-root",
        ),
        pytest.param(
            "import company.client\nimport fixlib\nimport helper",
            "helper.value()" + installed_dists.LIBRARY_VALUES,
            {"helper.py": _READS_LOCAL_IN_A_LAMBDA, "company/local.py": installed_dists.LOCAL},
            {},
            (1, "failed", "AttributeError: module 'company' has no attribute 'local'"),
            id="limit-helper-under-the-root-in-a-lambda",
        ),
        pytest.param(
            _BY_REFERENCE_IMPORTS,
            "helper.value()" + installed_dists.LIBRARY_VALUES,
            {"company/local.py": installed_dists.LOCAL},
            {"helper.py": _IMPORTS_THE_PACKAGE},
            (1, "failed", "AttributeError: module 'company' has no attribute 'local'"),
            id="limit-helper-importing-the-package-alone",
        ),
    ],
)
def test_a_no_fingerprint_stage_reads_the_projects_part_only_where_its_worker_binds_it(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    imports: str,
    expression: str,
    files: dict[str, str],
    extra_files: dict[str, str],
    outcome: tuple[int, str, str],
) -> None:
    """`company-client` is installed into site-packages from its path, and the project holds its
    part of `company`, `company/local.py`, in its root. A `@no_fingerprint` stage, whose code the
    walk does not read, reads that module through the package's name in its own code
    (`own-code`): `pivot repro` runs the stage, which writes `local-1:client-1:value-1`, refusing
    nothing, as a scan of the stage function's code finds the module
    (`fingerprint.submodule_entries`), which the worker imports by name before the stage runs. The
    scan reads the stage function's own code, not a helper module it calls. The project's module
    `helper`, in its root, which discovery pickles by value, imports `company.local` and reads it
    so, and the stage calls `helper.value()` (`helper-under-the-root`): the stage runs and writes
    the same, as cloudpickle, which pickles the helper's function by value, imports in the worker
    each submodule of a package that the function's own code names. Two cases pin the stated limit
    (`docs/architecture/fingerprinting.md`, "Imports a Stage Makes While It Runs") where nothing
    in the worker binds the module, which `pipeline.py` imports: that helper reads it only inside
    a lambda, whose names cloudpickle does not follow (`limit-helper-under-the-root-in-a-lambda`),
    or only the module `helper`, in a `PYTHONPATH` entry, which the worker imports by name, reads
    it, and imports the package alone (`import company`,
    `limit-helper-importing-the-package-alone`): `pivot repro` exits 1, the stage failing in its
    worker with `AttributeError: module 'company' has no attribute 'local'`, where a plain `python`
    run writes `local-1:client-1:value-1`."""
    project = _helper_project_with(
        tmp_path,
        monkeypatch,
        installed_dists.CLIENT_FILES,
        files,
        extra_files,
        installed_dists.pipeline_text(
            imports + "\nfrom pivot import no_fingerprint", expression, "@no_fingerprint()"
        ),
    )

    found, shown = installed_dists.repro_stage_outcome(project)

    assert found == outcome, shown
