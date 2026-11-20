"""The tests of how a stage's worker gets the project's part of a package it shares with an
installed library, through the `pivot` CLI on real installs, each against what a plain `python`
run of the project writes, or pinning what a stated limit gives.

Discovery pickles the project's `company/local.py` by value, as it pickles every module under the
project root, though the package `company` that holds it is a library's too, which cloudpickle
pickles by reference. The walk records each submodule a stage reads through such a package
(`submodule:` entries); the coordinator sends with the stage each such module that pickles alone,
or its name where cloudpickle pickles it by reference, which the worker imports before anything
of the stage runs, and the worker binds it on its package while the stage runs, where the package
holds nothing of that name, then removes the binding. So the stage writes what a plain run writes
when code cloudpickle pickles by value reads the module through the package, when the stage holds
a class of the module (`from company.local import Mode`) while other code reads it so, also code
the worker imports by name that imports the module itself, when a later stage in the same worker
reads the module, and when a second import of the module would fail; and an edit of the module
after discovery reruns the stage on the next run. A module sent by name whose import fails in the
worker is warned of in the output of each stage it is sent with and binds nothing, so only a stage
that reads it fails, with `AttributeError`: `test_installed_distributions_namespace.py`.

The stated limits (`docs/architecture/fingerprinting.md`, "Imports a Stage Makes While It Runs")
that a test here pins, each of which pivot 0.10.4 has too: a module that cannot be pickled is
missing where only code the worker imports by name reads it, without importing it itself, and
fails the stage that reads it so itself; code pickled by value and code imported by name that both
hold the module hold two copies; a module's import-time code does not run in the worker (pivot
0.10.4 fails that test's stage earlier, with `AttributeError`, as it binds nothing on the
package); and a `@no_fingerprint` stage's scan does not read a helper module it calls. The other
stated limits are documented only, pinned by no test: a stage body that imports the library's
package itself, a function of a sent module that rebinds one of its globals (`global COUNT`), a
project package below the shared one (`company.proj`), and cloudpickle's import, as the worker
unpickles the call, of a submodule it pickles by reference that the stage's own code names.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`,
`tests.shared_namespace`) and its own pivot project. The file runs pivot only through its CLI and
imports nothing that pivot 0.10.4 lacks, so it collects with `packages/pivot/src` at that release
too.
"""

import collections.abc
import pathlib

import pytest
from tests import installed_dists, shared_namespace

# The module `helper`, in a `PYTHONPATH` entry, which the worker imports by name, with `import
# company` alone (`_IMPORTS_THE_PACKAGE`), or with `import company.local`
# (`shared_namespace.READS_LOCAL` and the others), and code of it that reads `company.local`
# through the package's name: a function, a static method, a class method, a function under a
# `functools.wraps` decorator, the wrapper of a decorator without it, a lambda, a function under
# `functools.cache`, a property's getter. The stage calls that code; `_BY_REFERENCE_IMPORTS` are the
# imports of its module.
_IMPORTS_THE_PACKAGE = "import company\n\n\ndef value():\n    return company.local.value()\n"
_STATIC_METHOD = """\
import company.local


class Tools:
    @staticmethod
    def value():
        return company.local.value()
"""
_CLASS_METHOD = """\
import company.local


class Tools:
    @classmethod
    def value(cls):
        return company.local.value()
"""
_UNDER_WRAPS = """\
import functools

import company.local


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
import company.local


def prefixed(function):
    def wrapper():
        return company.local.value() + function()

    return wrapper


@prefixed
def value():
    return ""
"""
_LAMBDA = "import company.local\n\nvalue = lambda: company.local.value()\n"
_UNDER_CACHE = """\
import functools

import company.local


@functools.cache
def value():
    return company.local.value()
"""
_PROPERTY = """\
import company.local


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
    "read()" + shared_namespace.LIBRARY_VALUES,
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
# namespace package the project shares with `company-client`
# (`shared_namespace.CLIENT_AND_SUB_FILES`), is pickled with the stage function.
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
    "company/client/__init__.py": shared_namespace.CLIENT + "\n\nREGISTRY = {}\n"
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
# The project's `company/local.py` of `shared_namespace.LOCAL_WITH_LOCK` with `name`; the stage calls
# `name`, which `from company.local import name` bound, and `helper.value()`
# (`shared_namespace.READS_LOCAL`, in a `PYTHONPATH` entry), which reads the module through the
# package's name.
_LOCAL_WITH_LOCK_AND_NAME = shared_namespace.LOCAL_WITH_LOCK + '\n\ndef name():\n    return "n"\n'
_NAME_PIPELINE = installed_dists.pipeline_text(
    "import helper\nfrom company.local import name", 'name() + ":" + helper.value()'
)
# Two stages of `Pipeline("p")`: after the lines `{imports}`, `first` returns `{first}`, and
# `second`, which reads what `first` writes, so that it runs after `first`, returns `{second}`. The
# module `tools3`, in a `PYTHONPATH` entry, imports `company.local` itself.
_TWO_STAGES = """\
{imports}

from pivot import Pipeline, stage


@stage
def first() -> str:
    return {first}


@stage
def second(ready: str) -> str:
    return {second}


with Pipeline("p") as pipeline:
    second(first())
"""
_IMPORTS_LOCAL_ITSELF = 'import company.local\n\n\ndef t():\n    return "t3"\n'
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


def _helper_two_stages_outcome(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    files: collections.abc.Mapping[str, str],
    extra_files: collections.abc.Mapping[str, str],
    stages: tuple[str, str, str],
) -> tuple[tuple[int, dict[str, str], tuple[str | None, ...]], str]:
    """What one `pivot repro --json` gives of the two stages of `_TWO_STAGES`, whose imports,
    `first` and `second` are `stages`, run by one worker (`core.max_workers 1`) in a project with
    `company-client` (`shared_namespace.project_with`, `files` in its root, `extra_files` in a
    `PYTHONPATH` entry): its exit code, each stage's status by name, and what `first` and `second`
    wrote (None for nothing); and the command's output."""
    imports, first, second = stages
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        files,
        extra_files,
        _TWO_STAGES.format(imports=imports, first=first, second=second),
    )
    installed_dists.pivot_ok(project, "config", "set", "core.max_workers", "1")

    ran = installed_dists.run_pivot(project, "repro", "--json")

    statuses = {
        event["stage"]: event["status"] for event in installed_dists.stage_complete_events(ran)
    }
    written = [project.root / "data" / "p" / f"{stage}.txt" for stage in ("first", "second")]
    found = tuple(path.read_text() if path.exists() else None for path in written)
    return (ran.returncode, statuses, found), installed_dists.shown(ran)


# Each case writes what a plain run writes, `shared_namespace.SHARED_OUTPUT`, then its edit.
@pytest.mark.parametrize(
    ("helper", "call"),
    [
        pytest.param(shared_namespace.READS_LOCAL, "helper.value()", id="function-importing-it"),
        pytest.param(_STATIC_METHOD, "helper.Tools.value()", id="staticmethod"),
        pytest.param(_CLASS_METHOD, "helper.Tools.value()", id="classmethod"),
        pytest.param(_UNDER_WRAPS, "helper.value()", id="wraps-decorator"),
        pytest.param(_UNDER_A_DECORATOR, "helper.value()", id="decorator-without-wraps"),
        pytest.param(_LAMBDA, "helper.value()", id="lambda"),
        pytest.param(_UNDER_CACHE, "helper.value()", id="functools-cache"),
        pytest.param(_PROPERTY, "helper.Tools().value", id="property"),
    ],
)
def test_a_submodule_that_cannot_be_pickled_is_found_where_its_reader_imports_it_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, helper: str, call: str
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, and that module holds a lock no stage reads and
    that cloudpickle cannot pickle, so the coordinator does not send it. `pipeline.py` imports it,
    and the stage calls code of the module `helper`, in a `PYTHONPATH` entry, which the worker
    imports by name and which imports `company.local` itself; only that code reads `company.local`
    through the package's name: a function, a static method, a class method, a function under a
    `functools.wraps` decorator, the wrapper of a decorator without it, a lambda, a function under
    `functools.cache`, or a property's getter. The stage's pickle holds the module `helper` by
    reference and nothing of `company.local`. `pivot repro` runs the stage, which writes
    `local-1:client-1:value-1`, refusing nothing: the worker's import of `helper` imports
    `company.local` from its file, which binds it on the package, as in a plain run. After one byte
    of the project's module changes, the stage is stale on that module's `func:` entry alone, and
    `pivot repro` writes the new value. Were the module sent with the stage, the worker's call could
    not be pickled, for its lock."""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {"company/local.py": shared_namespace.LOCAL_WITH_LOCK},
        {"helper.py": helper},
        installed_dists.pipeline_text(
            _BY_REFERENCE_IMPORTS, call + shared_namespace.LIBRARY_VALUES
        ),
    )

    shared_namespace.assert_runs_then_reruns_on_an_edit(
        project, project.root / "company" / "local.py", "func:company.local.value"
    )


def test_a_submodule_that_cannot_be_pickled_is_missing_where_its_reader_imports_the_package_alone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stated limit, pivot 0.10.4's outcome. `company-client` is installed into site-packages from
    its path. The project's `company/local.py`, in its root, holds a lock no stage reads and that
    cloudpickle cannot pickle, so the coordinator does not send it. `pipeline.py` imports it, and
    the stage calls `helper.value()` of the module `helper`, in a `PYTHONPATH` entry, which the
    worker imports by name, which imports `company` alone and reads `company.local` through the
    package's name. Nothing in the worker binds the module there: `pivot repro` exits 1, the stage
    failing in its worker with `AttributeError: module 'company' has no attribute 'local'`, where a
    plain `python` run writes `local-1:client-1:value-1`."""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {"company/local.py": shared_namespace.LOCAL_WITH_LOCK},
        {"helper.py": _IMPORTS_THE_PACKAGE},
        installed_dists.pipeline_text(
            _BY_REFERENCE_IMPORTS, "helper.value()" + shared_namespace.LIBRARY_VALUES
        ),
    )

    found, shown = shared_namespace.repro_outcome(project)

    assert found == (1, "failed", shared_namespace.MISSING), shown


def test_a_lambda_of_a_module_sent_by_reference_that_reads_the_projects_part_gets_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root. The module `helper`, in a `PYTHONPATH` entry,
    binds a lambda as `read` that reads `company.local` through the package's name, and the stage
    calls that lambda (`from helper import read`). cloudpickle pickles the lambda by value, as it
    pickles every lambda, so no import of `helper` runs in the worker for it, and with it the
    module it reads through the package, by value, a module no package holds; the coordinator sends
    that module with the stage, the same object, and the worker binds it on the package. `pivot
    repro` runs the stage, which writes `local-1:client-1:value-1`, as a plain run does, refusing
    nothing. After one byte of the project's module changes, the stage is stale on that module's
    `func:` entry alone, and `pivot repro` writes the new value. Were the module not bound on the
    package, the stage would fail in the worker with
    `AttributeError: module 'company' has no attribute 'local'`."""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {"company/local.py": shared_namespace.LOCAL},
        {"helper.py": _LAMBDA_HELPER},
        _LAMBDA_HELPER_PIPELINE,
    )

    shared_namespace.assert_runs_then_reruns_on_an_edit(
        project, project.root / "company" / "local.py", "func:company.local.value"
    )


# Each case writes what a plain run writes, `fast`.
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
    `PYTHONPATH` entry, and `pipeline.py` imports and calls it. Its module binds `Mode` (`from
    company.local import Mode`), and the stage passes `Mode.FAST` to `company.local.run`, read
    through the package's name (`own-code`), also after asserting that the module it reads so holds
    the stage's `Mode` (`one-module`), or to `tools.run` of the module `tools` in the same entry,
    which reads `company.local` so and imports `company` alone (`through-a-helper`). cloudpickle
    pickles the stage function by value, and with it `Mode`, by value, as discovery pickles the
    project's part by value; the coordinator sends `company.local` with the stage, in the same
    pickle, and the worker binds it on the package. `pivot repro` runs the stage, which writes
    `fast`, refusing nothing, as a plain `python` run of `stages` does: the stage's `Mode` is the
    sent module's class. Were the stage's `Mode` a copy apart from the module the code reads
    through the package, `run` would compare the stage's `Mode.FAST` with another copy of it and
    the stage would write `slow`, or fail on its assertion."""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {"company/local.py": _MODES},
        {"stages.py": _STAGES.format(body=body), "tools.py": _TOOLS},
        _STAGES_PIPELINE,
    )

    installed_dists.assert_runs_with_no_refusal(project, "fast")


# Each case writes what a plain run writes, `output`.
@pytest.mark.parametrize(
    ("client", "files", "extra_files", "pipeline_text", "output"),
    [
        pytest.param(
            shared_namespace.CLIENT_AND_SUB_FILES,
            {"company/sub/local.py": _MODES},
            {},
            _SUB_MODES_PIPELINE,
            "fast",
            id="package-imported-by-name",
        ),
        pytest.param(
            shared_namespace.CLIENT_FILES,
            {"company/local.py": _MODES, "company/other.py": _READS_LOCAL_AT_IMPORT},
            {"tools.py": _READS_OTHER},
            _READS_OTHER_PIPELINE,
            "fast:fast",
            id="read-by-another-sent-module",
        ),
        pytest.param(
            shared_namespace.CLIENT_FILES,
            {"company/local.py": _MODES, "company/other.py": _IMPORTS_LOCAL},
            {"tools.py": _READS_OTHER},
            _READS_OTHER_PIPELINE,
            "g:fast",
            id="imported-by-another-sent-module",
        ),
        pytest.param(
            shared_namespace.CLIENT_FILES,
            {"company/local.py": _MODES, "company/other.py": _OTHER_HOLDS_MODE},
            {"tools.py": _READS_OTHER},
            _READS_OTHER_PIPELINE,
            "fast:fast",
            id="another-sent-module-holds-the-class",
        ),
        pytest.param(
            shared_namespace.CLIENT_FILES,
            {"company/local.py": _MODES},
            {"tools2.py": _HOLDS_MODE},
            _HOLDS_MODE_PIPELINE,
            "fast:fast",
            id="a-helper-imported-by-name-holds-the-class",
        ),
        pytest.param(
            shared_namespace.CLIENT_FILES,
            {"company/local.py": _MODES_AND_KIND, "company/other.py": _KINDS},
            {"tools.py": _CHECKS_KIND},
            _CHECKS_PIPELINE,
            "fast:a",
            id="a-class-another-part-holds",
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
    discovery pickles by value. The stage holds `Mode` (`from company.local import Mode`), and other
    code reads the module: the stage reads `company/sub/local.py` through `company` and
    `company.sub` inside a lambda, so that its pickle holds nothing of `company.sub`, which
    cloudpickle pickles by reference, so the coordinator sends its name, the worker imports it
    before the stage runs and then binds the sent `company.sub.local` on it
    (`package-imported-by-name`); or it reads `company/local.py` through the package's name beside
    `company/other.py`, which only `tools.g`, in a `PYTHONPATH` entry, reads through the package,
    so the coordinator sends both, and which reads `company.local` through the package's name as it
    is imported (`read-by-another-sent-module`), imports it itself
    (`imported-by-another-sent-module`), or binds
    `Mode` itself and passes it to `company.local.run` (`another-sent-module-holds-the-class`); or
    it writes `tools2.h()` and its own `Mode.FAST.value`, `tools2`, in a `PYTHONPATH` entry, binding
    `Mode` itself and passing it to `company.local.run`, so the worker's import of `tools2` imports
    `company.local` from its file, which the worker then leaves bound
    (`a-helper-imported-by-name-holds-the-class`); or `company/local.py` holds `Kind` of
    `company/other.py` (`from company.other import Kind`), and `tools.check`, in a `PYTHONPATH`
    entry, reads both modules through the package's name and checks that `company.local.kind()` is
    `company.other.Kind.A` (`a-class-another-part-holds`). `pivot repro` runs the stage, which
    writes what a plain `python` run of `pipeline.py` writes (`fast`, `fast:fast`, `g:fast`,
    `fast:fast`, `fast:fast`, `fast:a`), refusing nothing: the stage's pickle and the modules sent
    with it carry one copy of each by-value module and class, and the worker leaves bound what an
    import there bound. Were the sent module bound over the one `tools2`'s import bound, `run` of
    the sent copy would compare `tools2`'s `Mode.FAST` with its own (`slow:fast`); were
    `company/other.py` sent apart from the stage's pickle, `kind()` would return another copy's
    member (`not-a`)."""
    project = shared_namespace.project_with(
        tmp_path, monkeypatch, client, files, extra_files, pipeline_text
    )

    found, shown = shared_namespace.repro_outcome(project)

    assert found == (0, "ran", output), shown


def test_the_import_time_code_of_a_module_sent_by_value_does_not_run_in_the_worker(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stated limit, which pivot 0.10.4 has too: a module pickled by value runs no import-time
    code in the worker. `company-client`, whose `company.client` holds a registry, `REGISTRY`, is
    installed into site-packages from its path. The project's `company/other.py` registers its
    `Kind` there as it is imported, and `company/local.py`, which holds `Mode` and `run`
    (`_MODES`), imports it. The stage holds `Mode` and calls `tools.check`, in a `PYTHONPATH`
    entry, which reads both modules through the package's name, so the coordinator sends both:
    what `run` returns, and whether the registry holds the `Kind` that `company.other` holds.
    `pivot repro` runs the stage, which writes `fast:missing`, where a plain `python` run writes
    `fast:registered`: discovery pickles `company/other.py` by value, and its import-time code
    does not run in the worker, whose `company.client`, imported by name, holds an empty registry.
    Pivot 0.10.4 gives another outcome here: it binds neither module on the package, so the stage
    fails before it reads the registry, with
    `AttributeError: module 'company' has no attribute 'local'`."""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        _CLIENT_WITH_REGISTRY,
        {"company/local.py": _MODES_IMPORTING_KIND, "company/other.py": _REGISTERS_KIND},
        {"tools.py": _CHECKS_REGISTRY},
        _CHECKS_PIPELINE,
    )

    found, shown = shared_namespace.repro_outcome(project)

    assert found == (0, "ran", "fast:missing"), shown


def test_a_submodule_whose_function_the_stage_holds_and_that_cannot_be_pickled_is_imported_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path. The project's
    `company/local.py`, in its root, holds a lock no stage reads and that cloudpickle cannot
    pickle, `value` and `name`. The stage calls `name`, which `from company.local import name`
    bound in its module, and `helper.value()` of the module `helper`, in a `PYTHONPATH` entry,
    which imports `company.local` and reads it through the package's name. cloudpickle pickles
    `name` by value, with none of its module's other values, and the coordinator does not send the
    module, which does not pickle alone; the worker's import of `helper` imports it from its file.
    `pivot repro` runs the stage, which writes `n:local-1`, refusing nothing, as a plain `python`
    run does. Were the module sent with the stage, the stage would fail as the pickling of its
    worker's call fails: `Could not pickle the task to send it to the workers.`"""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {"company/local.py": _LOCAL_WITH_LOCK_AND_NAME},
        {"helper.py": shared_namespace.READS_LOCAL},
        _NAME_PIPELINE,
    )

    found, shown = shared_namespace.repro_outcome(project)

    assert found == (0, "ran", "n:local-1"), shown


@pytest.mark.parametrize(
    "second",
    [
        pytest.param('raise RuntimeError("a second import")', id="raises"),
        pytest.param("raise SystemExit(3)", id="exits"),
        pytest.param("raise KeyboardInterrupt", id="is-interrupted"),
    ],
)
def test_the_projects_part_runs_where_a_second_import_of_it_would_fail(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, second: str
) -> None:
    """`company-client` is installed into site-packages from its path. The project holds its part
    of `company`, `company/local.py`, in its root, which raises `RuntimeError` (`raises`),
    `SystemExit(3)` (`exits`) or `KeyboardInterrupt` (`is-interrupted`) when a second import in the
    project runs it, and the stage reads it through the package's name. `pivot repro` runs the
    stage, which writes `local-1:client-1:value-1`, refusing nothing, as a plain `python` run,
    which imports it once, does: discovery pickles the module by value, and the coordinator sends
    it with the stage, so nothing imports it in the worker. Were it pickled by reference, the
    worker would import it as it unpickles the stage, since cloudpickle carries with the stage
    function each submodule its own code names, and the stage would fail with loky's `A task has
    failed to un-serialize`."""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {"company/local.py": shared_namespace.SECOND_IMPORT.format(second=second)},
        {},
        shared_namespace.TOP_NAME_PIPELINE,
    )

    found, shown = shared_namespace.repro_outcome(project)

    assert found == (0, "ran", shared_namespace.SHARED_OUTPUT), shown


def test_a_stage_whose_own_code_reads_a_part_that_cannot_be_pickled_fails_as_its_pickling_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stated limit, pivot 0.10.4's outcome. `company-client` is installed into site-packages from
    its path. The project holds its part of `company`, `company/local.py`, in its root, which holds
    a lock that cloudpickle cannot pickle, and the stage reads it through the package's name
    (`company.local.value()`). cloudpickle pickles the stage function by value, and with it the
    module, which it pickles by value, as each submodule of a package the function holds that its
    own code names. `pivot repro` exits 1, and the stage fails as the pickling of its worker's call
    fails, with loky's `Could not pickle the task to send it to the workers.`, where a plain
    `python` run writes `local-1:client-1:value-1`."""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {"company/local.py": shared_namespace.LOCAL_WITH_LOCK},
        {},
        shared_namespace.TOP_NAME_PIPELINE,
    )

    found, shown = shared_namespace.repro_outcome(project)

    assert found == (1, "failed", "Could not pickle the task to send it to the workers."), shown


# Each case writes what a plain run of both stages in one process writes, `outputs`.
@pytest.mark.parametrize(
    ("files", "extra_files", "stages", "outputs"),
    [
        pytest.param(
            {"company/local.py": shared_namespace.LOCAL},
            {},
            (
                "import company.local\nfrom company import local",
                "company.local.value()",
                '"one" if company.local is local else "two"',
            ),
            ("local-1", "one"),
            id="each-stage-its-sent-module",
        ),
        pytest.param(
            {"company/local.py": _MODES, "company/other.py": _IMPORTS_LOCAL},
            {"tools.py": _READS_OTHER, "tools2.py": _HOLDS_MODE},
            (
                "import company.local\nimport company.other\nimport tools\nimport tools2\n"
                + "from company.local import Mode",
                'tools.g() + ":" + company.local.run(Mode.FAST)',
                "tools2.h()",
            ),
            ("g:fast", "fast"),
            id="a-later-import-by-name",
        ),
        pytest.param(
            {"company/local.py": _MODES},
            {"tools2.py": _HOLDS_MODE},
            (
                "import company.local\nimport tools2",
                '__import__("company.local").local.run(company.local.Mode.FAST)',
                "tools2.h()",
            ),
            ("fast", "fast"),
            id="an-import-in-the-first-stage-rebinds-it",
        ),
    ],
)
def test_a_later_stage_in_the_same_worker_reads_the_one_module_a_plain_run_holds(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    extra_files: dict[str, str],
    stages: tuple[str, str, str],
    outputs: tuple[str, str],
) -> None:
    """`company-client` is installed into site-packages from its path, and the project holds its
    part of `company`, `company/local.py`, in its root. One worker (`core.max_workers 1`) runs the
    stage `first`, then `second`, which reads what `first` writes, each reading `company.local`
    through the package's name, so the coordinator sends the module with each, two copies. `first`
    returns `company.local.value()`, and `second` returns `one` where the module it reads through
    the package is its own (`from company import local`), the copy sent with it
    (`each-stage-its-sent-module`); or `first` holds `Mode` (`_MODES`) and calls `tools.g`, which
    reads `company/other.py` through the package, which imports `company.local` itself, and
    `second` returns `tools2.h()`, `tools2`, in a `PYTHONPATH` entry, binding `Mode` itself and
    passing it to `company.local.run`, so the worker's import of `tools2` imports `company.local`
    from its file (`a-later-import-by-name`); or `first` imports `company.local` by name itself
    (`__import__`), which loads it from its file in the worker and binds that module on the
    package over the sent one, and `second` returns `tools2.h()`, whose import then finds that
    module in `sys.modules` and binds nothing (`an-import-in-the-first-stage-rebinds-it`).
    `pivot repro` runs both stages, which write what a plain `python` run of their functions in one
    process writes (`local-1` then `one`, `g:fast` then `fast`, `fast` then `fast`): the worker
    removes after `first` the binding it made for it, and only that one. Were the copy sent with
    `first` left bound on the package, `second` would read it where its own sent copy is not bound
    (`two`); were the module the import in `first` bound removed too, `second` would bind its own
    sent copy, whose `run` compares the `Mode` of the module `sys.modules` holds (`slow`)."""
    found, shown = _helper_two_stages_outcome(tmp_path, monkeypatch, files, extra_files, stages)

    assert found == (0, {"p/first": "ran", "p/second": "ran"}, outputs), shown


def test_code_pickled_by_value_and_code_imported_by_name_hold_two_copies_of_the_projects_part(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stated limit, pivot 0.10.4's outcome. `company-client` is installed into site-packages from
    its path, and the project holds `Mode` and `run` (`_MODES`) in its part of `company`,
    `company/local.py`. One worker (`core.max_workers 1`) runs the stage `first`, which holds `Mode`,
    pickled by value, calls `tools3.t()` of the module `tools3`, in a `PYTHONPATH` entry, which the
    worker imports by name and which imports `company.local` itself, from its file, and returns
    that with `company.local.run(Mode.FAST)`, read through the package's name; then `second`, which
    reads what `first` writes and returns `tools2.h()`, `tools2`, in a `PYTHONPATH` entry, binding
    `Mode` itself and passing it to `company.local.run`. `pivot repro` runs both stages: `first`
    writes `t3:slow`, where a plain `python` run writes `t3:fast`, as `run` of the module
    `tools3`'s import bound compares the stage's own copy of `Mode.FAST`; `second` writes `fast`,
    as a plain run does, as `tools2` holds the `Mode` of that same module."""
    found, shown = _helper_two_stages_outcome(
        tmp_path,
        monkeypatch,
        {"company/local.py": _MODES},
        {"tools3.py": _IMPORTS_LOCAL_ITSELF, "tools2.py": _HOLDS_MODE},
        (
            "import company.local\nimport tools2\nimport tools3\nfrom company.local import Mode",
            'tools3.t() + ":" + company.local.run(Mode.FAST)',
            "tools2.h()",
        ),
    )

    assert found == (0, {"p/first": "ran", "p/second": "ran"}, ("t3:slow", "fast")), shown


def test_an_edit_of_the_projects_part_after_discovery_runs_the_loaded_code_and_reruns_the_stage(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`company-client` is installed into site-packages from its path, and the project holds its
    part of `company`, `company/local.py`, in its root. The stage `edit` changes one byte of that
    file after discovery loaded it (`local-1` to `local-2`), and the stage `make` then reads
    `company.local` through the package's name. `pivot repro` runs `make`, which writes
    `local-1:client-1:value-1`, the code discovery loaded, as a plain `python` run of both stages
    in one process writes, with no warning and no mark in its lock; `pivot status` then reports
    `make` stale on `func:company.local.value` alone, as the next discovery loads the edited file,
    and `pivot repro` writes `local-2:client-1:value-1`. Were the module imported from its file in
    the worker, the first run would write `local-2` and report the edit; were the fingerprint taken
    from the file as it is when `make` starts, the next run would report `make` cached with the
    output of the code before the edit."""
    local = tmp_path / "project" / "company" / "local.py"
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        {"company/local.py": shared_namespace.LOCAL},
        {},
        _EDIT_THEN_READ.format(local=str(local), library_values=shared_namespace.LIBRARY_VALUES),
    )

    ran = installed_dists.run_pivot(project, "repro")

    first = installed_dists.shown(ran)
    written = (project.root / installed_dists.OUTPUT).read_text()
    assert (ran.returncode, written) == (0, shared_namespace.SHARED_OUTPUT), first
    manifest = installed_dists.lock_manifest(project, installed_dists.STAGE)
    assert [key for key in manifest if key.startswith("edited-source:")] == [], manifest
    installed_dists.assert_stale_on(project, installed_dists.STAGE, "func:company.local.value")
    installed_dists.assert_runs_and_writes(
        project, shared_namespace.SHARED_OUTPUT.replace("local-1", "local-2")
    )


@pytest.mark.parametrize(
    ("imports", "expression", "files", "extra_files", "outcome"),
    [
        # A plain run's output.
        pytest.param(
            "import company.client\nimport company.local\nimport fixlib",
            "company.local.value()" + shared_namespace.LIBRARY_VALUES,
            {"company/local.py": shared_namespace.LOCAL},
            {},
            (0, "ran", shared_namespace.SHARED_OUTPUT),
            id="own-code",
        ),
        # A plain run's output.
        pytest.param(
            _BY_REFERENCE_IMPORTS,
            "helper.value()" + shared_namespace.LIBRARY_VALUES,
            {"company/local.py": shared_namespace.LOCAL},
            {"helper.py": shared_namespace.READS_LOCAL},
            (0, "ran", shared_namespace.SHARED_OUTPUT),
            id="helper-importing-it-by-name",
        ),
        # Pivot 0.10.4's outcome, a stated limit.
        pytest.param(
            "import company.client\nimport fixlib\nimport helper",
            "helper.value()" + shared_namespace.LIBRARY_VALUES,
            {"helper.py": shared_namespace.READS_LOCAL, "company/local.py": shared_namespace.LOCAL},
            {},
            (1, "failed", shared_namespace.MISSING),
            id="limit-helper-under-the-root",
        ),
        # Pivot 0.10.4's outcome, a stated limit.
        pytest.param(
            _BY_REFERENCE_IMPORTS,
            "helper.value()" + shared_namespace.LIBRARY_VALUES,
            {"company/local.py": shared_namespace.LOCAL},
            {"helper.py": _IMPORTS_THE_PACKAGE},
            (1, "failed", shared_namespace.MISSING),
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
    (`fingerprint.submodule_entries`), which the coordinator sends with the stage. The scan reads
    the stage function's own code, not a helper module it calls: where the stage calls
    `helper.value()` of the module `helper`, in a `PYTHONPATH` entry, which the worker imports by
    name and which imports `company.local` itself and reads it through the package's name, the
    stage writes the same (`helper-importing-it-by-name`), as that import binds it. Two cases pin
    the stated limit (`docs/architecture/fingerprinting.md`, "Imports a Stage Makes While It Runs")
    where nothing in the worker binds the module, which `pipeline.py` imports: the project's module
    `helper`, in its root, which discovery pickles by value, imports `company.local` and reads it
    so (`limit-helper-under-the-root`), or `helper`, in a `PYTHONPATH` entry, imports the package
    alone (`import company`, `limit-helper-importing-the-package-alone`): `pivot repro` exits 1,
    the stage failing in its worker with
    `AttributeError: module 'company' has no attribute 'local'`, where a plain `python` run writes
    `local-1:client-1:value-1`."""
    project = shared_namespace.project_with(
        tmp_path,
        monkeypatch,
        shared_namespace.CLIENT_FILES,
        files,
        extra_files,
        installed_dists.pipeline_text(
            imports + "\nfrom pivot import no_fingerprint", expression, "@no_fingerprint()"
        ),
    )

    found, shown = shared_namespace.repro_outcome(project)

    assert found == outcome, shown
