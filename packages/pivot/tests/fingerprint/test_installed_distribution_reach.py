"""The `dist:` key of a library a stage reaches without reading the library's module, or a
callable of it that names its module, as a global: one parametrised case per form of that reach,
checked through the `pivot` CLI on git installs.

Each case makes its own venv with uv over this test environment (`tests.installed_dists`) and
installs one library from commit C1 of its git repository (`git+file://...@<commit>`, so its
`direct_url.json` has `vcs_info`). Its own pivot project has one stage, `p/make`, that reaches the
library through one form alone: the base class of a user class, the type hint of a user function,
an instance inside a value the stage reads, an input's pivot loader, the stage function itself,
or a function whose `__module__` is None. `pivot repro` runs the stage and `pivot status --json`
reports it `cached`. The library is then reinstalled from commit C2, which changes one byte of the
reached module: a literal in a module-level helper that the reached code calls, or, for the stage
function, a comment. `pivot status --json` reports the stage `stale`, `pivot status --explain
--json` names exactly the library's `dist:` key changed, and `pivot repro` runs the stage again.

The file runs pivot only through its CLI and imports nothing but `pathlib`, `pytest` and
`tests.installed_dists`.
"""

import pathlib

import pytest
from tests import installed_dists

# The base class form: the stage builds `Greeter`, a user class whose base is the library's
# `Base`, and calls the method `Greeter` inherits from it; C2 changes the literal of `_greeting`,
# the module-level helper that method calls.
_BASE_LIBRARY = """\
def _greeting():
    return "base-1"


class Base:
    def greet(self):
        return _greeting()
"""
_BASE_PIPELINE = """\
import reachbase

from pivot import Pipeline, stage


class Greeter(reachbase.Base):
    pass


@stage
def make() -> str:
    return Greeter().greet()


with Pipeline("p") as pipeline:
    make()
"""

# The type hint form: the stage calls `describe`, a user function whose parameter hint is the
# library's `Label`, and nothing else of the library: it builds that argument with the class it
# reads back from the hint (`typing.get_type_hints`), as a framework that reads hints does. C2
# changes the literal of `_prefix`, the module-level helper `Label` calls.
_HINT_LIBRARY = """\
def _prefix():
    return "hint-1"


class Label:
    def __init__(self, text):
        self.text = _prefix() + ":" + text
"""
_HINT_PIPELINE = """\
import typing

import reachhint

from pivot import Pipeline, stage


def describe(label: reachhint.Label) -> str:
    return label.text


@stage
def make() -> str:
    return describe(typing.get_type_hints(describe)["label"]("x"))


with Pipeline("p") as pipeline:
    make()
"""

# The value form: the stage reads `RENDER`, a partial of a user function whose keyword argument is
# an instance of the library's `Marker`, on which the user function calls `apply`; C2 changes the
# literal of `_mark`, the module-level helper `apply` calls.
_VALUE_LIBRARY = """\
def _mark():
    return "value-1"


class Marker:
    def apply(self, text):
        return _mark() + ":" + text
"""
_VALUE_PIPELINE = """\
import functools

import reachvalue

from pivot import Pipeline, stage


def render(text, marker):
    return marker.apply(text)


RENDER = functools.partial(render, marker=reachvalue.Marker())


@stage
def make() -> str:
    return RENDER("x")


with Pipeline("p") as pipeline:
    make()
"""

# The loader form: the stage's input is read by the library's pivot loader `Tagged`; C2 changes the
# literal of `_tag`, the module-level helper its `load` calls, which leaves the syntax tree of
# `load` as it was.
_LOADER_LIBRARY = """\
import dataclasses
import pathlib

from pivot import loaders


def _tag(text):
    return "loader-1:" + text


@dataclasses.dataclass(frozen=True)
class Tagged(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return _tag(path.read_text())
"""
_LOADER_PIPELINE = """\
import reachloader

from pivot import Pipeline, stage


@stage
def make(text: str) -> str:
    return text


with Pipeline("p") as pipeline:
    make(pipeline.input("words.txt", format=reachloader.Tagged()))
"""
_LOADER_INPUTS = {"data/raw/words.txt": "x"}

# The stage function form: the stage is the library's own `make`, which reads no global, calls
# nothing and whose only hint is `str`. C2 changes one byte of a comment in its module, which
# leaves the syntax tree of `make` as it was, so what the stage writes cannot change: it writes the
# same value when it runs again.
_STAGE_LIBRARY = """\
from pivot import stage

# Revision 1 of this module.


@stage
def make() -> str:
    return "made"
"""
_STAGE_PIPELINE = """\
from reachstage import make

from pivot import Pipeline

with Pipeline("p") as pipeline:
    make()
"""

# The form of a callable whose `__module__` is not a string: the stage reads, as a global, the
# library's module-level function `value`, whose `__module__` the library sets to None, so no
# `__module__` names the module that binds it; C2 changes the literal of `_suffix`, the
# module-level helper `value` calls.
_NO_MODULE_LIBRARY = """\
def _suffix():
    return "nomod-1"


def value():
    return "x:" + _suffix()


value.__module__ = None
"""
_NO_MODULE_PIPELINE = """\
from reachnomod import value

from pivot import Pipeline, stage


@stage
def make() -> str:
    return value()


with Pipeline("p") as pipeline:
    make()
"""


@pytest.mark.parametrize(
    ("name", "library", "edit", "pipeline", "outputs", "inputs"),
    [
        pytest.param(
            "reachbase",
            _BASE_LIBRARY,
            ('"base-1"', '"base-2"'),
            _BASE_PIPELINE,
            ("base-1", "base-2"),
            dict[str, str](),
            id="base-class",
        ),
        pytest.param(
            "reachhint",
            _HINT_LIBRARY,
            ('"hint-1"', '"hint-2"'),
            _HINT_PIPELINE,
            ("hint-1:x", "hint-2:x"),
            dict[str, str](),
            id="type-hint",
        ),
        pytest.param(
            "reachvalue",
            _VALUE_LIBRARY,
            ('"value-1"', '"value-2"'),
            _VALUE_PIPELINE,
            ("value-1:x", "value-2:x"),
            dict[str, str](),
            id="instance-in-a-value",
        ),
        pytest.param(
            "reachloader",
            _LOADER_LIBRARY,
            ('"loader-1:"', '"loader-2:"'),
            _LOADER_PIPELINE,
            ("loader-1:x", "loader-2:x"),
            _LOADER_INPUTS,
            id="pivot-loader",
        ),
        pytest.param(
            "reachstage",
            _STAGE_LIBRARY,
            ("# Revision 1", "# Revision 2"),
            _STAGE_PIPELINE,
            ("made", "made"),
            dict[str, str](),
            id="library-stage-function",
        ),
        pytest.param(
            "reachnomod",
            _NO_MODULE_LIBRARY,
            ('"nomod-1"', '"nomod-2"'),
            _NO_MODULE_PIPELINE,
            ("x:nomod-1", "x:nomod-2"),
            dict[str, str](),
            id="function-whose-module-is-none",
        ),
    ],
)
def test_a_library_reached_through_one_form_reruns_the_stage_after_a_one_byte_reinstall(
    tmp_path: pathlib.Path,
    name: str,
    library: str,
    edit: tuple[str, str],
    pipeline: str,
    outputs: tuple[str, str],
    inputs: dict[str, str],
) -> None:
    """Distribution `name`, whose package `name` has `library` as its `__init__.py`, installed from
    commit C1 of its git repository, and C2 replacing `edit[0]` with `edit[1]` in that file. In a
    project of `pipeline` and the files `inputs` (each path relative to the project's root, with
    its text), `pivot repro` runs `p/make`, which writes `outputs[0]`, and `pivot status` reports
    it cached. After the reinstall from C2, the stage is stale on `dist:<name>` alone, and `pivot
    repro` runs it again, which writes `outputs[1]`."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(tmp_path / "src", name, name, library)
    project = installed_dists.make_project_in(tmp_path, venv, pipeline)
    for relative, text in inputs.items():
        path = project.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    installed_dists.install(venv, installed_dists.init_git_requirement(source, project.home, name))
    installed_dists.assert_ran_then_cached(project, outputs[0])
    installed_dists.replace_once(source / name / "__init__.py", *edit)
    commit = installed_dists.commit_all(source, project.home, "C2")

    installed_dists.reinstall_from_commit(venv, name, source, commit, "C2")

    module = (venv.site_packages / name / "__init__.py").read_text()
    assert module == library.replace(*edit), f"the reinstall must change {edit}: {module!r}"
    installed_dists.assert_stale_on(project, installed_dists.STAGE, f"dist:{name}")
    installed_dists.assert_runs_and_writes(project, outputs[1], "the second `pivot repro`")
