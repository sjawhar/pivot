# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false, reportUnusedCallResult=false, reportAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportReturnType=false, reportUnusedImport=false, reportInvalidTypeForm=false, reportImplicitStringConcatenation=false, reportUnnecessaryTypeIgnoreComment=false, reportArgumentType=false

"""Tests for @fingerprint_by_name: a helper recorded by name only."""

import ast
import contextlib
import functools
import importlib
import inspect
import json
import linecache
import os
import pathlib
import shutil
import subprocess
import sys
import types
import typing

import cloudpickle
import pytest
import yaml

from pivot import (  # noqa: F401
    decorators,
    discovery,
    exceptions,
    fingerprint,
    fingerprint_by_name,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def module_dir(tmp_path: pathlib.Path):
    """Create temp directory for test modules and add to sys.path."""
    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))
    to_remove = [
        name
        for name in sys.modules
        if name.startswith(
            ("fbn_", "fbn2_", "fbnhelp_", "fbnstage_", "rk_", "rk4_", "rk5_", "rk6_")
        )
    ]
    for name in to_remove:
        del sys.modules[name]


def _import_fresh(name: str, *, parent_dir: pathlib.Path | None = None) -> types.ModuleType:
    """Import module fresh, clearing any cached version."""
    if name in sys.modules:
        del sys.modules[name]
    linecache.clearcache()
    importlib.invalidate_caches()
    # Clear __pycache__ so a rewritten file is not served from stale bytecode
    if parent_dir is not None:
        import shutil

        pycache = parent_dir / "__pycache__"
        if pycache.is_dir():
            shutil.rmtree(pycache)
        importlib.invalidate_caches()
    return importlib.import_module(name)


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _fresh(*names: str) -> types.ModuleType:
    """Import `names` again from their files, in order, and return the last."""
    for name in names:
        sys.modules.pop(name, None)
    linecache.clearcache()
    importlib.invalidate_caches()
    return [importlib.import_module(name) for name in names][-1]


# ---------------------------------------------------------------------------
# (a) Marked helper body change → cached
# ---------------------------------------------------------------------------


def test_marked_helper_body_change_is_cached(module_dir: pathlib.Path) -> None:
    """A @fingerprint_by_name helper's body change does not change the stage fingerprint."""
    _write(
        module_dir / "fbnhelp_a.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def process_start():\n"
        "    return 'v1'\n",
    )
    _write(
        module_dir / "fbnstage_a.py",
        "from fbnhelp_a import process_start\n\ndef my_stage():\n    return process_start()\n",
    )
    _import_fresh("fbnhelp_a")
    mod = _import_fresh("fbnstage_a")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(
        module_dir / "fbnhelp_a.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def process_start():\n"
        "    return 'v2-different-body'\n",
    )
    _import_fresh("fbnhelp_a")
    mod = _import_fresh("fbnstage_a")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 == fp2, "Body change of marked helper must not change fingerprint"


# ---------------------------------------------------------------------------
# (b) Rename, move, unmark → rerun
# ---------------------------------------------------------------------------


def test_marked_helper_rename_reruns(module_dir: pathlib.Path) -> None:
    """Renaming a @fingerprint_by_name helper changes the stage fingerprint."""
    _write(
        module_dir / "fbnhelp_b.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def process_start():\n"
        "    return 1\n",
    )
    _write(
        module_dir / "fbnstage_b.py",
        "from fbnhelp_b import process_start\n\ndef my_stage():\n    return process_start()\n",
    )
    _import_fresh("fbnhelp_b")
    mod = _import_fresh("fbnstage_b")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(
        module_dir / "fbnhelp_b.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def process_start_v2():\n"
        "    return 1\n",
    )
    _write(
        module_dir / "fbnstage_b.py",
        "from fbnhelp_b import process_start_v2 as process_start\n\n"
        "def my_stage():\n"
        "    return process_start()\n",
    )
    _import_fresh("fbnhelp_b")
    mod = _import_fresh("fbnstage_b")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 != fp2, "Renaming a marked helper must change fingerprint"


def test_marked_helper_move_to_another_module_reruns(
    module_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving a @fingerprint_by_name helper to another module changes the fingerprint.

    The `ext:` entry holds the module and the qualified name, so the module is part of it.
    """
    monkeypatch.setattr("pivot.project._project_root_cache", module_dir)
    _write(
        module_dir / "fbn_mod_a.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def helper():\n"
        "    return 1\n",
    )
    _write(
        module_dir / "fbnstage_mv.py",
        "from fbn_mod_a import helper\n\ndef my_stage():\n    return helper()\n",
    )
    _import_fresh("fbn_mod_a", parent_dir=module_dir)
    mod = _import_fresh("fbnstage_mv", parent_dir=module_dir)
    with fingerprint.fingerprint_pass():
        fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    sys.modules.pop("fbn_mod_a", None)
    sys.modules.pop("fbnstage_mv", None)
    _write(
        module_dir / "fbn_mod_b.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def helper():\n"
        "    return 1\n",
    )
    _write(
        module_dir / "fbnstage_mv.py",
        "from fbn_mod_b import helper\n\ndef my_stage():\n    return helper()\n",
    )
    _import_fresh("fbn_mod_b", parent_dir=module_dir)
    mod = _import_fresh("fbnstage_mv", parent_dir=module_dir)
    with fingerprint.fingerprint_pass():
        fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 != fp2, "Moving a marked helper to another module must change fingerprint"


def test_unmarking_reruns(module_dir: pathlib.Path) -> None:
    """Removing @fingerprint_by_name reruns (code hash now included)."""
    _write(
        module_dir / "fbnhelp_b2.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def helper():\n"
        "    return 42\n",
    )
    _write(
        module_dir / "fbnstage_b2.py",
        "from fbnhelp_b2 import helper\n\ndef my_stage():\n    return helper()\n",
    )
    _import_fresh("fbnhelp_b2")
    mod = _import_fresh("fbnstage_b2")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(module_dir / "fbnhelp_b2.py", "def helper():\n    return 42\n")
    _import_fresh("fbnhelp_b2")
    mod = _import_fresh("fbnstage_b2")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 != fp2, "Removing mark must change fingerprint"


# ---------------------------------------------------------------------------
# (c) Transitive helper change → cached
# ---------------------------------------------------------------------------


def test_transitive_helper_change_stays_cached(module_dir: pathlib.Path) -> None:
    """Code reachable only through a @fingerprint_by_name helper is not walked."""
    _write(module_dir / "fbn_inner_c.py", "def decode():\n    return 'v1'\n")
    _write(
        module_dir / "fbnhelp_c.py",
        "from pivot import fingerprint_by_name\n"
        "from fbn_inner_c import decode\n\n"
        "@fingerprint_by_name\n"
        "def process_start():\n"
        "    return decode()\n",
    )
    _write(
        module_dir / "fbnstage_c.py",
        "from fbnhelp_c import process_start\n\ndef my_stage():\n    return process_start()\n",
    )
    _import_fresh("fbn_inner_c")
    _import_fresh("fbnhelp_c")
    mod = _import_fresh("fbnstage_c")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(module_dir / "fbn_inner_c.py", "def decode():\n    return 'v2'\n")
    _import_fresh("fbn_inner_c")
    _import_fresh("fbnhelp_c")
    mod = _import_fresh("fbnstage_c")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 == fp2, "Transitive callee change must not change fingerprint"


# ---------------------------------------------------------------------------
# (d) Unmarked helper still reruns
# ---------------------------------------------------------------------------


def test_unmarked_helper_body_change_still_reruns(module_dir: pathlib.Path) -> None:
    """An unmarked helper coexisting with a marked one still reruns on change."""
    _write(
        module_dir / "fbnhelp_d_marked.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def marked():\n"
        "    return 'stable'\n",
    )
    _write(module_dir / "fbnhelp_d_unmarked.py", "def unmarked():\n    return 'v1'\n")
    _write(
        module_dir / "fbnstage_d.py",
        "from fbnhelp_d_marked import marked\n"
        "from fbnhelp_d_unmarked import unmarked\n\n"
        "def my_stage():\n"
        "    return marked() + unmarked()\n",
    )
    _import_fresh("fbnhelp_d_marked")
    _import_fresh("fbnhelp_d_unmarked")
    mod = _import_fresh("fbnstage_d")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(module_dir / "fbnhelp_d_unmarked.py", "def unmarked():\n    return 'v2'\n")
    _import_fresh("fbnhelp_d_unmarked")
    mod = _import_fresh("fbnstage_d")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 != fp2, "Unmarked helper body change must still change fingerprint"


# ---------------------------------------------------------------------------
# (e) #496 refusal: body import of user module
# ---------------------------------------------------------------------------


def test_body_import_of_user_module_is_refused(module_dir: pathlib.Path) -> None:
    """@fingerprint_by_name refuses a function with body imports of user modules."""
    _write(module_dir / "fbn_user_mod.py", "VALUE = 1\n")
    _write(
        module_dir / "fbnhelp_e.py",
        "def helper():\n    from fbn_user_mod import VALUE\n    return VALUE\n",
    )
    mod = _import_fresh("fbnhelp_e")
    with pytest.raises(exceptions.StageDefinitionError, match="fbn_user_mod"):
        decorators.fingerprint_by_name(mod.helper)


def test_body_import_of_library_module_is_accepted(module_dir: pathlib.Path) -> None:
    """@fingerprint_by_name accepts a function that body-imports only library modules."""
    _write(
        module_dir / "fbnhelp_e2.py",
        "def helper():\n    import json\n    return json.dumps({})\n",
    )
    mod = _import_fresh("fbnhelp_e2")
    result = decorators.fingerprint_by_name(mod.helper)
    assert result is mod.helper


# ---------------------------------------------------------------------------
# (g) Type refusals (kills m8, m9)
# ---------------------------------------------------------------------------


def test_non_callable_is_refused() -> None:
    """@fingerprint_by_name raises TypeError for a non-callable."""
    with pytest.raises(TypeError, match="requires a callable"):
        decorators.fingerprint_by_name(42)  # pyright: ignore[reportArgumentType]


def test_class_is_refused() -> None:
    """@fingerprint_by_name refuses a class."""

    class Foo:
        pass

    with pytest.raises(TypeError, match="module-level function"):
        decorators.fingerprint_by_name(Foo)  # pyright: ignore[reportArgumentType]


def test_module_level_lambda_is_refused() -> None:
    """@fingerprint_by_name refuses a module-level lambda.

    The lambda check stands on its own. A module-level lambda has qualname '<lambda>' and no
    '<locals>', so the nested-function check does not reach it.
    """
    # Create in a module so the lambda is module-level, not nested.
    _ns: dict[str, object] = {}
    exec("fn = lambda: 1", _ns)  # noqa: S102
    fn = _ns["fn"]
    assert callable(fn)
    qualname = getattr(fn, "__qualname__", "")
    assert "<locals>" not in qualname, "This lambda must be module-level for the test"
    with pytest.raises(TypeError, match="lambda"):
        decorators.fingerprint_by_name(fn)  # pyright: ignore[reportArgumentType]


def test_nested_function_is_refused() -> None:
    """@fingerprint_by_name refuses a nested function."""

    def outer():
        def inner():
            return 1

        return inner

    with pytest.raises(TypeError, match="module-level"):
        decorators.fingerprint_by_name(outer())


def test_partial_is_refused() -> None:
    """@fingerprint_by_name refuses a functools.partial.

    The mark needs a function to name. A partial is an object that holds one.
    """

    def helper():
        return 1

    with pytest.raises(TypeError, match="requires a function"):
        decorators.fingerprint_by_name(
            functools.partial(
                helper,
            )
        )


def test_builtin_is_refused() -> None:
    """@fingerprint_by_name refuses a builtin.

    The mark needs a Python function to name. A builtin has no module-level definition.
    """
    with pytest.raises(TypeError, match="requires a function"):
        decorators.fingerprint_by_name(len)  # pyright: ignore[reportArgumentType]


def test_callable_instance_is_refused() -> None:
    """@fingerprint_by_name refuses a callable instance.

    The mark needs a function to name. An instance carries state the walk reads instead.
    """

    class CallableObj:
        def __call__(self):
            return 1

    with pytest.raises(TypeError, match="requires a function"):
        decorators.fingerprint_by_name(CallableObj())


def test_method_is_refused(module_dir: pathlib.Path) -> None:
    """@fingerprint_by_name refuses a method defined in a class body.

    A method's qualified name holds its class, so it is not a module-level function.
    """
    _write(
        module_dir / "fbnhelp_method.py",
        "class Box:\n    def method(self):\n        return 1\n",
    )
    mod = _import_fresh("fbnhelp_method")
    with pytest.raises(TypeError, match="module-level function"):
        decorators.fingerprint_by_name(mod.Box.method)


def test_staticmethod_is_refused() -> None:
    """@fingerprint_by_name refuses a staticmethod object."""

    class Foo:
        @staticmethod
        def bar():
            return 1

    sm = Foo.__dict__["bar"]
    assert isinstance(sm, staticmethod)
    with pytest.raises(TypeError, match="module-level function"):
        decorators.fingerprint_by_name(sm)  # pyright: ignore[reportArgumentType]


def test_classmethod_is_refused() -> None:
    """@fingerprint_by_name refuses a classmethod object."""

    class Foo:
        @classmethod
        def bar(cls):
            return 1

    cm = Foo.__dict__["bar"]
    assert isinstance(cm, classmethod)
    with pytest.raises(TypeError, match="callable|module-level function"):
        decorators.fingerprint_by_name(cm)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# C3: Dynamic imports in body are refused
# ---------------------------------------------------------------------------


def test_dynamic_import_module_is_refused(module_dir: pathlib.Path) -> None:
    """@fingerprint_by_name refuses importlib.import_module in the body."""
    _write(
        module_dir / "fbnhelp_dyn.py",
        "import importlib\n\ndef helper():\n    return importlib.import_module('json')\n",
    )
    mod = _import_fresh("fbnhelp_dyn")
    with pytest.raises(exceptions.StageDefinitionError, match="dynamic imports"):
        decorators.fingerprint_by_name(mod.helper)


# ---------------------------------------------------------------------------
# R1: functools.wraps wrapper over a marked function
# ---------------------------------------------------------------------------


def test_wraps_wrapper_over_marked_helper_reruns_on_wrapper_body_change(
    module_dir: pathlib.Path,
) -> None:
    """A functools.wraps wrapper over a marked function: the wrapper body is
    fingerprinted. A change to the wrapper body reruns the stage.

    `functools.wraps` copies `__dict__`, and so the mark, into the wrapper. The mark names the
    function it was applied to, so the wrapper stays user code.
    """
    _write(
        module_dir / "fbnhelp_r1.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def process_start():\n"
        "    return 'v1'\n",
    )
    _write(
        module_dir / "fbnstage_r1.py",
        "import functools\n"
        "from fbnhelp_r1 import process_start\n\n"
        "def retry(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(*a, **kw):\n"
        "        return fn(*a, **kw)\n"
        "    return wrapper\n\n"
        "wrapped = retry(process_start)\n\n"
        "def my_stage():\n"
        "    return wrapped()\n",
    )
    _import_fresh("fbnhelp_r1")
    mod = _import_fresh("fbnstage_r1")
    # The wrapper must NOT carry the mark
    assert not fingerprint.fingerprinted_by_name(mod.wrapped), (
        "functools.wraps wrapper must not carry the mark"
    )
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    # Change the wrapper body
    _write(
        module_dir / "fbnstage_r1.py",
        "import functools\n"
        "from fbnhelp_r1 import process_start\n\n"
        "def retry(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(*a, **kw):\n"
        "        return fn(*a, **kw).upper()\n"
        "    return wrapper\n\n"
        "wrapped = retry(process_start)\n\n"
        "def my_stage():\n"
        "    return wrapped()\n",
    )
    _import_fresh("fbnhelp_r1")
    mod = _import_fresh("fbnstage_r1")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 != fp2, "Wrapper body change must change fingerprint"


# ---------------------------------------------------------------------------
# R2: Marked helpers as class attributes
# ---------------------------------------------------------------------------


def test_marked_contextmanager_as_class_attribute_cached(
    module_dir: pathlib.Path,
) -> None:
    """A marked @contextmanager helper held as a class attribute: body change → cached.

    The class walk reads each member through `_user_member_value`, which records a marked
    member by name.
    """
    _write(
        module_dir / "fbnhelp_r2cm.py",
        "import contextlib\n"
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "@contextlib.contextmanager\n"
        "def locked():\n"
        "    yield 'v1'\n",
    )
    _write(
        module_dir / "fbnstage_r2cm.py",
        "from fbnhelp_r2cm import locked\n\n"
        "class Locks:\n"
        "    lock = locked\n\n"
        "def my_stage():\n"
        "    with Locks.lock() as val:\n"
        "        return val\n",
    )
    _import_fresh("fbnhelp_r2cm")
    mod = _import_fresh("fbnstage_r2cm")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(
        module_dir / "fbnhelp_r2cm.py",
        "import contextlib\n"
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "@contextlib.contextmanager\n"
        "def locked():\n"
        "    yield 'v2-changed'\n",
    )
    _import_fresh("fbnhelp_r2cm")
    mod = _import_fresh("fbnstage_r2cm")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 == fp2, "Marked CM as class attribute: body change must stay cached"
    assert fp2["ext:fbnstage_r2cm.Locks.lock"] == "fbnhelp_r2cm.locked", fp2


def test_marked_cache_as_class_attribute_cached(module_dir: pathlib.Path) -> None:
    """A marked @functools.cache helper held as a class attribute: body change → cached.

    The class walk records the marked function behind the cache wrapper by name.
    """
    _write(
        module_dir / "fbnhelp_r2ca.py",
        "import functools\n"
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "@functools.cache\n"
        "def start(x):\n"
        "    return x * 2\n",
    )
    _write(
        module_dir / "fbnstage_r2ca.py",
        "from fbnhelp_r2ca import start\n\n"
        "class Tools:\n"
        "    compute = start\n\n"
        "def my_stage():\n"
        "    return Tools.compute(5)\n",
    )
    _import_fresh("fbnhelp_r2ca")
    mod = _import_fresh("fbnstage_r2ca")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(
        module_dir / "fbnhelp_r2ca.py",
        "import functools\n"
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "@functools.cache\n"
        "def start(x):\n"
        "    return x * 3\n",
    )
    _import_fresh("fbnhelp_r2ca")
    mod = _import_fresh("fbnstage_r2ca")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 == fp2, "Marked cache as class attribute: body change must stay cached"
    assert fp2["ext:fbnstage_r2ca.Tools.compute"] == "fbnhelp_r2ca.start", fp2


_HELD_HELPER = {
    "contextmanager": "import contextlib\n\nfrom pivot import fingerprint_by_name\n\n\n"
    "@fingerprint_by_name\n@contextlib.contextmanager\ndef member():\n    yield 'v1'\n",
    "cache": "import functools\n\nfrom pivot import fingerprint_by_name\n\n\n"
    "@fingerprint_by_name\n@functools.cache\ndef member(x):\n    return x * 2\n",
    "contextmanager_above_the_mark": "import contextlib\n\nfrom pivot import fingerprint_by_name"
    "\n\n\n@contextlib.contextmanager\n@fingerprint_by_name\ndef member():\n    yield 'v1'\n",
}

_HELD_STAGE = """\
from {module} import member


class Holder:
    held = member


def my_stage():
    holder = Holder()
    return holder.held
"""


@pytest.mark.parametrize("wrapper", list(_HELD_HELPER))
def test_marked_class_attribute_the_stage_never_names_is_recorded_by_name(
    module_dir: pathlib.Path, wrapper: str
) -> None:
    """A class holds a marked helper, and the stage reads the class without naming the member.
    The class walk records the member by its module and qualified name, so a move of the helper
    to another module changes the stage fingerprint."""
    helper = _HELD_HELPER[wrapper]
    _write(module_dir / f"fbnhelp_held1_{wrapper}.py", helper)
    _write(
        module_dir / f"fbnstage_held_{wrapper}.py",
        _HELD_STAGE.format(module=f"fbnhelp_held1_{wrapper}"),
    )
    names = (f"fbnhelp_held1_{wrapper}", f"fbnstage_held_{wrapper}")
    first = fingerprint.get_stage_fingerprint(_fresh(*names).my_stage)
    entry = f"ext:fbnstage_held_{wrapper}.Holder.held"
    assert first[entry] == f"fbnhelp_held1_{wrapper}.member", first

    _write(module_dir / f"fbnhelp_held2_{wrapper}.py", helper)
    _write(
        module_dir / f"fbnstage_held_{wrapper}.py",
        _HELD_STAGE.format(module=f"fbnhelp_held2_{wrapper}"),
    )
    # The rewritten stage file keeps its size and mtime second, so drop its bytecode.
    shutil.rmtree(module_dir / "__pycache__", ignore_errors=True)
    moved = (f"fbnhelp_held2_{wrapper}", f"fbnstage_held_{wrapper}")
    second = fingerprint.get_stage_fingerprint(_fresh(*moved).my_stage)
    assert second[entry] == f"fbnhelp_held2_{wrapper}.member", second
    assert first != second, "a move of the marked member kept the stage fingerprint"


# ---------------------------------------------------------------------------
# m2: identity keeps the body hash
# ---------------------------------------------------------------------------


def test_marked_helper_as_partial_argument_body_change_cached(
    module_dir: pathlib.Path,
) -> None:
    """A marked helper bound as a functools.partial argument: body change → cached.

    The identity of a marked function holds its module and qualified name, and no code hash.
    """
    _write(
        module_dir / "fbnhelp_m2.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def process_start():\n"
        "    return 'v1'\n",
    )
    _write(
        module_dir / "fbnstage_m2.py",
        "import functools\n"
        "from fbnhelp_m2 import process_start\n\n"
        "RUNNER = functools.partial(process_start)\n\n"
        "def my_stage():\n"
        "    return RUNNER()\n",
    )
    _import_fresh("fbnhelp_m2")
    mod = _import_fresh("fbnstage_m2")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(
        module_dir / "fbnhelp_m2.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def process_start():\n"
        "    return 'v2-body-changed'\n",
    )
    _import_fresh("fbnhelp_m2")
    mod = _import_fresh("fbnstage_m2")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 == fp2, "Marked helper as partial arg: body change must stay cached"


# ---------------------------------------------------------------------------
# #496 limit: marked helper → unmarked callee → body import
# ---------------------------------------------------------------------------


def test_496_limit_uncovered_import_through_marked_helper(
    module_dir: pathlib.Path,
) -> None:
    """A marked helper calls an unmarked function that body-imports a user module.

    The stage warns and reruns every run. Moving the import to the top of the
    module clears the warning and the stage caches.
    """
    _write(module_dir / "fbn_extra.py", "VALUE = 42\n")
    _write(
        module_dir / "fbn_inner.py",
        "def decode():\n    from fbn_extra import VALUE\n    return VALUE\n",
    )
    _write(
        module_dir / "fbnhelp_limit.py",
        "from pivot import fingerprint_by_name\n"
        "from fbn_inner import decode\n\n"
        "@fingerprint_by_name\n"
        "def outer():\n"
        "    return decode()\n",
    )
    _write(
        module_dir / "fbnstage_limit.py",
        "from fbnhelp_limit import outer\n\ndef my_stage():\n    return outer()\n",
    )
    _import_fresh("fbn_extra")
    _import_fresh("fbn_inner")
    _import_fresh("fbnhelp_limit")
    mod = _import_fresh("fbnstage_limit")
    fp = fingerprint.get_stage_fingerprint(mod.my_stage)

    # The walk does not descend into the marked helper, so fbn_extra
    # is not in the covered set.
    covered = fingerprint.covered_user_modules(fp)
    assert "fbn_extra" not in covered, (
        "fbn_extra must not be covered: the walk did not descend into the marked helper"
    )


# ---------------------------------------------------------------------------
# Accepted wrappers
# ---------------------------------------------------------------------------


def test_contextmanager_wrapper_is_accepted(module_dir: pathlib.Path) -> None:
    """@fingerprint_by_name works with @contextlib.contextmanager wrappers."""
    _write(
        module_dir / "fbnhelp_ctx.py",
        "import contextlib\n"
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "@contextlib.contextmanager\n"
        "def locked():\n"
        "    yield 'lock'\n",
    )
    mod = _import_fresh("fbnhelp_ctx")
    assert fingerprint.fingerprinted_by_name(mod.locked)
    assert not fingerprint.is_user_code(mod.locked)


def test_functools_cache_wrapper_is_accepted(module_dir: pathlib.Path) -> None:
    """@fingerprint_by_name works with @functools.cache wrappers."""
    _write(
        module_dir / "fbnhelp_cache.py",
        "import functools\n"
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "@functools.cache\n"
        "def cached_fn(x):\n"
        "    return x * 2\n",
    )
    mod = _import_fresh("fbnhelp_cache")
    assert fingerprint.fingerprinted_by_name(mod.cached_fn)
    assert not fingerprint.is_user_code(mod.cached_fn)


# ---------------------------------------------------------------------------
# Unreadable source
# ---------------------------------------------------------------------------


def test_unreadable_source_at_decoration_is_accepted() -> None:
    """When inspect.getsource fails, the mark is set without the body check."""
    code = compile("def helper(): return 1", "<test>", "exec")
    ns: dict[str, object] = {}
    exec(code, ns)  # noqa: S102
    fn = ns["helper"]
    assert callable(fn)
    result = decorators.fingerprint_by_name(fn)  # pyright: ignore[reportArgumentType]
    assert result is fn
    assert fingerprint.fingerprinted_by_name(fn)


# ---------------------------------------------------------------------------
# Marked stage is refused
# ---------------------------------------------------------------------------


def test_marked_stage_is_refused(module_dir: pathlib.Path) -> None:
    """A stage decorated with @fingerprint_by_name is refused by Pipeline."""
    _write(
        module_dir / "fbnstage_refuse.py",
        "import pathlib\n"
        "from pivot import fingerprint_by_name\n"
        "from pivot.compose import Pipeline, stage\n\n"
        "@fingerprint_by_name\n"
        "@stage\n"
        "def bad_stage() -> str:\n"
        "    return 'oops'\n\n"
        "with Pipeline('test', root=pathlib.Path(__file__).parent) as pipeline:\n"
        "    bad_stage()\n",
    )
    with pytest.raises(exceptions.StageDefinitionError, match="marks a helper, not a stage"):
        _import_fresh("fbnstage_refuse")


# ---------------------------------------------------------------------------
# Context manager body change in direct use (not class attribute)
# ---------------------------------------------------------------------------


def test_marked_helper_contextmanager_body_change_cached(
    module_dir: pathlib.Path,
) -> None:
    """A @contextlib.contextmanager helper marked with @fingerprint_by_name:
    body change does not change the stage fingerprint."""
    _write(
        module_dir / "fbnhelp_cm.py",
        "import contextlib\n"
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "@contextlib.contextmanager\n"
        "def locked():\n"
        "    yield 'v1'\n",
    )
    _write(
        module_dir / "fbnstage_cm.py",
        "from fbnhelp_cm import locked\n\n"
        "def my_stage():\n"
        "    with locked() as val:\n"
        "        return val\n",
    )
    _import_fresh("fbnhelp_cm")
    mod = _import_fresh("fbnstage_cm")
    fp1 = fingerprint.get_stage_fingerprint(mod.my_stage)

    _write(
        module_dir / "fbnhelp_cm.py",
        "import contextlib\n"
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "@contextlib.contextmanager\n"
        "def locked():\n"
        "    yield 'v2-changed'\n",
    )
    _import_fresh("fbnhelp_cm")
    mod = _import_fresh("fbnstage_cm")
    fp2 = fingerprint.get_stage_fingerprint(mod.my_stage)

    assert fp1 == fp2, "Marked CM body change must stay cached"


# =============================================================================
# CLI acceptance tests
# =============================================================================


_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_REPO = pathlib.Path(__file__).resolve().parents[4]
_UNCOVERED = "imported user modules its fingerprint does not cover"


def _run(
    project: pathlib.Path, *args: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING")
    }
    return subprocess.run(
        [str(_PIVOT), *args],
        cwd=project,
        env={**env, **(extra or {})},
        capture_output=True,
        text=True,
        timeout=300,
    )


def _project(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    project = tmp_path / "project"
    for rel, text in {**files, "data/raw/in.txt": "alpha beta\ngamma\n"}.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    init = _run(project, "init")
    assert init.returncode == 0, init.stdout + init.stderr
    return project


def _statuses(result: subprocess.CompletedProcess[str]) -> dict[str, tuple[str, str]]:
    """Each stage's (status, reason) in a `pivot repro --json` run."""
    assert result.returncode == 0, result.stdout + result.stderr
    found = dict[str, tuple[str, str]]()
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            event = json.loads(line)
            if event.get("type") == "stage_complete":
                found[event["stage"]] = (event["status"], event["reason"])
    return found


def _status(result: subprocess.CompletedProcess[str], stage: str) -> tuple[str, str]:
    found = _statuses(result)
    assert stage in found, f"no stage_complete for {stage}:\n{result.stdout}{result.stderr}"
    return found[stage]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _edit(path: pathlib.Path, old: str, new: str) -> None:
    """Replace `old` once. `old` and `new` differ in length, so no stale bytecode is used."""
    text = path.read_text()
    assert text.count(old) == 1, f"{path}: {old!r}"
    assert len(old) != len(new)
    path.write_text(text.replace(old, new))


def _section(rel: str, heading: str) -> str:
    """The flattened text of the section `heading` starts in `rel`, up to the next `## `."""
    text = (_REPO / rel).read_text()
    assert heading in text, f"{rel}: section {heading!r} not found"
    section = text[text.index(heading) :]
    following = section.find("\n## ", len(heading))
    return _flat(section if following < 0 else section[:following])


def _decorator_docstring() -> str:
    tree = ast.parse((_REPO / "packages/pivot/src/pivot/decorators.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fingerprint_by_name":
            return _flat(ast.get_docstring(node) or "")
    raise AssertionError("decorators.fingerprint_by_name not found")


_CONCEPTS = ("docs/concepts/fingerprinting.md", "## @pivot.fingerprint_by_name")
_SKILL = ("skills/writing-pivot-stages/SKILL.md", "## Helpers that do not affect output")


# ---------------------------------------------------------------------------------------------
# D1. A user's functools.wraps decorator above the mark loses its own fingerprint.
# ---------------------------------------------------------------------------------------------

_WRAPS_ABOVE = {
    "lib/__init__.py": "",
    "lib/deco.py": """\
import functools


def doubled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) * 2

    return wrapper
""",
    "lib/h.py": """\
from pivot import fingerprint_by_name

from lib.deco import doubled


@doubled
@fingerprint_by_name
def helper() -> int:
    return 5
""",
    "pipeline.py": """\
from pivot import Pipeline, stage

from lib import h


@stage
def s(text: str) -> str:
    return f"{h.helper()}:{len(text)}\\n"


with Pipeline("leak") as pipeline:
    s(pipeline.input("in.txt", t=str))
""",
}


def test_unmarked_decorator_above_a_marked_helper_is_still_fingerprinted(
    tmp_path: pathlib.Path,
) -> None:
    """`functools.wraps` copies the function's `__dict__`, and so the mark, onto the user's own
    wrapper. The mark names the function it was applied to, so the wrapper stays user code. The
    manifest holds the wrapper's code and the marked function's name, and an edit to the
    wrapper reruns the stage with the new output."""
    project = _project(tmp_path, _WRAPS_ABOVE)
    out = project / "data" / "leak" / "s.txt"
    assert _status(_run(project, "repro", "--json"), "leak/s")[0] == "ran"
    assert out.read_text() == "10:17\n"

    _edit(project / "lib" / "deco.py", "* 2\n", "* 30\n")
    assert _status(_run(project, "repro", "--json"), "leak/s") == ("ran", "Code changed")
    assert out.read_text() == "150:17\n"

    _edit(project / "lib" / "h.py", "@doubled\n", "")
    assert _status(_run(project, "repro", "--json"), "leak/s") == ("ran", "Code changed")
    assert out.read_text() == "5:17\n"


# ---------------------------------------------------------------------------------------------
# D2. A marked method is accepted. The contract, the design and the PR body say it is refused.
# ---------------------------------------------------------------------------------------------

_METHOD_CASES = {
    "method": "class C:\n    @fingerprint_by_name\n    def m(self):\n        return 1\n",
    "staticmethod_mark_below": (
        "class C:\n    @staticmethod\n    @fingerprint_by_name\n    def m():\n        return 1\n"
    ),
    "staticmethod_mark_above": (
        "class C:\n    @fingerprint_by_name\n    @staticmethod\n    def m():\n        return 1\n"
    ),
    "classmethod_mark_below": (
        "class C:\n    @classmethod\n    @fingerprint_by_name\n    def m(cls):\n        return 1\n"
    ),
}

_IMPORTS_LIB_H = """\
from pivot import Pipeline, stage

import lib.h


@stage
def s(text: str) -> str:
    return text


with Pipeline("m") as pipeline:
    s(pipeline.input("in.txt", t=str))
"""


def test_marked_method_is_refused(tmp_path: pathlib.Path) -> None:
    """`@fingerprint_by_name` marks a module-level function. A method, a staticmethod and a
    classmethod are refused when the module loads, with a message that names the decorator and
    the method. Without the refusal the class walk records the method by name, and a change in
    what the method calls leaves the stage cached with a stale output."""
    for case, body in _METHOD_CASES.items():
        files = {
            "lib/__init__.py": "",
            "lib/h.py": "from pivot import fingerprint_by_name\n\n\n" + body,
            "pipeline.py": _IMPORTS_LIB_H,
        }
        project = _project(tmp_path / case, files)
        result = _run(project, "list")
        text = _flat(result.stdout + result.stderr)
        assert result.returncode != 0, f"{case}: a marked method was accepted: {text}"
        assert "@fingerprint_by_name" in text and "C.m" in text, f"{case}: {text}"


# ---------------------------------------------------------------------------------------------
# D3. The every-run rerun of the stated limit is in no doc a user reads.
# ---------------------------------------------------------------------------------------------

_LIMIT = {
    "lib/__init__.py": "",
    "lib/table.py": "X = 7\n",
    "lib/inner.py": """\
def decode() -> int:
    from lib import table
    return table.X
""",
    "lib/box.py": """\
from pivot import fingerprint_by_name

from lib import inner


@fingerprint_by_name
def process_start() -> int:
    return inner.decode()
""",
    "pipeline.py": """\
from pivot import Pipeline, stage

import lib.table
from lib import box


@stage
def s(text: str) -> str:
    return f"{box.process_start()}:{len(text)}\\n"


with Pipeline("tr") as pipeline:
    s(pipeline.input("in.txt", t=str))
""",
}


def test_stated_limit_reruns_every_run_and_both_docs_say_so(tmp_path: pathlib.Path) -> None:
    """Code reached only through a marked helper is not walked. When that code imports a user
    module in its body, the stage warns and reruns on every run. The decorator's sections in
    docs/concepts/fingerprinting.md and skills/writing-pivot-stages/SKILL.md must say that the
    stage reruns on every (or each) run."""
    project = _project(tmp_path, _LIMIT)
    for _ in range(2):
        result = _run(project, "repro")
        assert result.returncode == 0, result.stdout + result.stderr
        assert _UNCOVERED in _flat(result.stdout + result.stderr)
    for rel, heading in (_CONCEPTS, _SKILL):
        section = _section(rel, heading)
        assert "every run" in section or "each run" in section, (
            f"{rel}: the every-run rerun of the stated limit is not stated: {section}"
        )


# ---------------------------------------------------------------------------------------------
# D4. Marking a helper removes main's refusal of a dynamic import.
# ---------------------------------------------------------------------------------------------

_DYNAMIC = {
    "lib/__init__.py": "",
    "lib/table.py": "X = 7\n",
    "lib/h.py": """\
import importlib

from pivot import fingerprint_by_name


@fingerprint_by_name
def helper() -> int:
    return importlib.import_module("lib.table").X
""",
    "pipeline.py": """\
from pivot import Pipeline, stage

import lib.h
import lib.table


@stage
def s(text: str) -> str:
    return f"{lib.h.helper()}:{len(text)}\\n"


with Pipeline("bi") as pipeline:
    s(pipeline.input("in.txt", t=str))
""",
}


def test_marked_helper_that_imports_a_user_module_dynamically_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A marked helper whose body loads a user module with `importlib.import_module` is
    refused, as an unmarked one is. The message names the helper. Without the refusal the stage
    runs, then warns and reruns on every run."""
    project = _project(tmp_path, _DYNAMIC)
    result = _run(project, "repro")
    text = _flat(result.stdout + result.stderr)
    assert result.returncode != 0, f"the marked helper was accepted: {text}"
    assert "helper" in text, text
    assert not (project / "data" / "bi" / "s.txt").exists()


# ---------------------------------------------------------------------------------------------
# D5. The docs say the helper's file is recorded. A file move with the same module name is
#     not seen.
# ---------------------------------------------------------------------------------------------

_MOVE = {
    "lib/__init__.py": "",
    "lib/procs.py": """\
from pivot import fingerprint_by_name


@fingerprint_by_name
def process_start() -> int:
    return 1
""",
    "pipeline.py": """\
from pivot import Pipeline, stage

from lib import procs


@stage
def s(text: str) -> str:
    return f"{procs.process_start()}:{len(text)}\\n"


with Pipeline("mv") as pipeline:
    s(pipeline.input("in.txt", t=str))
""",
}


def test_recorded_file_claim_matches_behavior(tmp_path: pathlib.Path) -> None:
    """A marked helper is recorded by its module and qualified name, and by no file. The
    manifest entry is `ext:<binding> = <module>.<qualname>`, so moving `lib/procs.py` to
    `lib/procs/__init__.py` leaves the stage cached. Neither the decorator's docstring nor the
    fingerprinting guide says that the file is part of the record. BET's helpers sit outside
    the project root, where no project-relative file exists."""
    project = _project(tmp_path, _MOVE)
    assert _status(_run(project, "repro", "--json"), "mv/s")[0] == "ran"
    package = project / "lib" / "procs"
    package.mkdir()
    (package / "__init__.py").write_text((project / "lib" / "procs.py").read_text())
    (project / "lib" / "procs.py").unlink()
    moved = _status(_run(project, "repro", "--json"), "mv/s")
    claims = [
        where
        for where, text in (
            ("docstring", _decorator_docstring()),
            (_CONCEPTS[0], _section(*_CONCEPTS)),
        )
        if "qualified name and file" in text
    ]
    assert moved[0] == "ran" or not claims, (
        f"a file move with the same module name gave {moved}, but {claims} say the file is recorded"
    )


# ---------------------------------------------------------------------------------------------
# D6. A marked StageParams default_factory still has its body hashed.
# ---------------------------------------------------------------------------------------------

_FACTORY = {
    "lib/__init__.py": "",
    "lib/defaults.py": """\
from pivot import fingerprint_by_name


@fingerprint_by_name
def default_cols() -> list[str]:
    return ["a", "b"]
""",
    "pipeline.py": """\
import pydantic

from pivot import Pipeline, StageParams, stage

from lib import defaults


class JoinParams(StageParams):
    cols: list[str] = pydantic.Field(default_factory=defaults.default_cols)


@stage
def join(text: str, params: JoinParams) -> str:
    return f"{','.join(params.cols)}:{len(text)}\\n"


with Pipeline("params") as pipeline:
    join(pipeline.input("in.txt", t=str), params=JoinParams())
""",
}


def test_marked_default_factory_body_edit_keeps_the_stage_cached(tmp_path: pathlib.Path) -> None:
    """A pydantic field's `default_factory` is hashed with `hash_function_ast`, which hashes a
    marked factory by its name. An edit to the marked factory's body that keeps its value
    leaves the stage cached. The value the factory returns is in the lock's params, so a value
    change still reruns the stage."""
    project = _project(tmp_path, _FACTORY)
    assert _status(_run(project, "repro", "--json"), "params/join")[0] == "ran"

    _edit(
        project / "lib" / "defaults.py",
        '    return ["a", "b"]\n',
        '    cols = ["a", "b"]\n    return cols\n',
    )
    assert _status(_run(project, "repro", "--json"), "params/join")[0] == "cached"

    _edit(project / "lib" / "defaults.py", '    cols = ["a", "b"]\n', '    cols = ["a", "cc"]\n')
    assert _status(_run(project, "repro", "--json"), "params/join")[0] == "ran"
    assert (project / "data" / "params" / "join.txt").read_text() == "a,cc:17\n"


# ---------------------------------------------------------------------------------------------
# D7. test_upgrade_preserves_fingerprint_schema cannot fail for the upgrade defect it names.
# ---------------------------------------------------------------------------------------------

_UPGRADE = {
    "lib/__init__.py": "",
    "lib/util.py": """\
import contextlib
import dataclasses
import functools
import re

SCALE = 3
PATTERN = re.compile(r"\\w+")


@dataclasses.dataclass(frozen=True)
class Box:
    width: int = 2

    def area(self, h: int) -> int:
        return self.width * h


@functools.cache
def vocab() -> tuple[str, ...]:
    return ("a", "b", "c")


@contextlib.contextmanager
def guard():
    yield SCALE


def words(text: str) -> list[str]:
    return PATTERN.findall(text)


def scaled(n: int, factor: int = SCALE) -> int:
    return n * factor
""",
    "pipeline.py": """\
from typing import Annotated, TypedDict

import pydantic

from pivot import Pipeline, StageParams, metric, stage

from lib import util
from lib.util import Box, guard, scaled, vocab


def default_tags() -> list[str]:
    return ["x", "y"]


class CountParams(StageParams):
    factor: int = 2
    tags: list[str] = pydantic.Field(default_factory=default_tags)


class Split(TypedDict):
    words: list
    stats: Annotated[dict, metric]


@stage
def count(text: str, params: CountParams) -> dict:
    with guard() as g:
        return {"n": scaled(len(util.words(text)), params.factor) + g, "tags": params.tags}


@stage
def split(text: str) -> Split:
    ws = util.words(text)
    return {"words": ws, "stats": {"count": len(ws), "vocab": len(vocab())}}


@stage
def area(text: str) -> dict:
    return {"a": Box(width=4).area(len(text))}


with Pipeline("up") as pipeline:
    raw = pipeline.input("in.txt", t=str)
    count(raw, params=CountParams())
    split(raw)
    area(raw)
""",
}

# The locks `pivot repro` with pivot 0.5.8 (PyPI, CPython 3.13) wrote for `_UPGRADE`, in a
# fresh tree with a fresh `.pivot/state.lmdb`.
_LOCKS_0_5_8 = {
    "area": """\
schema_version: 2
code_manifest:
  class:lib.util.Box: 545ea4838dab4252
  default:lib.util.Box.width: 6021b5621680598b
  dep:text:loader:pivot.loaders.Text:empty: f346543fc28fb29c
  dep:text:loader:pivot.loaders.Text:load: 69b78870ffe7932d
  dep:text:loader:pivot.loaders.Text:save: f187f23051043f60
  func:lib.util.Box.area: 99e24a5dad70eace
  module:lib: ''
  module:lib.util: ''
  out:up/area:loader:pivot.loaders.YAML:config: cfc61af456ae42e3
  out:up/area:loader:pivot.loaders.YAML:empty: a66d45b00a1a3a57
  out:up/area:loader:pivot.loaders.YAML:load: 4a31bd3a307ad2f0
  out:up/area:loader:pivot.loaders.YAML:save: 5240a272b83ad042
  out:up/area:tag: data
  self:area: 3e6cc2221e9ed6c5
  wire:text: in.txt
  wire:text:path: data/raw/in.txt
params: {}
deps:
- producer: in.txt
  key: null
  hash: d014bdbeea62a8b9
outs:
- key: null
  hash: 15c3ec44bec25f26
  path: data/up/area.yaml
""",
    "count": """\
schema_version: 2
code_manifest:
  class:pipeline.CountParams: 11b98d3f45f56c21
  const:lib.util.PATTERN: '["re.Pattern", "\\\\w+", 32]'
  const:lib.util.SCALE: '3'
  default:lib.util.scaled.factor: '3'
  dep:text:loader:pivot.loaders.Text:empty: f346543fc28fb29c
  dep:text:loader:pivot.loaders.Text:load: 69b78870ffe7932d
  dep:text:loader:pivot.loaders.Text:save: f187f23051043f60
  ext:contextlib._GeneratorContextManager: contextlib._GeneratorContextManager
  func:lib.util.guard: e05da719ea96cb62
  func:lib.util.scaled: d6cf10ce59ab5b3f
  func:lib.util.words: 8e7304dd2c0f8fbf
  func:pipeline.guard.<wrapper>: 927552b05dc91bbb
  module:lib: ''
  module:lib.util: ''
  module:pipeline: ''
  out:up/count:loader:pivot.loaders.YAML:config: cfc61af456ae42e3
  out:up/count:loader:pivot.loaders.YAML:empty: a66d45b00a1a3a57
  out:up/count:loader:pivot.loaders.YAML:load: 4a31bd3a307ad2f0
  out:up/count:loader:pivot.loaders.YAML:save: 5240a272b83ad042
  out:up/count:tag: data
  schema:pipeline.CountParams: 7da5beb5efe4a65a
  self:count: 66d0a83d97c551ad
  wire:text: in.txt
  wire:text:path: data/raw/in.txt
params:
  factor: 2
  tags:
  - x
  - y
deps:
- producer: in.txt
  key: null
  hash: d014bdbeea62a8b9
outs:
- key: null
  hash: d3f61ff182894f6a
  path: data/up/count.yaml
""",
    "split": """\
schema_version: 2
code_manifest:
  class:pipeline.Split: c7db050b60e18908
  const:lib.util.PATTERN: '["re.Pattern", "\\\\w+", 32]'
  dep:text:loader:pivot.loaders.Text:empty: f346543fc28fb29c
  dep:text:loader:pivot.loaders.Text:load: 69b78870ffe7932d
  dep:text:loader:pivot.loaders.Text:save: f187f23051043f60
  func:lib.util.vocab: b46760a3301568d7
  func:lib.util.words: 8e7304dd2c0f8fbf
  module:lib: ''
  module:lib.util: ''
  module:pipeline: ''
  out:up/split:stats:loader:pivot.loaders.YAML:config: cfc61af456ae42e3
  out:up/split:stats:loader:pivot.loaders.YAML:empty: a66d45b00a1a3a57
  out:up/split:stats:loader:pivot.loaders.YAML:load: 4a31bd3a307ad2f0
  out:up/split:stats:loader:pivot.loaders.YAML:save: 5240a272b83ad042
  out:up/split:stats:tag: metric
  out:up/split:words:loader:pivot.loaders.YAML:config: cfc61af456ae42e3
  out:up/split:words:loader:pivot.loaders.YAML:empty: a66d45b00a1a3a57
  out:up/split:words:loader:pivot.loaders.YAML:load: 4a31bd3a307ad2f0
  out:up/split:words:loader:pivot.loaders.YAML:save: 5240a272b83ad042
  out:up/split:words:tag: data
  self:split: 49663be953ae81c5
  wire:text: in.txt
  wire:text:path: data/raw/in.txt
params: {}
deps:
- producer: in.txt
  key: null
  hash: d014bdbeea62a8b9
outs:
- key: stats
  hash: 9f688aaee9660f88
  path: metrics/up/split/stats.yaml
- key: words
  hash: 5b955e9ae8aa08d4
  path: data/up/split/words.yaml
""",
}


def test_upgrade_from_0_5_8_reruns_nothing(tmp_path: pathlib.Path) -> None:
    """A project with no mark that 0.5.8 ran to completion is all cached under this version,
    from a fresh `.pivot/state.lmdb`. The comparison is of whole manifests, value by value, so
    a change to any unmarked manifest value fails it."""
    assert _LOCKS_0_5_8, "the 0.5.8 locks are not recorded"
    project = _project(tmp_path, _UPGRADE)
    first = _statuses(_run(project, "repro", "--json"))
    assert sorted(first) == sorted(f"up/{name}" for name in _LOCKS_0_5_8), first
    stages = project / ".pivot" / "stages" / "up"
    for name, text in _LOCKS_0_5_8.items():
        head = yaml.safe_load((stages / f"{name}.lock").read_text())["code_manifest"]
        assert head == yaml.safe_load(text)["code_manifest"], f"{name}: manifest differs"
        (stages / f"{name}.lock").write_text(text)
    shutil.rmtree(project / ".pivot" / "state.lmdb")
    upgraded = _statuses(_run(project, "repro", "--json"))
    assert {status for status, _ in upgraded.values()} == {"cached"}, upgraded


# ---------------------------------------------------------------------------------------------
# A C-level cache wrapper above the mark (PR512_RECHECK.md F1).
# ---------------------------------------------------------------------------------------------

_CACHE_ABOVE = """\
import functools

import rk_inner
from pivot import fingerprint_by_name


@{cache}
@fingerprint_by_name
def start():
    return rk_inner.decode() + {body}
"""


@pytest.mark.parametrize("cache", ["functools.cache", "functools.lru_cache(maxsize=8)"])
def test_cache_wrapper_above_the_mark_records_the_marked_function_by_name(
    module_dir: pathlib.Path, cache: str
) -> None:
    """A cache wrapper above the mark is unmarked user code, so the walk reads it. Through its
    `__wrapped__` the walk reaches the marked function and records it by name. An edit to the
    marked function's body, or to a function that only the marked function calls, keeps the
    stage fingerprint. The manifest holds the name of the marked function and no code."""
    _write(module_dir / "rk_inner.py", "def decode():\n    return 'i1'\n")
    _write(module_dir / "rk_cache_h.py", _CACHE_ABOVE.format(cache=cache, body="'b1'"))
    _write(
        module_dir / "rk_cache_s.py",
        "from rk_cache_h import start\n\n\ndef my_stage():\n    return start()\n",
    )
    names = ("rk_inner", "rk_cache_h", "rk_cache_s")
    first = fingerprint.get_stage_fingerprint(_fresh(*names).my_stage)

    _write(module_dir / "rk_cache_h.py", _CACHE_ABOVE.format(cache=cache, body="'b2-changed'"))
    second = fingerprint.get_stage_fingerprint(_fresh(*names).my_stage)
    assert first == second, "a body edit of the marked function changed the fingerprint"

    _write(module_dir / "rk_inner.py", "def decode():\n    return 'i2-changed'\n")
    third = fingerprint.get_stage_fingerprint(_fresh(*names).my_stage)
    assert second == third, "an edit to a callee of the marked function changed the fingerprint"
    assert {key: value for key, value in third.items() if not key.startswith("self:")} == {
        "ext:rk_cache_s.start": "rk_cache_h.start"
    }, third


def test_the_walk_of_a_cache_wrapper_above_the_mark_reads_no_marked_body(
    module_dir: pathlib.Path,
) -> None:
    """`get_stage_fingerprint` of a cache wrapper itself walks what the wrapper reads. The
    unwrap behind the wrapper stops at the marked function, so the manifest holds the wrapper's
    own name hash and nothing the marked body reads."""
    _write(module_dir / "rk_inner.py", "def decode():\n    return 'i1'\n")
    _write(
        module_dir / "rk_direct_h.py",
        _CACHE_ABOVE.format(cache="functools.cache", body="'b1'").replace("rk_cache", "rk_direct"),
    )
    wrapper = _fresh("rk_inner", "rk_direct_h").start
    manifest = fingerprint.get_stage_fingerprint(wrapper)
    assert [key for key in manifest if not key.startswith("self:")] == [], manifest


# ---------------------------------------------------------------------------------------------
# The canonical form of a value that holds a copy of the mark (PR512_RECHECK.md F2, F3).
# ---------------------------------------------------------------------------------------------


def test_the_canonical_form_of_a_cache_wrapper_above_the_mark_holds_no_address(
    module_dir: pathlib.Path,
) -> None:
    """`functools.cache` copies the marked function's `__dict__`, and so the mark, into its
    wrapper. The canonical form of a partial that holds the wrapper reads that `__dict__`. The
    canonical form is the same in every process, so it holds no memory address and no mark."""
    _write(
        module_dir / "rk_canon_h.py",
        "import functools\n\nfrom pivot import fingerprint_by_name\n\n\n"
        + "@functools.cache\n@fingerprint_by_name\ndef start():\n    return 1\n",
    )
    wrapper = _fresh("rk_canon_h").start
    text = fingerprint._serialize_value_for_hash(functools.partial(print, wrapper))
    for address in (id(wrapper), id(wrapper.__wrapped__)):
        assert str(address) not in text, f"the canonical form holds address {address}: {text}"
    assert fingerprint._BY_NAME_MARK not in text, f"the canonical form holds the mark: {text}"


_OBJECT_WITH_MARK = """\
import functools

from pivot import fingerprint_by_name


@fingerprint_by_name
def start():
    return 1


class Wrapping:
    def __init__(self, func):
        functools.update_wrapper(self, func)


HELD = Wrapping(start)
"""


def test_the_canonical_form_of_an_object_that_holds_a_copy_of_the_mark_omits_the_mark(
    module_dir: pathlib.Path,
) -> None:
    """`functools.update_wrapper` copies `__dict__`, and so the mark, into any object. The
    canonical form of a user object reads its `__dict__`, and it leaves the mark out, so no
    lock holds it."""
    _write(module_dir / "rk_obj_h.py", _OBJECT_WITH_MARK)
    held = _fresh("rk_obj_h").HELD
    assert fingerprint._BY_NAME_MARK in vars(held)
    text = fingerprint._serialize_value_for_hash(held)
    assert "rk_obj_h.Wrapping" in text, text
    assert fingerprint._BY_NAME_MARK not in text, f"the canonical form holds the mark: {text}"


_CANON_IN_CHILD = """\
import functools
import sys

sys.path.insert(0, sys.argv[1])
import rk_canon_h
from pivot import fingerprint

print(fingerprint._serialize_value_for_hash(functools.partial(print, rk_canon_h.start)))
"""


def test_the_canonical_form_of_a_marked_partial_argument_is_the_same_in_two_processes(
    module_dir: pathlib.Path,
) -> None:
    """A lock holds the canonical form of a stage's partial arguments, and a later run reads it
    in another process. The canonical form of a partial that binds a cache wrapper of a marked
    helper is the same in both processes."""
    _write(
        module_dir / "rk_canon_h.py",
        "import functools\n\nfrom pivot import fingerprint_by_name\n\n\n"
        + "@functools.cache\n@fingerprint_by_name\ndef start():\n    return 1\n",
    )
    here = fingerprint._serialize_value_for_hash(
        functools.partial(print, _fresh("rk_canon_h").start)
    )
    child = subprocess.run(
        [sys.executable, "-c", _CANON_IN_CHILD, str(module_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.strip() == here, f"{child.stdout.strip()!r} != {here!r}"


_CACHED_PARTIAL = {
    "lib/__init__.py": "",
    "lib/h.py": """\
import functools

from pivot import fingerprint_by_name


@functools.cache
@fingerprint_by_name
def process_start() -> int:
    return 5


def apply(fn) -> int:
    return fn()


RUNNER = functools.partial(apply, process_start)
""",
    "pipeline.py": """\
from pivot import Pipeline, stage

from lib import h


@stage
def s(text: str) -> str:
    return f"{h.RUNNER()}:{len(text)}\\n"


with Pipeline("leak") as pipeline:
    s(pipeline.input("in.txt", t=str))
""",
}


def test_a_cached_marked_helper_in_a_partial_argument_is_cached_on_the_next_run(
    tmp_path: pathlib.Path,
) -> None:
    """A stage reads a `functools.cache` wrapper of a marked helper through a partial argument.
    The lock records the partial's arguments in canonical form, which is the same in every
    process. The second run is cached, and an edit to the marked body keeps the stage cached."""
    project = _project(tmp_path, _CACHED_PARTIAL)
    assert _status(_run(project, "repro", "--json"), "leak/s")[0] == "ran"
    assert _status(_run(project, "repro", "--json"), "leak/s") == ("cached", "unchanged")

    helper = project / "lib" / "h.py"
    helper.write_text(
        helper.read_text().replace("    return 5\n", "    value = 5\n    return value\n")
    )
    assert _status(_run(project, "repro", "--json"), "leak/s") == ("cached", "unchanged")
    assert (project / "data" / "leak" / "s.txt").read_text() == "5:17\n"


# ---------------------------------------------------------------------------------------------
# The mark names the object it was applied to, also after that object is freed.
# ---------------------------------------------------------------------------------------------


def test_a_function_that_receives_a_copy_of_the_mark_never_reads_as_marked() -> None:
    """The mark names the function it was applied to. Another function that receives a copy of
    the mark's `__dict__` entry (a clone made with `__dict__.update`, or a wrapper whose
    `__wrapped__` is gone) is a different object. It never reads as marked, also when it takes
    the memory address of the freed original."""
    code_marked = compile("def helper():\n    return 1\n", "<rk-marked>", "exec").co_consts[0]
    code_other = compile("def other():\n    return 2\n", "<rk-other>", "exec").co_consts[0]
    namespace: dict[str, object] = {"__name__": "rk_copied_mark"}
    for attempt in range(50):
        original = decorators.fingerprint_by_name(types.FunctionType(code_marked, namespace))
        address = id(original)
        copied = dict(vars(original))
        del original
        clone = types.FunctionType(code_other, namespace)
        clone.__dict__.update(copied)
        assert not fingerprint.fingerprinted_by_name(clone), (
            f"attempt {attempt}: a clone with a copied mark reads as marked "
            f"(at the original's address: {id(clone) == address})"
        )


# ---------------------------------------------------------------------------------------------
# `__import__` of a user module in a marked body (PR512_RECHECK.md F4).
# ---------------------------------------------------------------------------------------------


def test_dunder_import_of_a_user_module_in_a_marked_body_is_refused(
    module_dir: pathlib.Path,
) -> None:
    """Decoration refuses a marked helper whose body loads a user module with `__import__`, as
    the decorator's guide, the skill and the PR body say. The message names the helper."""
    _write(module_dir / "rk_user_mod.py", "VALUE = 1\n")
    _write(
        module_dir / "rk_dunder.py",
        "def helper():\n    return __import__('rk_user_mod').VALUE\n",
    )
    mod = _fresh("rk_user_mod", "rk_dunder")
    with pytest.raises(exceptions.StageDefinitionError, match="helper"):
        decorators.fingerprint_by_name(mod.helper)


# ---------------------------------------------------------------------------------------------
# A marked helper as a partial argument and as a default value: its identity has no code.
# ---------------------------------------------------------------------------------------------

_PARTIAL_ARG_HELPER = """\
from pivot import fingerprint_by_name


@fingerprint_by_name
def process_start():
    return {body}
"""

_PARTIAL_ARG_STAGE = """\
import functools

from rk_arg_h import process_start


def _apply(fn):
    return fn()


RUNNER = functools.partial(_apply, process_start)


def my_stage(start=process_start):
    return RUNNER() + start()
"""


def test_a_marked_helper_bound_as_a_partial_argument_keeps_the_fingerprint(
    module_dir: pathlib.Path,
) -> None:
    """A marked helper bound as a `functools.partial` argument, and held as a stage default
    value, is named by module and qualified name, with no code. An edit to its body keeps the
    stage fingerprint and the stage's manifest-cache key."""
    _write(module_dir / "rk_arg_h.py", _PARTIAL_ARG_HELPER.format(body="'v1'"))
    _write(module_dir / "rk_arg_s.py", _PARTIAL_ARG_STAGE)
    stage = _fresh("rk_arg_h", "rk_arg_s").my_stage
    first = fingerprint.get_stage_fingerprint(stage)
    first_key = fingerprint._make_manifest_cache_key("arg/s", stage)

    _write(module_dir / "rk_arg_h.py", _PARTIAL_ARG_HELPER.format(body="'v2-body-changed'"))
    stage = _fresh("rk_arg_h", "rk_arg_s").my_stage
    assert fingerprint.get_stage_fingerprint(stage) == first
    assert fingerprint._make_manifest_cache_key("arg/s", stage) == first_key


# ---------------------------------------------------------------------------------------------
# Discovery's pickle-by-value decision (PR512_RECHECK.md item 6, mutant mP).
# ---------------------------------------------------------------------------------------------


def test_discovery_pickles_a_module_with_a_marked_helper_by_value_as_without_the_mark(
    tmp_path: pathlib.Path,
) -> None:
    """Discovery chooses pickling by value per project module. A project module that holds a
    marked helper is pickled by value, as the same module without the mark is."""
    found = dict[str, bool]()
    for variant, mark in (("marked", "@fingerprint_by_name\n"), ("plain", "")):
        root = tmp_path / variant
        (root / ".git").mkdir(parents=True)
        (root / ".pivot").mkdir()
        module = f"rk_pbv_{variant}"
        (root / f"{module}.py").write_text(
            "from pivot import fingerprint_by_name\n\n\n"
            + mark
            + "def helper() -> str:\n    return 'payload'\n"
        )
        (root / "pipeline.py").write_text(
            "import pathlib\n\n"
            + "from pivot.compose import Pipeline, stage\n"
            + f"from {module} import helper\n\n\n"
            + "@stage\ndef build() -> str:\n    return helper()\n\n\n"
            + "with Pipeline('test', root=pathlib.Path(__file__).parent) as pipeline:\n"
            + "    build()\n"
        )
        try:
            assert discovery.discover_pipeline(project_root=root) is not None
            found[variant] = module in cloudpickle.list_registry_pickle_by_value()
        finally:
            discovery.discover_pipeline(project_root=tmp_path / "empty")
            sys.modules.pop(module, None)
    assert found == {"marked": True, "plain": True}, found


# ---------------------------------------------------------------------------------------------
# A stage with the mark anywhere on its `__wrapped__` chain is refused.
# ---------------------------------------------------------------------------------------------

_STAGE_BELOW_DECORATOR = """\
import functools
import pathlib

from pivot import fingerprint_by_name
from pivot.compose import Pipeline, stage


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


@stage
{decorators}@fingerprint_by_name
def bad_stage() -> str:
    return "oops"


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    bad_stage()
"""


@pytest.mark.parametrize("depth", [1, 2])
def test_a_stage_marked_below_user_decorators_is_refused(
    module_dir: pathlib.Path, depth: int
) -> None:
    """Pipeline refuses a stage with `@fingerprint_by_name` anywhere on its `__wrapped__` chain,
    with the message a stage marked directly gets. A `functools.wraps` decorator between
    `@stage` and the mark copies the mark's `__dict__` entry, which names the function below
    it, so each function on the chain is asked, at any depth. Without the refusal the walk
    records the stage's own body by name only, and an edit to that body leaves the stage
    cached with an old output."""
    module = f"rk_r3a_stage_{depth}"
    _write(
        module_dir / f"{module}.py",
        _STAGE_BELOW_DECORATOR.format(decorators="@traced\n" * depth),
    )
    with pytest.raises(exceptions.StageDefinitionError, match="marks a helper, not a stage"):
        _fresh(module)


# ---------------------------------------------------------------------------------------------
# The manifest-cache replay of step "u" stops where the walk stops.
# ---------------------------------------------------------------------------------------------

_REPLAY_HELPER = """\
import functools

from pivot import fingerprint_by_name

K = 1


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def _impl() -> str:
    return f"k{K}"


_inner = traced(_impl)
helper = fingerprint_by_name(_inner)
start = functools.cache(traced(helper))
"""

_REPLAY_PIPELINE = """\
from pivot import Pipeline, stage

from rk_r3b_ext import h


@stage
def s(text: str) -> str:
    return f"{h.start()}:{len(text)}\\n"


with Pipeline("mr") as pipeline:
    s(pipeline.input("in.txt", t=str))
"""


def test_removing_the_mark_behind_a_cache_and_a_decorator_reruns_after_a_manifest_cache_hit(
    tmp_path: pathlib.Path,
) -> None:
    """The helper's file is outside the project root, so each new process takes the manifest
    cache and replays the recorded reads. The stage reaches a marked helper through
    `functools.cache` and a `functools.wraps` decorator. Removing the mark reruns the stage
    once. After that, a change to a value the helper reads reruns the stage with the new
    output."""
    ext = tmp_path / "ext"
    _write(ext / "rk_r3b_ext" / "__init__.py", "")
    _write(ext / "rk_r3b_ext" / "h.py", _REPLAY_HELPER)
    extra = {"PYTHONPATH": str(ext)}
    project = tmp_path / "project"
    _write(project / "pipeline.py", _REPLAY_PIPELINE)
    _write(project / "data" / "raw" / "in.txt", "alpha beta\ngamma\n")
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    init = _run(project, "init", extra=extra)
    assert init.returncode == 0, init.stdout + init.stderr
    out = project / "data" / "mr" / "s.txt"

    assert _status(_run(project, "repro", "--json", extra=extra), "mr/s")[0] == "ran"
    assert _status(_run(project, "repro", "--json", extra=extra), "mr/s")[0] == "cached"
    assert out.read_text() == "k1:17\n"

    _edit(ext / "rk_r3b_ext" / "h.py", "fingerprint_by_name(_inner)", "_inner")
    assert _status(_run(project, "repro", "--json", extra=extra), "mr/s") == (
        "ran",
        "Code changed",
    )
    assert _status(_run(project, "repro", "--json", extra=extra), "mr/s")[0] == "cached"

    _edit(ext / "rk_r3b_ext" / "h.py", "K = 1\n", "K = 22\n")
    assert _status(_run(project, "repro", "--json", extra=extra), "mr/s") == (
        "ran",
        "Code changed",
    )
    assert out.read_text() == "k22:17\n"


def test_the_replay_of_a_cache_over_a_decorator_over_the_mark_reads_what_the_walk_read(
    module_dir: pathlib.Path,
) -> None:
    """A manifest-cache hit replays each read the walk recorded and compares tokens. Behind
    `functools.cache` the walk's unwrap (step "u") stops at the user decorator above the mark,
    and the replay stops at the same function, so each recorded read gives its recorded token
    again while nothing changed."""
    _write(module_dir / "rk_replay_h.py", _REPLAY_HELPER)
    _write(
        module_dir / "rk_replay_s.py",
        "import rk_replay_h as h\n\n\ndef my_stage():\n    return h.start()\n",
    )
    stage = _fresh("rk_replay_h", "rk_replay_s").my_stage
    reads = list[tuple[tuple[str, ...], str]]()
    token = fingerprint._recorded_reads.set(reads)
    try:
        fingerprint.get_stage_fingerprint(stage)
    finally:
        fingerprint._recorded_reads.reset(token)
    assert any(path[-1] == "u" for path, _ in reads), reads
    assert fingerprint._recorded_reads_hold(stage, [[list(path), read] for path, read in reads])


# ---------------------------------------------------------------------------------------------
# A user `functools.wraps` decorator directly above the mark: its own lines are read, and the
# marked body is not.
# ---------------------------------------------------------------------------------------------

_C1 = {
    "rk_r3c1_settings.py": "FACTOR = 2\nBASE = 5\nOTHER = 5\n",
    "rk_r3c1_deco.py": """\
import functools

import rk_r3c1_settings as settings


def scaled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) * settings.FACTOR

    return wrapper
""",
    "rk_r3c1_h.py": """\
import rk_r3c1_settings as settings
from pivot import fingerprint_by_name
from rk_r3c1_deco import scaled


@scaled
@fingerprint_by_name
def helper():
    return settings.BASE
""",
    "rk_r3c1_s.py": """\
import rk_r3c1_h as h


def my_stage():
    return h.helper()
""",
}
_C1_MODULES = ("rk_r3c1_settings", "rk_r3c1_deco", "rk_r3c1_h", "rk_r3c1_s")


def test_a_user_decorator_above_the_mark_reads_no_module_attribute_of_the_marked_body(
    module_dir: pathlib.Path,
) -> None:
    """The attributes the walk reads through a module come from the decorator's own lines,
    never from the marked body below it. An edit that makes the marked body read another
    module attribute keeps the stage fingerprint."""
    for rel, text in _C1.items():
        _write(module_dir / rel, text)
    first = fingerprint.get_stage_fingerprint(_fresh(*_C1_MODULES).my_stage)

    _edit(module_dir / "rk_r3c1_h.py", "return settings.BASE", "return settings.OTHER")
    second = fingerprint.get_stage_fingerprint(_fresh(*_C1_MODULES).my_stage)
    assert second == first, sorted(
        key for key in set(first) | set(second) if first.get(key) != second.get(key)
    )


def test_a_user_decorator_above_the_mark_adds_its_own_module_reads(
    module_dir: pathlib.Path,
) -> None:
    """A decorator above the mark is user code, and its own lines are in the fingerprint. The
    decorator multiplies by `settings.FACTOR`, so a change to FACTOR changes the stage
    fingerprint."""
    for rel, text in _C1.items():
        _write(module_dir / rel, text)
    first = fingerprint.get_stage_fingerprint(_fresh(*_C1_MODULES).my_stage)

    _edit(module_dir / "rk_r3c1_settings.py", "FACTOR = 2\n", "FACTOR = 30\n")
    second = fingerprint.get_stage_fingerprint(_fresh(*_C1_MODULES).my_stage)
    assert second != first


# ---------------------------------------------------------------------------------------------
# `functools.cache` above a user `functools.wraps` decorator above the mark: the hash of the
# cache wrapper and the walk behind it stop at the decorator.
# ---------------------------------------------------------------------------------------------

_C2 = {
    "rk_r3c2_deco.py": """\
import functools


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
""",
    "rk_r3c2_h.py": """\
import functools

from pivot import fingerprint_by_name
from rk_r3c2_deco import traced


@functools.cache
@traced
@fingerprint_by_name
def start():
    return 5
""",
    "rk_r3c2_s.py": """\
import rk_r3c2_h as h


def my_stage():
    return h.start()
""",
}
_C2_MODULES = ("rk_r3c2_deco", "rk_r3c2_h", "rk_r3c2_s")


def test_a_cache_over_a_user_decorator_over_the_mark_hashes_no_marked_body(
    module_dir: pathlib.Path,
) -> None:
    """The cache wrapper has no code of its own, so its hash is of the source behind it. That
    read stops at the user decorator above the mark. An edit to the marked body keeps the
    stage fingerprint."""
    for rel, text in _C2.items():
        _write(module_dir / rel, text)
    first = fingerprint.get_stage_fingerprint(_fresh(*_C2_MODULES).my_stage)

    _edit(module_dir / "rk_r3c2_h.py", "    return 5\n", "    return 2 + 3\n")
    second = fingerprint.get_stage_fingerprint(_fresh(*_C2_MODULES).my_stage)
    assert second == first, sorted(
        key for key in set(first) | set(second) if first.get(key) != second.get(key)
    )


def test_a_cache_over_a_user_decorator_over_the_mark_adds_the_decorator_code(
    module_dir: pathlib.Path,
) -> None:
    """A decorator above the mark is user code, also below `functools.cache`: its own lines
    are in the fingerprint. An edit to that decorator changes the stage fingerprint."""
    for rel, text in _C2.items():
        _write(module_dir / rel, text)
    first = fingerprint.get_stage_fingerprint(_fresh(*_C2_MODULES).my_stage)

    _edit(
        module_dir / "rk_r3c2_deco.py",
        "        return fn(*args, **kwargs)\n",
        "        return fn(*args, **kwargs) * 2\n",
    )
    second = fingerprint.get_stage_fingerprint(_fresh(*_C2_MODULES).my_stage)
    assert second != first


_C1_CACHED_STAGES = {
    "called": _C1["rk_r3c1_s.py"],
    "class_member": (
        "import rk_r3c1_h as h\n\n\nclass Holder:\n    held = h.helper\n\n\n"
        "def my_stage():\n    holder = Holder()\n    return holder.held()\n"
    ),
}


@pytest.mark.parametrize("stage", list(_C1_CACHED_STAGES))
def test_a_cache_over_a_user_decorator_over_the_mark_adds_the_decorator_module_reads(
    module_dir: pathlib.Path, stage: str
) -> None:
    """Behind `functools.cache` the walk enters the user decorator above the mark, as it
    enters a decorator that the stage reads directly, also when the stage reaches the cache
    as a class member it never names. The decorator multiplies by `settings.FACTOR`, so a
    change to FACTOR changes the stage fingerprint, and the marked function stays a name."""
    files = {
        **{rel: text for rel, text in _C1.items() if rel != "rk_r3c1_h.py"},
        "rk_r3c1_h.py": _C1["rk_r3c1_h.py"]
        .replace("import rk_r3c1_settings", "import functools\n\nimport rk_r3c1_settings")
        .replace("@scaled\n", "@functools.cache\n@scaled\n"),
        "rk_r3c1_s.py": _C1_CACHED_STAGES[stage],
    }
    for rel, text in files.items():
        _write(module_dir / rel, text)
    first = fingerprint.get_stage_fingerprint(_fresh(*_C1_MODULES).my_stage)
    names = {value for key, value in first.items() if key.startswith("ext:")}
    assert "rk_r3c1_h.helper" in names, first

    _edit(module_dir / "rk_r3c1_settings.py", "FACTOR = 2\n", "FACTOR = 30\n")
    second = fingerprint.get_stage_fingerprint(_fresh(*_C1_MODULES).my_stage)
    assert second != first


# ---------------------------------------------------------------------------------------------
# The type hints and the refusal checks of a user decorator above the mark.
# ---------------------------------------------------------------------------------------------


def test_the_type_hints_a_decorator_copies_from_the_mark_stay_out_of_the_fingerprint(
    module_dir: pathlib.Path,
) -> None:
    """`functools.wraps` copies the marked function's annotations into the decorator. They are
    the marked function's type hints, so the walk does not read them: an edit to a class that
    only those hints name keeps the stage fingerprint, as with the mark alone."""
    _write(module_dir / "rk_hint_cfg.py", "class Cfg:\n    X = 1\n")
    _write(
        module_dir / "rk_hint_h.py",
        "import functools\n\n"
        "from pivot import fingerprint_by_name\n"
        "from rk_hint_cfg import Cfg\n\n\n"
        "def traced(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(*args, **kwargs):\n"
        "        return fn(*args, **kwargs)\n\n"
        "    return wrapper\n\n\n"
        "@traced\n@fingerprint_by_name\n"
        "def helper(cfg: Cfg | None = None) -> int:\n"
        "    return 1\n",
    )
    _write(
        module_dir / "rk_hint_s.py",
        "import rk_hint_h as h\n\n\ndef my_stage():\n    return h.helper()\n",
    )
    names = ("rk_hint_cfg", "rk_hint_h", "rk_hint_s")
    first = fingerprint.get_stage_fingerprint(_fresh(*names).my_stage)
    assert not [key for key in first if "rk_hint_cfg" in key], first

    _edit(module_dir / "rk_hint_cfg.py", "    X = 1\n", "    X = 22\n")
    second = fingerprint.get_stage_fingerprint(_fresh(*names).my_stage)
    assert second == first


def test_a_user_decorator_above_the_mark_that_calls_globals_is_refused(
    module_dir: pathlib.Path,
) -> None:
    """The walk's refusals read the decorator's own lines, which the fingerprint holds. A
    decorator above the mark that calls `globals()` is refused, as any walked function is."""
    _write(
        module_dir / "rk_dyn_h.py",
        "import functools\n\n"
        "from pivot import fingerprint_by_name\n\n\n"
        "def traced(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(*args, **kwargs):\n"
        "        globals()\n"
        "        return fn(*args, **kwargs)\n\n"
        "    return wrapper\n\n\n"
        "@traced\n@fingerprint_by_name\n"
        "def helper():\n"
        "    return 1\n",
    )
    _write(
        module_dir / "rk_dyn_s.py",
        "import rk_dyn_h as h\n\n\ndef my_stage():\n    return h.helper()\n",
    )
    stage = _fresh("rk_dyn_h", "rk_dyn_s").my_stage
    with pytest.raises(exceptions.StageDefinitionError, match=r"uses globals\(\)"):
        fingerprint.get_stage_fingerprint(stage)


# ---------------------------------------------------------------------------------------------
# No source or type-hint read of the walk reaches a marked function.
# ---------------------------------------------------------------------------------------------

_SPY_DECO = """\
import functools

import rk_spy_settings as settings


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) * settings.FACTOR

    return wrapper
"""

_SPY_HELPER = """\
import contextlib
import functools

import rk_spy_settings as settings
from pivot import fingerprint_by_name
from rk_spy_cfg import Cfg
from rk_spy_deco import traced


{decorators}
def target(cfg: Cfg | None = None):
    {verb} settings.BASE
"""

_SPY_SHAPES = {
    "decorator": ("@traced\n@fingerprint_by_name", "return"),
    "two_decorators": ("@traced\n@traced\n@fingerprint_by_name", "return"),
    "cache_decorator": ("@functools.cache\n@traced\n@fingerprint_by_name", "return"),
    "cache": ("@functools.cache\n@fingerprint_by_name", "return"),
    "contextmanager_above": ("@contextlib.contextmanager\n@fingerprint_by_name", "yield"),
    "contextmanager_below": ("@fingerprint_by_name\n@contextlib.contextmanager", "yield"),
    "cache_contextmanager": (
        "@functools.cache\n@contextlib.contextmanager\n@fingerprint_by_name",
        "yield",
    ),
}

_SPY_STAGES = {
    "called": "import rk_spy_h as h\n\n\ndef my_stage():\n    return h.target()\n",
    "class_member": (
        "import rk_spy_h as h\n\n\nclass Holder:\n    held = h.target\n\n\n"
        "def my_stage():\n    return Holder.held()\n"
    ),
}


@pytest.mark.parametrize("stage", list(_SPY_STAGES))
@pytest.mark.parametrize("shape", list(_SPY_SHAPES))
def test_no_source_or_hint_read_of_the_walk_reaches_a_marked_function(
    module_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch, shape: str, stage: str
) -> None:
    """`inspect.getsource` and `typing.get_type_hints` follow `__wrapped__` with no stop. Each
    object the walk gives them stops above the mark: a user decorator's own code, or no read.
    No call reaches the marked function, through any wrapper, for a stage that calls the
    helper or reads it as a class member."""
    decorators, verb = _SPY_SHAPES[shape]
    _write(module_dir / "rk_spy_settings.py", "FACTOR = 2\nBASE = 5\n")
    _write(module_dir / "rk_spy_cfg.py", "class Cfg:\n    X = 1\n")
    _write(module_dir / "rk_spy_deco.py", _SPY_DECO)
    _write(module_dir / "rk_spy_h.py", _SPY_HELPER.format(decorators=decorators, verb=verb))
    _write(module_dir / "rk_spy_s.py", _SPY_STAGES[stage])
    my_stage = _fresh(
        "rk_spy_settings", "rk_spy_cfg", "rk_spy_deco", "rk_spy_h", "rk_spy_s"
    ).my_stage

    sources = list[object]()
    hinted = list[object]()
    getsourcelines = inspect.getsourcelines
    get_type_hints = typing.get_type_hints

    def spy_sourcelines(obj: object) -> object:
        sources.append(obj)
        return getsourcelines(obj)

    def spy_type_hints(obj: object, *args: object, **kwargs: object) -> object:
        hinted.append(obj)
        return get_type_hints(obj, *args, **kwargs)

    monkeypatch.setattr(inspect, "getsourcelines", spy_sourcelines)
    monkeypatch.setattr(typing, "get_type_hints", spy_type_hints)
    fingerprint.get_stage_fingerprint(my_stage)
    monkeypatch.undo()

    def reaches_mark(obj: object) -> bool:
        if isinstance(obj, types.CodeType):
            return False
        end = inspect.unwrap(obj, stop=fingerprint.fingerprinted_by_name)
        return fingerprint.fingerprinted_by_name(end)

    assert my_stage in sources, sources
    assert [obj for obj in sources if reaches_mark(obj)] == []
    assert [obj for obj in hinted if reaches_mark(obj)] == []


# ---------------------------------------------------------------------------------------------
# The pickle-by-value decision of a module that holds a marked helper.
# ---------------------------------------------------------------------------------------------


def test_the_mark_leaves_the_pickle_by_value_decision_of_its_module_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    """Discovery pickles a project module by value and a module outside the project root by
    reference. A module that holds a marked helper gets the same decision as the same module
    without the mark, inside and outside the root."""
    found = dict[tuple[str, str], bool]()
    for place in ("inroot", "outroot"):
        for variant, mark in (("marked", "@fingerprint_by_name\n"), ("plain", "")):
            case = tmp_path / f"{place}_{variant}"
            root = case / "root"
            (root / ".git").mkdir(parents=True)
            (root / ".pivot").mkdir()
            home = root if place == "inroot" else case / "ext"
            module = f"rk_r3d_{place}_{variant}"
            _write(
                home / f"{module}.py",
                "from pivot import fingerprint_by_name\n\n\n"
                + mark
                + "def helper() -> str:\n    return 'payload'\n",
            )
            _write(
                root / "pipeline.py",
                "import pathlib\n\n"
                + "from pivot.compose import Pipeline, stage\n"
                + f"from {module} import helper\n\n\n"
                + "@stage\ndef build() -> str:\n    return helper()\n\n\n"
                + "with Pipeline('test', root=pathlib.Path(__file__).parent) as pipeline:\n"
                + "    build()\n",
            )
            if place == "outroot":
                sys.path.insert(0, str(home))
            try:
                assert discovery.discover_pipeline(project_root=root) is not None
                found[(place, variant)] = module in cloudpickle.list_registry_pickle_by_value()
            finally:
                discovery.discover_pipeline(project_root=tmp_path / "empty")
                held = sys.modules.pop(module, None)
                if held is not None:
                    with contextlib.suppress(ValueError):
                        cloudpickle.unregister_pickle_by_value(held)
                if place == "outroot":
                    sys.path.remove(str(home))
    assert found == {
        ("inroot", "marked"): True,
        ("inroot", "plain"): True,
        ("outroot", "marked"): False,
        ("outroot", "plain"): False,
    }, found


# ---------------------------------------------------------------------------------------------
# A class-based decorator above the mark, behind `functools.cache`, is user code.
# ---------------------------------------------------------------------------------------------

_R4A_DECO = """\
import functools


class Traced:
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs) * 2
"""

_R4A_HELPER = """\
import functools

from pivot import fingerprint_by_name
from rk4_a_deco import Traced


{top}@Traced
@fingerprint_by_name
def helper() -> int:
    return 5
"""

_R4A_STAGE = "import rk4_a_h as h\n\n\ndef my_stage():\n    return h.helper()\n"
_R4A_MODULES = ("rk4_a_deco", "rk4_a_h", "rk4_a_s")


@pytest.mark.parametrize(
    "top",
    ["@functools.cache\n", "@functools.lru_cache(maxsize=8)\n"],
    ids=["cache", "lru_cache"],
)
def test_a_class_decorator_above_the_mark_behind_a_cache_is_fingerprinted_or_refused(
    module_dir: pathlib.Path, top: str
) -> None:
    """A decorator above the mark is user code, and its own code is in the fingerprint, also
    when `functools.cache` wraps it. Here the decorator is a user class, and its own code is
    `Traced.__call__`. Pivot either puts that code in the fingerprint, so an edit to it changes
    the stage fingerprint, or refuses the stage as a non-scalar capture. It never records the
    chain by name: then the edit leaves the stage cached with the old output."""
    _write(module_dir / "rk4_a_deco.py", _R4A_DECO)
    _write(module_dir / "rk4_a_h.py", _R4A_HELPER.format(top=top))
    _write(module_dir / "rk4_a_s.py", _R4A_STAGE)
    try:
        first = fingerprint.get_stage_fingerprint(_fresh(*_R4A_MODULES).my_stage)
    except exceptions.StageDefinitionError as refused:
        assert "Traced" in str(refused), refused
        return

    _edit(module_dir / "rk4_a_deco.py", "* 2\n", "* 30\n")
    second = fingerprint.get_stage_fingerprint(_fresh(*_R4A_MODULES).my_stage)
    assert second != first, first


# ---------------------------------------------------------------------------------------------
# A mark on a `functools.wraps` wrapper keeps that wrapper's own code out of every fingerprint.
# ---------------------------------------------------------------------------------------------

_LOCK_HELPER = """\
import functools

from pivot import fingerprint_by_name


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def _impl():
    return "v1"


process_start = fingerprint_by_name(traced(_impl))
"""

_LOCK_STAGES = {
    "partial_and_default": """\
import functools

from rk4_lock_h import process_start


def _apply(fn):
    return fn()


RUNNER = functools.partial(_apply, process_start)


def my_stage(start=process_start):
    return RUNNER() + start()
""",
    "default_factory": """\
import pydantic

from pivot import StageParams
from rk4_lock_h import process_start


class P(StageParams):
    tag: str = pydantic.Field(default_factory=process_start)


def my_stage(params: P) -> str:
    return params.tag
""",
}


@pytest.mark.parametrize("stage", list(_LOCK_STAGES))
def test_a_mark_on_a_wraps_wrapper_keeps_the_fingerprint_when_its_own_code_changes(
    module_dir: pathlib.Path, stage: str
) -> None:
    """The mark covers the object it is applied to. When that object is a `functools.wraps`
    wrapper, the wrapper's own code is marked. An edit to it keeps the stage fingerprint and
    manifest-cache key for a partial argument and a default, and for a pydantic
    `default_factory`."""
    _write(module_dir / "rk4_lock_h.py", _LOCK_HELPER)
    _write(module_dir / "rk4_lock_s.py", _LOCK_STAGES[stage])
    names = ("rk4_lock_h", "rk4_lock_s")
    my_stage = _fresh(*names).my_stage
    first = fingerprint.get_stage_fingerprint(my_stage)
    first_key = fingerprint._make_manifest_cache_key("lock/s", my_stage)

    _edit(
        module_dir / "rk4_lock_h.py",
        "        return fn(*args, **kwargs)\n",
        "        return fn(*args, **kwargs) + ''\n",
    )
    my_stage = _fresh(*names).my_stage
    second = fingerprint.get_stage_fingerprint(my_stage)
    assert second == first, sorted(
        key for key in set(first) | set(second) if first.get(key) != second.get(key)
    )
    assert fingerprint._make_manifest_cache_key("lock/s", my_stage) == first_key


# ---------------------------------------------------------------------------------------------
# A class-based decorator above the mark is refused however the stage reaches it: one decision
# (`_mark_decision`) for every path that reads a value, so no path drops it.
# ---------------------------------------------------------------------------------------------

_R5_DECO = """\
import functools


class Traced:
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs) * 2


class TracedGen:
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        for value in self.fn(*args, **kwargs):
            yield value * 3


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
"""

_R5_HELPER = """\
import contextlib
import functools

from pivot import fingerprint_by_name
from rk5_deco import Traced, TracedGen, traced


{decorators}@fingerprint_by_name
def helper(*_args):
    {verb} 5
"""

# Decorator lines above the mark, the helper's verb, the edit to the class's __call__, the class.
_R5_SHAPES = {
    "cache": (
        "@functools.cache\n@Traced\n",
        "return",
        ("kwargs) * 2\n", "kwargs) * 20\n"),
        "Traced",
    ),
    "lru_cache": (
        "@functools.lru_cache(maxsize=8)\n@Traced\n",
        "return",
        ("kwargs) * 2\n", "kwargs) * 20\n"),
        "Traced",
    ),
    "contextmanager": (
        "@contextlib.contextmanager\n@TracedGen\n",
        "yield",
        ("value * 3\n", "value * 30\n"),
        "TracedGen",
    ),
    "cache_over_class_over_function": (
        "@functools.cache\n@Traced\n@traced\n",
        "return",
        ("kwargs) * 2\n", "kwargs) * 20\n"),
        "Traced",
    ),
}

_R5_HOLDER = "import rk5_h as h\n\n\nclass Holder:\n    held = h.helper\n\n\n"
_R5_FORMS = {
    "called": "import rk5_h as h\n\n\ndef my_stage():\n    return h.helper()\n",
    "class_member_named": _R5_HOLDER + "def my_stage():\n    return Holder.held()\n",
    "class_member_unnamed": _R5_HOLDER
    + "def my_stage():\n    holder = Holder()\n    return holder.held()\n",
    "partial_argument": (
        "import functools\n\nimport rk5_h as h\n\n\ndef _apply(fn):\n    return fn()\n\n\n"
        "RUNNER = functools.partial(_apply, h.helper)\n\n\ndef my_stage():\n    return RUNNER()\n"
    ),
    "default_argument": "import rk5_h as h\n\n\ndef my_stage(start=h.helper):\n    return start()\n",
}
_R5_MODULES = ("rk5_deco", "rk5_h", "rk5_s")


@pytest.mark.parametrize("form", list(_R5_FORMS))
@pytest.mark.parametrize("shape", list(_R5_SHAPES))
def test_a_class_decorator_above_the_mark_is_never_left_out_of_the_fingerprint(
    module_dir: pathlib.Path, shape: str, form: str
) -> None:
    """A class-based decorator above the mark under a library wrapper is refused, or its own
    code is in the fingerprint, however the stage reaches the helper: called, as a named or
    an unnamed class member, as a partial argument or as a default. It is never left out of
    both, which would leave the stage cached with an old output after an edit to the class's
    `__call__`."""
    decorators, verb, (old, new), cls = _R5_SHAPES[shape]
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R5_HELPER.format(decorators=decorators, verb=verb))
    _write(module_dir / "rk5_s.py", _R5_FORMS[form])
    try:
        first = fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)
    except exceptions.StageDefinitionError as refused:
        assert cls in str(refused), refused
        return

    _edit(module_dir / "rk5_deco.py", old, new)
    second = fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)
    assert second != first, first


@pytest.mark.parametrize("shape", ["cache", "lru_cache"])
def test_the_refusal_of_a_class_decorator_above_the_mark_names_the_class_and_the_mark(
    module_dir: pathlib.Path, shape: str
) -> None:
    """The refusal exists only because of the mark, so it names the class-based decorator and
    `@fingerprint_by_name`: the user knows what to change."""
    decorators, verb, _edit_pair, cls = _R5_SHAPES[shape]
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R5_HELPER.format(decorators=decorators, verb=verb))
    _write(module_dir / "rk5_s.py", _R5_FORMS["called"])
    with pytest.raises(exceptions.StageDefinitionError) as refused:
        fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)
    message = str(refused.value)
    assert cls in message, message
    assert "fingerprint_by_name" in message, message


_R6_SHAPES = {
    **{name: (lines, verb, cls) for name, (lines, verb, _edit_pair, cls) in _R5_SHAPES.items()},
    "cache_over_function_over_class": ("@functools.cache\n@traced\n@Traced\n", "return", "Traced"),
    "class_alone": ("@Traced\n", "return", "Traced"),
}

_R6_FORMS = {
    **_R5_FORMS,
    "default_factory": (
        "import typing\n\nimport pydantic\n\nimport rk5_h as h\nfrom pivot import StageParams\n\n\n"
        "class P(StageParams):\n    tag: typing.Any = pydantic.Field(default_factory=h.helper)\n\n\n"
        "def my_stage(params: P):\n    return params.tag\n"
    ),
    "partial_function": (
        "import functools\n\nimport rk5_h as h\n\n\nRUNNER = functools.partial(h.helper)\n\n\n"
        "def my_stage():\n    return RUNNER()\n"
    ),
}


@pytest.mark.parametrize("form", list(_R6_FORMS))
@pytest.mark.parametrize("shape", list(_R6_SHAPES))
def test_get_stage_fingerprint_refuses_a_class_decorator_above_the_mark_in_every_stage_form(
    module_dir: pathlib.Path, shape: str, form: str
) -> None:
    """`get_stage_fingerprint` alone refuses a user class instance above the mark, with or
    without a library wrapper over it, and with a function decorator on either side of it,
    however the stage reaches the helper, a pydantic `default_factory` and a partial's function
    included. The message names the class and the mark and gives the fixes that work, not the
    advice of the non-scalar refusal (StageParams, unsafe fingerprinting)."""
    decorators, verb, cls = _R6_SHAPES[shape]
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R5_HELPER.format(decorators=decorators, verb=verb))
    _write(module_dir / "rk5_s.py", _R6_FORMS[form])
    with pytest.raises(exceptions.StageDefinitionError) as refused:
        fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)
    message = str(refused.value)
    assert f"rk5_deco.{cls}" in message, message
    assert "rk5_h.helper" in message, message
    assert "@fingerprint_by_name" in message, message
    assert "functools.wraps" in message, message
    assert "StageParams" not in message, message
    assert "PIVOT_UNSAFE_FINGERPRINTING" not in message, message


def test_unsafe_fingerprinting_does_not_suppress_the_refusal_of_a_class_above_the_mark(
    module_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe fingerprinting turns a non-scalar refusal into a warning. It does not turn this
    refusal into one: nothing would then record the class's code, and the stage would be
    cached with an old output after an edit to it."""
    monkeypatch.setenv("PIVOT_UNSAFE_FINGERPRINTING", "1")
    decorators, verb, _edit_pair, _cls = _R5_SHAPES["cache"]
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R5_HELPER.format(decorators=decorators, verb=verb))
    _write(module_dir / "rk5_s.py", _R5_FORMS["called"])
    with pytest.raises(exceptions.StageDefinitionError, match="fingerprint_by_name"):
        fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)


_R6_FIX_HELPER = """\
import contextlib
import functools

from pivot import fingerprint_by_name
from rk5_deco import Traced, TracedGen


{decorators}def helper(*_args):
    {verb} 5
"""

# The refusal's first fix: the mark above the class-based decorator, so the whole helper is
# recorded by name. Decorator lines, the helper's verb, the edit to the class's __call__.
_R6_FIX_SHAPES = {
    "cache_over_marked_class": (
        "@functools.cache\n@fingerprint_by_name\n@Traced\n",
        "return",
        ("kwargs) * 2\n", "kwargs) * 20\n"),
    ),
    "marked_class": (
        "@fingerprint_by_name\n@Traced\n",
        "return",
        ("kwargs) * 2\n", "kwargs) * 20\n"),
    ),
    "contextmanager_over_marked_class": (
        "@contextlib.contextmanager\n@fingerprint_by_name\n@TracedGen\n",
        "yield",
        ("value * 3\n", "value * 30\n"),
    ),
}


@pytest.mark.parametrize("form", list(_R6_FORMS))
@pytest.mark.parametrize("shape", list(_R6_FIX_SHAPES))
def test_the_mark_above_a_class_based_decorator_records_the_whole_helper_by_name(
    module_dir: pathlib.Path, shape: str, form: str
) -> None:
    """The refusal's first fix works in every stage form: with `@fingerprint_by_name` above the
    class-based decorator, Pivot accepts the stage and records the helper by name, so an edit
    to the class's `__call__` or to the helper's body keeps the stage fingerprint. Where the
    stage reads the helper as a value, the manifest names it."""
    decorators, verb, (old, new) = _R6_FIX_SHAPES[shape]
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R6_FIX_HELPER.format(decorators=decorators, verb=verb))
    _write(module_dir / "rk5_s.py", _R6_FORMS[form])
    first = fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)
    if form in ("called", "class_member_named", "class_member_unnamed", "default_argument"):
        named = {value for key, value in first.items() if key.startswith("ext:")}
        assert "rk5_h.helper" in named, first

    _edit(module_dir / "rk5_deco.py", old, new)
    second = fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)
    assert second == first, sorted(
        key for key in set(first) | set(second) if first.get(key) != second.get(key)
    )

    _edit(module_dir / "rk5_h.py", f"    {verb} 5\n", f"    {verb} 2 + 3\n")
    third = fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)
    assert third == first, sorted(
        key for key in set(first) | set(third) if first.get(key) != third.get(key)
    )


@pytest.mark.parametrize("shape", ["cache", "lru_cache"])
def test_without_the_mark_a_cache_over_a_class_decorator_is_not_refused(
    module_dir: pathlib.Path, shape: str
) -> None:
    """The refusal's last fix, removing the mark, gives main's decision for the same chain:
    a library wrapper over a class-based decorator with no mark below is not refused."""
    decorators, verb, _edit_pair, _cls = _R5_SHAPES[shape]
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(
        module_dir / "rk5_h.py",
        _R5_HELPER.format(decorators=decorators, verb=verb).replace("@fingerprint_by_name\n", ""),
    )
    _write(module_dir / "rk5_s.py", _R5_FORMS["called"])
    fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)


@pytest.mark.parametrize("shape", ["cache", "lru_cache", "contextmanager"])
def test_the_canonical_form_refuses_a_class_decorator_above_the_mark(
    module_dir: pathlib.Path, shape: str
) -> None:
    """A loader field reaches the canonical form without the walk's value classification. The
    canonical form takes the mark decision before it names a library callable, so a library
    wrapper over a class-based decorator over the mark is refused there too, rather than named
    by the wrapper alone with the class's code left out."""
    decorators, verb, _edit_pair, cls = _R5_SHAPES[shape]
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R5_HELPER.format(decorators=decorators, verb=verb))
    helper = _fresh("rk5_deco", "rk5_h").helper
    with pytest.raises(exceptions.StageDefinitionError, match=f"rk5_deco.{cls}"):
        fingerprint._canonicalize_value_for_hash({"field": helper})


_R6_DUMP_HELPER = """\
from pivot import fingerprint_by_name
from rk5_deco import Traced


class TracedDump(Traced):
    def model_dump(self):
        return {"wrapped": self.fn.__qualname__}


@TracedDump
@fingerprint_by_name
def helper(*_args):
    return 5
"""


@pytest.mark.parametrize("form", ["partial_argument", "default_argument"])
def test_a_class_decorator_above_the_mark_with_model_dump_is_refused(
    module_dir: pathlib.Path, form: str
) -> None:
    """The canonical form takes the mark decision before it dumps a value with `model_dump`.
    A class-based decorator that defines `model_dump` would otherwise be canonicalized as its
    dump, which records neither its code nor the state its `__call__` reads, and a partial
    argument holding it would be accepted with the class left out."""
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R6_DUMP_HELPER)
    _write(module_dir / "rk5_s.py", _R5_FORMS[form])
    with pytest.raises(exceptions.StageDefinitionError, match="rk5_h.TracedDump"):
        fingerprint.get_stage_fingerprint(_fresh(*_R5_MODULES).my_stage)


_R6_LIBRARY = """\
class Memo:
    def __init__(self, fn):
        self.__wrapped__ = fn

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)
"""

_R6_LIBRARY_HELPER = """\
from pivot import fingerprint_by_name
from rk5_deco import Traced
from rk6_lib import Memo


@Memo
@Traced
@fingerprint_by_name
def helper(*_args):
    return 5
"""


@pytest.mark.parametrize("form", list(_R6_FORMS))
def test_a_library_wrapper_over_a_class_decorator_over_the_mark_is_refused(
    module_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch, form: str
) -> None:
    """A library's class-based wrapper that sets `__wrapped__` keeps its own module, so it is
    not user code. Over a class-based decorator over the mark the chain is still refused
    however the stage reaches it, a partial's function included, rather than named as a
    library callable with the class's code left out. Without the mark, main refuses the
    wrapper as a callable instance, so the mark must not relax that refusal."""
    library = module_dir / "site-packages"
    _write(library / "rk6_lib.py", _R6_LIBRARY)
    monkeypatch.syspath_prepend(str(library))
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R6_LIBRARY_HELPER)
    _write(module_dir / "rk5_s.py", _R6_FORMS[form])
    stage = _fresh("rk6_lib", *_R5_MODULES).my_stage
    assert not fingerprint.is_user_code(sys.modules["rk6_lib"].Memo)
    with pytest.raises(exceptions.StageDefinitionError) as refused:
        fingerprint.get_stage_fingerprint(stage)
    message = str(refused.value)
    assert "rk5_deco.Traced" in message, message
    assert "rk5_h.helper" in message, message


_R6_BOUND_HELPER = """\
from pivot import fingerprint_by_name


@fingerprint_by_name
def member(cls):
    return cls.__name__


class Helpers:
    run = classmethod(member)
"""

_R6_BOUND_STAGE = """\
from {module} import Helpers


class Holder:
    held = Helpers.run


def my_stage():
    holder = Holder()
    return holder.held()
"""


def test_a_marked_function_bound_to_a_class_and_held_by_a_class_is_recorded_by_name(
    module_dir: pathlib.Path,
) -> None:
    """A class attribute holds a marked function bound to a class (`Helpers.run`, a
    classmethod), and the stage reads the holder without naming the member. The member takes
    the mark decision, which records the bound method by name, so the member is in the manifest
    and a move of the helper to another module changes the stage fingerprint. An unwrap of the
    bound method finds no `__wrapped__` and would drop it."""
    _write(module_dir / "rk6_bound1.py", _R6_BOUND_HELPER)
    _write(module_dir / "rk6_bound_s.py", _R6_BOUND_STAGE.format(module="rk6_bound1"))
    first = fingerprint.get_stage_fingerprint(_fresh("rk6_bound1", "rk6_bound_s").my_stage)
    entry = "ext:rk6_bound_s.Holder.held"
    assert first[entry] == "rk6_bound1.member[rk6_bound1.Helpers]", first

    _write(module_dir / "rk6_bound2.py", _R6_BOUND_HELPER)
    _write(module_dir / "rk6_bound_s.py", _R6_BOUND_STAGE.format(module="rk6_bound2"))
    # The rewritten stage file keeps its size and mtime second, so drop its bytecode.
    shutil.rmtree(module_dir / "__pycache__", ignore_errors=True)
    second = fingerprint.get_stage_fingerprint(_fresh("rk6_bound2", "rk6_bound_s").my_stage)
    assert second[entry] == "rk6_bound2.member[rk6_bound2.Helpers]", second
    assert first != second, "a move of the marked member kept the stage fingerprint"


@pytest.mark.parametrize("shape", list(_R6_SHAPES))
def test_get_stage_fingerprint_given_the_refused_chain_itself_refuses_it(
    module_dir: pathlib.Path, shape: str
) -> None:
    """Given the helper itself, a chain with a user class instance above the mark,
    `get_stage_fingerprint` refuses it with the class-above-mark message. The source reads
    (`_source_object`) take the mark decision first, so none of them stops at the instance and
    asks it for a code object it does not have."""
    decorators, verb, cls = _R6_SHAPES[shape]
    _write(module_dir / "rk5_deco.py", _R5_DECO)
    _write(module_dir / "rk5_h.py", _R5_HELPER.format(decorators=decorators, verb=verb))
    helper = _fresh("rk5_deco", "rk5_h").helper
    with pytest.raises(exceptions.StageDefinitionError) as refused:
        fingerprint.get_stage_fingerprint(helper)
    message = str(refused.value)
    assert f"rk5_deco.{cls}" in message, message
    assert "@fingerprint_by_name" in message, message
