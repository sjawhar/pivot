# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false, reportUnusedCallResult=false, reportAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportReturnType=false, reportUnusedImport=false, reportInvalidTypeForm=false, reportImplicitStringConcatenation=false, reportUnnecessaryTypeIgnoreComment=false, reportArgumentType=false

"""Tests for @fingerprint_by_name: a helper recorded by name only."""

import ast
import functools
import importlib
import json
import linecache
import os
import pathlib
import shutil
import subprocess
import sys
import types

import pytest
import yaml

from pivot import decorators, exceptions, fingerprint, fingerprint_by_name  # noqa: F401

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
        name for name in sys.modules if name.startswith(("fbn_", "fbn2_", "fbnhelp_", "fbnstage_"))
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

    Kills mutant m6 (ext: entry drops the module).
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

    Kills mutant m8: the lambda check is separate from the nested-function check.
    A module-level lambda has qualname '<lambda>' (no '<locals>').
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

    Kills mutant m9.
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

    Kills mutant m9.
    """
    with pytest.raises(TypeError, match="requires a function"):
        decorators.fingerprint_by_name(len)  # pyright: ignore[reportArgumentType]


def test_callable_instance_is_refused() -> None:
    """@fingerprint_by_name refuses a callable instance.

    Kills mutant m9.
    """

    class CallableObj:
        def __call__(self):
            return 1

    with pytest.raises(TypeError, match="requires a function"):
        decorators.fingerprint_by_name(CallableObj())


def test_method_is_refused(module_dir: pathlib.Path) -> None:
    """@fingerprint_by_name refuses a method defined in a class body.

    Kills C1: qualname != name for methods.
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

    Kills R1: the mark must not spread through __dict__ copying.
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

    assert fp1 != fp2, "Wrapper body change must change fingerprint (R1)"


# ---------------------------------------------------------------------------
# R2: Marked helpers as class attributes
# ---------------------------------------------------------------------------


def test_marked_contextmanager_as_class_attribute_cached(
    module_dir: pathlib.Path,
) -> None:
    """A marked @contextmanager helper held as a class attribute: body change → cached.

    Kills R2: _user_member_value must check fingerprinted_by_name before unwrapping.
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

    assert fp1 == fp2, "Marked CM as class attribute: body change must stay cached (R2)"


def test_marked_cache_as_class_attribute_cached(module_dir: pathlib.Path) -> None:
    """A marked @functools.cache helper held as a class attribute: body change → cached.

    Kills R2.
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

    assert fp1 == fp2, "Marked cache as class attribute: body change must stay cached (R2)"


# ---------------------------------------------------------------------------
# m2: identity keeps the body hash
# ---------------------------------------------------------------------------


def test_marked_helper_as_partial_argument_body_change_cached(
    module_dir: pathlib.Path,
) -> None:
    """A marked helper bound as a functools.partial argument: body change → cached.

    Kills mutant m2 (_function_identity keeps body hash for marked function).
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

    assert fp1 == fp2, "Marked helper as partial arg: body change must stay cached (m2)"


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
# Pickling: pickle-by-value decision unchanged
# ---------------------------------------------------------------------------


def test_pickle_by_value_decision_unchanged_for_marked_helper(
    module_dir: pathlib.Path,
) -> None:
    """The pickle-by-value decision for a marked helper's module is the same
    with and without the mark. It fails if the decision becomes per function.
    """
    import cloudpickle

    _write(
        module_dir / "fbnhelp_pbv.py",
        "from pivot import fingerprint_by_name\n\n"
        "@fingerprint_by_name\n"
        "def helper():\n"
        "    return 'test'\n",
    )
    _import_fresh("fbnhelp_pbv")
    marked_registered = "fbnhelp_pbv" in cloudpickle.list_registry_pickle_by_value()

    _write(
        module_dir / "fbnhelp_pbv.py",
        "def helper():\n    return 'test'\n",
    )
    _import_fresh("fbnhelp_pbv")
    unmarked_registered = "fbnhelp_pbv" in cloudpickle.list_registry_pickle_by_value()

    assert marked_registered == unmarked_registered, (
        "pickle-by-value decision must be the same with and without the mark"
    )


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
# CLI acceptance tests (from PR512 tester)
# =============================================================================


_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_REPO = pathlib.Path(__file__).resolve().parents[4]
_UNCOVERED = "imported user modules its fingerprint does not cover"


def _run(project: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING")
    }
    return subprocess.run(
        [str(_PIVOT), *args], cwd=project, env=env, capture_output=True, text=True, timeout=300
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
    text = path.read_text()
    assert text.count(old) == 1, f"{path}: {old!r}"
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
    wrapper. The wrapper is unmarked user code, but the stage records it by name only: an edit to
    `doubled` changes the output, and the stage stays cached with the old output. Main records
    `func:lib.h.helper` and `func:lib.h.helper.<wrapper>` and reruns."""
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
    """At f54054fc each case loads: `pivot list` exits 0. For the plain method, the class walk
    then records the method by name only, so a change in what the method calls changes the
    output and the stage stays cached (head kept `5:17`, main reran and gave `6:17`). Each case
    must be refused when the module loads, with a message that names the decorator and the
    method."""
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
    module in its body, the stage warns and reruns on every run. The CLI does this at f54054fc,
    but the decorator's sections in docs/concepts/fingerprinting.md and
    skills/writing-pivot-stages/SKILL.md do not say it, and no test covered it. Each section must
    say that the stage reruns on every (or each) run."""
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
    """Main refuses this helper unmarked (`'helper' uses dynamic imports`). At f54054fc the mark
    skips that check: the stage runs, then warns and reruns on every run. The contract says a
    marked helper whose body imports a user module is refused. It must be refused, and the
    message must name the helper."""
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
    """The settled contract, the docstring and the fingerprinting guide say a marked helper is
    recorded by its module, qualified name and file. The manifest entry is
    `ext:<binding> = <module>.<qualname>` and holds no file: moving `lib/procs.py` to
    `lib/procs/__init__.py` leaves the stage cached. The text and the behavior must agree: either
    the move reruns the stage, or neither the docstring nor the guide says the file is recorded.
    Which of the two is a contract ruling; BET's helpers are outside the project root, where no
    project-relative file exists."""
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
    """`fingerprint._schema_hash` hashes a field's `default_factory` with `hash_function_ast`,
    which never asks `fingerprinted_by_name`. At f54054fc an edit to the marked factory's body
    that keeps its value reruns the stage as "Code changed". The value the factory returns is in
    the lock's params, so a value change still reruns the stage."""
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
# fresh tree with a fresh `.pivot/state.lmdb` on pivot-slices. The PR 512 tester recorded them.
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
    from a fresh `.pivot/state.lmdb`. `test_upgrade_preserves_fingerprint_schema` asserts only
    that the schema constant is 30 and that keys start with known prefixes, so a change to any
    unmarked manifest value passes it: the tester showed it passes with `fingerprinted_by_name`
    also true for every `functools.wraps` wrapper. This test compares whole manifests."""
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
