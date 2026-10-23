import dataclasses
import enum
import functools
import importlib.machinery
import json
import logging
import math
import os
import pathlib
import re
import subprocess
import sys
import types
import typing
from collections.abc import Callable
from typing import ClassVar
from unittest import mock

import numpy as np
import pandas as pd
import pydantic
import pytest
from tests.conftest import init_git_repo

from pivot import exceptions, fingerprint, run_history
from pivot.config import models
from pivot.storage import state


@dataclasses.dataclass
class MutableConfig:
    value: int


@dataclasses.dataclass(frozen=True)
class FrozenConfig:
    value: int


class MutableModel(pydantic.BaseModel):
    value: int


class FrozenModel(pydantic.BaseModel):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)
    value: int


class CapturedMode(enum.Enum):
    DEFAULT = "default"


class CallableConfig:
    def __call__(self) -> int:
        return 7


MUTABLE_DICT = {"a": 1}
MUTABLE_LIST = [1, 2]
MUTABLE_SET = {1, 2}
MUTABLE_DATACLASS = MutableConfig(value=1)
MUTABLE_PYDANTIC = MutableModel(value=1)
FROZEN_DATACLASS = FrozenConfig(value=1)
FROZEN_PYDANTIC = FrozenModel(value=1)
IMMUTABLE_TUPLE = (1, 2)
IMMUTABLE_FROZENSET = frozenset({1, 2})
PRIMITIVE_INT = 42
ENUM_MEMBER = CapturedMode.DEFAULT
CALLABLE_INSTANCE = CallableConfig()


class Scaler:
    factor: int

    def __init__(self, factor: int) -> None:
        self.factor = factor

    def apply(self, x: int) -> int:
        return x * self.factor


BOUND_METHOD = Scaler(3).apply
SEEDED_NORMAL = np.random.default_rng(0).normal
LOGGER_INFO = logging.getLogger(__name__).info
COMMA_JOIN = ",".join
LIBRARY_NORMAL = np.random.normal


class _ExternalCallable:
    __module__ = "math"

    def __call__(self) -> int:
        return 7


EXTERNAL_CALLABLE = _ExternalCallable()
COMPILED_PATTERN = re.compile("x")
BYTES_PATTERN = re.compile(rb"\x00+", re.MULTILINE)
LONG_PATTERN = re.compile("a" * 300)
STR_TUPLE = ("patchable", "squash")
STR_FROZENSET = frozenset({"tasks", "runs", "models"})
TUPLE_OF_TUPLE = (("a",),)
FROZENSET_OF_TUPLE = frozenset({("a",)})
TUPLE_OF_OBJECT = (FROZEN_DATACLASS,)
NUMPY_LOG = np.log  # What `from numpy import log` binds: a ufunc, a callable instance.


def _scaled(value: float) -> float:
    return value * 2


# np.frompyfunc products carry no `__module__`, like scipy.special's ufuncs (`expit`).
MODULELESS_UFUNC = np.frompyfunc(_scaled, 1, 1)


def _stage_uses_moduleless_ufunc() -> object:
    return MODULELESS_UFUNC(1.0)


def _install_library_module(monkeypatch: pytest.MonkeyPatch, binds: dict[str, object]) -> None:
    """Register a third-party module (it lives under site-packages) that binds `binds`."""
    module = types.ModuleType("fake_special")
    module.__file__ = "/opt/venv/lib/python3.13/site-packages/fake_special/_ufuncs.py"
    module.__spec__ = importlib.machinery.ModuleSpec("fake_special", loader=None)
    vars(module).update(binds)
    monkeypatch.setitem(sys.modules, "fake_special", module)


def _callable_helper() -> int:
    return 7


def _stage_uses_mutable_dict() -> int:
    return MUTABLE_DICT["a"]


def _stage_uses_mutable_list() -> int:
    return len(MUTABLE_LIST)


def _stage_uses_mutable_set() -> int:
    return len(MUTABLE_SET)


def _stage_uses_mutable_dataclass() -> int:
    return MUTABLE_DATACLASS.value


def _stage_uses_mutable_pydantic() -> int:
    return MUTABLE_PYDANTIC.value


def _stage_uses_frozen_dataclass() -> int:
    return FROZEN_DATACLASS.value


def _stage_uses_frozen_pydantic() -> int:
    return FROZEN_PYDANTIC.value


def _stage_uses_tuple() -> int:
    return len(IMMUTABLE_TUPLE)


def _stage_uses_frozenset() -> int:
    return len(IMMUTABLE_FROZENSET)


def _stage_uses_primitive() -> int:
    return PRIMITIVE_INT


def _stage_uses_callable() -> int:
    return _callable_helper()


def _stage_uses_enum_member() -> str:
    return ENUM_MEMBER.value


def _stage_uses_callable_instance() -> int:
    return CALLABLE_INSTANCE()


def _stage_uses_bound_method() -> int:
    return BOUND_METHOD(1)


def _stage_uses_seeded_normal() -> float:
    return float(SEEDED_NORMAL())


def _make_stage_with_bound_method_cell() -> Callable[[], int]:
    cell_method = Scaler(4).apply

    def stage() -> int:
        return cell_method(1)

    return stage


def _apply_each(method: Callable[[int], int], items: list[int]) -> int:
    return sum(method(item) for item in items)


_PARTIAL_OVER_METHOD = functools.partial(_apply_each, Scaler(5).apply)


def _stage_uses_partial_over_bound_method() -> int:
    return _PARTIAL_OVER_METHOD([1])


def _stage_uses_stateless_bound_methods() -> int:
    LOGGER_INFO("x")
    return len(COMMA_JOIN(["a"])) + int(LIBRARY_NORMAL() * 0)


def _stage_uses_three_non_scalar_values() -> int:
    return len(MUTABLE_LIST) + len(MUTABLE_SET) + FROZEN_DATACLASS.value


CHILD_BAD: list[int] = []


def _child_with_non_scalar_capture() -> int:
    return len(CHILD_BAD)


def _make_parent_with_nested_non_scalar_capture() -> Callable[[], int]:
    PARENT_BAD = list[int]()

    def parent() -> int:
        return _child_with_non_scalar_capture() + len(PARENT_BAD)

    return parent


def _decorator(func: Callable[[], int]) -> Callable[[], int]:
    @functools.wraps(func)
    def wrapper() -> int:
        return func()

    return wrapper


@_decorator
def _decorated_user_function() -> int:
    return 1


def _stage_uses_common_callables() -> int:
    frame = pd.DataFrame({"value": [1, 2]})
    return (
        int(np.mean(frame["value"]))
        + len(json.dumps({"value": 1}))
        + int(math.sqrt(4))
        + typing.cast("int", 1)
        + int(NUMPY_LOG(1))
        + _decorated_user_function()
    )


def _stage_uses_external_callable() -> int:
    return EXTERNAL_CALLABLE()


def _stage_uses_compiled_pattern() -> bool:
    return COMPILED_PATTERN.search("x") is not None


def _stage_uses_bytes_pattern() -> bool:
    return BYTES_PATTERN.search(b"\x00") is not None


def _stage_uses_long_pattern() -> bool:
    return LONG_PATTERN.search("a") is not None


def _stage_uses_str_tuple() -> int:
    return len(STR_TUPLE)


def _stage_uses_str_frozenset() -> int:
    return len(STR_FROZENSET)


def _stage_uses_tuple_of_tuple() -> int:
    return len(TUPLE_OF_TUPLE)


def _stage_uses_frozenset_of_tuple() -> int:
    return len(FROZENSET_OF_TUPLE)


def _stage_uses_tuple_of_object() -> int:
    return len(TUPLE_OF_OBJECT)


@pytest.mark.parametrize(
    ("func", "var_name", "type_name"),
    [
        pytest.param(
            _stage_uses_mutable_dict,
            "MUTABLE_DICT",
            "dict",
            id="mutable-dict",
        ),
        pytest.param(
            _stage_uses_mutable_list,
            "MUTABLE_LIST",
            "list",
            id="mutable-list",
        ),
        pytest.param(
            _stage_uses_mutable_set,
            "MUTABLE_SET",
            "set",
            id="mutable-set",
        ),
        pytest.param(
            _stage_uses_mutable_dataclass,
            "MUTABLE_DATACLASS",
            "MutableConfig",
            id="mutable-dataclass",
        ),
        pytest.param(
            _stage_uses_mutable_pydantic,
            "MUTABLE_PYDANTIC",
            "MutableModel",
            id="mutable-pydantic",
        ),
        pytest.param(
            _stage_uses_frozen_dataclass,
            "FROZEN_DATACLASS",
            "FrozenConfig",
            id="frozen-dataclass",
        ),
        pytest.param(
            _stage_uses_frozen_pydantic,
            "FROZEN_PYDANTIC",
            "FrozenModel",
            id="frozen-pydantic",
        ),
        pytest.param(_stage_uses_tuple_of_tuple, "TUPLE_OF_TUPLE", "tuple", id="tuple-of-tuple"),
        pytest.param(
            _stage_uses_frozenset_of_tuple,
            "FROZENSET_OF_TUPLE",
            "frozenset",
            id="frozenset-of-tuple",
        ),
        pytest.param(_stage_uses_tuple_of_object, "TUPLE_OF_OBJECT", "tuple", id="tuple-of-object"),
        pytest.param(_stage_uses_enum_member, "ENUM_MEMBER", "CapturedMode", id="enum-member"),
        pytest.param(
            _stage_uses_callable_instance,
            "CALLABLE_INSTANCE",
            "CallableConfig",
            id="callable-instance",
        ),
        pytest.param(_stage_uses_bound_method, "BOUND_METHOD", "method", id="bound-method"),
        pytest.param(_stage_uses_seeded_normal, "SEEDED_NORMAL", "method", id="seeded-generator"),
        pytest.param(
            _make_stage_with_bound_method_cell(), "cell_method", "method", id="bound-method-cell"
        ),
        pytest.param(
            _stage_uses_partial_over_bound_method,
            "_PARTIAL_OVER_METHOD.args[0]",
            "method",
            id="bound-method-in-partial",
        ),
    ],
)
def test_non_scalar_closure_capture_raises(
    func: Callable[[], int],
    var_name: str,
    type_name: str,
) -> None:
    with pytest.raises(exceptions.StageDefinitionError) as exc:
        fingerprint.get_stage_fingerprint_cached("train", func)

    message = str(exc.value)
    assert "Stage 'train'" in message, "Should include stage name"
    assert var_name in message, "Should include captured variable name"
    assert f"type: {type_name}" in message, "Should include captured variable type"
    assert "Fix: move this value into the stage's StageParams" in message, (
        "Should include suggestion"
    )
    assert "PIVOT_UNSAFE_FINGERPRINTING=1" in message, "Should include suppression hint"


def test_non_scalar_capture_errors_are_aggregated() -> None:
    with pytest.raises(exceptions.StageDefinitionError) as exc:
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_three_non_scalar_values)

    message = str(exc.value)
    for variable in ("MUTABLE_LIST", "MUTABLE_SET", "FROZEN_DATACLASS"):
        assert variable in message


def test_nested_non_scalar_capture_errors_are_aggregated() -> None:
    with pytest.raises(exceptions.StageDefinitionError) as exc:
        fingerprint.get_stage_fingerprint_cached(
            "train",
            _make_parent_with_nested_non_scalar_capture(),
        )

    message = str(exc.value)
    assert "CHILD_BAD" in message
    assert "PARENT_BAD" in message


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(_stage_uses_primitive, id="primitive"),
        pytest.param(_stage_uses_callable, id="callable"),
    ],
)
def test_scalar_and_callable_closure_capture_allows_fingerprint(func: Callable[[], object]) -> None:
    fingerprint.get_stage_fingerprint_cached("train", func)


@pytest.mark.parametrize(
    ("func", "variable", "expected"),
    [
        pytest.param(_stage_uses_str_tuple, "STR_TUPLE", ["tuple", ["patchable", "squash"]]),
        pytest.param(_stage_uses_tuple, "IMMUTABLE_TUPLE", ["tuple", [1, 2]], id="int-tuple"),
        pytest.param(
            _stage_uses_str_frozenset,
            "STR_FROZENSET",
            ["frozenset", ["models", "runs", "tasks"]],
            id="str-frozenset",
        ),
        pytest.param(
            _stage_uses_compiled_pattern,
            "COMPILED_PATTERN",
            ["re.Pattern", "x", re.UNICODE],
            id="str-pattern",
        ),
        pytest.param(
            _stage_uses_bytes_pattern,
            "BYTES_PATTERN",
            ["re.Pattern", ["bytes", repr(rb"\x00+")], re.MULTILINE],
            id="bytes-pattern",
        ),
    ],
)
def test_immutable_capture_is_tracked_by_value(
    func: Callable[[], object], variable: str, expected: object
) -> None:
    manifest = fingerprint.get_stage_fingerprint_cached("train", func)

    assert manifest[f"const:{__name__}.{variable}"] == json.dumps(expected)


@pytest.mark.parametrize(
    ("func", "variable", "same", "edited"),
    [
        pytest.param(
            _stage_uses_str_tuple,
            "STR_TUPLE",
            ("patchable", "squash"),
            ("patchable", "squash", "extra"),
            id="tuple-member-added",
        ),
        pytest.param(
            _stage_uses_str_tuple,
            "STR_TUPLE",
            ("patchable", "squash"),
            ("squash", "patchable"),
            id="tuple-reordered",
        ),
        pytest.param(
            _stage_uses_str_frozenset,
            "STR_FROZENSET",
            frozenset({"runs", "models", "tasks"}),
            frozenset({"runs", "models", "jobs"}),
            id="frozenset-member-changed",
        ),
        pytest.param(
            _stage_uses_compiled_pattern,
            "COMPILED_PATTERN",
            re.compile("x"),
            re.compile("y"),
            id="pattern-text",
        ),
        pytest.param(
            _stage_uses_compiled_pattern,
            "COMPILED_PATTERN",
            re.compile("x"),
            re.compile("x", re.IGNORECASE),
            id="pattern-flags",
        ),
        pytest.param(
            _stage_uses_bytes_pattern,
            "BYTES_PATTERN",
            re.compile(rb"\x00+", re.MULTILINE),
            re.compile(rb"\x01+", re.MULTILINE),
            id="bytes-pattern-text",
        ),
        pytest.param(
            _stage_uses_long_pattern,
            "LONG_PATTERN",
            re.compile("a" * 300),
            re.compile("a" * 299 + "b"),
            id="pattern-past-repr-truncation",
        ),
    ],
)
def test_editing_immutable_capture_changes_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    func: Callable[[], object],
    variable: str,
    same: object,
    edited: object,
) -> None:
    fingerprint._getclosurevars_cache.clear()
    original = fingerprint.get_stage_fingerprint(func)

    monkeypatch.setattr(sys.modules[__name__], variable, same)
    fingerprint._getclosurevars_cache.clear()
    assert fingerprint.get_stage_fingerprint(func) == original, "an equal value is not a change"

    monkeypatch.setattr(sys.modules[__name__], variable, edited)
    fingerprint._getclosurevars_cache.clear()
    assert fingerprint.get_stage_fingerprint(func) != original


def test_common_function_and_class_references_are_allowed() -> None:
    fingerprint.get_stage_fingerprint(_stage_uses_common_callables)


def test_methods_bound_to_a_logger_scalar_or_library_are_allowed() -> None:
    """A Logger's, a scalar's or a library module's own bound method carries no stage state."""
    manifest = fingerprint.get_stage_fingerprint(_stage_uses_stateless_bound_methods)

    assert manifest[f"ext:{__name__}.COMMA_JOIN"] == "builtins.str.join[',']", manifest
    assert manifest[f"ext:{__name__}.LIBRARY_NORMAL"] == "numpy.random.RandomState.normal"


def test_nonlocal_stdlib_module_is_allowed() -> None:
    import time

    def stage() -> None:
        time.sleep(0)

    assert "self:stage" in fingerprint.get_stage_fingerprint(stage)


def test_moduleless_ufunc_exported_by_library_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`from scipy.special import expit`: the library binds the ufunc under its own name."""
    _install_library_module(monkeypatch, {MODULELESS_UFUNC.__name__: MODULELESS_UFUNC})

    assert "self:_stage_uses_moduleless_ufunc" in fingerprint.get_stage_fingerprint(
        _stage_uses_moduleless_ufunc
    )


def test_moduleless_ufunc_bound_under_another_name_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A library binding the object under a different name does not make it the library's."""
    _install_library_module(monkeypatch, {"alias": MODULELESS_UFUNC})

    with pytest.raises(exceptions.StageDefinitionError, match="MODULELESS_UFUNC"):
        fingerprint.get_stage_fingerprint(_stage_uses_moduleless_ufunc)


@pytest.mark.parametrize(
    ("func", "variable", "type_name"),
    [
        pytest.param(
            _stage_uses_external_callable,
            "EXTERNAL_CALLABLE",
            "_ExternalCallable",
            id="external-callable-instance",
        ),
        pytest.param(
            _stage_uses_moduleless_ufunc,
            "MODULELESS_UFUNC",
            "ufunc",
            id="frompyfunc-of-user-function",
        ),
    ],
)
def test_non_scalar_non_user_values_are_refused(
    func: Callable[[], object],
    variable: str,
    type_name: str,
) -> None:
    with pytest.raises(exceptions.StageDefinitionError) as exc:
        fingerprint.get_stage_fingerprint(func)

    message = str(exc.value)
    assert variable in message
    assert f"type: {type_name}" in message


def test_unsafe_env_allows_non_scalar_capture(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PIVOT_UNSAFE_FINGERPRINTING", "1")
    with caplog.at_level(logging.WARNING):
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_mutable_list)

    assert any("non-scalar variable" in record.message for record in caplog.records), (
        "Should warn when unsafe fingerprinting is enabled"
    )


def test_unsafe_capture_does_not_populate_manifest_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = dict[str, dict[str, str]]()

    class ManifestWrites(list[tuple[bytes, bytes]]):
        @typing.override
        def append(self, entry: tuple[bytes, bytes]) -> None:
            cached["unsafe-cache"] = json.loads(entry[1])["m"]

    def cache_hit(stage_name: str, _func: Callable[..., object]) -> dict[str, str] | None:
        return cached.get(stage_name)

    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", ManifestWrites())
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", cache_hit)
    monkeypatch.setenv("PIVOT_UNSAFE_FINGERPRINTING", "1")
    fingerprint.get_stage_fingerprint_cached("unsafe-cache", _stage_uses_mutable_list)

    assert "unsafe-cache" not in cached

    monkeypatch.delenv("PIVOT_UNSAFE_FINGERPRINTING")
    with pytest.raises(exceptions.StageDefinitionError):
        fingerprint.get_stage_fingerprint_cached("unsafe-cache", _stage_uses_mutable_list)


NESTED_MUTABLE_TUPLE = (1, [2, 3])


def _stage_uses_nested_mutable_tuple() -> int:
    return len(NESTED_MUTABLE_TUPLE)


def _stage_uses_mutable_dict_for_config_test() -> int:
    return MUTABLE_DICT["a"]


def test_unsafe_config_allows_non_scalar_capture(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Config-based unsafe_fingerprinting=true downgrades errors to warnings."""
    unsafe_config = models.PivotConfig.get_default()
    unsafe_config.core.unsafe_fingerprinting = True

    with (
        mock.patch("pivot.config.io.get_merged_config", autospec=True, return_value=unsafe_config),
        caplog.at_level(logging.WARNING),
    ):
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_mutable_set)

    assert any("non-scalar variable" in record.message for record in caplog.records), (
        "Should warn when unsafe fingerprinting via config is enabled"
    )


def test_nested_mutable_in_tuple_raises() -> None:
    with pytest.raises(exceptions.StageDefinitionError, match="NESTED_MUTABLE_TUPLE"):
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_nested_mutable_tuple)


DEFAULT_COLUMNS = ["a"]
DEFAULT_FACTOR = 2
DEFAULT_TUPLE = ("patchable", "squash")


def _stage_with_list_default(cols: list[str] = DEFAULT_COLUMNS) -> list[str]:
    return cols


def _stage_with_keyword_list_default(*, cols: list[str] = DEFAULT_COLUMNS) -> list[str]:
    return cols


def _stage_with_literal_defaults(
    factor: int = 3, keys: tuple[str, ...] = ("a", "b"), *, limit: float = 0.5
) -> int:
    return factor + len(keys) + int(limit)


def _stage_with_scalar_module_default(factor: int = DEFAULT_FACTOR) -> int:
    return factor


def _stage_with_tuple_module_default(keys: tuple[str, ...] = DEFAULT_TUPLE) -> int:
    return len(keys)


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(_stage_with_list_default, id="positional"),
        pytest.param(_stage_with_keyword_list_default, id="keyword-only"),
    ],
)
def test_non_scalar_module_value_as_default_raises(func: Callable[..., object]) -> None:
    with pytest.raises(exceptions.StageDefinitionError) as exc:
        fingerprint.get_stage_fingerprint(func)

    message = str(exc.value)
    assert "captures non-scalar variable 'DEFAULT_COLUMNS'" in message
    assert "as the default of parameter 'cols'" in message


def test_literal_defaults_are_allowed() -> None:
    manifest = fingerprint.get_stage_fingerprint(_stage_with_literal_defaults)

    ref = f"{__name__}._stage_with_literal_defaults"
    assert manifest[f"default:{ref}.factor"] == "3"
    assert manifest[f"default:{ref}.limit"] == "0.5"
    assert f"default:{ref}.keys" not in manifest, "a literal default is covered by the AST hash"


def test_scalar_module_default_is_tracked_by_value() -> None:
    manifest = fingerprint.get_stage_fingerprint(_stage_with_scalar_module_default)

    assert manifest[f"default:{__name__}._stage_with_scalar_module_default.factor"] == "2"


def test_tuple_module_default_is_tracked_by_value() -> None:
    manifest = fingerprint.get_stage_fingerprint(_stage_with_tuple_module_default)

    key = f"default:{__name__}._stage_with_tuple_module_default.keys"
    assert manifest[key] == json.dumps(["tuple", ["patchable", "squash"]])


def test_wrapped_function_default_is_checked_by_value() -> None:
    """A `functools.wraps` wrapper is hashed as bytecode, which holds no default values."""

    def _wrap(func: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(func)
        def wrapper(keys: list[str] = DEFAULT_COLUMNS) -> object:
            return func(keys)

        return wrapper

    wrapped = _wrap(_stage_with_list_default)

    with pytest.raises(exceptions.StageDefinitionError, match="parameter 'keys'"):
        fingerprint.get_stage_fingerprint(wrapped)


@functools.cache
def _cached_helper_with_list_default(cols: list[str] = DEFAULT_COLUMNS) -> int:
    return len(cols)


@functools.lru_cache(maxsize=2)
def _cached_helper_reading_list() -> int:
    return len(DEFAULT_COLUMNS)


def _stage_calls_cached_helpers() -> int:
    return _cached_helper_with_list_default() + _cached_helper_reading_list()


def test_cached_helper_globals_and_defaults_are_checked() -> None:
    """A C-level cache wrapper hides the helper from getclosurevars; its function is walked."""
    with pytest.raises(exceptions.StageDefinitionError) as exc:
        fingerprint.get_stage_fingerprint(_stage_calls_cached_helpers)

    message = str(exc.value)
    assert "as the default of parameter 'cols'" in message
    assert "captures non-scalar variable 'DEFAULT_COLUMNS' (type: list)" in message


_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")

_UNSAFE_LIST_PIPELINE = """\
from pivot import Pipeline, stage

SUFFIXES = ["!"]


@stage
def shout(text: str) -> str:
    return text.strip().upper() + SUFFIXES[0]


with Pipeline("u") as pipeline:
    shout(pipeline.input("input.txt", t=str))
"""

_PARAMS_LIST_PIPELINE = """\
from pivot import Pipeline, StageParams, stage


class ShoutParams(StageParams):
    suffixes: list[str] = ["!"]


@stage
def shout(text: str, params: ShoutParams) -> str:
    return text.strip().upper() + params.suffixes[0]


with Pipeline("u") as pipeline:
    shout(pipeline.input("input.txt", t=str), params=ShoutParams())
"""


def _helper_pivot_ok(
    cwd: pathlib.Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    full_env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING")
    }
    full_env.update(env or {})
    proc = subprocess.run(
        [_PIVOT_BIN, *args],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"pivot {' '.join(args)} failed:\n{proc.stdout}{proc.stderr}"
    return proc


def _helper_shout_project(root: pathlib.Path, pipeline_source: str) -> pathlib.Path:
    init_git_repo(root)
    _helper_pivot_ok(root, "init")
    (root / "pipeline.py").write_text(pipeline_source)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "input.txt").write_text("hello\n")
    return root


def _helper_run_cache_entries(
    root: pathlib.Path, stage_name: str
) -> list[run_history.RunCacheEntry]:
    """Every run-cache entry filed for ``stage_name``, under any input hash."""
    prefix = f"runcache:{stage_name}:".encode()
    with state.StateDB(root / ".pivot", readonly=True) as db:
        raw_entries = db.iter_prefix(prefix)
    return [run_history.deserialize_run_cache_entry(value) for _key, value in raw_entries]


@pytest.mark.parametrize(
    ("command", "switch"),
    [
        pytest.param("repro", "config", id="repro-config"),
        pytest.param("repro", "env", id="repro-env"),
        pytest.param("commit", "config", id="commit-config"),
    ],
)
def test_unsafe_result_is_kept_out_of_the_run_cache(
    tmp_path: pathlib.Path, command: str, switch: str
) -> None:
    """A result fingerprinted with a refusal downgraded to a warning is never filed in the run
    cache, by a worker (`pivot repro`) or by `pivot commit`: the run cache would hand it to the
    next run whose fingerprint matches, though the captured list may hold other values."""
    root = _helper_shout_project(tmp_path, _UNSAFE_LIST_PIPELINE)
    env = dict[str, str]()
    if switch == "config":
        (root / ".pivot" / "config.yaml").write_text("core:\n  unsafe_fingerprinting: true\n")
    else:
        env["PIVOT_UNSAFE_FINGERPRINTING"] = "1"
    if command == "commit":
        _helper_pivot_ok(root, "repro", "--no-commit", env=env)
    _helper_pivot_ok(root, command, env=env)

    output = root / "data" / "u" / "shout.txt"
    assert output.read_text() == "HELLO!", "the unsafe stage still runs and writes its output"
    entries = _helper_run_cache_entries(root, "u/shout")
    assert entries == [], f"an unsafe result reached the run cache: {entries}"


def test_safe_result_is_filed_in_the_run_cache(tmp_path: pathlib.Path) -> None:
    """Control: the same stage with its list in StageParams, the flag off, files one entry."""
    root = _helper_shout_project(tmp_path, _PARAMS_LIST_PIPELINE)
    _helper_pivot_ok(root, "repro")

    assert (root / "data" / "u" / "shout.txt").read_text() == "HELLO!", "the stage ran"
    entries = _helper_run_cache_entries(root, "u/shout")
    assert len(entries) == 1, f"a safe result is filed once in the run cache: {entries}"


def test_config_set_accepts_unsafe_fingerprinting(tmp_path: pathlib.Path) -> None:
    """The documented switch for unsafe fingerprinting can be set from the CLI."""
    root = _helper_shout_project(tmp_path, _UNSAFE_LIST_PIPELINE)
    _helper_pivot_ok(root, "config", "set", "core.unsafe_fingerprinting", "true")
    got = _helper_pivot_ok(root, "config", "get", "core.unsafe_fingerprinting", "--json")
    assert json.loads(got.stdout)["value"] is True, got.stdout
