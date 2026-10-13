import dataclasses
import enum
import functools
import json
import logging
import math
import re
import typing
from collections.abc import Callable
from typing import ClassVar
from unittest import mock

import pydantic
import pytest

import numpy as np
import pandas as pd

from pivot import exceptions, fingerprint
from pivot.config import models


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


class _ExternalCallable:
    __module__ = "math"

    def __call__(self) -> int:
        return 7


EXTERNAL_CALLABLE = _ExternalCallable()
COMPILED_PATTERN = re.compile("x")


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


def _stage_uses_three_non_scalar_values() -> int:
    return len(IMMUTABLE_TUPLE) + len(IMMUTABLE_FROZENSET) + FROZEN_DATACLASS.value


CHILD_BAD = []


def _child_with_non_scalar_capture() -> int:
    return len(CHILD_BAD)


def _make_parent_with_nested_non_scalar_capture() -> Callable[[], int]:
    PARENT_BAD = ()

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
        + typing.cast(int, 1)
        + _decorated_user_function()
    )


def _stage_uses_external_callable() -> int:
    return EXTERNAL_CALLABLE()


def _stage_uses_compiled_pattern() -> bool:
    return COMPILED_PATTERN.search("x") is not None


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
        pytest.param(_stage_uses_tuple, "IMMUTABLE_TUPLE", "tuple", id="tuple"),
        pytest.param(_stage_uses_frozenset, "IMMUTABLE_FROZENSET", "frozenset", id="frozenset"),
        pytest.param(_stage_uses_enum_member, "ENUM_MEMBER", "CapturedMode", id="enum-member"),
        pytest.param(
            _stage_uses_callable_instance,
            "CALLABLE_INSTANCE",
            "CallableConfig",
            id="callable-instance",
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
    assert "Fix: pass this data via StageParams" in message, "Should include suggestion"
    assert "PIVOT_UNSAFE_FINGERPRINTING=1" in message, "Should include suppression hint"


def test_non_scalar_capture_errors_are_aggregated() -> None:
    with pytest.raises(exceptions.StageDefinitionError) as exc:
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_three_non_scalar_values)

    message = str(exc.value)
    for variable in ("IMMUTABLE_TUPLE", "IMMUTABLE_FROZENSET", "FROZEN_DATACLASS"):
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


def test_common_function_and_class_references_are_allowed() -> None:
    fingerprint.get_stage_fingerprint(_stage_uses_common_callables)


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
            _stage_uses_compiled_pattern,
            "COMPILED_PATTERN",
            "Pattern",
            id="compiled-regex",
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
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_tuple)

    assert any("non-scalar variable" in record.message for record in caplog.records), (
        "Should warn when unsafe fingerprinting is enabled"
    )


def test_unsafe_capture_does_not_populate_manifest_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = dict[str, dict[str, str]]()

    class ManifestWrites(list[tuple[bytes, bytes]]):
        def append(self, entry: tuple[bytes, bytes]) -> None:
            cached["unsafe-cache"] = json.loads(entry[1])["m"]

    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", ManifestWrites())
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda stage_name: cached.get(stage_name))
    monkeypatch.setenv("PIVOT_UNSAFE_FINGERPRINTING", "1")
    fingerprint.get_stage_fingerprint_cached("unsafe-cache", _stage_uses_tuple)

    assert "unsafe-cache" not in cached

    monkeypatch.delenv("PIVOT_UNSAFE_FINGERPRINTING")
    with pytest.raises(exceptions.StageDefinitionError):
        fingerprint.get_stage_fingerprint_cached("unsafe-cache", _stage_uses_tuple)


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
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_frozenset)

    assert any("non-scalar variable" in record.message for record in caplog.records), (
        "Should warn when unsafe fingerprinting via config is enabled"
    )


def test_nested_mutable_in_tuple_raises() -> None:
    with pytest.raises(exceptions.StageDefinitionError, match="NESTED_MUTABLE_TUPLE"):
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_nested_mutable_tuple)
