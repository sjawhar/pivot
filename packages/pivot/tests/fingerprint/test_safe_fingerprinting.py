import dataclasses
import logging
from collections.abc import Callable
from typing import ClassVar
from unittest import mock

import pydantic
import pytest

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
    ],
)
def test_mutable_closure_capture_raises(
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


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(_stage_uses_frozen_dataclass, id="frozen-dataclass"),
        pytest.param(_stage_uses_frozen_pydantic, id="frozen-pydantic"),
        pytest.param(_stage_uses_tuple, id="tuple"),
        pytest.param(_stage_uses_frozenset, id="frozenset"),
        pytest.param(_stage_uses_primitive, id="primitive"),
        pytest.param(_stage_uses_callable, id="callable"),
    ],
)
def test_immutable_closure_capture_allows_fingerprint(func: Callable[[], int]) -> None:
    fingerprint.get_stage_fingerprint_cached("train", func)


def test_unsafe_env_allows_mutable_capture(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PIVOT_UNSAFE_FINGERPRINTING", "1")
    with caplog.at_level(logging.WARNING):
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_mutable_dict)

    assert any(
        "closure captures mutable variable" in record.message for record in caplog.records
    ), "Should warn when unsafe fingerprinting is enabled"


NESTED_MUTABLE_TUPLE = (1, [2, 3])


def _stage_uses_nested_mutable_tuple() -> int:
    return len(NESTED_MUTABLE_TUPLE)


def _stage_uses_mutable_dict_for_config_test() -> int:
    return MUTABLE_DICT["a"]


def test_unsafe_config_allows_mutable_capture(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Config-based unsafe_fingerprinting=true downgrades errors to warnings."""
    unsafe_config = models.PivotConfig.get_default()
    unsafe_config.core.unsafe_fingerprinting = True

    with (
        mock.patch("pivot.config.io.get_merged_config", autospec=True, return_value=unsafe_config),
        caplog.at_level(logging.WARNING),
    ):
        fingerprint.get_stage_fingerprint_cached("train", _stage_uses_mutable_dict_for_config_test)

    assert any(
        "closure captures mutable variable" in record.message for record in caplog.records
    ), "Should warn when unsafe fingerprinting via config is enabled"


def test_nested_mutable_in_tuple_allows_fingerprint() -> None:
    """Tuple containing mutable list is allowed — tuples are immutable at top level."""
    fingerprint.get_stage_fingerprint_cached("train", _stage_uses_nested_mutable_tuple)
