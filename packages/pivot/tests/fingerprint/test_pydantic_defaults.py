# pyright: reportUnusedFunction=false, reportUnusedParameter=false
"""Tests for fingerprinting behavior with Pydantic model schemas.

Pydantic model defaults in type hints are captured via schema hashing.
Changes to default values trigger cache invalidation.
"""

import pydantic

from pivot import fingerprint


def _definition_key(prefix: str, value: object) -> str:
    return f"{prefix}:{value.__module__}.{value.__qualname__}"  # type: ignore[attr-defined] - test types


# --- Scenario 1: Direct list constant as default ---

ITEMS_V1 = ["item1", "item2"]


class ParamsWithListDefault(pydantic.BaseModel):
    items: list[str] = ITEMS_V1


def _stage_with_pydantic_param_v1(params: ParamsWithListDefault) -> list[str]:
    """Stage function that receives a Pydantic model with list default."""
    return params.items


# --- Scenario 2: Pydantic model instances in list ---


class ItemConfig(pydantic.BaseModel):
    name: str
    value: int


CONFIGS_V1 = [
    ItemConfig(name="first", value=1),
    ItemConfig(name="second", value=2),
]


class ParamsWithConfigList(pydantic.BaseModel):
    configs: list[ItemConfig] = CONFIGS_V1


def _stage_with_config_list(params: ParamsWithConfigList) -> list[ItemConfig]:
    """Stage function with Pydantic model instances as default."""
    return params.configs


STRING_CONST = "hello"


def _stage_references_string() -> str:
    return STRING_CONST


def test_string_constants_are_captured():
    """String constants ARE captured in the fingerprint."""
    fp = fingerprint.get_stage_fingerprint(_stage_references_string)

    key = f"const:{__name__}.STRING_CONST"
    assert key in fp, f"String constants should be captured. Got keys: {list(fp.keys())}"
    assert fp[key] == "'hello'"


def test_pydantic_class_captured_from_type_hint():
    """Pydantic classes in type hints ARE captured."""
    fp = fingerprint.get_stage_fingerprint(_stage_with_pydantic_param_v1)

    assert _definition_key("class", ParamsWithListDefault) in fp, (
        f"Pydantic class should be captured. Got keys: {list(fp.keys())}"
    )


def test_pydantic_default_data_captured():
    """Data in Pydantic field defaults is captured via schema hashing."""
    fp1 = fingerprint.get_stage_fingerprint(_stage_with_pydantic_param_v1)
    fp2 = fingerprint.get_stage_fingerprint(_stage_with_config_list)

    assert _definition_key("schema", ParamsWithListDefault) in fp1, (
        f"Pydantic schema should be captured. Got keys: {list(fp1.keys())}"
    )
    assert _definition_key("schema", ParamsWithConfigList) in fp2, (
        f"Pydantic schema should be captured. Got keys: {list(fp2.keys())}"
    )
    assert _definition_key("schema", ItemConfig) in fp2
    assert _definition_key("class", ItemConfig) in fp2


def test_pydantic_default_change_triggers_different_hash():
    """Changing a Pydantic default value changes the fingerprint hash."""

    # Create two models with different defaults
    class ParamsV1(pydantic.BaseModel):
        items: list[str] = ["a", "b"]

    class ParamsV2(pydantic.BaseModel):
        items: list[str] = ["a", "b", "c"]

    def stage_v1(params: ParamsV1) -> list[str]:
        return params.items

    def stage_v2(params: ParamsV2) -> list[str]:
        return params.items

    fp1 = fingerprint.get_stage_fingerprint(stage_v1)
    fp2 = fingerprint.get_stage_fingerprint(stage_v2)

    # The schema hashes should be different
    assert fp1[_definition_key("schema", ParamsV1)] != fp2[_definition_key("schema", ParamsV2)], (
        "Different default values should produce different hashes"
    )


def test_pydantic_nested_model_defaults_captured():
    """Nested Pydantic model instances in defaults are captured."""
    fp = fingerprint.get_stage_fingerprint(_stage_with_config_list)

    # The nested model schema should be hashed
    params_key = _definition_key("schema", ParamsWithConfigList)
    assert params_key in fp
    assert _definition_key("schema", ItemConfig) in fp

    hash_val = fp[params_key]
    assert len(hash_val) == 16, f"Expected 16-char hash, got {hash_val}"
    assert all(c in "0123456789abcdef" for c in hash_val)


def test_fingerprint_includes_class_and_defaults():
    """Fingerprint for Pydantic param stages includes class and defaults."""
    fp = fingerprint.get_stage_fingerprint(_stage_with_pydantic_param_v1)

    # Should have: self:, class:, schema:
    assert "self:_stage_with_pydantic_param_v1" in fp
    assert _definition_key("class", ParamsWithListDefault) in fp
    assert _definition_key("schema", ParamsWithListDefault) in fp
    assert len(fp) == 3, f"Expected 3 entries, got {len(fp)}: {list(fp.keys())}"


def test_default_factory_is_tracked():
    """Fields using default_factory are tracked by hashing the factory function."""

    class ParamsWithFactory(pydantic.BaseModel):
        items: list[str] = pydantic.Field(default_factory=lambda: ["a", "b"])

    def stage(params: ParamsWithFactory) -> list[str]:
        return params.items

    fp = fingerprint.get_stage_fingerprint(stage)

    # default_factory should be captured
    assert _definition_key("schema", ParamsWithFactory) in fp, (
        f"default_factory should be captured. Got keys: {list(fp.keys())}"
    )


def test_default_factory_change_triggers_different_hash():
    """Changing a default_factory function changes the fingerprint."""

    class ParamsV1(pydantic.BaseModel):
        items: list[str] = pydantic.Field(default_factory=lambda: ["a"])

    class ParamsV2(pydantic.BaseModel):
        items: list[str] = pydantic.Field(default_factory=lambda: ["a", "b"])

    def stage_v1(params: ParamsV1) -> list[str]:
        return params.items

    def stage_v2(params: ParamsV2) -> list[str]:
        return params.items

    fp1 = fingerprint.get_stage_fingerprint(stage_v1)
    fp2 = fingerprint.get_stage_fingerprint(stage_v2)

    # Different factories should produce different hashes
    assert fp1[_definition_key("schema", ParamsV1)] != fp2[_definition_key("schema", ParamsV2)], (
        "Different default_factory functions should produce different hashes"
    )


def test_none_default_is_tracked():
    """None as an explicit default is tracked (not skipped)."""

    class ParamsWithNone(pydantic.BaseModel):
        value: str | None = None

    def stage(params: ParamsWithNone) -> str | None:
        return params.value

    fp = fingerprint.get_stage_fingerprint(stage)

    # None default should be captured
    assert _definition_key("schema", ParamsWithNone) in fp, (
        f"None default should be captured. Got keys: {list(fp.keys())}"
    )


def test_frozenset_default_is_deterministic():
    """frozenset defaults produce deterministic hashes regardless of iteration order."""

    class ParamsWithSet(pydantic.BaseModel):
        tags: frozenset[str] = frozenset({"c", "a", "b"})

    def stage(params: ParamsWithSet) -> frozenset[str]:
        return params.tags

    # Run multiple times to verify determinism
    fp1 = fingerprint.get_stage_fingerprint(stage)
    fp2 = fingerprint.get_stage_fingerprint(stage)
    fp3 = fingerprint.get_stage_fingerprint(stage)

    assert fp1 == fp2 == fp3, "frozenset defaults should be deterministic"
