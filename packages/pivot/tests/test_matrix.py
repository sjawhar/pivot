from __future__ import annotations

from typing import Any

from pivot import matrix

# =============================================================================
# normalize_dimension_keys tests
# =============================================================================


def test_normalize_dimension_keys_list_of_strings() -> None:
    """List of strings returns string keys."""
    result = matrix.normalize_dimension_keys(["bert", "gpt"])
    assert result == ["bert", "gpt"]


def test_normalize_dimension_keys_list_with_mixed_types() -> None:
    """List with mixed types converts all to strings."""
    result = matrix.normalize_dimension_keys(["a", True, 42, 0.5])
    assert result == ["a", "True", "42", "0.5"]


def test_normalize_dimension_keys_dict_returns_keys() -> None:
    """Dict returns its keys."""
    result = matrix.normalize_dimension_keys({"bert": {"hidden": 768}, "gpt": None})
    assert result == ["bert", "gpt"]


def test_normalize_dimension_keys_empty_list() -> None:
    """Empty list returns empty list."""
    result = matrix.normalize_dimension_keys([])
    assert result == []


def test_normalize_dimension_keys_empty_dict() -> None:
    """Empty dict returns empty list."""
    result = matrix.normalize_dimension_keys({})
    assert result == []


def test_normalize_dimension_keys_single_item_list() -> None:
    """Single item list works."""
    result = matrix.normalize_dimension_keys(["only"])
    assert result == ["only"]


# =============================================================================
# parse_stage_name_template tests
# =============================================================================


def test_parse_stage_name_template_no_template() -> None:
    """Stage name without @ returns no template."""
    base, template = matrix.parse_stage_name_template("train")
    assert base == "train"
    assert template is None


def test_parse_stage_name_template_with_template() -> None:
    """Stage name with @ extracts template."""
    base, template = matrix.parse_stage_name_template("train@{model}_{dataset}")
    assert base == "train"
    assert template == "{model}_{dataset}"


def test_parse_stage_name_template_complex_template() -> None:
    """Complex template with multiple variables."""
    base, template = matrix.parse_stage_name_template("process@{a}-{b}_{c}")
    assert base == "process"
    assert template == "{a}-{b}_{c}"


def test_parse_stage_name_template_with_static_text() -> None:
    """Template can have static text mixed with variables."""
    base, template = matrix.parse_stage_name_template("train@v1_{model}")
    assert base == "train"
    assert template == "v1_{model}"


def test_parse_stage_name_template_empty_base_name() -> None:
    """Empty base name with template."""
    base, template = matrix.parse_stage_name_template("@{dim}")
    assert base == ""
    assert template == "{dim}"


# =============================================================================
# expand_matrix_names tests
# =============================================================================


def test_expand_matrix_names_single_dimension_list() -> None:
    """Single dimension list produces simple names."""
    names = matrix.expand_matrix_names("train", {"model": ["bert", "gpt"]})
    assert names is not None
    assert set(names) == {"train@bert", "train@gpt"}


def test_expand_matrix_names_single_dimension_dict() -> None:
    """Single dimension dict uses keys for names."""
    names = matrix.expand_matrix_names("train", {"model": {"bert": {"hidden": 768}, "gpt": None}})
    assert names is not None
    assert set(names) == {"train@bert", "train@gpt"}


def test_expand_matrix_names_two_dimensions() -> None:
    """Two dimensions produces cartesian product."""
    names = matrix.expand_matrix_names(
        "train", {"model": ["bert", "gpt"], "dataset": ["swe", "human"]}
    )
    assert names is not None
    assert set(names) == {
        "train@bert_swe",
        "train@bert_human",
        "train@gpt_swe",
        "train@gpt_human",
    }


def test_expand_matrix_names_three_dimensions() -> None:
    """Three dimensions produces full cartesian product."""
    names = matrix.expand_matrix_names("stage", {"a": ["1", "2"], "b": ["x", "y"], "c": ["p"]})
    assert names is not None
    assert set(names) == {
        "stage@1_x_p",
        "stage@1_y_p",
        "stage@2_x_p",
        "stage@2_y_p",
    }


def test_expand_matrix_names_with_explicit_template() -> None:
    """Custom template is respected."""
    names = matrix.expand_matrix_names(
        "train",
        {"model": ["bert", "gpt"], "dataset": ["swe"]},
        name_template="{dataset}-{model}",
    )
    assert names is not None
    assert set(names) == {"train@swe-bert", "train@swe-gpt"}


def test_expand_matrix_names_empty_matrix() -> None:
    """Empty matrix returns empty list."""
    names = matrix.expand_matrix_names("train", {})
    assert names == []


def test_expand_matrix_names_empty_dimension() -> None:
    """Empty dimension returns empty list."""
    names = matrix.expand_matrix_names("train", {"model": []})
    assert names == []


def test_expand_matrix_names_mixed_types_in_list() -> None:
    """Mixed types in list are converted to strings for names."""
    names = matrix.expand_matrix_names("stage", {"val": [1, True, "str"]})
    assert names is not None
    assert set(names) == {"stage@1", "stage@True", "stage@str"}


def test_expand_matrix_names_single_value_dimension() -> None:
    """Single value dimension works."""
    names = matrix.expand_matrix_names("train", {"model": ["bert"]})
    assert names == ["train@bert"]


def test_expand_matrix_names_preserves_dimension_order() -> None:
    """Auto-generated names use dimension insertion order."""
    names = matrix.expand_matrix_names("train", {"first": ["a"], "second": ["b"], "third": ["c"]})
    assert names == ["train@a_b_c"]


def test_expand_matrix_names_template_with_unknown_variable() -> None:
    """Template with unknown variable returns None for fallback."""
    names = matrix.expand_matrix_names("train", {"model": ["bert"]}, name_template="{unknown_var}")
    assert names is None, "Should return None when template has unknown variables"


# =============================================================================
# extract_stage_names tests
# =============================================================================


def test_extract_stage_names_simple_stages() -> None:
    """Extract names from simple stages (no matrix)."""
    config = {
        "stages": {
            "preprocess": {"python": "stages.preprocess"},
            "train": {"python": "stages.train"},
        }
    }
    names = matrix.extract_stage_names(config)
    assert names is not None
    assert set(names) == {"preprocess", "train"}


def test_extract_stage_names_matrix_stages() -> None:
    """Extract names from matrix stages."""
    config = {
        "stages": {
            "train": {
                "python": "stages.train",
                "matrix": {"model": ["bert", "gpt"]},
            }
        }
    }
    names = matrix.extract_stage_names(config)
    assert names is not None
    assert set(names) == {"train@bert", "train@gpt"}


def test_extract_stage_names_mixed_simple_and_matrix() -> None:
    """Extract names from mix of simple and matrix stages."""
    config = {
        "stages": {
            "preprocess": {"python": "stages.preprocess"},
            "train": {
                "python": "stages.train",
                "matrix": {"model": ["bert", "gpt"]},
            },
        }
    }
    names = matrix.extract_stage_names(config)
    assert names is not None
    assert set(names) == {"preprocess", "train@bert", "train@gpt"}


def test_extract_stage_names_with_template() -> None:
    """Extract names from stage with explicit template."""
    config = {
        "stages": {
            "train@{m}": {
                "python": "stages.train",
                "matrix": {"m": ["bert", "gpt"]},
            }
        }
    }
    names = matrix.extract_stage_names(config)
    assert names is not None
    assert set(names) == {"train@bert", "train@gpt"}


def test_extract_stage_names_empty_stages() -> None:
    """Empty stages dict returns empty list."""
    config: dict[str, Any] = {"stages": {}}
    names = matrix.extract_stage_names(config)
    assert names == []


def test_extract_stage_names_missing_stages_key() -> None:
    """Missing stages key returns empty list."""
    config: dict[str, Any] = {}
    names = matrix.extract_stage_names(config)
    assert names == []


def test_extract_stage_names_none_config() -> None:
    """None config returns empty list."""
    names = matrix.extract_stage_names(None)
    assert names == []


def test_extract_stage_names_skips_variants() -> None:
    """Stages with variants field are skipped (require fallback)."""
    config = {
        "stages": {
            "simple": {"python": "stages.simple"},
            "dynamic": {"python": "stages.dynamic", "variants": "stages.get_variants"},
        }
    }
    names = matrix.extract_stage_names(config)
    assert names == ["simple"]


def test_extract_stage_names_malformed_stages_list() -> None:
    """Malformed stages (list instead of dict) returns None for fallback."""
    config: dict[str, Any] = {"stages": ["stage1", "stage2"]}
    names = matrix.extract_stage_names(config)
    assert names is None, "Should return None when stages is not a dict"


def test_extract_stage_names_skips_non_dict_stage_config() -> None:
    """Non-dict stage configs are skipped."""
    config: dict[str, Any] = {
        "stages": {
            "valid": {"python": "stages.valid"},
            "invalid_string": "not a dict",
            "invalid_none": None,
        }
    }
    names = matrix.extract_stage_names(config)
    assert names is not None
    assert names == ["valid"], "Should skip non-dict stage configs"


def test_extract_stage_names_template_error_returns_none() -> None:
    """Template with unknown variable returns None for fallback."""
    config = {
        "stages": {
            "train@{unknown}": {
                "python": "stages.train",
                "matrix": {"model": ["bert"]},
            }
        }
    }
    names = matrix.extract_stage_names(config)
    assert names is None, "Should return None when template has unknown variables"


# =============================================================================
# needs_fallback tests
# =============================================================================


def test_needs_fallback_simple_config() -> None:
    """Simple config without variants doesn't need fallback."""
    config = {
        "stages": {
            "preprocess": {"python": "stages.preprocess"},
            "train": {"python": "stages.train"},
        }
    }
    assert matrix.needs_fallback(config) is False


def test_needs_fallback_matrix_config() -> None:
    """Matrix config without variants doesn't need fallback."""
    config = {"stages": {"train": {"python": "stages.train", "matrix": {"model": ["bert"]}}}}
    assert matrix.needs_fallback(config) is False


def test_needs_fallback_with_variants() -> None:
    """Config with variants needs fallback."""
    config = {"stages": {"train": {"python": "stages.train", "variants": "stages.get_variants"}}}
    assert matrix.needs_fallback(config) is True


def test_needs_fallback_mixed_with_variants() -> None:
    """Mixed config with any variants needs fallback."""
    config = {
        "stages": {
            "simple": {"python": "stages.simple"},
            "dynamic": {"python": "stages.dynamic", "variants": "stages.get_variants"},
        }
    }
    assert matrix.needs_fallback(config) is True


def test_needs_fallback_empty_stages() -> None:
    """Empty stages doesn't need fallback."""
    config: dict[str, Any] = {"stages": {}}
    assert matrix.needs_fallback(config) is False


def test_needs_fallback_none_config() -> None:
    """None config doesn't need fallback."""
    assert matrix.needs_fallback(None) is False


def test_needs_fallback_missing_stages() -> None:
    """Missing stages key doesn't need fallback."""
    config: dict[str, Any] = {}
    assert matrix.needs_fallback(config) is False


def test_needs_fallback_malformed_stages_list() -> None:
    """Malformed stages (list instead of dict) needs fallback."""
    config: dict[str, Any] = {"stages": ["stage1", "stage2"]}
    assert matrix.needs_fallback(config) is True, "Should need fallback when stages is not a dict"
