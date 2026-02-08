from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, TypedDict, TypeGuard, cast

# Type aliases matching pipeline_config.py types
MatrixPrimitive = str | int | float | bool
# Dict value is DimensionOverrides (field overrides) or None, but we use Any here
# to avoid importing from pipeline_config and keep this module dependency-free
MatrixDimension = list[MatrixPrimitive] | dict[str, Any]


class _StageConfigDict(TypedDict, total=False):
    """Minimal stage config structure for YAML parsing (internal use only)."""

    python: str
    deps: list[str]
    outs: list[str]
    matrix: dict[str, MatrixDimension]
    variants: str


if TYPE_CHECKING:
    # Type alias for parsed pivot.yaml config - dict[str, Any] because YAML parsing
    # produces unvalidated data that may have arbitrary keys
    type PivotConfigDict = dict[str, Any]


def normalize_dimension_keys(dim_value: MatrixDimension) -> list[str]:
    """Extract string keys from a matrix dimension.

    Handles both list form ["a", "b", true, 0.5] and dict form {"a": {...}, "b": None}.
    """
    if isinstance(dim_value, dict):
        return list(dim_value.keys())
    return [str(v) for v in dim_value]


def parse_stage_name_template(name: str) -> tuple[str, str | None]:
    """Parse stage name for template pattern.

    Returns:
        Tuple of (base_name, template_or_None)
        Example: "train@{model}_{dataset}" -> ("train", "{model}_{dataset}")
    """
    if "@" not in name:
        return name, None
    base_name, template = name.split("@", 1)
    return base_name, template


def expand_matrix_names(
    base_name: str,
    matrix: dict[str, MatrixDimension],
    name_template: str | None = None,
) -> list[str] | None:
    """Expand matrix configuration into stage names only (fast path).

    This function only generates names, not full stage configurations.
    Used for shell completion where we need names quickly.

    Returns None if template has invalid variables (triggers fallback to full discovery).
    """
    if not matrix:
        return []

    dim_names = list(matrix.keys())
    dim_keys = [normalize_dimension_keys(matrix[dim]) for dim in dim_names]

    if any(len(keys) == 0 for keys in dim_keys):
        return []

    result = list[str]()
    for combo in itertools.product(*dim_keys):
        string_values = dict(zip(dim_names, combo, strict=True))
        try:
            if name_template is not None:
                variant_name = name_template.format(**string_values)
            else:
                variant_name = "_".join(string_values[d] for d in dim_names)
        except KeyError:
            return None
        result.append(f"{base_name}@{variant_name}")

    return result


def extract_stage_names(config: dict[str, Any] | None) -> list[str] | None:
    """Extract all stage names from a pivot.yaml config dict.

    Stages with 'variants' field are skipped as they require Python imports.
    Returns None to trigger fallback on malformed config or template errors.
    """
    if config is None:
        return []

    stages = config.get("stages", {})
    if not isinstance(stages, dict) or not stages:
        return None if stages else []

    stages_dict = cast("dict[str, Any]", stages)
    result = list[str]()
    for name, stage_config in stages_dict.items():
        if not _is_stage_config_dict(stage_config):
            continue

        if "variants" in stage_config:
            continue

        matrix_config = stage_config.get("matrix")
        if matrix_config:
            base_name, template = parse_stage_name_template(name)
            expanded = expand_matrix_names(base_name, matrix_config, template)
            if expanded is None:
                return None
            result.extend(expanded)
        else:
            result.append(name)

    return result


def _is_stage_config_dict(value: object) -> TypeGuard[_StageConfigDict]:
    """Type guard to validate stage config is a dict."""
    return isinstance(value, dict)


def needs_fallback(config: dict[str, Any] | None) -> bool:
    """Check if config requires full discovery (has variants field or malformed stages)."""
    if config is None:
        return False

    stages = config.get("stages", {})
    if not isinstance(stages, dict):
        return True
    if not stages:
        return False

    stages_dict = cast("dict[str, Any]", stages)
    return any(
        isinstance(stage_config, dict) and "variants" in stage_config
        for stage_config in stages_dict.values()
    )
