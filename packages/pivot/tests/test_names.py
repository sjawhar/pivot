"""Tests for pivot.names — stage name display and resolution helpers."""

from __future__ import annotations

from pivot import names

# =============================================================================
# display_stage_name
# =============================================================================


def test_display_stage_name_strips_prefix() -> None:
    assert names.display_stage_name("horizon/train", "horizon") == "train"


def test_display_stage_name_no_strip_when_none() -> None:
    assert names.display_stage_name("horizon/train", None) == "horizon/train"


def test_display_stage_name_no_strip_when_different_prefix() -> None:
    assert names.display_stage_name("horizon/train", "other") == "horizon/train"


def test_display_stage_name_no_strip_partial_prefix() -> None:
    """Don't strip 'hor' from 'horizon/train' — must match full prefix."""
    assert names.display_stage_name("horizon/train", "hor") == "horizon/train"


def test_display_stage_name_bare_name_unchanged() -> None:
    """Stage without any slash is returned as-is."""
    assert names.display_stage_name("train", "horizon") == "train"


# =============================================================================
# resolve_stage_name
# =============================================================================


def test_resolve_stage_name_exact_match() -> None:
    stages: dict[str, object] = {"horizon/train": {}, "horizon/eval": {}}
    assert names.resolve_stage_name("horizon/train", stages) == "horizon/train"


def test_resolve_stage_name_bare_name() -> None:
    stages: dict[str, object] = {"horizon/train": {}, "horizon/eval": {}}
    assert names.resolve_stage_name("train", stages) == "horizon/train"


def test_resolve_stage_name_ambiguous_falls_through() -> None:
    stages: dict[str, object] = {"a/train": {}, "b/train": {}}
    assert names.resolve_stage_name("train", stages) == "train"


def test_resolve_stage_name_no_match_falls_through() -> None:
    stages: dict[str, object] = {"horizon/train": {}}
    assert names.resolve_stage_name("missing", stages) == "missing"


def test_resolve_stage_name_bare_name_with_key_syntax() -> None:
    """Bare name that includes a colon for key — still resolves by /{name} suffix."""
    stages: dict[str, object] = {"horizon/train": {}}
    # "train" should resolve even though the caller will later parse :key
    assert names.resolve_stage_name("train", stages) == "horizon/train"


def test_resolve_stage_name_no_slash_stages() -> None:
    """Stages without pipeline prefix — exact match only."""
    stages: dict[str, object] = {"train": {}, "eval": {}}
    assert names.resolve_stage_name("train", stages) == "train"
