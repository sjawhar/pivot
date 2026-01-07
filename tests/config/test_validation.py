"""Tests for stage registration input validation.

Tests validation of stage names, dependency paths, output paths,
and detection of conflicts like duplicate stage names or output conflicts.
"""

import pytest

from pivot import registry
from pivot.exceptions import ValidationError
from pivot.registry import ValidationMode


def test_duplicate_stage_name_raises_error() -> None:
    """Should raise error when registering stage with duplicate name."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    def stage2() -> None:
        pass

    reg.register(stage1, name="process", deps=list[str](), outs=list[str](), params=None)

    with pytest.raises(ValidationError, match="already registered"):
        reg.register(stage2, name="process", deps=list[str](), outs=list[str](), params=None)


def test_duplicate_stage_name_with_warning_mode() -> None:
    """Should log warning but allow registration in WARN mode."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    def stage1() -> None:
        pass

    def stage2() -> None:
        pass

    reg.register(stage1, name="process", deps=list[str](), outs=list[str](), params=None)

    # Should not raise, just warn
    reg.register(stage2, name="process", deps=list[str](), outs=list[str](), params=None)

    # Second registration should overwrite
    assert reg.get("process")["func"] is stage2


def test_invalid_dep_path_with_parent_traversal() -> None:
    """Should raise error for paths with '..' (path traversal)."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    with pytest.raises(ValidationError, match="path traversal"):
        reg.register(
            stage1,
            name="process",
            deps=["../secrets/passwords.txt"],
            outs=list[str](),
            params=None,
        )


def test_invalid_out_path_with_parent_traversal() -> None:
    """Should raise error for output paths with '..' (path traversal)."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    with pytest.raises(ValidationError, match="path traversal"):
        reg.register(
            stage1,
            name="process",
            deps=list[str](),
            outs=["../system/file.txt"],
            params=None,
        )


def test_invalid_path_with_null_byte() -> None:
    """Should raise error for paths with null bytes."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    with pytest.raises(ValidationError, match="null byte"):
        reg.register(
            stage1,
            name="process",
            deps=["data\x00.csv"],
            outs=list[str](),
            params=None,
        )


def test_invalid_path_with_newline() -> None:
    """Should raise error for paths with newline characters."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    with pytest.raises(ValidationError, match="newline"):
        reg.register(
            stage1,
            name="process",
            deps=["file\nname.csv"],
            outs=list[str](),
            params=None,
        )


def test_output_conflict_raises_error() -> None:
    """Should raise error when two stages produce same output."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    def stage2() -> None:
        pass

    reg.register(stage1, name="process1", deps=list[str](), outs=["output.txt"], params=None)

    with pytest.raises(ValidationError, match="produced by both"):
        reg.register(stage2, name="process2", deps=list[str](), outs=["output.txt"], params=None)


def test_output_conflict_with_warning_mode() -> None:
    """Should log warning but allow registration in WARN mode."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    def stage1() -> None:
        pass

    def stage2() -> None:
        pass

    reg.register(stage1, name="process1", deps=list[str](), outs=["output.txt"], params=None)

    # Should not raise, just warn
    reg.register(stage2, name="process2", deps=list[str](), outs=["output.txt"], params=None)


def test_empty_stage_name_raises_error() -> None:
    """Should raise error for empty stage name."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    with pytest.raises(ValidationError, match="cannot be empty"):
        reg.register(stage1, name="", deps=list[str](), outs=list[str](), params=None)


def test_whitespace_only_stage_name_raises_error() -> None:
    """Should raise error for whitespace-only stage name."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    with pytest.raises(ValidationError, match="cannot be empty"):
        reg.register(stage1, name="   ", deps=list[str](), outs=list[str](), params=None)


def test_valid_inputs_pass_silently() -> None:
    """Should register stage without errors when all inputs valid."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    # Should not raise
    reg.register(
        stage1,
        name="process",
        deps=["data/input.csv"],
        outs=["data/output.csv"],
        params=None,
    )

    assert "process" in reg.list_stages()


def test_invalid_path_with_parent_traversal_warn_mode() -> None:
    """Should warn but not raise for path with '..' in WARN mode."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    def stage1() -> None:
        pass

    # Should not raise, just warn
    reg.register(
        stage1,
        name="process",
        deps=["../external/data.csv"],
        outs=list[str](),
        params=None,
    )

    assert "process" in reg.list_stages()


def test_invalid_path_with_null_byte_warn_mode() -> None:
    """Should warn but not raise for path with null byte in WARN mode."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    def stage1() -> None:
        pass

    # Should not raise, just warn
    reg.register(stage1, name="process", deps=["bad\x00file.csv"], outs=list[str](), params=None)

    assert "process" in reg.list_stages()
