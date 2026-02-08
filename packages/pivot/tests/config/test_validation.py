import pathlib
from typing import Annotated, TypedDict

import pytest

from pivot import loaders, outputs, registry
from pivot.exceptions import SecurityValidationError, ValidationError
from pivot.registry import ValidationMode

# --- TypedDicts for outputs ---


class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _DataOutput(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("data/output.csv", loaders.PathOnly())]


# --- Helper stage functions with parent traversal deps ---


def _helper_dep_parent_traversal(
    secrets: Annotated[pathlib.Path, outputs.Dep("../secrets/passwords.txt", loaders.PathOnly())],
) -> None:
    _ = secrets


def _helper_out_parent_traversal() -> Annotated[
    pathlib.Path, outputs.Out("../system/file.txt", loaders.PathOnly())
]:
    return pathlib.Path("../system/file.txt")


def _helper_dep_null_byte(
    data: Annotated[pathlib.Path, outputs.Dep("data\x00.csv", loaders.PathOnly())],
) -> None:
    _ = data


def _helper_dep_newline(
    file: Annotated[pathlib.Path, outputs.Dep("file\nname.csv", loaders.PathOnly())],
) -> None:
    _ = file


def _helper_output_txt(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    _ = input_file
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _helper_with_data_output(
    input_file: Annotated[pathlib.Path, outputs.Dep("data/input.csv", loaders.PathOnly())],
) -> _DataOutput:
    _ = input_file
    return _DataOutput(output=pathlib.Path("data/output.csv"))


def _helper_dep_external_parent_traversal(
    external: Annotated[pathlib.Path, outputs.Dep("../external/data.csv", loaders.PathOnly())],
) -> None:
    _ = external


def _helper_dep_bad_null_byte(
    bad: Annotated[pathlib.Path, outputs.Dep("bad\x00file.csv", loaders.PathOnly())],
) -> None:
    _ = bad


def _helper_dep_newline_file(
    file: Annotated[pathlib.Path, outputs.Dep("file\nname.csv", loaders.PathOnly())],
) -> None:
    _ = file


def _helper_dep_carriage_return(
    file: Annotated[pathlib.Path, outputs.Dep("file\rname.csv", loaders.PathOnly())],
) -> None:
    _ = file


# --- Tests ---


def test_duplicate_stage_name_raises_error() -> None:
    """Should raise error when registering stage with duplicate name."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    def stage2() -> None:
        pass

    reg.register(stage1, name="process")

    with pytest.raises(ValidationError, match="already registered"):
        reg.register(stage2, name="process")


def test_duplicate_stage_name_with_warning_mode() -> None:
    """Should log warning but allow registration in WARN mode."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    def stage1() -> None:
        pass

    def stage2() -> None:
        pass

    reg.register(stage1, name="process")

    # Should not raise, just warn
    reg.register(stage2, name="process")

    # Second registration should overwrite
    assert reg.get("process")["func"] is stage2


def test_invalid_dep_path_with_parent_traversal() -> None:
    """Should raise SecurityValidationError for paths with '..' (path traversal)."""
    reg = registry.StageRegistry()

    with pytest.raises(SecurityValidationError, match="path traversal"):
        reg.register(_helper_dep_parent_traversal, name="process")


def test_invalid_out_path_with_parent_traversal() -> None:
    """Should raise SecurityValidationError for output paths with '..' (path traversal)."""
    reg = registry.StageRegistry()

    with pytest.raises(SecurityValidationError, match="path traversal"):
        reg.register(_helper_out_parent_traversal, name="process")


def test_invalid_path_with_null_byte() -> None:
    """Should raise SecurityValidationError for paths with null bytes."""
    reg = registry.StageRegistry()

    with pytest.raises(SecurityValidationError, match="null byte"):
        reg.register(_helper_dep_null_byte, name="process")


def test_invalid_path_with_newline() -> None:
    """Should raise SecurityValidationError for paths with newline characters."""
    reg = registry.StageRegistry()

    with pytest.raises(SecurityValidationError, match="newline"):
        reg.register(_helper_dep_newline, name="process")


def test_output_conflict_raises_error() -> None:
    """Should raise error when two stages produce same output."""
    reg = registry.StageRegistry()

    reg.register(_helper_output_txt, name="process1")

    # Second stage with same output path
    def _helper_output_txt_2(
        input_file: Annotated[pathlib.Path, outputs.Dep("input2.txt", loaders.PathOnly())],
    ) -> _OutputTxt:
        _ = input_file
        return _OutputTxt(output=pathlib.Path("output.txt"))

    # Second registration succeeds (validation is deferred)
    reg.register(_helper_output_txt_2, name="process2")

    # Output validation is deferred until validate_outputs() is called
    with pytest.raises(ValidationError, match="produced by both"):
        reg.validate_outputs()


def test_output_conflict_with_warning_mode() -> None:
    """Should log warning but allow registration in WARN mode."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    reg.register(_helper_output_txt, name="process1")

    # Second stage with same output path
    def _helper_output_txt_2(
        input_file: Annotated[pathlib.Path, outputs.Dep("input2.txt", loaders.PathOnly())],
    ) -> _OutputTxt:
        _ = input_file
        return _OutputTxt(output=pathlib.Path("output.txt"))

    # Should not raise, just warn
    reg.register(_helper_output_txt_2, name="process2")


def test_empty_stage_name_raises_error() -> None:
    """Should raise error for empty stage name."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    with pytest.raises(ValidationError, match="must start with a letter"):
        reg.register(stage1, name="")


def test_whitespace_only_stage_name_raises_error() -> None:
    """Should raise error for whitespace-only stage name."""
    reg = registry.StageRegistry()

    def stage1() -> None:
        pass

    with pytest.raises(ValidationError, match="must start with a letter"):
        reg.register(stage1, name="   ")


def test_valid_inputs_pass_silently() -> None:
    """Should register stage without errors when all inputs valid."""
    reg = registry.StageRegistry()

    # Should not raise
    reg.register(_helper_with_data_output, name="process")

    assert "process" in reg.list_stages()


def test_invalid_path_with_parent_traversal_warn_mode() -> None:
    """Path traversal should always error, even in WARN mode (security check)."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    with pytest.raises(SecurityValidationError, match="path traversal"):
        reg.register(_helper_dep_external_parent_traversal, name="process")


def test_invalid_path_with_null_byte_warn_mode() -> None:
    """Null byte in path should always error, even in WARN mode (security check)."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    with pytest.raises(SecurityValidationError, match="null byte"):
        reg.register(_helper_dep_bad_null_byte, name="process")


def test_newline_in_path_always_errors_regardless_of_mode() -> None:
    """Newline in path should always error, even in WARN mode (security check)."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    with pytest.raises(SecurityValidationError, match="newline"):
        reg.register(_helper_dep_newline_file, name="process")


def test_carriage_return_in_path_always_errors_regardless_of_mode() -> None:
    """Carriage return in path should always error, even in WARN mode (security check)."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    with pytest.raises(SecurityValidationError, match="newline"):
        reg.register(_helper_dep_carriage_return, name="process")


def test_non_security_validation_respects_warn_mode() -> None:
    """Non-security validations (like duplicate names) should still respect WARN mode."""
    reg = registry.StageRegistry(validation_mode=ValidationMode.WARN)

    def stage1() -> None:
        pass

    def stage2() -> None:
        pass

    reg.register(stage1, name="process")
    reg.register(stage2, name="process")

    assert reg.get("process")["func"] is stage2, "Second registration should overwrite in WARN mode"
