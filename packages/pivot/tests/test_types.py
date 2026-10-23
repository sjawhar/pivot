"""Tests for pivot.types module."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from typing import get_args

from pivot import loaders, types
from pivot.types import CompletionType, StageStatus


def test_completion_type_includes_ran() -> None:
    """CompletionType should include RAN."""
    status: CompletionType = StageStatus.RAN
    assert status == StageStatus.RAN


def test_completion_type_includes_all_terminal_statuses() -> None:
    """CompletionType includes all terminal status values."""
    assert StageStatus.RAN in get_args(CompletionType)
    assert StageStatus.CACHED in get_args(CompletionType)
    assert StageStatus.BLOCKED in get_args(CompletionType)
    assert StageStatus.CANCELLED in get_args(CompletionType)
    assert StageStatus.FAILED in get_args(CompletionType)


def test_completion_type_includes_failed() -> None:
    """CompletionType should include FAILED."""
    status: CompletionType = StageStatus.FAILED
    assert status == StageStatus.FAILED


def test_directory_file_source_identity_uses_stored_directory() -> None:
    """A one-file directory ref carries its source identity instead of parsing its own key."""
    directory = types.ArtifactIdentity("p/split", "parts")
    ref = types.ArtifactRef(
        identity=types.directory_file_identity(directory, "train.csv"),
        format=loaders.CSV(),
        python_type=object,
        tag=types.ArtifactTag.DIRECTORY,
        directory=directory,
    )

    assert types.source_identity(ref) == directory
