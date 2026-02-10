from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from conftest import isolated_pivot_dir
from pivot import cli, import_artifact, status
from pivot.status import ImportCheckStatus
from pivot.storage import track

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture


# =============================================================================
# Unit Tests: get_import_status
# =============================================================================


def test_status_import_update_available(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Import with available update should report update available."""
    pvt_data = track.PvtData(
        path="model.pkl",
        hash="aaa",
        size=100,
        source=track.ImportSource(
            repo="https://github.com/org/repo",
            rev="main",
            rev_lock="abc123def456",
            stage="train",
            path="models/model.pkl",
            remote="s3://bucket/cache",
        ),
    )
    mocker.patch.object(
        track,
        "discover_import_pvt_files",
        autospec=True,
        return_value={str(tmp_path / "models" / "model.pkl"): pvt_data},
    )
    mocker.patch.object(
        import_artifact,
        "check_for_update",
        new=AsyncMock(
            return_value=import_artifact.UpdateCheck(
                available=True,
                current_rev="abc123def456",
                latest_rev="def456abc789",
            )
        ),
    )

    results = status.get_import_status(tmp_path)

    assert len(results) == 1, "Should have one import result"
    assert results[0].status is ImportCheckStatus.UPDATE_AVAILABLE
    assert results[0].current_rev == "abc123de"
    assert results[0].latest_rev == "def456ab"


def test_status_import_up_to_date(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Import with no available update should report up to date."""
    pvt_data = track.PvtData(
        path="data.csv",
        hash="bbb",
        size=200,
        source=track.ImportSource(
            repo="https://github.com/org/repo",
            rev="main",
            rev_lock="abc123def456",
            stage="prepare",
            path="data/train.csv",
            remote="s3://bucket/cache",
        ),
    )
    mocker.patch.object(
        track,
        "discover_import_pvt_files",
        autospec=True,
        return_value={str(tmp_path / "data" / "train.csv"): pvt_data},
    )
    mocker.patch.object(
        import_artifact,
        "check_for_update",
        new=AsyncMock(
            return_value=import_artifact.UpdateCheck(
                available=False,
                current_rev="abc123def456",
                latest_rev="abc123def456",
            )
        ),
    )

    results = status.get_import_status(tmp_path)

    assert len(results) == 1, "Should have one import result"
    assert results[0].status is ImportCheckStatus.UP_TO_DATE


def test_status_import_network_error(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Network failure should produce error status, not raise."""
    pvt_data = track.PvtData(
        path="model.pkl",
        hash="ccc",
        size=300,
        source=track.ImportSource(
            repo="https://github.com/org/repo",
            rev="main",
            rev_lock="abc123def456",
            stage="train",
            path="models/model.pkl",
            remote="s3://bucket/cache",
        ),
    )
    mocker.patch.object(
        track,
        "discover_import_pvt_files",
        autospec=True,
        return_value={str(tmp_path / "models" / "model.pkl"): pvt_data},
    )
    mocker.patch.object(
        import_artifact,
        "check_for_update",
        new=AsyncMock(side_effect=ConnectionError("Network unreachable")),
    )

    results = status.get_import_status(tmp_path)

    assert len(results) == 1, "Should have one import result"
    assert results[0].status is ImportCheckStatus.ERROR
    assert "Network unreachable" in results[0].error


# =============================================================================
# CLI Tests: --check-imports flag
# =============================================================================


def test_status_no_flag_no_api(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Without --check-imports flag, check_for_update is never called."""
    check_mock = mocker.patch.object(
        import_artifact,
        "check_for_update",
        new=AsyncMock(),
    )
    # Also mock discover to ensure it's not even called
    discover_mock = mocker.patch.object(
        track,
        "discover_import_pvt_files",
        autospec=True,
        return_value={},
    )

    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text("""\
from __future__ import annotations
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline('test')
""")
        result = runner.invoke(cli.cli, ["status"])

    assert result.exit_code == 0
    check_mock.assert_not_called()
    discover_mock.assert_not_called()


def test_status_check_imports_shows_update(
    runner: click.testing.CliRunner,
    mock_discovery: object,
    mocker: MockerFixture,
) -> None:
    """With --check-imports, shows import status section with update available."""
    from pivot import project

    project_root = project.get_project_root()
    pvt_data = track.PvtData(
        path="model.pkl",
        hash="aaa",
        size=100,
        source=track.ImportSource(
            repo="https://github.com/org/repo",
            rev="main",
            rev_lock="abc123de",
            stage="train",
            path="models/model.pkl",
            remote="s3://bucket/cache",
        ),
    )
    mocker.patch.object(
        track,
        "discover_import_pvt_files",
        autospec=True,
        return_value={str(project_root / "models" / "model.pkl"): pvt_data},
    )
    mocker.patch.object(
        import_artifact,
        "check_for_update",
        new=AsyncMock(
            return_value=import_artifact.UpdateCheck(
                available=True,
                current_rev="abc123de",
                latest_rev="def456ab",
            )
        ),
    )

    result = runner.invoke(cli.cli, ["status", "--check-imports"])

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert "Imports" in result.output
    assert "update available" in result.output
    assert "abc123de" in result.output
    assert "def456ab" in result.output


def test_status_check_imports_shows_up_to_date(
    runner: click.testing.CliRunner,
    mock_discovery: object,
    mocker: MockerFixture,
) -> None:
    """With --check-imports, shows up to date imports."""
    from pivot import project

    project_root = project.get_project_root()
    pvt_data = track.PvtData(
        path="data.csv",
        hash="bbb",
        size=200,
        source=track.ImportSource(
            repo="https://github.com/org/repo",
            rev="main",
            rev_lock="abc123def456",
            stage="prepare",
            path="data/train.csv",
            remote="s3://bucket/cache",
        ),
    )
    mocker.patch.object(
        track,
        "discover_import_pvt_files",
        autospec=True,
        return_value={str(project_root / "data" / "train.csv"): pvt_data},
    )
    mocker.patch.object(
        import_artifact,
        "check_for_update",
        new=AsyncMock(
            return_value=import_artifact.UpdateCheck(
                available=False,
                current_rev="abc123def456",
                latest_rev="abc123def456",
            )
        ),
    )

    result = runner.invoke(cli.cli, ["status", "--check-imports"])

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert "Imports" in result.output
    assert "up to date" in result.output


def test_status_check_imports_network_error_doesnt_fail(
    runner: click.testing.CliRunner,
    mock_discovery: object,
    mocker: MockerFixture,
) -> None:
    """Network error during import check shows warning but command succeeds."""
    from pivot import project

    project_root = project.get_project_root()
    pvt_data = track.PvtData(
        path="model.pkl",
        hash="ccc",
        size=300,
        source=track.ImportSource(
            repo="https://github.com/org/repo",
            rev="main",
            rev_lock="abc123def456",
            stage="train",
            path="models/model.pkl",
            remote="s3://bucket/cache",
        ),
    )
    mocker.patch.object(
        track,
        "discover_import_pvt_files",
        autospec=True,
        return_value={str(project_root / "models" / "model.pkl"): pvt_data},
    )
    mocker.patch.object(
        import_artifact,
        "check_for_update",
        new=AsyncMock(side_effect=ConnectionError("Network unreachable")),
    )

    result = runner.invoke(cli.cli, ["status", "--check-imports"])

    assert result.exit_code == 0, f"Should succeed despite network error: {result.output}"
    assert "Imports" in result.output
    assert "check failed" in result.output
