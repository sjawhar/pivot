from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from pivot import cli, import_artifact, project
from pivot.import_artifact import UpdateCheck, UpdateResult
from pivot.storage import track
from pivot.storage.track import ImportSource, PvtData

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from click.testing import CliRunner


def _make_import_pvt(
    path: str = "data.csv",
    rev_lock: str = "abc123def456",
) -> PvtData:
    return PvtData(
        path=path,
        hash="deadbeef",
        size=100,
        source=ImportSource(
            repo="https://github.com/org/repo",
            rev="main",
            rev_lock=rev_lock,
            stage="train",
            path=path,
            remote="s3://bucket/cache",
        ),
    )


def _make_non_import_pvt(path: str = "local.csv") -> PvtData:
    return PvtData(path=path, hash="cafebabe", size=50)


def _helper_get_project_root(tmp_path: pathlib.Path) -> Callable[[], pathlib.Path]:
    def _get_project_root() -> pathlib.Path:
        return tmp_path

    return _get_project_root


def _helper_read_pvt(
    pvt_path: pathlib.Path,
    pvt_data: PvtData,
) -> Callable[[pathlib.Path], PvtData | None]:
    def _read(path: pathlib.Path) -> PvtData | None:
        return pvt_data if path == pvt_path else None

    return _read


def _helper_is_import(data: PvtData) -> bool:
    return "source" in data


def _helper_get_pvt_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(path.suffix + ".pvt")


def _helper_normalize_path(
    tmp_path: pathlib.Path,
) -> Callable[[str | pathlib.Path, pathlib.Path | None], pathlib.Path]:
    def _normalize(path: str | pathlib.Path, base: pathlib.Path | None = None) -> pathlib.Path:
        path_obj = pathlib.Path(path)
        if path_obj.is_absolute():
            return path_obj
        return tmp_path / path_obj

    return _normalize


def _helper_discover_imports(
    imports: dict[str, PvtData],
) -> Callable[[pathlib.Path], dict[str, PvtData]]:
    def _discover(root: pathlib.Path) -> dict[str, PvtData]:
        return imports

    return _discover


# =============================================================================
# Single target
# =============================================================================


def test_cli_update_single_target(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update with a single target calls update_import for that path."""
    pvt_data = _make_import_pvt()
    pvt_path = tmp_path / "data.csv.pvt"

    monkeypatch.setattr(project, "get_project_root", _helper_get_project_root(tmp_path))
    monkeypatch.setattr(track, "read_pvt_file", _helper_read_pvt(pvt_path, pvt_data))
    monkeypatch.setattr(track, "is_import", _helper_is_import)
    monkeypatch.setattr(track, "get_pvt_path", _helper_get_pvt_path)
    monkeypatch.setattr(
        project,
        "normalize_path",
        _helper_normalize_path(tmp_path),
    )

    mock_update = AsyncMock(
        return_value=UpdateResult(
            downloaded=True,
            metadata_updated=True,
            updated=True,
            old_rev="abc123def456",
            new_rev="999888777666",
            path=str(tmp_path / "data.csv"),
        )
    )
    monkeypatch.setattr(import_artifact, "update_import", mock_update)

    result = runner.invoke(cli.cli, ["update", "data.csv"])
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Updated:" in result.output
    assert "abc123de" in result.output
    assert "99988877" in result.output
    mock_update.assert_called_once_with(pvt_path, new_rev=None)


# =============================================================================
# Discover all imports
# =============================================================================


def test_cli_update_all_discovers(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update with no targets discovers all import .pvt files."""
    pvt_data = _make_import_pvt("model.pkl", rev_lock="aaa111bbb222")
    data_path_str = str(tmp_path / "model.pkl")

    monkeypatch.setattr(project, "get_project_root", _helper_get_project_root(tmp_path))
    monkeypatch.setattr(
        track,
        "discover_import_pvt_files",
        _helper_discover_imports({data_path_str: pvt_data}),
    )
    monkeypatch.setattr(track, "get_pvt_path", _helper_get_pvt_path)

    mock_update = AsyncMock(
        return_value=UpdateResult(
            downloaded=True,
            metadata_updated=True,
            updated=True,
            old_rev="aaa111bbb222",
            new_rev="ccc333ddd444",
            path=data_path_str,
        )
    )
    monkeypatch.setattr(import_artifact, "update_import", mock_update)

    result = runner.invoke(cli.cli, ["update"])
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Updated:" in result.output
    mock_update.assert_called_once()


# =============================================================================
# Skip non-import
# =============================================================================


def test_cli_update_skips_non_import(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update skips targets whose .pvt has no source (non-import)."""
    pvt_data = _make_non_import_pvt()
    pvt_path = tmp_path / "local.csv.pvt"

    monkeypatch.setattr(project, "get_project_root", _helper_get_project_root(tmp_path))
    monkeypatch.setattr(track, "read_pvt_file", _helper_read_pvt(pvt_path, pvt_data))
    monkeypatch.setattr(track, "is_import", _helper_is_import)
    monkeypatch.setattr(track, "get_pvt_path", _helper_get_pvt_path)
    monkeypatch.setattr(
        project,
        "normalize_path",
        _helper_normalize_path(tmp_path),
    )

    result = runner.invoke(cli.cli, ["update", "local.csv"])
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Skipping non-import:" in result.output


# =============================================================================
# Dry run
# =============================================================================


def test_cli_update_dry_run(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dry-run calls check_for_update, not update_import."""
    pvt_data = _make_import_pvt(rev_lock="old11111old22")

    monkeypatch.setattr(project, "get_project_root", _helper_get_project_root(tmp_path))
    monkeypatch.setattr(
        track,
        "discover_import_pvt_files",
        _helper_discover_imports({str(tmp_path / "data.csv"): pvt_data}),
    )
    monkeypatch.setattr(track, "get_pvt_path", _helper_get_pvt_path)

    mock_check = AsyncMock(
        return_value=UpdateCheck(
            available=True,
            current_rev="old11111old22222",
            latest_rev="new33333new44444",
        )
    )
    monkeypatch.setattr(import_artifact, "check_for_update", mock_check)

    mock_update = AsyncMock()
    monkeypatch.setattr(import_artifact, "update_import", mock_update)

    result = runner.invoke(cli.cli, ["update", "--dry-run"])
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Update available:" in result.output
    assert "old11111" in result.output
    assert "new33333" in result.output
    mock_check.assert_called_once_with(pvt_data)
    mock_update.assert_not_called()


# =============================================================================
# No imports found
# =============================================================================


def test_cli_update_no_imports_found(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update with no imports shows 'No imports found' message."""
    monkeypatch.setattr(project, "get_project_root", _helper_get_project_root(tmp_path))
    monkeypatch.setattr(track, "discover_import_pvt_files", _helper_discover_imports({}))

    result = runner.invoke(cli.cli, ["update"])
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "No imports found to update." in result.output
