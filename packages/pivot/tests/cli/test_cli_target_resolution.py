from __future__ import annotations

from typing import TYPE_CHECKING

import click
import pytest

from pivot import project
from pivot.cli import remote as remote_mod

if TYPE_CHECKING:
    import pathlib


def _helper_setup_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".pivot").mkdir()
    (project_dir / ".git").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", None)
    monkeypatch.chdir(project_dir)
    return project_dir


def test_normalize_cli_targets_known_stage_passed_through(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(
        ("train_model",), known_stages={"train_model", "evaluate"}
    )

    assert result == ("train_model",)


def test_normalize_cli_targets_pvt_suffix_stripped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(("data/foo.csv.pvt",))

    assert result == (str(project_dir / "data" / "foo.csv"),)


def test_normalize_cli_targets_relative_path_resolved_from_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(("data/input.csv",))

    assert result == (str(project_dir / "data" / "input.csv"),)


def test_normalize_cli_targets_absolute_path_within_project(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)
    target_path = project_dir / "data" / "bar.csv"

    result = remote_mod._normalize_cli_targets((str(target_path),))

    assert result == (str(target_path),)


def test_normalize_cli_targets_path_outside_project_raises(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _helper_setup_project(tmp_path, monkeypatch)

    with pytest.raises(click.ClickException, match="resolves outside project root"):
        remote_mod._normalize_cli_targets(("../outside.csv",))


def test_normalize_cli_targets_dotdot_within_project_works(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(("data/../data/file.csv",))

    assert result == (str(project_dir / "data" / "file.csv"),)


def test_normalize_cli_targets_root_level_file_treated_as_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(("output.csv",))

    assert result == (str(project_dir / "output.csv"),), (
        "Root-level file without slash should be normalized as a path"
    )


def test_normalize_cli_targets_root_level_pvt_stripped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(("output.csv.pvt",))

    assert result == (str(project_dir / "output.csv"),), (
        ".pvt suffix should be stripped even without slash"
    )


def test_normalize_cli_targets_unknown_target_without_slash_normalized(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(("output.csv",), known_stages={"train_model"})

    assert result == (str(project_dir / "output.csv"),), (
        "Non-stage target without slash should be normalized"
    )


def test_normalize_cli_targets_no_pipeline_all_treated_as_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(("train_model", "data/input.csv"), known_stages=None)

    assert result == (
        str(project_dir / "train_model"),
        str(project_dir / "data" / "input.csv"),
    ), "Without pipeline, all targets should be treated as paths"


def test_normalize_cli_targets_backslash_path_normalized(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)

    result = remote_mod._normalize_cli_targets(("data\\foo.csv",))

    assert result == (str(project_dir / "data" / "foo.csv"),), (
        "Backslash paths should be normalized to POSIX"
    )


def test_normalize_cli_targets_from_subdirectory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)
    subdir = project_dir / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    result = remote_mod._normalize_cli_targets(("../data/foo.csv",))

    assert result == (str(project_dir / "data" / "foo.csv"),), (
        "Relative path with .. from subdirectory should resolve correctly"
    )


def test_normalize_cli_targets_from_subdirectory_pvt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _helper_setup_project(tmp_path, monkeypatch)
    subdir = project_dir / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    result = remote_mod._normalize_cli_targets(("../data/foo.csv.pvt",))

    assert result == (str(project_dir / "data" / "foo.csv"),), (
        ".pvt should be stripped and path resolved from subdirectory"
    )
