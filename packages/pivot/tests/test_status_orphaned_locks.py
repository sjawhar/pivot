"""Orphaned lock detection scans the state directories the engine writes locks to."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from pivot import config, status
from pivot.registry import RegistryStageInfo

if TYPE_CHECKING:
    import pathlib

    import pytest


def _stage_info(name: str, state_dir: pathlib.Path | None) -> RegistryStageInfo:
    def _stage_func() -> None:
        return None

    return RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        outs=[],
        params=None,
        resources={},
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint=None,
        params_arg_name=None,
        state_dir=state_dir,
        collection_params={},
        no_fingerprint=False,
        code_deps=[],
    )


def _touch_lock(state_dir: pathlib.Path, stage_name: str) -> None:
    lock_path = state_dir / "stages" / f"{stage_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("")


def test_find_orphaned_lock_files_scans_each_stage_state_dir(
    set_project_root: pathlib.Path,
) -> None:
    sub_state_dir = set_project_root / "sub" / ".pivot"
    all_stages = {
        "main/train": _stage_info("main/train", set_project_root / ".pivot"),
        "sub/train": _stage_info("sub/train", sub_state_dir),
    }
    _touch_lock(set_project_root / ".pivot", "main/train")
    _touch_lock(sub_state_dir, "sub/train")
    _touch_lock(sub_state_dir, "sub/renamed_away")

    assert status.find_orphaned_lock_files(all_stages) == ["sub/renamed_away"]


def test_find_orphaned_lock_files_includes_default_dir_with_only_custom_stages(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom stage dir does not suppress cleanup of stale locks in the configured default."""
    default_state_dir = set_project_root / ".pivot"
    custom_state_dir = set_project_root / "custom" / ".pivot"
    monkeypatch.setattr(config, "get_state_dir", lambda: default_state_dir)
    all_stages = {"custom/train": _stage_info("custom/train", custom_state_dir)}
    _touch_lock(custom_state_dir, "custom/train")
    _touch_lock(default_state_dir, "deleted/train")

    assert status.find_orphaned_lock_files(all_stages) == ["deleted/train"]


def test_find_orphaned_lock_files_scans_default_dir_without_stages(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting every stage does not hide stale locks in the configured default directory."""
    default_state_dir = set_project_root / ".pivot"
    monkeypatch.setattr(config, "get_state_dir", lambda: default_state_dir)
    _touch_lock(default_state_dir, "deleted/train")

    assert status.find_orphaned_lock_files({}) == ["deleted/train"]
