# pyright: reportMissingImports=false, reportImplicitRelativeImport=false
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import yaml

from pivot import project, types
from pivot.storage import restore

if TYPE_CHECKING:
    import pytest

    from conftest import GitRepo


def test_entry_display_uses_identity_key() -> None:
    entry = cast(
        "types.OutEntry",
        cast(
            "object",
            {"key": "model", "hash": "abc123", "tag": "data", "path": "legacy/path"},
        ),
    )

    assert restore._entry_display(entry) == "model"


def test_resolve_targets_stage_uses_identity_keys(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_path, commit = git_repo

    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "code_manifest": {"func:main": "abc123"},
        "params": {},
        "deps": [],
        "outs": [
            {"key": None, "hash": "hash1", "tag": "data"},
            {"key": "model", "hash": "hash2", "tag": "data"},
        ],
    }
    (repo_path / ".pivot" / "stages" / "train.lock").write_text(yaml.dump(lock_content))

    sha = commit("add lock file")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    state_dir = repo_path / ".pivot"

    targets = restore.resolve_targets(["train"], sha, state_dir)

    assert len(targets) == 1
    paths = set(targets[0]["paths"])
    assert paths == {"train", "train:model"}
    assert set(targets[0]["hashes"].keys()) == paths


def test_resolve_targets_multiple_stages(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple stages should each produce a TargetInfo."""
    repo_path, commit = git_repo

    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    train_lock = {
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [{"key": None, "hash": "train_hash", "tag": "data"}],
    }
    eval_lock = {
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [{"key": None, "hash": "eval_hash", "tag": "data"}],
    }
    (repo_path / ".pivot" / "stages" / "train.lock").write_text(yaml.dump(train_lock))
    (repo_path / ".pivot" / "stages" / "eval.lock").write_text(yaml.dump(eval_lock))

    sha = commit("add lock files")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    state_dir = repo_path / ".pivot"

    targets = restore.resolve_targets(["train", "eval"], sha, state_dir)

    assert len(targets) == 2
    stage_names = {t["original_target"] for t in targets}
    assert stage_names == {"train", "eval"}


def test_resolve_targets_stage_with_only_metrics(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stages with only METRIC outputs should still be included."""
    repo_path, commit = git_repo

    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [{"key": "score", "hash": "metric_hash", "tag": "metric"}],
    }
    (repo_path / ".pivot" / "stages" / "eval.lock").write_text(yaml.dump(lock_content))

    sha = commit("add lock file")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    state_dir = repo_path / ".pivot"

    targets = restore.resolve_targets(["eval"], sha, state_dir)

    assert len(targets) == 1
    assert targets[0]["original_target"] == "eval"
    # Metrics are included in paths
    assert "eval:score" in targets[0]["paths"]
