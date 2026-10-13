# pyright: reportMissingImports=false, reportImplicitRelativeImport=false
from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from pivot import project
from pivot.storage import restore

if TYPE_CHECKING:
    import pathlib

    import pytest

    from conftest import GitRepo


def _stage_target(
    repo_path: pathlib.Path, stage_name: str, paths_by_key: dict[str | None, str]
) -> restore.StageTarget:
    return restore.StageTarget(
        stage_name=stage_name, state_dir=repo_path / ".pivot", paths_by_key=paths_by_key
    )


def test_resolve_targets_stage_matches_lock_entries_by_identity_key(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each lock entry's hash lands at the workspace path of the output with the same key."""
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
    stage_targets = {
        "train": _stage_target(
            repo_path, "train", {None: "data/train.csv", "model": "data/train/model.pkl"}
        )
    }

    targets = restore.resolve_targets(["train"], sha, stage_targets)

    assert len(targets) == 1
    assert targets[0]["hashes"] == {
        "data/train.csv": {"hash": "hash1"},
        "data/train/model.pkl": {"hash": "hash2"},
    }
    assert set(targets[0]["paths"]) == set(targets[0]["hashes"])


def test_resolve_targets_keyed_target_restores_only_that_output(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``stage:key`` target maps only that key, so other lock entries are not restored."""
    repo_path, commit = git_repo

    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [
            {"key": "model", "hash": "hash2", "tag": "data"},
            {"key": "table", "hash": "hash3", "tag": "data"},
        ],
    }
    (repo_path / ".pivot" / "stages" / "train.lock").write_text(yaml.dump(lock_content))

    sha = commit("add lock file")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    stage_targets = {
        "train:model": _stage_target(repo_path, "train", {"model": "data/train/model.pkl"})
    }

    targets = restore.resolve_targets(["train:model"], sha, stage_targets)

    assert targets[0]["paths"] == ["data/train/model.pkl"]


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
    stage_targets = {
        "train": _stage_target(repo_path, "train", {None: "data/train.csv"}),
        "eval": _stage_target(repo_path, "eval", {None: "data/eval.csv"}),
    }

    targets = restore.resolve_targets(["train", "eval"], sha, stage_targets)

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
    stage_targets = {"eval": _stage_target(repo_path, "eval", {"score": "metrics/eval/score.json"})}

    targets = restore.resolve_targets(["eval"], sha, stage_targets)

    assert len(targets) == 1
    assert targets[0]["original_target"] == "eval"
    # Metrics are included in paths
    assert targets[0]["paths"] == ["metrics/eval/score.json"]
