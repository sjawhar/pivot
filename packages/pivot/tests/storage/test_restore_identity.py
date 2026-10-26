# pyright: reportMissingImports=false, reportImplicitRelativeImport=false
from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from pivot import project, types
from pivot.storage import listing, restore

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
    """Each lock entry's record and committed listing land at the output with the same key."""
    repo_path, commit = git_repo

    stages_dir = repo_path / ".pivot" / "stages"
    stages_dir.mkdir(parents=True)
    directory_members = listing.DirListing(
        files=[
            types.DirManifestEntry(
                relpath="weights.bin", hash="2222222222222222", size=7, isexec=False
            )
        ],
        empty_dirs=[],
    )
    directory_data = listing.encode(directory_members)
    directory_record = listing.record_of(directory_members, directory_data)
    lock_content = {
        "schema_version": 3,
        "code_manifest": {"func:main": "abc123"},
        "params": {},
        "deps": [],
        "outs": [
            {"key": None, "hash": "1111111111111111", "tag": "data"},
            {
                "key": "model",
                "tag": "data",
                "listing": directory_record["hash"],
                "size": directory_record["size"],
                "nfiles": directory_record["nfiles"],
            },
        ],
    }
    (stages_dir / "train.lock").write_text(yaml.dump(lock_content))
    listing.write(
        listing.output_listing_path(stages_dir, types.ArtifactIdentity("train", "model")),
        directory_data,
    )

    sha = commit("add lock and listing")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    stage_targets = {
        "train": _stage_target(
            repo_path, "train", {None: "data/train.csv", "model": "data/train/model"}
        )
    }

    targets = restore.resolve_targets(["train"], sha, stage_targets)

    assert len(targets) == 1
    assert targets[0]["hashes"] == {
        "data/train.csv": {"hash": "1111111111111111"},
        "data/train/model": directory_record,
    }
    assert targets[0]["members"] == {"data/train/model": directory_members}
    assert set(targets[0]["paths"]) == set(targets[0]["hashes"])


def test_resolve_targets_keyed_target_restores_only_that_output(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``stage:key`` target maps only that key, so other lock entries are not restored."""
    repo_path, commit = git_repo

    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "schema_version": 3,
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [
            {"key": "model", "hash": "2222222222222222", "tag": "data"},
            {"key": "table", "hash": "3333333333333333", "tag": "data"},
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
        "schema_version": 3,
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [{"key": None, "hash": "7a17a17a17a17a10", "tag": "data"}],
    }
    eval_lock = {
        "schema_version": 3,
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [{"key": None, "hash": "e1a1e1a1e1a1e1a1", "tag": "data"}],
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
        "schema_version": 3,
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [{"key": "score", "hash": "3e7c3e7c3e7c3e7c", "tag": "metric"}],
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
