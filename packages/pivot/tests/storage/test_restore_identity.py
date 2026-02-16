from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import yaml

from pivot import project, types
from pivot.storage import restore

if TYPE_CHECKING:
    from conftest import GitRepo


def test_entry_display_uses_identity_key() -> None:
    entry = cast(
        "types.OutEntry",
        {"key": "model", "hash": "abc123", "tag": "data", "path": "legacy/path"},
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
