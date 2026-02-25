# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import pathlib
from typing import Any

from pivot import loaders, types
from pivot.executor import worker
from pivot.storage import state
from pivot.storage import store as store_mod


def _helper_noop() -> None:
    return None


class _HelperStageLock:
    def __init__(self, changed: bool, reason: str) -> None:
        self._changed = changed
        self._reason = reason

    def is_changed_with_lock_data(
        self,
        _lock_data: types.LockData,
        _current_fingerprint: dict[str, str],
        _current_params: dict[str, Any],
        _dep_hashes: dict[str, types.HashInfo],
        _out_paths: list[str],
    ) -> tuple[bool, str]:
        return self._changed, self._reason


def _helper_ref(producer: str, key: str | None) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer, key),
        format=loaders.Text(),
        python_type=str,
        tag=types.ArtifactTag.DATA,
    )


def _helper_lock_data(identity: types.ArtifactIdentity) -> types.LockData:
    return types.LockData(
        code_manifest={"self": "abc"},
        params={},
        dep_hashes={},
        output_hashes={identity: types.FileHash(hash="out-hash")},
        merkle_id=None,
    )


def test_hash_dependencies_dict_path_uses_workspace_store(tmp_path: pathlib.Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello")

    dep_ref = _helper_ref("upstream", None)
    store = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={"upstream": "input.txt"},
    )

    hashes, missing, unreadable, file_entries = worker.hash_dependencies({"data": dep_ref}, store)

    assert missing == []
    assert unreadable == []
    assert "upstream" in hashes
    assert file_entries, "File dependencies should produce file hash entries"


def test_hash_dependencies_list_path_hashes_files_and_directories(
    set_project_root: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(set_project_root)
    (set_project_root / "nested").mkdir()
    (set_project_root / "nested" / "part.txt").write_text("part")
    (set_project_root / "data.txt").write_text("value")

    hashes, missing, unreadable, file_entries = worker.hash_dependencies(["data.txt", "nested"])

    assert missing == []
    assert unreadable == []
    assert "data.txt" in hashes
    assert "nested" in hashes
    assert "manifest" in hashes["nested"], "Directory dependency should include manifest"
    assert any(entry[0].endswith("data.txt") for entry in file_entries)


def test_can_skip_via_generation_handles_dict_deps(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    dep_ref = _helper_ref("upstream", "train")
    dep_key = types.identity_key(dep_ref.identity)

    with state.StateDB(db_path) as db:
        db.increment_generation(dep_key)
        db.record_dep_generations("stage", {dep_key: 1})
        can_skip = worker.can_skip_via_generation(
            stage_name="stage",
            fingerprint={"self": "abc"},
            deps={"x": dep_ref},
            outs_paths=["stage:out"],
            current_params={},
            lock_data=_helper_lock_data(types.ArtifactIdentity("stage", "out")),
            state_db=db,
        )

    assert can_skip is True


def test_can_skip_via_generation_fails_on_generation_mismatch(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    dep_ref = _helper_ref("upstream", "train")
    dep_key = types.identity_key(dep_ref.identity)

    with state.StateDB(db_path) as db:
        db.increment_generation(dep_key)
        db.record_dep_generations("stage", {dep_key: 1})
        db.increment_generation(dep_key)
        can_skip = worker.can_skip_via_generation(
            stage_name="stage",
            fingerprint={"self": "abc"},
            deps={"x": dep_ref},
            outs_paths=["stage:out"],
            current_params={},
            lock_data=_helper_lock_data(types.ArtifactIdentity("stage", "out")),
            state_db=db,
        )

    assert can_skip is False


def test_check_skip_or_run_and_make_result_paths() -> None:
    dep_ref = _helper_ref("upstream", "train")
    stage_info = worker.WorkerStageInfo(
        func=_helper_noop,
        fingerprint={"self": "abc"},
        deps={"x": dep_ref},
        outs=[_helper_ref("stage", "out")],
        store_spec={
            "kind": "workspace",
            "cache_dir": "/tmp/cache",
            "project_root": "/tmp",
            "pipeline_name": "pipe",
            "input_bindings": {},
        },
        signature=None,
        params=None,
        variant=None,
        overrides={},
        checkout_modes=[],
        run_id="run",
        force=False,
        no_commit=True,
        params_arg_name=None,
        project_root=pathlib.Path("/tmp"),
        state_dir=pathlib.Path("/tmp/.pivot"),
        collection_params={},
    )
    dep_hashes = {types.identity_key(dep_ref.identity): types.FileHash(hash="dep-hash")}

    skip_reason, run_reason, input_hash = worker._check_skip_or_run(
        stage_info,
        _HelperStageLock(changed=True, reason="deps changed"),  # pyright: ignore[reportArgumentType] - test double
        None,
        {"self": "abc"},
        {},
        dep_hashes,
    )
    assert skip_reason is None
    assert run_reason == "No previous run"
    assert input_hash

    skip_reason, run_reason, _ = worker._check_skip_or_run(
        stage_info,
        _HelperStageLock(changed=False, reason=""),  # pyright: ignore[reportArgumentType] - test double
        _helper_lock_data(types.ArtifactIdentity("stage", "out")),
        {"self": "abc"},
        {},
        dep_hashes,
    )
    assert skip_reason == "unchanged"
    assert run_reason == ""

    result = worker._make_result(
        types.StageStatus.CACHED,
        "unchanged",
        worker._OutputRingBuffer(max_lines=10),
        accessed_dep_keys={"group": {"a.txt"}},
    )
    assert result["status"] == types.StageStatus.CACHED
    assert result.get("accessed_dep_keys") == {"group": {"a.txt"}}
