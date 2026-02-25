from __future__ import annotations

import asyncio
import pathlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from pivot import import_artifact, loaders, registry, status, types
from pivot.engine import graph as engine_graph
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import track

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _helper_stage_func() -> None:
    return None


def _artifact_ref(identity_key: str) -> types.ArtifactRef:
    identity = types.identity_from_key(identity_key)
    return types.ArtifactRef(
        identity=identity,
        format=loaders.PathOnly(),
        python_type=str,
        tag=types.ArtifactTag.DATA,
    )


def _stage_info(
    name: str,
    *,
    deps: list[str] | None = None,
    outs: list[str] | None = None,
    fingerprint: dict[str, str] | None = None,
) -> registry.RegistryStageInfo:
    dep_keys = deps or []
    out_keys = outs or []
    return registry.RegistryStageInfo(
        func=_helper_stage_func,
        name=name,
        deps={f"d{i}": _artifact_ref(dep) for i, dep in enumerate(dep_keys)},
        outs=[_artifact_ref(out) for out in out_keys],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint=fingerprint or {"self:stage": "fp"},
        params_arg_name=None,
        state_dir=None,
        collection_params={},
    )


def test_discovers_tracked_files_only_when_allow_missing(mocker: MockerFixture) -> None:
    discovered = {"/tmp/a": track.PvtData(path="a", hash="h", size=1)}
    discover = mocker.patch.object(
        track, "discover_pvt_files", autospec=True, return_value=discovered
    )

    tracked, trie = status._discover_tracked_files(False)  # noqa: SLF001
    assert tracked is None and trie is None

    tracked, trie = status._discover_tracked_files(True)  # noqa: SLF001
    assert tracked == discovered
    assert trie is None
    discover.assert_called_once()


def test_get_explanations_in_parallel_handles_worker_errors(mocker: MockerFixture) -> None:
    a = _stage_info("a", outs=["a.txt"])
    b = _stage_info("b", deps=["a.txt"], outs=["b.txt"])
    all_stages = {"a": a, "b": b}

    def _fake_explain(stage_name: str, *_args: Any, **_kwargs: Any) -> types.StageExplanation:
        if stage_name == "b":
            raise RuntimeError("boom")
        return types.StageExplanation(
            stage_name=stage_name,
            will_run=False,
            is_forced=False,
            reason="",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        )

    mocker.patch.object(status.explain, "get_stage_explanation", side_effect=_fake_explain)
    mocker.patch.object(status.parameters, "load_params_yaml", autospec=True, return_value=None)

    explanations = status._get_explanations_in_parallel(  # noqa: SLF001
        ["a", "b"],
        overrides=None,
        all_stages=all_stages,
    )

    assert explanations["a"]["will_run"] is False
    assert explanations["b"]["will_run"] is True
    assert "Error: boom" in explanations["b"]["reason"]


def test_pipeline_status_and_explanations_propagate_upstream_staleness(
    mocker: MockerFixture,
) -> None:
    stages = {
        "a": _stage_info("a", outs=["a.txt"]),
        "b": _stage_info("b", deps=["a.txt"], outs=["b.txt"]),
    }
    stage_registry = registry.StageRegistry()
    stage_registry.add_existing(stages["a"])
    stage_registry.add_existing(stages["b"])
    graph = engine_graph.build_graph(stages)

    def _fake_explain(stage_name: str, *_args: Any, **_kwargs: Any) -> types.StageExplanation:
        return types.StageExplanation(
            stage_name=stage_name,
            will_run=(stage_name == "a"),
            is_forced=False,
            reason="Code changed" if stage_name == "a" else "",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        )

    mocker.patch.object(status.explain, "get_stage_explanation", side_effect=_fake_explain)
    mocker.patch.object(status.parameters, "load_params_yaml", autospec=True, return_value=None)

    statuses, _stage_graph = status.get_pipeline_status(
        stages=None,
        single_stage=False,
        all_stages=stages,
        stage_registry=stage_registry,
        graph=graph,
    )
    explanations = status.get_pipeline_explanations(
        stages=None,
        single_stage=False,
        all_stages=stages,
        stage_registry=stage_registry,
        graph=graph,
    )

    assert [s["name"] for s in statuses] == ["a", "b"]
    assert statuses[0]["status"] is types.PipelineStatus.STALE
    assert statuses[1]["status"] is types.PipelineStatus.STALE
    assert statuses[1]["upstream_stale"] == ["a"]
    assert "Upstream stale" in statuses[1]["reason"]
    assert explanations[1]["upstream_stale"] == ["a"]


def test_get_tracked_files_status_reports_clean_modified_and_missing(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path
    (root / ".pivot").mkdir()
    clean_file = root / "clean.txt"
    modified_file = root / "modified.txt"
    missing_file = root / "missing.txt"
    clean_file.write_text("clean")
    modified_file.write_text("modified")

    tracked = {
        str(clean_file): track.PvtData(path="clean.txt", hash="h1", size=1),
        str(modified_file): track.PvtData(path="modified.txt", hash="h2", size=2),
        str(missing_file): track.PvtData(path="missing.txt", hash="h3", size=3),
    }
    mocker.patch.object(track, "discover_pvt_files", autospec=True, return_value=tracked)
    mocker.patch.object(
        status.config, "get_state_db_path", autospec=True, return_value=root / ".pivot" / "state.db"
    )

    def _hash_file(path: pathlib.Path, *_args: Any, **_kwargs: Any) -> tuple[str, int]:
        if path == clean_file:
            return "h1", 0
        if path == modified_file:
            return "different", 0
        raise FileNotFoundError(path)

    mocker.patch.object(status.cache, "hash_file", autospec=True, side_effect=_hash_file)
    progress: list[tuple[int, int]] = []

    results = status.get_tracked_files_status(
        root, on_progress=lambda done, total: progress.append((done, total))
    )
    by_path = {row["path"]: row["status"] for row in results}

    assert by_path["clean.txt"] is types.TrackedFileStatus.CLEAN
    assert by_path["modified.txt"] is types.TrackedFileStatus.MODIFIED
    assert by_path["missing.txt"] is types.TrackedFileStatus.MISSING
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_remote_status_covers_no_remote_empty_cache_and_sync(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    list_remotes = mocker.patch.object(
        remote_config, "list_remotes", autospec=True, return_value=[]
    )
    with pytest.raises(status.exceptions.RemoteNotConfiguredError):
        status.get_remote_status(None, tmp_path)

    list_remotes.return_value = ["origin"]
    mocker.patch.object(
        transfer,
        "create_remote_from_name",
        autospec=True,
        return_value=(object(), "origin"),
    )
    mocker.patch.object(remote_config, "get_remote_url", autospec=True, return_value="s3://bucket")
    local_hashes = mocker.patch.object(
        transfer, "get_local_cache_hashes", autospec=True, return_value=set()
    )
    empty = status.get_remote_status(None, tmp_path)
    assert empty["push_count"] == 0 and empty["pull_count"] == 0

    local_hashes.return_value = {"a"}
    mocker.patch.object(
        transfer,
        "compare_status",
        new=AsyncMock(return_value={"local_only": {"l"}, "remote_only": {"r1", "r2"}}),
    )
    sync = status.get_remote_status(None, tmp_path)
    assert sync["push_count"] == 1
    assert sync["pull_count"] == 2


def test_suggestions_pluralization_and_what_if_changed() -> None:
    suggestions = status.get_suggestions(
        stale_count=1, modified_count=2, push_count=1, pull_count=3
    )

    assert "1 stale stage" in suggestions[0]
    assert "2 modified files" in suggestions[1]
    assert "1 file" in suggestions[2]
    assert "3 files" in suggestions[3]

    affected = status.what_if_changed(
        [pathlib.Path("x")],
        all_stages={"b": _stage_info("b"), "a": _stage_info("a")},
        graph=None,
    )
    assert affected == ["a", "b"]


def test_import_status_batch_handles_update_up_to_date_and_error(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    in_project = tmp_path / "data" / "artifact.bin"
    outside = pathlib.Path("/tmp/outside.bin")
    pvt_a = track.PvtData(path="artifact.bin", hash="a", size=1)
    pvt_b = track.PvtData(path="outside.bin", hash="b", size=2)
    import_pvts = {str(in_project): pvt_a, str(outside): pvt_b}

    async def _check_for_update(pvt_data: track.PvtData) -> import_artifact.UpdateCheck:
        if pvt_data["hash"] == "a":
            return import_artifact.UpdateCheck(
                available=True,
                current_rev="1234567890",
                latest_rev="abcdef1234",
            )
        raise RuntimeError("network")

    mocker.patch.object(
        import_artifact, "check_for_update", new=AsyncMock(side_effect=_check_for_update)
    )

    batch = asyncio.run(status._check_imports_batch(import_pvts, tmp_path))  # noqa: SLF001
    by_path = {item.path: item for item in batch}
    assert by_path["data/artifact.bin"].status is status.ImportCheckStatus.UPDATE_AVAILABLE
    assert by_path["data/artifact.bin"].current_rev == "12345678"
    assert by_path["/tmp/outside.bin"].status is status.ImportCheckStatus.ERROR

    mocker.patch.object(track, "discover_import_pvt_files", autospec=True, return_value={})
    assert status.get_import_status(tmp_path) == []
