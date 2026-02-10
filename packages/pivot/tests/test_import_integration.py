# pyright: reportMissingImports=false
"""Integration tests for cross-repo import: DAG recognition, discovery, roundtrip, full flow."""

from __future__ import annotations

import pathlib
import subprocess
from typing import TYPE_CHECKING

import pytest
import xxhash
import yaml

import conftest
from pivot import exceptions, import_artifact, loaders, outputs
from pivot.remote import storage as remote_mod

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from types_aiobotocore_s3 import S3Client
from pivot.engine import graph
from pivot.registry import RegistryStageInfo
from pivot.storage import track


def _create_stage(name: str, deps: list[str], outs: list[str]) -> RegistryStageInfo:
    """Create a minimal RegistryStageInfo for graph tests."""
    return RegistryStageInfo(
        func=lambda: None,
        name=name,
        deps={f"_{i}": d for i, d in enumerate(deps)},
        deps_paths=deps,
        outs=[
            outputs.require_expanded(outputs.Out(path=out, loader=loaders.PathOnly()))
            for out in outs
        ],
        outs_paths=outs,
        params=None,
        mutex=list[str](),
        variant=None,
        signature=None,
        fingerprint=dict[str, str](),
        dep_specs={},
        out_specs=dict[str, outputs.BaseOut](),
        params_arg_name=None,
        state_dir=None,
    )


def _make_import_source(
    *,
    repo: str = "https://github.com/org/upstream",
    rev: str = "main",
    rev_lock: str = "abc123deadbeef",
    stage: str = "train",
    path: str = "data/output.csv",
    remote: str = "s3://bucket/prefix",
) -> track.ImportSource:
    return track.ImportSource(
        repo=repo, rev=rev, rev_lock=rev_lock, stage=stage, path=path, remote=remote
    )


def _lock_bytes(*, outs: list[dict[str, object]]) -> bytes:
    data = {
        "code_manifest": {"stage": "hash"},
        "params": {},
        "deps": [],
        "outs": outs,
    }
    return yaml.safe_dump(data).encode()


def _config_bytes(remote_url: str) -> bytes:
    data = {
        "remotes": {"origin": remote_url},
        "default_remote": "origin",
    }
    return yaml.safe_dump(data).encode()


# --- Test 1: Import .pvt files recognized as DAG dependencies ---


def test_import_pvt_recognized_as_dag_dep(tmp_path: pathlib.Path) -> None:
    """Imported .pvt file is accepted as a valid dependency in the DAG (validate=True)."""
    imported_data = tmp_path / "imported_data.csv"
    output_file = tmp_path / "result.csv"

    # Build tracked_files dict as discover_pvt_files would produce it,
    # including an ImportSource to prove imports work identically to regular tracked files.
    tracked_files: dict[str, track.PvtData] = {
        str(imported_data): track.PvtData(
            path="imported_data.csv",
            hash="abc123",
            size=1024,
            source=_make_import_source(path="data/imported_data.csv"),
        )
    }

    stages = {
        "consumer": _create_stage("consumer", [str(imported_data)], [str(output_file)]),
    }

    # With validate=True, a missing dep that isn't in tracked_files would raise.
    # This must NOT raise because the import .pvt is in tracked_files.
    g = graph.build_graph(stages, validate=True, tracked_files=tracked_files)

    assert "stage:consumer" in g, "Consumer stage should be in the graph"
    assert graph.artifact_node(imported_data) in g, "Imported artifact should be a graph node"


def test_import_pvt_missing_from_tracked_raises(tmp_path: pathlib.Path) -> None:
    """Without tracked_files entry, a missing imported dep raises on validate."""
    missing_dep = tmp_path / "not_tracked.csv"
    output_file = tmp_path / "result.csv"

    stages = {
        "consumer": _create_stage("consumer", [str(missing_dep)], [str(output_file)]),
    }

    with pytest.raises(exceptions.DependencyNotFoundError):
        graph.build_graph(stages, validate=True)


def test_import_pvt_chain_through_dag(tmp_path: pathlib.Path) -> None:
    """Import dep feeds into a stage whose output feeds into another — full chain works."""
    imported = tmp_path / "external.csv"
    intermediate = tmp_path / "cleaned.csv"
    final = tmp_path / "model.pkl"

    tracked_files: dict[str, track.PvtData] = {
        str(imported): track.PvtData(
            path="external.csv",
            hash="aaa",
            size=100,
            source=_make_import_source(),
        )
    }

    stages = {
        "clean": _create_stage("clean", [str(imported)], [str(intermediate)]),
        "train": _create_stage("train", [str(intermediate)], [str(final)]),
    }

    g = graph.build_graph(stages, validate=True, tracked_files=tracked_files)
    stage_dag = graph.get_stage_dag(g)

    # train depends on clean
    assert stage_dag.has_edge("train", "clean"), "Train should depend on clean"
    # Execution order should be clean -> train
    order = graph.get_execution_order(stage_dag)
    assert order.index("clean") < order.index("train")


# --- Test 2: discover_import_pvt_files end-to-end ---


@pytest.mark.usefixtures("set_project_root")
def test_discover_import_pvt_files_end_to_end(tmp_path: pathlib.Path) -> None:
    """discover_import_pvt_files finds only .pvt files with source (imports)."""
    # Create an import .pvt file
    track.write_pvt_file(
        tmp_path / "imported.csv.pvt",
        track.PvtData(
            path="imported.csv",
            hash="hash_imported",
            size=500,
            source=_make_import_source(path="data/imported.csv"),
        ),
    )

    # Create a second import in a subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    track.write_pvt_file(
        subdir / "model.pkl.pvt",
        track.PvtData(
            path="model.pkl",
            hash="hash_model",
            size=2000,
            source=_make_import_source(stage="eval", path="models/model.pkl"),
        ),
    )

    # Create a non-import (tracked-only) .pvt file
    track.write_pvt_file(
        tmp_path / "local.csv.pvt",
        track.PvtData(path="local.csv", hash="hash_local", size=100),
    )

    # Create another non-import .pvt
    track.write_pvt_file(
        tmp_path / "another.csv.pvt",
        track.PvtData(path="another.csv", hash="hash_another", size=200),
    )

    result = track.discover_import_pvt_files(tmp_path)

    assert len(result) == 2, f"Expected 2 import files, got {len(result)}: {list(result.keys())}"
    # Verify all results have source
    for path, pvt_data in result.items():
        assert "source" in pvt_data, f"Import pvt at {path} should have source"
        assert track.is_import(pvt_data), f"Import pvt at {path} should pass is_import()"


# --- Test 3: Import .pvt roundtrip (write + read) ---


def test_import_pvt_roundtrip(tmp_path: pathlib.Path) -> None:
    """PvtData with ImportSource writes to disk and reads back identically."""
    pvt_path = tmp_path / "data.csv.pvt"
    source = _make_import_source(
        repo="https://github.com/org/upstream",
        rev="v2.0",
        rev_lock="deadbeef12345678",
        stage="preprocess",
        path="outputs/data.csv",
        remote="s3://my-bucket/cache",
    )
    original = track.PvtData(
        path="data.csv",
        hash="xxhash_value_here",
        size=4096,
        source=source,
    )

    track.write_pvt_file(pvt_path, original)
    loaded = track.read_pvt_file(pvt_path)

    assert loaded is not None, "Should successfully read back the .pvt file"
    assert loaded["path"] == original["path"]
    assert loaded["hash"] == original["hash"]
    assert loaded["size"] == original["size"]
    assert "source" in loaded, "Loaded data should contain source"

    loaded_source = loaded["source"]
    assert loaded_source["repo"] == source["repo"]
    assert loaded_source["rev"] == source["rev"]
    assert loaded_source["rev_lock"] == source["rev_lock"]
    assert loaded_source["stage"] == source["stage"]
    assert loaded_source["path"] == source["path"]
    assert loaded_source["remote"] == source["remote"]

    assert track.is_import(loaded), "Roundtripped data should pass is_import()"


# --- Test 4: Import from real local git repo (no mocks of git/resolve) ---


def _setup_source_repo(
    source_dir: pathlib.Path,
    remote_url: str,
    stage_name: str,
    outs: list[dict[str, object]],
) -> str:
    """Create a git repo with .pivot/ config and lock file committed. Returns HEAD SHA."""
    source_dir.mkdir(exist_ok=True)
    conftest.init_git_repo(source_dir)
    # git archive --remote with SHAs requires this for local repos
    subprocess.run(
        ["git", "config", "uploadArchive.allowUnreachable", "true"],
        cwd=source_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "main"],
        cwd=source_dir,
        check=False,
        capture_output=True,
    )

    pivot_dir = source_dir / ".pivot"
    pivot_dir.mkdir()
    (pivot_dir / "stages").mkdir()

    config = {"remotes": {"origin": remote_url}}
    (pivot_dir / "config.yaml").write_text(yaml.safe_dump(config))

    lock_data = {
        "schema_version": 1,
        "code_manifest": {"self:run": "abcdef1234567890"},
        "params": {},
        "deps": [],
        "outs": outs,
    }
    (pivot_dir / "stages" / f"{stage_name}.lock").write_text(yaml.safe_dump(lock_data))

    subprocess.run(["git", "add", "."], cwd=source_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=source_dir,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


async def test_import_from_local_git_repo(
    tmp_path: pathlib.Path,
    moto_s3_bucket: str,
    aioboto3_s3_client: S3Client,
    mocker: MockerFixture,
) -> None:
    """Import an artifact from a real local git repo with moto S3 — no GitHub API mocks."""
    content = b"real csv data\ncol1,col2\n1,2\n3,4\n"
    file_hash = xxhash.xxh64(content).hexdigest()

    remote_url = f"s3://{moto_s3_bucket}/cache/"
    source_dir = tmp_path / "source_repo"
    consumer_dir = tmp_path / "consumer"
    consumer_dir.mkdir()

    head_sha = _setup_source_repo(
        source_dir,
        remote_url,
        "prepare",
        [{"path": "data/output.csv", "hash": file_hash}],
    )

    key = remote_mod._hash_to_key("cache/", file_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key, Body=content)

    result = await import_artifact.import_artifact(
        str(source_dir),
        "data/output.csv",
        rev="main",
        project_root=consumer_dir,
    )

    data_path = pathlib.Path(result["data_path"])
    assert data_path.read_bytes() == content, "Downloaded content should match"

    pvt_path = pathlib.Path(result["pvt_path"])
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert pvt_data["hash"] == file_hash
    assert pvt_data["size"] == len(content), "Size should come from actual downloaded file"
    assert "source" in pvt_data
    source = pvt_data["source"]
    assert source["repo"] == str(source_dir)
    assert source["rev"] == "main"
    assert source["rev_lock"] == head_sha, "rev_lock should be resolved HEAD SHA"
    assert source["stage"] == "prepare"
    assert source["path"] == "data/output.csv"
    assert source["remote"] == remote_url


async def test_import_from_local_repo_then_update(
    tmp_path: pathlib.Path,
    moto_s3_bucket: str,
    aioboto3_s3_client: S3Client,
    mocker: MockerFixture,
) -> None:
    """Import, then update after source repo gets a new commit with changed output."""
    v1_content = b"version one"
    v1_hash = xxhash.xxh64(v1_content).hexdigest()
    v2_content = b"version two with more data"
    v2_hash = xxhash.xxh64(v2_content).hexdigest()

    remote_url = f"s3://{moto_s3_bucket}/cache/"
    source_dir = tmp_path / "source_repo"
    consumer_dir = tmp_path / "consumer"
    consumer_dir.mkdir()

    _setup_source_repo(
        source_dir,
        remote_url,
        "train",
        [{"path": "models/weights.bin", "hash": v1_hash}],
    )

    key_v1 = remote_mod._hash_to_key("cache/", v1_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key_v1, Body=v1_content)

    result = await import_artifact.import_artifact(
        str(source_dir),
        "models/weights.bin",
        rev="main",
        project_root=consumer_dir,
    )

    data_path = pathlib.Path(result["data_path"])
    pvt_path = pathlib.Path(result["pvt_path"])
    assert data_path.read_bytes() == v1_content

    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    old_rev = pvt_data["source"]["rev_lock"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

    lock_v2 = {
        "schema_version": 1,
        "code_manifest": {"self:run": "abcdef1234567890"},
        "params": {},
        "deps": [],
        "outs": [{"path": "models/weights.bin", "hash": v2_hash}],
    }
    (source_dir / ".pivot" / "stages" / "train.lock").write_text(yaml.safe_dump(lock_v2))
    subprocess.run(["git", "add", "."], cwd=source_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "update output"],
        cwd=source_dir,
        check=True,
        capture_output=True,
    )

    key_v2 = remote_mod._hash_to_key("cache/", v2_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key_v2, Body=v2_content)

    check = await import_artifact.check_for_update(pvt_data)
    assert check["available"] is True, "Should detect update after new commit"

    update_result = await import_artifact.update_import(pvt_path)
    assert update_result["downloaded"] is True
    assert update_result["old_rev"] == old_rev
    assert data_path.read_bytes() == v2_content

    updated_pvt = track.read_pvt_file(pvt_path)
    assert updated_pvt is not None
    assert updated_pvt["hash"] == v2_hash
    assert updated_pvt["size"] == len(v2_content), "Size should reflect v2 file"


async def test_import_from_local_repo_not_found_path(
    tmp_path: pathlib.Path,
    moto_s3_bucket: str,
    aioboto3_s3_client: S3Client,
    mocker: MockerFixture,
) -> None:
    """Importing a path that doesn't exist in any stage lock file raises PivotError."""
    content = b"data"
    file_hash = xxhash.xxh64(content).hexdigest()

    remote_url = f"s3://{moto_s3_bucket}/cache/"
    source_dir = tmp_path / "source_repo"
    consumer_dir = tmp_path / "consumer"
    consumer_dir.mkdir()

    _setup_source_repo(
        source_dir,
        remote_url,
        "train",
        [{"path": "data/output.csv", "hash": file_hash}],
    )

    with pytest.raises(exceptions.PivotError, match="not found in remote outputs"):
        await import_artifact.import_artifact(
            str(source_dir),
            "data/nonexistent.csv",
            rev="main",
            project_root=consumer_dir,
        )


# --- Test 5: Full import -> check_for_update -> update_import flow (mocked GitHub) ---


async def test_full_import_flow_mocked(
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    moto_s3_bucket: str,
    aioboto3_s3_client: S3Client,
) -> None:
    """End-to-end: import_artifact -> check_for_update -> update_import with mocked network."""
    repo_url = "https://github.com/org/upstream"
    remote_url = f"s3://{moto_s3_bucket}/test-prefix/"
    initial_content = b"initial data"
    initial_hash = xxhash.xxh64(initial_content).hexdigest()
    updated_content = b"updated data with more stuff"
    updated_hash = xxhash.xxh64(updated_content).hexdigest()

    # --- Step 1: import_artifact ---

    lock_bytes_v1 = _lock_bytes(outs=[{"path": "data/output.csv", "hash": initial_hash}])

    mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
        return_value=["train.lock"],
    )
    resolve_ref_mock = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
        return_value="sha_v1",
    )

    call_count = 0

    async def _read_file_v1(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        nonlocal call_count
        call_count += 1
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == ".pivot/stages/train.lock":
            return lock_bytes_v1
        raise AssertionError(f"Unexpected path: {path}")

    read_file_mock = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
        side_effect=_read_file_v1,
    )

    key_v1 = remote_mod._hash_to_key("test-prefix/", initial_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key_v1, Body=initial_content)

    s3_cls = mocker.patch(
        "pivot.import_artifact.remote_storage.S3Remote",
        autospec=True,
        return_value=remote_mod.S3Remote(remote_url),
    )

    result = await import_artifact.import_artifact(
        repo_url, "data/output.csv", rev="main", project_root=tmp_path
    )

    data_path = pathlib.Path(result["data_path"])
    pvt_path = pathlib.Path(result["pvt_path"])
    assert result["downloaded"] is True
    assert data_path.read_bytes() == initial_content

    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert pvt_data["hash"] == initial_hash
    assert pvt_data["source"]["rev_lock"] == "sha_v1"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert pvt_data["size"] == len(initial_content), "Size should come from downloaded file"

    # --- Step 2: check_for_update (no change) ---

    resolve_ref_mock.return_value = "sha_v1"

    check = await import_artifact.check_for_update(pvt_data)

    assert check["available"] is False, "No update when rev_lock matches"

    # --- Step 3: check_for_update (new commit available) ---

    resolve_ref_mock.return_value = "sha_v2"

    check = await import_artifact.check_for_update(pvt_data)

    assert check["available"] is True, "Update available when rev_lock differs"
    assert check["current_rev"] == "sha_v1"
    assert check["latest_rev"] == "sha_v2"

    # --- Step 4: update_import ---

    lock_bytes_v2 = _lock_bytes(outs=[{"path": "data/output.csv", "hash": updated_hash}])

    async def _read_file_v2(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == ".pivot/stages/train.lock":
            return lock_bytes_v2
        raise AssertionError(f"Unexpected path: {path}")

    read_file_mock.side_effect = _read_file_v2

    key_v2 = remote_mod._hash_to_key("test-prefix/", updated_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key_v2, Body=updated_content)

    s3_cls.return_value = remote_mod.S3Remote(remote_url)

    update_result = await import_artifact.update_import(pvt_path)

    assert update_result["downloaded"] is True, "Should mark downloaded when hash changed"
    assert update_result["old_rev"] == "sha_v1"
    assert update_result["new_rev"] == "sha_v2"
    assert data_path.read_bytes() == updated_content

    # Verify .pvt file is updated
    updated_pvt = track.read_pvt_file(pvt_path)
    assert updated_pvt is not None
    assert updated_pvt["hash"] == updated_hash
    assert updated_pvt["source"]["rev_lock"] == "sha_v2"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert updated_pvt["source"]["repo"] == repo_url  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert updated_pvt["size"] == len(updated_content), "Size should come from downloaded file"
