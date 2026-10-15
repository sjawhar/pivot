from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot.storage import lock
from pivot.types import ArtifactIdentity, DirHash, LockData, StorageLockData

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from pivot.types import HashInfo


def _id(producer: str, key: str | None = None) -> ArtifactIdentity:
    return ArtifactIdentity(producer, key)


def test_lock_file_creation(tmp_path: Path) -> None:
    """Lock file is created on first write."""
    stage_lock = lock.StageLock("preprocess", tmp_path)

    stage_lock.write(
        LockData(
            code_manifest={"self:preprocess": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={},
        )
    )

    assert stage_lock.path.exists()
    assert stage_lock.path.name == "preprocess.lock"


def test_lock_file_read(set_project_root: Path) -> None:
    """Lock file contents can be read back."""
    cache_dir = set_project_root / ".cache"
    stage_lock = lock.StageLock("train", cache_dir)
    dep_hashes: dict[ArtifactIdentity, HashInfo] = {_id("input"): {"hash": "0123456789abcdef"}}
    data = LockData(
        code_manifest={"self:train": "def456", "func:helper": "ghi789"},
        params={"learning_rate": 0.01},
        dep_hashes=dep_hashes,
        output_hashes={},
    )

    stage_lock.write(data)
    result = stage_lock.read()

    assert result == data


def test_lock_file_read_missing(tmp_path: Path) -> None:
    """Reading non-existent lock file returns None."""
    stage_lock = lock.StageLock("missing", tmp_path)

    result = stage_lock.read()

    assert result is None


def test_manifest_preservation(tmp_path: Path) -> None:
    """Code manifest is preserved exactly through write/read cycle."""
    stage_lock = lock.StageLock("evaluate", tmp_path)
    manifest = {
        "self:evaluate": "hash1",
        "func:compute_metrics": "hash2",
        "func:load_model": "hash3",
        "mod:sklearn.metrics": "hash4",
        "const:THRESHOLD": "hash5",
    }

    data = LockData(
        code_manifest=manifest,
        params={},
        dep_hashes={},
        output_hashes={},
    )
    stage_lock.write(data)
    result = stage_lock.read()

    assert result is not None
    assert result["code_manifest"] == manifest


def test_parallel_lock_writes(tmp_path: Path) -> None:
    """Multiple stages can write locks in parallel without corruption."""
    stages = [f"stage_{i}" for i in range(10)]
    errors = list[Exception]()

    def write_lock(name: str) -> None:
        try:
            stage_lock = lock.StageLock(name, tmp_path)
            stage_lock.write(
                LockData(
                    code_manifest={f"self:{name}": f"hash_{name}"},
                    params={},
                    dep_hashes={},
                    output_hashes={},
                )
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_lock, args=(name,)) for name in stages]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during parallel writes: {errors}"

    for name in stages:
        stage_lock = lock.StageLock(name, tmp_path)
        result = stage_lock.read()
        assert result == LockData(
            code_manifest={f"self:{name}": f"hash_{name}"},
            params={},
            dep_hashes={},
            output_hashes={},
        )


def test_atomic_write_no_partial_file(tmp_path: Path) -> None:
    """Write failure should not leave partial lock file."""
    stage_lock = lock.StageLock("atomic_test", tmp_path)

    stage_lock.write(
        {
            "code_manifest": {"self:atomic_test": "hash"},
            "params": {},
            "dep_hashes": {},
            "output_hashes": {},
        }
    )

    # Verify no .tmp file remains
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Temporary files remain: {tmp_files}"


def test_lock_directory_created(tmp_path: Path) -> None:
    """Lock file parent directories are created automatically."""
    stages_dir = tmp_path / "deep" / "nested" / "stages"
    stage_lock = lock.StageLock("nested_stage", stages_dir)

    stage_lock.write(
        {
            "code_manifest": {},
            "params": {},
            "dep_hashes": {},
            "output_hashes": {},
        }
    )

    assert stage_lock.path.exists()
    assert stage_lock.path.parent == stages_dir


@pytest.mark.parametrize(
    "invalid_name",
    [
        pytest.param("", id="empty"),
        pytest.param("../etc/passwd", id="path_traversal"),
        pytest.param("stage with spaces", id="spaces"),
        pytest.param("stage\nwith\nnewlines", id="newlines"),
        pytest.param("stage\n", id="trailing_newline"),
        pytest.param("../../traversal", id="double_traversal"),
        pytest.param("/f", id="leading_slash"),
        pytest.param("/etc/f", id="absolute"),
        pytest.param("a//f", id="empty_segment"),
        pytest.param("a/", id="trailing_slash"),
        pytest.param("./f", id="dot_segment"),
        pytest.param("a/./f", id="inner_dot_segment"),
        pytest.param("   ", id="spaces_only"),
    ],
)
def test_invalid_stage_name_rejected(tmp_path: Path, invalid_name: str) -> None:
    """Stage names with path traversal or special chars are rejected."""
    with pytest.raises(ValueError, match="Invalid stage name"):
        lock.StageLock(invalid_name, tmp_path)


@pytest.mark.parametrize(
    "valid_name",
    [
        pytest.param("preprocess", id="simple"),
        pytest.param("train_model", id="underscore"),
        pytest.param("evaluate-v2", id="dash"),
        pytest.param("Stage123", id="mixed_case_digits"),
        pytest.param("a", id="single_char"),
        pytest.param("A-B_C", id="mixed_separators"),
        pytest.param("stage.with.dots", id="dots"),
        pytest.param("plot@0.5", id="at_decimal"),
        pytest.param("wrangle@swe_bench", id="at_underscore"),
        pytest.param("pipeline/stage", id="pipeline_prefixed"),
        pytest.param("mr_time_horizon_1_1/compute_task_weights", id="pipeline_prefixed_real"),
    ],
)
def test_valid_stage_names_accepted(tmp_path: Path, valid_name: str) -> None:
    """Valid stage names with alphanumeric, underscore, dash, dot, @, / are accepted."""
    stage_lock = lock.StageLock(valid_name, tmp_path)
    assert stage_lock.stage_name == valid_name


def test_write_failure_no_orphaned_tmp(tmp_path: Path, mocker: MockerFixture) -> None:
    """Write failure cleans up temporary file."""
    stage_lock = lock.StageLock("failing", tmp_path)

    mocker.patch("yaml.dump", side_effect=RuntimeError("dump failed"), autospec=True)
    with pytest.raises(RuntimeError, match="dump failed"):
        stage_lock.write(
            {
                "code_manifest": {},
                "params": {},
                "dep_hashes": {},
                "output_hashes": {},
            }
        )

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Orphaned temp files: {tmp_files}"


def test_concurrent_same_stage_writes(tmp_path: Path) -> None:
    """Concurrent writes to same stage don't corrupt each other."""
    errors = list[Exception]()
    results = list[int]()

    def write_value(value: int) -> None:
        try:
            stage_lock = lock.StageLock("shared", tmp_path)
            stage_lock.write(
                {
                    "code_manifest": {},
                    "params": {"thread_id": value},
                    "dep_hashes": {},
                    "output_hashes": {},
                }
            )
            results.append(value)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_value, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent writes: {errors}"

    stage_lock = lock.StageLock("shared", tmp_path)
    final = stage_lock.read()
    assert final is not None
    assert final["params"]["thread_id"] in range(20), (
        "Final value should be from one of the threads"
    )

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Orphaned temp files: {tmp_files}"


def test_read_corrupted_non_dict_returns_none(tmp_path: Path) -> None:
    """Lock file with non-dict YAML returns None (treated as missing)."""
    stage_lock = lock.StageLock("corrupted", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("just a string\n")

    result = stage_lock.read()

    assert result is None


def test_read_corrupted_list_returns_none(tmp_path: Path) -> None:
    """Lock file with list YAML returns None."""
    stage_lock = lock.StageLock("corrupted", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("- item1\n- item2\n")

    result = stage_lock.read()

    assert result is None


def test_read_binary_garbage_returns_none(tmp_path: Path) -> None:
    """Lock file with binary garbage returns None."""
    stage_lock = lock.StageLock("binary", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_bytes(b"\xff\xfe\x00\x01\x80\x81")

    result = stage_lock.read()

    assert result is None


def test_read_invalid_yaml_returns_none(tmp_path: Path) -> None:
    """Lock file with invalid YAML syntax returns None."""
    stage_lock = lock.StageLock("invalid", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("key: [unclosed bracket\n")

    result = stage_lock.read()

    assert result is None


def test_read_empty_file_returns_none(tmp_path: Path) -> None:
    """Empty lock file returns None."""
    stage_lock = lock.StageLock("empty", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("")

    result = stage_lock.read()

    assert result is None


def test_is_lock_data_accepts_missing_dep_generations() -> None:
    """Lock data without dep_generations is valid."""
    data: StorageLockData = {
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [],
    }

    assert lock.is_lock_data(data)


def test_read_lock_with_dep_generations_ignored(tmp_path: Path) -> None:
    """Legacy lock files with dep_generations are still readable."""
    stage_lock = lock.StageLock("legacy", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(
        "code_manifest: {}\nparams: {}\ndeps: []\nouts: []\ndep_generations: {}\n"
    )

    result = stage_lock.read()

    assert result is not None
    assert "dep_generations" not in result


def test_read_lock_with_extra_keys_accepted(tmp_path: Path) -> None:
    """Lock files with extra keys (forward compatibility) are accepted."""
    stage_lock = lock.StageLock("future", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(
        "code_manifest: {}\nparams: {}\ndeps: []\nouts: []\n"
        + "future_key: some_value\nanother_new_field: 123\n"
    )

    result = stage_lock.read()

    assert result is not None
    assert result["code_manifest"] == {}
    assert result["params"] == {}


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(
            {
                "code_manifest": {},
                "params": {},
                "deps": [{"producer": ["p/a"], "key": None, "hash": "0123456789abcdef"}],
                "outs": [],
            },
            id="list-producer",
        ),
        pytest.param(
            {
                "code_manifest": {},
                "params": {},
                "deps": [],
                "outs": [{"key": {"k": 1}, "hash": "fedcba9876543210"}],
            },
            id="mapping-key",
        ),
    ],
)
def test_lock_read_treats_wrongly_typed_identity_fields_as_missing(
    document: dict[str, object], tmp_path: Path
) -> None:
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    stage_lock.path.write_text(yaml.safe_dump(document))
    assert stage_lock.read() is None


@pytest.mark.parametrize(
    ("code_manifest", "params"),
    [
        pytest.param({}, [], id="params-list"),
        pytest.param({}, 3, id="params-int"),
        pytest.param(["self:x"], {}, id="code-manifest-list"),
    ],
)
def test_lock_read_treats_non_mapping_manifest_or_params_as_missing(
    code_manifest: object, params: object, tmp_path: Path
) -> None:
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    document = {"code_manifest": code_manifest, "params": params, "deps": [], "outs": []}
    stage_lock.path.write_text(yaml.safe_dump(document))
    assert stage_lock.read() is None


_DEP: dict[str, object] = {"producer": "p/a", "key": None, "hash": "0123456789abcdef"}
_OUT: dict[str, object] = {"key": "o", "hash": "fedcba9876543210"}
_MANIFEST_ENTRY: dict[str, object] = {
    "relpath": "f.txt",
    "hash": "00112233445566ff",
    "size": 1,
    "isexec": False,
}


@pytest.mark.parametrize(
    ("field", "document"),
    [
        pytest.param("code_manifest", {"code_manifest": {"self:x": 1}}, id="manifest-int-value"),
        pytest.param("code_manifest", {"code_manifest": {1: "h"}}, id="manifest-int-key"),
        pytest.param("params", {"params": {1: "v"}}, id="params-int-key"),
        pytest.param("deps", {"deps": {"producer": "p/a"}}, id="deps-mapping"),
        pytest.param("outs", {"outs": "o"}, id="outs-string"),
        pytest.param(
            "deps[0].producer", {"deps": [{**_DEP, "producer": None}]}, id="null-producer"
        ),
        pytest.param("deps[0].producer", {"deps": [{**_DEP, "producer": ""}]}, id="empty-producer"),
        pytest.param("deps[0].key", {"deps": [{**_DEP, "key": 3}]}, id="int-dep-key"),
        pytest.param("deps[0].hash", {"deps": [{**_DEP, "hash": ["aa"]}]}, id="list-dep-hash"),
        pytest.param("deps[0].hash", {"deps": [{**_DEP, "hash": "abc"}]}, id="short-dep-hash"),
        pytest.param("outs[0].hash", {"outs": [{**_OUT, "hash": 7}]}, id="int-out-hash"),
        pytest.param("outs[0].hash", {"outs": [{**_OUT, "hash": "abc"}]}, id="short-out-hash"),
        pytest.param(
            "outs[0].hash",
            {"outs": [{**_OUT, "hash": "0123456789ABCDEF"}]},
            id="uppercase-out-hash",
        ),
        pytest.param(
            "outs[0].hash",
            {"outs": [{**_OUT, "hash": "0123456789abcdefa"}]},
            id="long-out-hash",
        ),
        pytest.param(
            "outs[0].hash",
            {"outs": [{**_OUT, "hash": "../../etc/passwd"}]},
            id="traversal-out-hash",
        ),
        pytest.param(
            "outs[0].manifest",
            {"outs": [{**_OUT, "manifest": dict[str, object]()}]},
            id="mapping-manifest",
        ),
        pytest.param(
            "outs[0].manifest[0].size",
            {"outs": [{**_OUT, "manifest": [{**_MANIFEST_ENTRY, "size": "1"}]}]},
            id="string-manifest-size",
        ),
        pytest.param(
            "outs[0].manifest[0].relpath",
            {"outs": [{**_OUT, "manifest": [{**_MANIFEST_ENTRY, "relpath": None}]}]},
            id="null-manifest-relpath",
        ),
        pytest.param(
            "outs[0].manifest[0].hash",
            {"outs": [{**_OUT, "manifest": [{**_MANIFEST_ENTRY, "hash": "abc"}]}]},
            id="short-manifest-hash",
        ),
        pytest.param("schema_version", {"schema_version": "2"}, id="string-schema-version"),
    ],
)
def test_lock_read_treats_wrongly_typed_fields_as_missing_and_names_them(
    field: str,
    document: dict[str, object],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every field is type-checked once; the debug log names the field that failed."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    base: dict[str, object] = {"code_manifest": {}, "params": {}, "deps": [], "outs": []}
    stage_lock.path.write_text(yaml.safe_dump({**base, **document}))

    with caplog.at_level(logging.DEBUG, logger="pivot.storage.lock"):
        assert stage_lock.read() is None

    assert any(field in record.getMessage() for record in caplog.records), caplog.text


def test_lock_read_accepts_every_written_field(tmp_path: Path) -> None:
    """The validator accepts each optional field the writer produces, and ignores fields older
    locks carry (`display`, `tag`, `size`, `merkle_id`)."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    document = {
        "schema_version": 2,
        "code_manifest": {"self:stage": "h"},
        "params": {"lr": 0.1, "nested": {"a": [1, 2]}},
        "deps": [{**_DEP, "display": "data/raw/a.csv", "size": 3}],
        "outs": [
            {**_OUT, "key": None, "manifest": [_MANIFEST_ENTRY]},
            {**_OUT, "key": "m", "tag": "metric", "size": 4},
        ],
        "merkle_id": "abc",
    }
    stage_lock.path.write_text(yaml.safe_dump(document))

    result = stage_lock.read()

    assert result is not None
    assert result["params"] == {"lr": 0.1, "nested": {"a": [1, 2]}}
    assert sorted(result) == ["code_manifest", "dep_hashes", "output_hashes", "params"]


def test_written_lock_outs_record_key_and_hash_only(tmp_path: Path) -> None:
    """Outs carry no constant `tag`; the pipeline, not the lock, knows an output's kind."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:stage": "h"},
            params={},
            dep_hashes={_id("raw.csv"): {"hash": "0123456789abcdef"}},
            output_hashes={_id("p/stage", "m"): {"hash": "fedcba9876543210"}},
        )
    )

    document = yaml.safe_load(stage_lock.path.read_text())

    assert document["outs"] == [{"key": "m", "hash": "fedcba9876543210"}]
    assert document["deps"] == [{"producer": "raw.csv", "key": None, "hash": "0123456789abcdef"}]
    assert stage_lock.read() is not None


def test_older_v2_lock_with_out_tag_and_display_still_parses(tmp_path: Path) -> None:
    """Locks written before `tag` and `display` were dropped stay valid: extra keys are ignored."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    stage_lock.path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "code_manifest": {"self:stage": "h"},
                "params": {},
                "deps": [{**_DEP, "display": "data/raw/a.csv"}],
                "outs": [{**_OUT, "tag": "data", "display": "data/p/stage/o.csv"}],
            }
        )
    )

    result = stage_lock.read()

    assert result is not None
    assert result["output_hashes"] == {_id("p/stage", "o"): {"hash": "fedcba9876543210"}}


# ==============================================================================
# Directory outputs: lock display fields
# ==============================================================================


def test_lock_read_ignores_display_fields(set_project_root: Path) -> None:
    """Display fields should be ignored on read (write-only schema field)."""
    cache_dir = set_project_root / ".cache"
    stage_lock = lock.StageLock("dir_out_stage", cache_dir)

    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(
        "code_manifest: {self:dir_out_stage: fingerprint}\n"
        + "params: {}\n"
        + "schema_version: 2\n"
        + "deps:\n"
        + "  - producer: input\n"
        + "    key: null\n"
        + "    hash: d0d0d0d0d0d0d0d0\n"
        + "    display: data/raw/input.csv\n"
        + "outs:\n"
        + "  - key: results\n"
        + "    hash: 0e0e0e0e0e0e0e0e\n"
        + "    tag: directory\n"
        + "    display: results/\n"
        + "    manifest:\n"
        + "      - relpath: task_a.json\n"
        + "        hash: a0a0a0a0a0a0a0a0\n"
        + "        size: 100\n"
        + "        isexec: false\n"
    )

    result = stage_lock.read()

    assert result is not None
    assert result["dep_hashes"] == {_id("input"): {"hash": "d0d0d0d0d0d0d0d0"}}
    assert result["output_hashes"] == {
        _id("dir_out_stage", "results"): DirHash(
            hash="0e0e0e0e0e0e0e0e",
            manifest=[
                {"relpath": "task_a.json", "hash": "a0a0a0a0a0a0a0a0", "size": 100, "isexec": False}
            ],
        )
    }


def test_pipeline_prefixed_stage_name_creates_subdirectory(tmp_path: Path) -> None:
    """Pipeline-prefixed stage names (with /) create subdirectory lock files.

    When --all mode prefixes stage names as 'pipeline/stage', the lock file
    is stored at stages_dir/pipeline/stage.lock. This test verifies the
    full write/read roundtrip works and the subdirectory is created.
    """
    stage_lock = lock.StageLock("alpha/my_stage", tmp_path)
    data = LockData(
        code_manifest={"self:alpha/my_stage": "abc123"},
        params={"lr": 0.01},
        dep_hashes={},
        output_hashes={},
    )

    stage_lock.write(data)

    # Verify subdirectory structure
    expected_path = tmp_path / "alpha" / "my_stage.lock"
    assert expected_path.exists(), f"Expected lock file at {expected_path}"
    assert stage_lock.path == expected_path

    # Verify roundtrip read
    result = stage_lock.read()
    assert result is not None
    assert result["code_manifest"] == data["code_manifest"]
    assert result["params"] == data["params"]


# =============================================================================
# ArtifactIdentity-keyed lock roundtrip tests
# =============================================================================


def test_lock_roundtrip_identity_keys(tmp_path: Path) -> None:
    """LockData dep_hashes and output_hashes use ArtifactIdentity keys through roundtrip."""
    dep_hashes: dict[ArtifactIdentity, HashInfo] = {
        _id("input"): {"hash": "aaaaaaaaaaaaaaaa"},
        _id("features", "embeddings"): {"hash": "bbbbbbbbbbbbbbbb"},
    }
    output_hashes: dict[ArtifactIdentity, HashInfo] = {
        _id("train"): {"hash": "cccccccccccccccc"},
        _id("train", "metrics"): {"hash": "dddddddddddddddd"},
    }
    data = LockData(
        code_manifest={"self:train": "fp1"},
        params={"lr": 0.01},
        dep_hashes=dep_hashes,
        output_hashes=output_hashes,
    )
    stage_lock = lock.StageLock("train", tmp_path)
    stage_lock.write(data)
    loaded = stage_lock.read()

    assert loaded is not None
    # Keys must be ArtifactIdentity, not strings
    assert _id("input") in loaded["dep_hashes"]
    assert _id("features", "embeddings") in loaded["dep_hashes"]
    assert loaded["dep_hashes"][_id("input")]["hash"] == "aaaaaaaaaaaaaaaa"
    assert loaded["dep_hashes"][_id("features", "embeddings")]["hash"] == "bbbbbbbbbbbbbbbb"

    assert _id("train") in loaded["output_hashes"]
    assert _id("train", "metrics") in loaded["output_hashes"]
    assert loaded["output_hashes"][_id("train")]["hash"] == "cccccccccccccccc"
    assert loaded["output_hashes"][_id("train", "metrics")]["hash"] == "dddddddddddddddd"


def test_lock_roundtrip_dir_hash_with_identity_keys(tmp_path: Path) -> None:
    """DirHash manifests survive roundtrip with ArtifactIdentity keys."""
    dep_hashes: dict[ArtifactIdentity, HashInfo] = {
        _id("data_dir"): DirHash(
            hash="aaaaaaaaaaaaaaa0",
            manifest=[
                {"relpath": "a.csv", "hash": "1111111111111111", "size": 100, "isexec": False}
            ],
        )
    }
    data = LockData(
        code_manifest={},
        params={},
        dep_hashes=dep_hashes,
        output_hashes={},
    )
    stage_lock = lock.StageLock("consume", tmp_path)
    stage_lock.write(data)
    loaded = stage_lock.read()

    assert loaded is not None
    info = loaded["dep_hashes"][_id("data_dir")]
    assert info["hash"] == "aaaaaaaaaaaaaaa0"
    assert "manifest" in info


# --- find_orphaned_locks ---


def test_find_orphaned_locks_empty_dir(tmp_path: Path) -> None:
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    assert lock.find_orphaned_locks(stages_dir, {"train", "test"}) == []


def test_find_orphaned_locks_nonexistent_dir(tmp_path: Path) -> None:
    assert lock.find_orphaned_locks(tmp_path / "nope", {"train"}) == []


def test_find_orphaned_locks_all_registered(tmp_path: Path) -> None:
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    (stages_dir / "train.lock").touch()
    (stages_dir / "test.lock").touch()
    assert lock.find_orphaned_locks(stages_dir, {"train", "test"}) == []


def test_find_orphaned_locks_detects_orphans(tmp_path: Path) -> None:
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    (stages_dir / "train.lock").touch()
    (stages_dir / "old_stage.lock").touch()
    (stages_dir / "renamed@legacy.lock").touch()

    orphans = lock.find_orphaned_locks(stages_dir, {"train"})
    assert orphans == ["old_stage", "renamed@legacy"]


def test_find_orphaned_locks_nested_directories(tmp_path: Path) -> None:
    stages_dir = tmp_path / "stages"
    (stages_dir / "base").mkdir(parents=True)
    (stages_dir / "base" / "base_train.lock").touch()
    (stages_dir / "base" / "base_test.lock").touch()
    (stages_dir / "old_root_stage.lock").touch()

    orphans = lock.find_orphaned_locks(stages_dir, {"base/base_train", "base/base_test"})
    assert orphans == ["old_root_stage"]
