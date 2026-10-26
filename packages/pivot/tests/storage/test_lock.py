from __future__ import annotations

import logging
import posixpath
import threading
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import exceptions
from pivot.storage import listing, lock
from pivot.types import (
    ArtifactIdentity,
    DepEntry,
    DirHash,
    DirManifestEntry,
    FileHash,
    LockData,
    OutEntry,
    StorageLockData,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from pivot.types import HashInfo


def _id(producer: str, key: str | None = None) -> ArtifactIdentity:
    return ArtifactIdentity(producer, key)


def _hashed_directory(
    files: list[DirManifestEntry], empty_dirs: list[str] | None = None
) -> listing.HashedDirectory:
    """A directory record with the listing bytes its lock must write."""
    members = listing.DirListing(
        files=files,
        empty_dirs=empty_dirs if empty_dirs is not None else list[str](),
    )
    data = listing.encode(members)
    return listing.HashedDirectory(
        record=listing.record_of(members, data), listing=members, data=data
    )


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
    """A current-schema lock without dep_generations is valid."""
    data: StorageLockData = {
        "schema_version": lock.SCHEMA_VERSION,
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [],
    }

    assert lock.is_lock_data(data)


def test_read_lock_with_dep_generations_ignored(tmp_path: Path) -> None:
    """Current-schema locks ignore the legacy dep_generations field."""
    stage_lock = lock.StageLock("legacy", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(
        f"schema_version: {lock.SCHEMA_VERSION}\n"
        + "code_manifest: {}\nparams: {}\ndeps: []\nouts: []\ndep_generations: {}\n"
    )

    result = stage_lock.read()

    assert result is not None
    assert "dep_generations" not in result


def test_read_lock_with_extra_keys_accepted(tmp_path: Path) -> None:
    """Current-schema lock files ignore unknown top-level keys."""
    stage_lock = lock.StageLock("future", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(
        f"schema_version: {lock.SCHEMA_VERSION}\n"
        + "code_manifest: {}\nparams: {}\ndeps: []\nouts: []\n"
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
                "schema_version": lock.SCHEMA_VERSION,
                "code_manifest": {},
                "params": {},
                "deps": [{"producer": ["p/a"], "key": None, "hash": "0123456789abcdef"}],
                "outs": [],
            },
            id="list-producer",
        ),
        pytest.param(
            {
                "schema_version": lock.SCHEMA_VERSION,
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
    document = {
        "schema_version": lock.SCHEMA_VERSION,
        "code_manifest": code_manifest,
        "params": params,
        "deps": [],
        "outs": [],
    }
    stage_lock.path.write_text(yaml.safe_dump(document))
    assert stage_lock.read() is None


_DEP: dict[str, object] = {"producer": "p/a", "key": None, "hash": "0123456789abcdef"}
_OUT: dict[str, object] = {"key": "o", "hash": "fedcba9876543210"}
_DIR_OUT: dict[str, object] = {**_OUT, "size": 1, "nfiles": 1}


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
            "outs[0].nfiles",
            {"outs": [{**_DIR_OUT, "nfiles": "1"}]},
            id="string-directory-file-count",
        ),
        pytest.param(
            "outs[0].nfiles",
            {"outs": [{**_DIR_OUT, "nfiles": True}]},
            id="bool-directory-file-count",
        ),
        pytest.param(
            "outs[0].nfiles",
            {"outs": [{**_DIR_OUT, "nfiles": -1}]},
            id="negative-directory-file-count",
        ),
        pytest.param(
            "outs[0].size",
            {"outs": [{**_DIR_OUT, "size": "1"}]},
            id="string-directory-size",
        ),
        pytest.param(
            "outs[0].size",
            {"outs": [{**_DIR_OUT, "size": -1}]},
            id="negative-directory-size",
        ),
        pytest.param(
            "outs[0].size",
            {"outs": [{**_OUT, "nfiles": 1}]},
            id="directory-missing-size",
        ),
    ],
)
def test_lock_read_treats_invalid_fields_as_missing_and_names_them(
    field: str,
    document: dict[str, object],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every current-schema field is type-checked once; the debug log names a rejected field."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    base: dict[str, object] = {
        "schema_version": lock.SCHEMA_VERSION,
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [],
    }
    stage_lock.path.write_text(yaml.safe_dump({**base, **document}))

    with caplog.at_level(logging.DEBUG, logger="pivot.storage.lock"):
        assert stage_lock.read() is None

    assert any(field in record.getMessage() for record in caplog.records), caplog.text


def test_old_schema_lock_warns_and_reads_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An old lock is stale and names its path in the migration warning."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    stage_lock.path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "code_manifest": {"self:stage": "h"},
                "params": {},
                "deps": [],
                "outs": [],
            }
        )
    )

    with caplog.at_level(logging.WARNING, logger="pivot.storage.lock"):
        assert stage_lock.read() is None

    (record,) = caplog.records
    assert str(stage_lock.path) in record.getMessage()
    assert "was written by an older pivot; pivot reads it as missing" in record.getMessage()


def _directory_lock_data(
    relpath: str, *, empty_dir: str | None = None
) -> tuple[LockData, dict[str, listing.HashedDirectory]]:
    """A directory output record and the listing bytes its lock writes."""
    identity = _id("p/stage", "o")
    hashed = _hashed_directory(
        [DirManifestEntry(relpath=relpath, hash="00112233445566ff", size=1, isexec=False)],
        [empty_dir] if empty_dir is not None else None,
    )
    return (
        LockData(
            code_manifest={"self:stage": "h"},
            params={},
            dep_hashes={},
            output_hashes={identity: hashed["record"]},
            output_paths={identity: "data/p/stage/o"},
        ),
        {hashed["record"]["hash"]: hashed},
    )


@pytest.mark.parametrize(
    "relpath",
    [
        pytest.param("Screenshot 2024-01-01 at 10.00.00\u202fAM.png", id="narrow-no-break-space"),
        pytest.param("caf\u00e9\u00a0menu.csv", id="no-break-space"),
        pytest.param("\u8cc7\u6599\u3000\u4e00\u89a7.png", id="ideographic-space"),
        pytest.param("team\u200dlead.csv", id="zero-width-joiner"),
        pytest.param("soft\u00adhyphen.csv", id="soft-hyphen"),
        pytest.param("\ufeffbom.csv", id="byte-order-mark"),
        pytest.param("line\u2028sep.csv", id="line-separator"),
    ],
)
def test_lock_records_and_reads_a_member_name_with_any_non_control_character(
    relpath: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A space separator or format character is part of an ordinary file name (a macOS
    screenshot holds U+202F): the lock records it without a warning and reads it back."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    data, hashed_dirs = _directory_lock_data(relpath)

    with caplog.at_level(logging.WARNING, logger="pivot.storage.lock"):
        stage_lock.write(data, hashed_dirs=hashed_dirs)
        assert stage_lock.read() == data

    assert caplog.records == [], caplog.text


@pytest.mark.parametrize(
    "relpath",
    [
        pytest.param("a\tb.csv", id="tab"),
        pytest.param("x\x01y.csv", id="soh"),
        pytest.param("e\x1b[31mred.csv", id="esc"),
        pytest.param("d\x7fel.csv", id="del"),
        pytest.param("n\x85l.csv", id="nel"),
    ],
)
def test_lock_records_a_control_character_name_with_one_warning_and_reads_it_back(
    relpath: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A control character is recorded and read back, as on 0.5.2, with one warning that names
    the file under its output's path and says a later release will refuse it."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    data, hashed_dirs = _directory_lock_data(relpath)

    with caplog.at_level(logging.WARNING, logger="pivot.storage.lock"):
        stage_lock.write(data, hashed_dirs=hashed_dirs)
        assert stage_lock.read() == data

    (record,) = caplog.records
    assert record.getMessage() == (
        f"Stage 'p/stage' records {'data/p/stage/o/' + relpath!r}, whose name holds a control "
        + "character; a later release will refuse such names, so rename it"
    )


def test_lock_write_names_every_control_character_name_in_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A member and an empty directory with control characters share one warning."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    data, hashed_dirs = _directory_lock_data("a\tb.csv", empty_dir="e\x1bmpty")

    with caplog.at_level(logging.WARNING, logger="pivot.storage.lock"):
        stage_lock.write(data, hashed_dirs=hashed_dirs)

    (record,) = caplog.records
    assert record.getMessage() == (
        "Stage 'p/stage' records 'data/p/stage/o/a\\tb.csv', 'data/p/stage/o/e\\x1bmpty', "
        + "whose names hold a control character; a later release will refuse such names, so "
        + "rename them"
    )


@pytest.mark.parametrize(
    ("relpath", "reason"),
    [
        pytest.param("b\\s.csv", "it contains a backslash", id="backslash"),
        pytest.param("nul\0.csv", "it contains a NUL byte", id="nul"),
        pytest.param("../escaped.csv", "it has an empty, '.' or '..' segment", id="traversal"),
        pytest.param("/abs.csv", "it is absolute", id="absolute"),
    ],
)
def test_lock_write_refuses_a_name_read_would_refuse_and_leaves_the_old_lock(
    relpath: str, reason: str, tmp_path: Path
) -> None:
    """Pivot never writes a lock its own read refuses: the write names the file and why, and
    the lock already on disk is kept."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    previous, previous_dirs = _directory_lock_data("plain.csv")
    stage_lock.write(previous, hashed_dirs=previous_dirs)
    rejected, rejected_dirs = _directory_lock_data(relpath)

    with pytest.raises(exceptions.InvalidPathError) as excinfo:
        stage_lock.write(rejected, hashed_dirs=rejected_dirs)

    shown = posixpath.join("data/p/stage/o", relpath)
    assert str(excinfo.value) == (f"Stage 'p/stage' cannot record {shown!r}: {reason}; rename it")
    assert stage_lock.read() == previous


def test_lock_write_names_a_dependency_member_with_its_dependency(tmp_path: Path) -> None:
    """A directory dependency's member is named with the dependency it belongs to."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    identity = _id("p/a", "parts")
    hashed = _hashed_directory(
        [DirManifestEntry(relpath="b\\s.csv", hash="00112233445566ff", size=1, isexec=False)]
    )

    with pytest.raises(exceptions.InvalidPathError) as excinfo:
        stage_lock.write(
            LockData(
                code_manifest={},
                params={},
                dep_hashes={identity: hashed["record"]},
                output_hashes={},
            ),
            hashed_dirs={hashed["record"]["hash"]: hashed},
            input_paths={identity: "data/raw/parts"},
        )

    assert str(excinfo.value) == (
        "Stage 'p/stage' cannot record 'b\\\\s.csv' in dependency 'p/a:parts': it contains a "
        + "backslash; rename it"
    )


def test_lock_read_accepts_every_written_field(tmp_path: Path) -> None:
    """The current schema accepts directory records and ignores unrelated old fields."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    document = {
        "schema_version": lock.SCHEMA_VERSION,
        "code_manifest": {"self:stage": "h"},
        "params": {"lr": 0.1, "nested": {"a": [1, 2]}},
        "deps": [{**_DEP, "display": "data/raw/a.csv", "size": 3}],
        "outs": [
            {**_OUT, "key": None, "size": 1, "nfiles": 1},
            {**_OUT, "key": "m", "tag": "metric", "size": 4},
        ],
        "merkle_id": "abc",
    }
    stage_lock.path.write_text(yaml.safe_dump(document))

    result = stage_lock.read()

    assert result is not None
    assert result["params"] == {"lr": 0.1, "nested": {"a": [1, 2]}}
    assert result["output_hashes"][_id("p/stage")] == DirHash(
        hash="fedcba9876543210", size=1, nfiles=1
    )


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

    assert document["schema_version"] == lock.SCHEMA_VERSION
    assert document["outs"] == [{"key": "m", "hash": "fedcba9876543210"}]
    assert document["deps"] == [{"producer": "raw.csv", "key": None, "hash": "0123456789abcdef"}]
    assert stage_lock.read() is not None


def test_written_lock_records_project_relative_output_path(tmp_path: Path) -> None:
    """A production lock records each output's workspace path in portable form."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    identity = _id("p/stage", "m")
    stage_lock.write(
        LockData(
            code_manifest={"self:stage": "h"},
            params={},
            dep_hashes={},
            output_hashes={identity: {"hash": "fedcba9876543210"}},
            output_paths={identity: "data/p/stage/m.txt"},
        )
    )

    document = yaml.safe_load(stage_lock.path.read_text())
    parsed = stage_lock.read()

    assert document["outs"] == [
        {"key": "m", "hash": "fedcba9876543210", "path": "data/p/stage/m.txt"}
    ]
    assert parsed is not None
    assert "output_paths" in parsed
    assert parsed["output_paths"] == {identity: "data/p/stage/m.txt"}


def test_lock_rejects_absolute_output_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A current-schema output path is validated and named in its warning."""
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    stage_lock.path.write_text(
        yaml.safe_dump(
            {
                "schema_version": lock.SCHEMA_VERSION,
                "code_manifest": {"self:stage": "h"},
                "params": {},
                "deps": [],
                "outs": [{**_OUT, "path": "/tmp/output.txt"}],
            }
        )
    )

    with caplog.at_level(logging.WARNING, logger="pivot.storage.lock"):
        assert stage_lock.read() is None

    (record,) = caplog.records
    assert str(stage_lock.path) in record.getMessage()
    assert (
        "outs[0].path: '/tmp/output.txt' is not a project-relative POSIX path"
        in record.getMessage()
    )


# ==============================================================================
# Directory outputs: lock display fields
# ==============================================================================


def test_lock_read_ignores_display_fields(set_project_root: Path) -> None:
    """Current-schema lock reads ignore display fields outside the directory record."""
    cache_dir = set_project_root / ".cache"
    stage_lock = lock.StageLock("dir_out_stage", cache_dir)

    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(
        yaml.safe_dump(
            {
                "schema_version": lock.SCHEMA_VERSION,
                "code_manifest": {"self:dir_out_stage": "fingerprint"},
                "params": {},
                "deps": [{**_DEP, "producer": "input", "display": "data/raw/input.csv"}],
                "outs": [
                    {
                        **_OUT,
                        "key": "results",
                        "hash": "0e0e0e0e0e0e0e0e",
                        "size": 100,
                        "nfiles": 1,
                        "tag": "directory",
                        "display": "results/",
                    }
                ],
            }
        )
    )

    result = stage_lock.read()

    assert result is not None
    assert result["dep_hashes"] == {_id("input"): {"hash": "0123456789abcdef"}}
    assert result["output_hashes"] == {
        _id("dir_out_stage", "results"): DirHash(hash="0e0e0e0e0e0e0e0e", size=100, nfiles=1)
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


def test_lock_roundtrip_dir_record_with_identity_keys(tmp_path: Path) -> None:
    """A directory lock keeps its record and writes its dependency listing."""
    identity = _id("data_dir")
    hashed = _hashed_directory(
        [DirManifestEntry(relpath="a.csv", hash="1111111111111111", size=100, isexec=False)]
    )
    data = LockData(
        code_manifest={},
        params={},
        dep_hashes={identity: hashed["record"]},
        output_hashes={},
    )
    stage_lock = lock.StageLock("consume", tmp_path)
    input_path = "data/raw/data_dir"
    input_listing = listing.input_listing_path(tmp_path, input_path)
    files_dir = tmp_path / "cache" / "files"
    stage_lock.write(
        data,
        hashed_dirs={hashed["record"]["hash"]: hashed},
        input_paths={identity: input_path},
        cache_dir=files_dir,
    )
    loaded = stage_lock.read()

    assert loaded is not None
    assert loaded["dep_hashes"][identity] == hashed["record"]
    assert listing.read(input_listing, hashed["record"], None) == hashed["listing"]
    assert listing.read(None, hashed["record"], files_dir) == hashed["listing"]


def test_lock_writes_listings_for_directory_deps_and_outs_before_the_lock(tmp_path: Path) -> None:
    """Directory members and empty dirs live in listings, written before their lock."""
    dep_identity = _id("upstream")
    out_identity = _id("consume")
    hashed = _hashed_directory(
        [DirManifestEntry(relpath="a.csv", hash="1111111111111111", size=100, isexec=False)],
        ["bundle/classifier_source", "logs"],
    )
    data = LockData(
        code_manifest={},
        params={},
        dep_hashes={dep_identity: hashed["record"]},
        output_hashes={out_identity: hashed["record"]},
    )
    stage_lock = lock.StageLock("consume", tmp_path)
    input_path = "data/raw/upstream"
    output_listing = stage_lock.output_listing_path(out_identity)
    input_listing = listing.input_listing_path(tmp_path, input_path)

    written = stage_lock.write(
        data,
        hashed_dirs={hashed["record"]["hash"]: hashed},
        input_paths={dep_identity: input_path},
    )
    loaded = stage_lock.read()

    assert written == [output_listing, input_listing, stage_lock.path]
    assert loaded is not None
    assert loaded["dep_hashes"][dep_identity] == hashed["record"]
    assert loaded["output_hashes"][out_identity] == hashed["record"]
    assert listing.read(output_listing, hashed["record"], None) == hashed["listing"]
    assert listing.read(input_listing, hashed["record"], None) == hashed["listing"]


def test_written_lock_records_directory_size_file_count_and_listing(tmp_path: Path) -> None:
    """A directory output records only its hash, total size, and file count in its lock."""
    entry = DirManifestEntry(relpath="a.csv", hash="1111111111111111", size=100, isexec=False)
    hashed = _hashed_directory([entry])
    identity = _id("p/stage")
    stage_lock = lock.StageLock("p/stage", tmp_path)
    output_listing = stage_lock.output_listing_path(identity)
    written = stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={identity: hashed["record"]},
        ),
        hashed_dirs={hashed["record"]["hash"]: hashed},
    )

    document = yaml.safe_load(stage_lock.path.read_text())

    assert written == [output_listing, stage_lock.path]
    assert document["schema_version"] == lock.SCHEMA_VERSION
    assert document["outs"] == [{"key": None, **hashed["record"]}]
    assert "manifest" not in document["outs"][0]
    assert "empty_dirs" not in document["outs"][0]
    assert listing.read(output_listing, hashed["record"], None) == hashed["listing"]


def test_lock_keeps_its_old_listing_when_lock_serialization_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    identity = _id("p/stage")
    previous = _hashed_directory(
        [DirManifestEntry(relpath="old.csv", hash="1111111111111111", size=1, isexec=False)]
    )
    replacement = _hashed_directory(
        [DirManifestEntry(relpath="new.csv", hash="2222222222222222", size=1, isexec=False)]
    )
    stage_lock = lock.StageLock("p/stage", tmp_path)
    old_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={identity: previous["record"]},
    )
    stage_lock.write(old_data, hashed_dirs={previous["record"]["hash"]: previous})

    mocker.patch("yaml.dump", side_effect=RuntimeError("dump failed"), autospec=True)
    with pytest.raises(RuntimeError, match="dump failed"):
        stage_lock.write(
            LockData(
                code_manifest={},
                params={},
                dep_hashes={},
                output_hashes={identity: replacement["record"]},
            ),
            hashed_dirs={replacement["record"]["hash"]: replacement},
        )

    assert stage_lock.read() == old_data
    assert (
        listing.read(stage_lock.output_listing_path(identity), previous["record"], None)
        == previous["listing"]
    )


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(
            OutEntry(key=None, hash="fedcba9876543210"),
            FileHash(hash="fedcba9876543210"),
            id="file",
        ),
        pytest.param(
            OutEntry(key=None, hash="fedcba9876543210", size=1),
            FileHash(hash="fedcba9876543210"),
            id="file-with-size",
        ),
        pytest.param(
            OutEntry(key=None, hash="fedcba9876543210", size=1, nfiles=1),
            DirHash(hash="fedcba9876543210", size=1, nfiles=1),
            id="directory",
        ),
        pytest.param(
            DepEntry(producer="p/a", key=None, hash="fedcba9876543210", size=0, nfiles=0),
            DirHash(hash="fedcba9876543210", size=0, nfiles=0),
            id="empty-directory",
        ),
    ],
)
def test_entry_hash_info_uses_file_count_to_identify_a_directory(
    entry: DepEntry | OutEntry, expected: HashInfo
) -> None:
    """Only nfiles identifies a directory entry; its members are in its listing."""
    assert lock.entry_hash_info(entry) == expected


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


# =============================================================================
# Locks read from git history or another repository
# =============================================================================


def _helper_old_lock(**outs_entry: object) -> dict[str, object]:
    return {
        "schema_version": 2,
        "code_manifest": {"self:f": "abc"},
        "params": {},
        "deps": [{"producer": "src", "key": None, "hash": "0123456789abcdef"}],
        "outs": [{"key": None, "hash": "fedcba9876543210", **outs_entry}],
    }


def test_parse_committed_reads_this_format_with_no_inline_listings() -> None:
    document: dict[str, object] = {**_helper_old_lock(), "schema_version": lock.SCHEMA_VERSION}

    parsed = lock.parse_committed(document)

    assert isinstance(parsed, lock.CommittedLock)
    assert parsed.data == document
    assert parsed.listings == {}


def test_parse_committed_reads_an_older_lock_keeping_its_file_entries() -> None:
    parsed = lock.parse_committed(_helper_old_lock(path="data/p/out.txt"))

    assert isinstance(parsed, lock.CommittedLock)
    assert parsed.data["schema_version"] == lock.SCHEMA_VERSION
    assert parsed.data["outs"] == [
        {"key": None, "hash": "fedcba9876543210", "path": "data/p/out.txt"}
    ]
    assert parsed.data["deps"] == [{"producer": "src", "key": None, "hash": "0123456789abcdef"}]


def test_parse_committed_names_an_unsafe_path_in_an_older_lock() -> None:
    parsed = lock.parse_committed(_helper_old_lock(path="../outside"))

    assert isinstance(parsed, lock.UnsafeLockPath)


def test_parse_committed_refuses_a_malformed_older_inline_listing() -> None:
    parsed = lock.parse_committed(
        _helper_old_lock(manifest=[{"relpath": "a", "hash": "short", "size": 1, "isexec": False}])
    )

    assert parsed is None
