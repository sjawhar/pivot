"""A lock records the objects its stage read by hash (`pivot.read_by_hash`) under `reads`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from pivot.storage import lock
from pivot.types import ArtifactIdentity, FileHash, LockData

if TYPE_CHECKING:
    from pathlib import Path

_FIRST = "0123456789abcdef"
_SECOND = "fedcba9876543210"


def _helper_lock_data(reads: list[str] | None) -> LockData:
    data = LockData(
        code_manifest={"self:plan": "aaaaaaaaaaaaaaaa"},
        params={},
        dep_hashes={ArtifactIdentity("counted.pin", None): FileHash(hash="1111111111111111")},
        output_hashes={ArtifactIdentity("p/plan", None): FileHash(hash="2222222222222222")},
    )
    if reads is not None:
        data["reads"] = reads
    return data


def test_a_lock_writes_its_reads_once_each_in_hash_order(tmp_path: Path) -> None:
    stage_lock = lock.StageLock("p/plan", tmp_path)

    _ = stage_lock.write(_helper_lock_data([_SECOND, _FIRST, _SECOND]))

    document = yaml.safe_load(stage_lock.path.read_text())
    assert document["reads"] == [_FIRST, _SECOND]
    read = stage_lock.read()
    assert read is not None and "reads" in read
    assert read["reads"] == [_FIRST, _SECOND]


def test_the_same_reads_in_another_order_write_the_same_lock_bytes(tmp_path: Path) -> None:
    first = lock.StageLock("p/plan", tmp_path / "a")
    second = lock.StageLock("p/plan", tmp_path / "b")

    _ = first.write(_helper_lock_data([_FIRST, _SECOND]))
    _ = second.write(_helper_lock_data([_SECOND, _FIRST, _FIRST]))

    assert first.path.read_bytes() == second.path.read_bytes()


@pytest.mark.parametrize("reads", [pytest.param(None, id="absent"), pytest.param([], id="empty")])
def test_a_lock_of_a_stage_that_read_nothing_has_no_reads_key(
    tmp_path: Path, reads: list[str] | None
) -> None:
    stage_lock = lock.StageLock("p/plan", tmp_path)

    _ = stage_lock.write(_helper_lock_data(reads))

    assert "reads" not in yaml.safe_load(stage_lock.path.read_text())
    read = stage_lock.read()
    assert read is not None
    assert "reads" not in read


def test_a_lock_that_records_reads_keeps_them_beside_a_directory_record(tmp_path: Path) -> None:
    """A lock that records a directory keeps its entries under `records`; `reads` stays at the
    top level, where every reader finds it."""
    stage_lock = lock.StageLock("p/plan", tmp_path)
    document = {
        "schema_version": lock.SCHEMA_VERSION,
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [],
        "reads": [_FIRST],
        "records": {
            "deps": [],
            "outs": [{"key": None, "listing": _SECOND, "size": 1, "nfiles": 1}],
        },
    }
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(yaml.safe_dump(document))

    read = stage_lock.read()

    assert read is not None and "reads" in read
    assert read["reads"] == [_FIRST]


@pytest.mark.parametrize(
    "reads",
    [
        pytest.param("0123456789abcdef", id="not-a-list"),
        pytest.param(["../../etc/passwd"], id="not-a-hash"),
        pytest.param([None], id="null"),
        pytest.param(["0123456789ABCDEF"], id="uppercase"),
    ],
)
def test_a_lock_whose_reads_are_malformed_reads_as_missing(tmp_path: Path, reads: object) -> None:
    stage_lock = lock.StageLock("p/plan", tmp_path)
    _ = stage_lock.write(_helper_lock_data([_FIRST]))
    document = yaml.safe_load(stage_lock.path.read_text())
    document["reads"] = reads
    stage_lock.path.write_text(yaml.safe_dump(document))

    assert lock.lock_data_error(document) == "reads"
    assert stage_lock.read() is None
