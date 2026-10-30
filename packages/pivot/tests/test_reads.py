"""`pivot.read_by_hash`: the object a content hash names, read inside a stage run, the check
of whether the objects a lock records are still in the cache or the remote, and
`pivot.output_hash`, the content hash a stage's committed lock records for an output."""

from __future__ import annotations

import asyncio
import multiprocessing
import pathlib
import subprocess
import sys
import threading
import uuid
from typing import Protocol, cast

import boto3
import pytest
import xxhash
import yaml

import pivot
from pivot import compose, exceptions, executor, reads
from pivot.executor import core as executor_core
from pivot.remote import keys, protocol
from pivot.storage import cache, listing, lock, state
from pivot.types import ArtifactIdentity, FileHash, LockData, RemoteHashes, UnavailableReads


class _MotoS3Client(Protocol):
    """The boto3 S3 client calls these tests make; boto3 ships no stubs."""

    def create_bucket(self, **kwargs: object) -> object: ...

    def put_object(self, **kwargs: object) -> object: ...


def _helper_hash(data: bytes) -> str:
    return xxhash.xxh64(data).hexdigest()


def _helper_put(files_dir: pathlib.Path, data: bytes) -> str:
    """Publish ``data`` as a read-only object in the cache at ``files_dir``; return its hash."""
    content_hash = _helper_hash(data)
    path = cache.get_cache_path(files_dir, content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o444)
    return content_hash


def _helper_remote(
    root: pathlib.Path, endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[_MotoS3Client, str]:
    """Give the project at ``root`` one remote, a fresh bucket on the moto server at
    ``endpoint``. Returns a client for the bucket and its name."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    bucket = f"reads-{uuid.uuid4().hex[:8]}"
    s3 = cast(
        "_MotoS3Client",
        boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1"),
    )
    s3.create_bucket(Bucket=bucket)
    (root / ".pivot" / "config.yaml").write_text(
        yaml.safe_dump({"remotes": {"origin": f"s3://{bucket}/cache"}})
    )
    return s3, bucket


@pytest.fixture
def files_dir(set_project_root: pathlib.Path) -> pathlib.Path:
    return set_project_root / ".pivot" / "cache" / "files"


# =============================================================================
# Reading inside a stage run
# =============================================================================


def test_read_by_hash_outside_a_stage_run_names_where_it_works() -> None:
    with pytest.raises(exceptions.ObjectReadError, match="only inside a stage"):
        pivot.read_by_hash("0123456789abcdef")


def test_a_read_returns_the_cache_object_and_records_its_hash(files_dir: pathlib.Path) -> None:
    first = _helper_put(files_dir, b"one\n")
    second = _helper_put(files_dir, b"two\n")

    with reads.recording(files_dir) as recorded:
        path = pivot.read_by_hash(second)
        _ = pivot.read_by_hash(first)
        _ = pivot.read_by_hash(second)

    assert path.read_bytes() == b"two\n"
    assert recorded.hashes() == sorted([first, second])
    assert recorded.fetched() == [], "nothing came from a remote"


def test_a_read_from_a_thread_the_stage_started_is_recorded(files_dir: pathlib.Path) -> None:
    content_hash = _helper_put(files_dir, b"one\n")
    paths = list[pathlib.Path]()

    with reads.recording(files_dir) as recorded:
        thread = threading.Thread(target=lambda: paths.append(pivot.read_by_hash(content_hash)))
        thread.start()
        thread.join()

    assert [path.read_bytes() for path in paths] == [b"one\n"]
    assert recorded.hashes() == [content_hash]


def test_one_recording_at_a_time(files_dir: pathlib.Path) -> None:
    with (
        reads.recording(files_dir),
        pytest.raises(RuntimeError, match="already recording"),
        reads.recording(files_dir),
    ):
        pass


def test_a_read_in_a_process_the_stage_started_names_the_rule(files_dir: pathlib.Path) -> None:
    """A process the stage starts (joblib's processes backend, a `subprocess` running Python)
    cannot record its reads in the stage's lock, so its read is refused with that reason, not
    with "outside a stage"."""
    with reads.recording(files_dir):
        child = subprocess.run(
            [sys.executable, "-c", "import pivot; pivot.read_by_hash('0123456789abcdef')"],
            capture_output=True,
            text=True,
            check=False,
        )

    assert child.returncode != 0
    assert "only in the stage's own process and its threads" in child.stderr, child.stderr


def _forked_read(content_hash: str, queue: multiprocessing.Queue[str]) -> None:
    try:
        _ = pivot.read_by_hash(content_hash)
        queue.put("read")
    except exceptions.ObjectReadError as error:
        queue.put(f"refused: {error}")


def test_a_read_in_a_forked_child_is_refused(files_dir: pathlib.Path) -> None:
    """A process the stage forks gets a copy of the recording that no lock ever reads, so a
    read there would leave the lock without `reads`, and `pivot status` would call the stage
    cached after the object left the cache and the remote. Reads are recorded only in the
    stage's own process and its threads, so the child's read is refused, naming that rule."""
    content_hash = _helper_put(files_dir, b"one\n")
    ctx = multiprocessing.get_context("fork")
    queue: multiprocessing.Queue[str] = ctx.Queue()

    with reads.recording(files_dir) as recorded:
        child = ctx.Process(target=_forked_read, args=(content_hash, queue))
        child.start()
        outcome = queue.get(timeout=60)
        child.join(timeout=60)

    assert outcome.startswith("refused: "), outcome
    assert "only in the stage's own process and its threads" in outcome, outcome
    assert recorded.hashes() == []


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("../../etc/passwd", id="path"),
        pytest.param("0123456789ABCDEF", id="uppercase"),
        pytest.param("0123456789abcde", id="short"),
        pytest.param(" 0123456789abcdef", id="space"),
    ],
)
def test_a_value_that_is_not_a_content_hash_fails_the_read(
    files_dir: pathlib.Path, value: str
) -> None:
    with (
        reads.recording(files_dir) as recorded,
        pytest.raises(exceptions.ObjectReadError, match="is not a content hash"),
    ):
        pivot.read_by_hash(value)
    assert recorded.hashes() == []


def test_a_cache_object_whose_bytes_do_not_match_its_hash_fails_the_read(
    files_dir: pathlib.Path,
) -> None:
    content_hash = _helper_hash(b"expected\n")
    path = cache.get_cache_path(files_dir, content_hash)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupt\n")

    with (
        reads.recording(files_dir) as recorded,
        pytest.raises(exceptions.ObjectReadError, match=f"hashes to {_helper_hash(b'corrupt\n')}"),
    ):
        pivot.read_by_hash(content_hash)
    assert recorded.hashes() == []


def test_an_object_missing_with_no_remote_configured_fails_the_read(
    files_dir: pathlib.Path,
) -> None:
    content_hash = _helper_hash(b"absent\n")

    with (
        reads.recording(files_dir),
        pytest.raises(
            exceptions.ObjectReadError,
            match=f"Object {content_hash} is not in the cache, and no remote is configured",
        ),
    ):
        pivot.read_by_hash(content_hash)


def test_a_read_with_no_cache_fails() -> None:
    with (
        reads.recording(None),
        pytest.raises(exceptions.ObjectReadError, match="has no cache"),
    ):
        pivot.read_by_hash("0123456789abcdef")


def test_a_read_fetches_a_missing_object_from_the_remote_into_the_cache(
    files_dir: pathlib.Path,
    set_project_root: pathlib.Path,
    moto_services: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3, bucket = _helper_remote(set_project_root, moto_services["s3"], monkeypatch)
    content_hash = _helper_hash(b"snapshot\n")
    s3.put_object(Bucket=bucket, Key=keys.hash_to_key("cache/", content_hash), Body=b"snapshot\n")

    with reads.recording(files_dir) as recorded:
        path = pivot.read_by_hash(content_hash)

    assert path == cache.get_cache_path(files_dir, content_hash)
    assert path.read_bytes() == b"snapshot\n"
    assert recorded.hashes() == [content_hash]
    assert recorded.fetched() == [
        RemoteHashes(
            name="origin",
            url=protocol.open_remote(f"s3://{bucket}/cache").url,
            hashes=[content_hash],
        )
    ], "the coordinator records what a read fetched in the remote index, as `pivot pull` does"


def test_a_read_records_each_object_it_fetched_once(
    files_dir: pathlib.Path,
    set_project_root: pathlib.Path,
    moto_services: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An object read twice is fetched once, and one read from the cache is fetched never."""
    s3, bucket = _helper_remote(set_project_root, moto_services["s3"], monkeypatch)
    cached = _helper_put(files_dir, b"cached\n")
    first, second = _helper_hash(b"first\n"), _helper_hash(b"second\n")
    for content_hash, body in ((first, b"first\n"), (second, b"second\n")):
        s3.put_object(Bucket=bucket, Key=keys.hash_to_key("cache/", content_hash), Body=body)

    with reads.recording(files_dir) as recorded:
        for content_hash in (second, cached, first, second):
            _ = pivot.read_by_hash(content_hash)

    assert recorded.fetched() == [
        RemoteHashes(
            name="origin",
            url=protocol.open_remote(f"s3://{bucket}/cache").url,
            hashes=sorted([first, second]),
        )
    ]
    assert recorded.hashes() == sorted([cached, first, second])


def test_a_read_from_a_coroutine_fetches_the_object_from_the_remote(
    files_dir: pathlib.Path,
    set_project_root: pathlib.Path,
    moto_services: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage whose helper code runs an event loop calls `read_by_hash`, a plain function, from
    a coroutine. With the object only on the remote, the read still fetches it: the download
    cannot start a second event loop on the thread already running one, so it runs on a thread
    of its own."""
    s3, bucket = _helper_remote(set_project_root, moto_services["s3"], monkeypatch)
    content_hash = _helper_hash(b"snapshot\n")
    s3.put_object(Bucket=bucket, Key=keys.hash_to_key("cache/", content_hash), Body=b"snapshot\n")

    async def stage_helper() -> bytes:
        return pivot.read_by_hash(content_hash).read_bytes()

    with reads.recording(files_dir) as recorded:
        data = asyncio.run(stage_helper())

    assert data == b"snapshot\n"
    assert recorded.hashes() == [content_hash]


def test_a_remote_object_whose_bytes_do_not_match_its_hash_fails_the_read(
    files_dir: pathlib.Path,
    set_project_root: pathlib.Path,
    moto_services: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3, bucket = _helper_remote(set_project_root, moto_services["s3"], monkeypatch)
    content_hash = _helper_hash(b"snapshot\n")
    s3.put_object(Bucket=bucket, Key=keys.hash_to_key("cache/", content_hash), Body=b"tampered\n")

    with (
        reads.recording(files_dir) as recorded,
        pytest.raises(exceptions.ObjectReadError, match="Hash mismatch"),
    ):
        pivot.read_by_hash(content_hash)
    assert recorded.hashes() == []
    assert recorded.fetched() == [], "a download that failed puts nothing in the remote index"
    assert not cache.get_cache_path(files_dir, content_hash).exists()


def test_an_object_the_remote_lacks_fails_the_read_naming_the_remote(
    files_dir: pathlib.Path,
    set_project_root: pathlib.Path,
    moto_services: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _helper_remote(set_project_root, moto_services["s3"], monkeypatch)
    content_hash = _helper_hash(b"absent\n")

    with (
        reads.recording(files_dir),
        pytest.raises(exceptions.ObjectReadError, match="remote 'origin'"),
    ):
        pivot.read_by_hash(content_hash)


# =============================================================================
# Whether the objects a lock records are still available
# =============================================================================


def _helper_unreachable_remote(root: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Give the project at ``root`` one remote that refuses every connection; return its URL as
    the remote index records it."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    url = "s3://unreachable/cache"
    (root / ".pivot" / "config.yaml").write_text(yaml.safe_dump({"remotes": {"origin": url}}))
    return protocol.open_remote(url).url


def _helper_index(root: pathlib.Path, url: str, content_hash: str) -> None:
    """Record in the remote index that remote `origin`, at ``url``, holds ``content_hash``, as
    `pivot push` does."""
    with state.StateDB(root / ".pivot") as state_db:
        state_db.remote_set_url("origin", url)
        state_db.remote_hashes_add("origin", [content_hash])


def test_objects_in_the_cache_are_available(files_dir: pathlib.Path) -> None:
    content_hash = _helper_put(files_dir, b"one\n")

    assert reads.unavailable([content_hash], files_dir) is None


def test_objects_missing_with_no_remote_are_unavailable(files_dir: pathlib.Path) -> None:
    present = _helper_put(files_dir, b"one\n")
    first, second = sorted([_helper_hash(b"absent 1\n"), _helper_hash(b"absent 2\n")])

    unavailable = reads.unavailable([second, present, first], files_dir)

    assert unavailable == UnavailableReads(
        reason=f"Read objects missing from the cache: {first}, {second} (no remote is configured)",
        pullable=False,
    ), "with no remote, `pivot pull` cannot fetch them"


def test_objects_the_remote_index_records_are_available_without_asking_the_remote(
    files_dir: pathlib.Path, set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check is local, as for outputs: the remote here refuses every connection, and an
    object the index records on it counts as there at once."""
    url = _helper_unreachable_remote(set_project_root, monkeypatch)
    content_hash = _helper_hash(b"snapshot\n")
    _helper_index(set_project_root, url, content_hash)

    assert reads.unavailable([content_hash], files_dir) is None


def test_objects_the_remote_index_does_not_record_are_unavailable_unasked(
    files_dir: pathlib.Path,
    set_project_root: pathlib.Path,
    moto_services: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote holds the object, but nothing recorded that in the index, and the check asks
    the remote nothing: plain `status` and `repro` stay local."""
    s3, bucket = _helper_remote(set_project_root, moto_services["s3"], monkeypatch)
    content_hash = _helper_hash(b"snapshot\n")
    s3.put_object(Bucket=bucket, Key=keys.hash_to_key("cache/", content_hash), Body=b"snapshot\n")

    unavailable = reads.unavailable([content_hash], files_dir)

    assert unavailable == UnavailableReads(
        reason=f"Read objects missing from the cache: {content_hash} (pivot has not seen them on "
        + "remote 'origin'; `pivot pull` fetches them)",
        pullable=True,
    )


def test_an_index_entry_for_another_url_of_the_remote_does_not_count(
    files_dir: pathlib.Path, set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _helper_unreachable_remote(set_project_root, monkeypatch)
    content_hash = _helper_hash(b"snapshot\n")
    _helper_index(set_project_root, "s3://elsewhere/cache/", content_hash)

    assert reads.unavailable([content_hash], files_dir) is not None


def _plan_pinned(pin: str) -> dict[str, list[str]]:
    return {"rows": [pivot.read_by_hash(content_hash).read_text() for content_hash in pin.split()]}


def test_what_a_stage_of_another_root_fetched_counts_as_on_the_project_remote(
    files_dir: pathlib.Path,
    set_project_root: pathlib.Path,
    moto_services: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A pipeline included from another root keeps its stages' records in that root's state
    directory, but the remote index that `pivot status`, `pull` and `push` read is the
    project's: what such a stage's read fetched is recorded there, so the cache losing the object
    leaves it available."""
    s3, bucket = _helper_remote(set_project_root, moto_services["s3"], monkeypatch)
    snapshot = _helper_hash(b"snapshot\n")
    s3.put_object(Bucket=bucket, Key=keys.hash_to_key("cache/", snapshot), Body=b"snapshot\n")
    (set_project_root / "data" / "raw").mkdir(parents=True)
    (set_project_root / "data" / "raw" / "counted.pin").write_text(f"{snapshot}\n")
    other = tmp_path_factory.mktemp("other")
    (other / ".pivot").mkdir()
    with compose.Pipeline("other", root=other) as included:
        compose.stage(_plan_pinned)(included.input("counted.pin", t=str))
    pipeline = compose.Pipeline("all", root=set_project_root)
    pipeline.include(included)
    monkeypatch.chdir(set_project_root)
    # Workers a pool reuses keep the environment they started with; the read in the stage needs
    # this test's endpoint for the remote, and later tests need workers without it.
    _ = executor_core.create_fresh_executor(1)
    try:
        results = executor.run(pipeline=pipeline)
    finally:
        monkeypatch.undo()
        _ = executor_core.create_fresh_executor(1)

    assert results["other/_plan_pinned"]["status"] == "ran", results
    with state.StateDB(set_project_root / ".pivot") as state_db:
        assert state_db.remote_hashes_intersection("origin", {snapshot}) == {snapshot}
    cache.get_cache_path(files_dir, snapshot).unlink()
    assert reads.unavailable([snapshot], files_dir) is None


# =============================================================================
# The content hash a stage's committed lock records for an output
# =============================================================================

_PLAN = "0123456789abcdef"


def _helper_write_lock(root: pathlib.Path) -> str:
    """Commit `p/plan`'s lock: a file output `plan` and a directory output `index`. Returns the
    directory's hash, its listing's."""
    members = listing.DirListing(files=[], empty_dirs=["empty"])
    data = listing.encode(members)
    record = listing.record_of(members, data)
    _ = lock.StageLock("p/plan", root / ".pivot" / "stages").write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={
                ArtifactIdentity("p/plan", "plan"): FileHash(hash=_PLAN),
                ArtifactIdentity("p/plan", "index"): record,
            },
        ),
        hashed_dirs={
            record["hash"]: listing.HashedDirectory(record=record, listing=members, data=data)
        },
    )
    return record["hash"]


def test_output_hash_is_the_hash_the_stage_lock_records_for_that_output(
    set_project_root: pathlib.Path,
) -> None:
    index = _helper_write_lock(set_project_root)

    assert pivot.output_hash("p/plan", "plan") == _PLAN
    assert pivot.output_hash("p/plan", "index") == index, "a directory's is its listing's"


def test_output_hash_inside_a_stage_run_is_refused(
    set_project_root: pathlib.Path, files_dir: pathlib.Path
) -> None:
    """Another stage's lock is no input of the stage that reads it, so a new output there would
    never rerun that stage, and status would call its stale output cached. Inside a stage run,
    `output_hash` refuses, naming the pin file to declare instead."""
    _ = _helper_write_lock(set_project_root)

    with (
        reads.recording(files_dir),
        pytest.raises(exceptions.ObjectReadError, match="write the hash into a pin file"),
    ):
        _ = pivot.output_hash("p/plan", "plan")


def test_output_hash_in_a_process_a_stage_started_is_refused(
    set_project_root: pathlib.Path, files_dir: pathlib.Path
) -> None:
    _ = _helper_write_lock(set_project_root)

    with reads.recording(files_dir):
        child = subprocess.run(
            [sys.executable, "-c", "import pivot; pivot.output_hash('p/plan', 'plan')"],
            cwd=set_project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    assert child.returncode != 0
    assert "write the hash into a pin file" in child.stderr, child.stderr


@pytest.mark.parametrize(
    ("stage", "key", "message"),
    [
        pytest.param("p/other", None, "Stage 'p/other' has no lock", id="no-lock"),
        pytest.param("p/plan", "absent", "records no output 'absent'", id="no-output"),
    ],
)
def test_output_hash_of_an_output_no_lock_records_names_it(
    set_project_root: pathlib.Path, stage: str, key: str | None, message: str
) -> None:
    _ = _helper_write_lock(set_project_root)

    with pytest.raises(exceptions.OutputMissingError, match=message):
        pivot.output_hash(stage, key)


@pytest.mark.usefixtures("set_project_root")
def test_output_hash_reads_the_project_at_root_not_the_current_one(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    other = tmp_path_factory.mktemp("other")
    (other / ".pivot").mkdir()
    index = _helper_write_lock(other)

    assert pivot.output_hash("p/plan", "index", root=other) == index
    with pytest.raises(exceptions.OutputMissingError, match="has no lock"):
        pivot.output_hash("p/plan", "index")
