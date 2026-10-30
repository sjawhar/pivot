"""Objects a stage reads by content hash (``pivot.read_by_hash``), and what its lock records.

A stage whose declared input names a content hash (a pin holding the hash of a snapshot, a
pointer holding the hash of an index, a record naming the hashes of other objects) reads each
object with ``pivot.read_by_hash``. Pivot finds the object in the cache, or fetches it from the
project's remote into the cache, checks that its bytes hash to that hash, and hands the stage
the cache object's path. Each hash a run reads is recorded (``recording``), and the lock the run
writes lists them under ``reads`` beside the stage's declared dependencies (``storage/lock.py``),
as does the run-cache entry it files and the ``nocommit:`` record of a ``--no-commit`` run.

The declared input that names the hash is what reruns the stage when it changes: that input is a
dependency like any other, hashed as Pivot hashes every dependency. ``reads`` records what the run
read, so the lock says exactly which objects the outputs came from. A stage whose lock records an
object that is in neither the cache nor the remote cannot be run again as it ran, so every skip
decision counts it as changed (``unavailable``, passed to ``skip.check_stage``): ``status`` reports
it stale and ``repro`` reruns it, where the read then fails, naming the object.

The run's recording is process state, not a ``ContextVar``: a stage may read from threads it
starts (a thread pool, joblib's threading backend), and on Python 3.13 a new thread does not
inherit its parent's context. A worker runs one stage at a time, so the one recording of its
process is that stage's.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import TYPE_CHECKING, final

from pivot import exceptions, project, stage_io, types
from pivot import state_dir as state_dir_mod
from pivot.remote import config as remote_config
from pivot.remote import sync
from pivot.storage import cache, lock, state

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Collection, Generator

    from pivot.remote import protocol


@final
class Recording:
    """The objects one stage run reads by hash from the cache at ``cache_dir``."""

    cache_dir: pathlib.Path | None
    _hashes: set[str]
    _lock: threading.Lock

    def __init__(self, cache_dir: pathlib.Path | None) -> None:
        self.cache_dir = cache_dir
        self._hashes = set[str]()
        self._lock = threading.Lock()

    def add(self, content_hash: str) -> None:
        with self._lock:
            self._hashes.add(content_hash)

    def hashes(self) -> list[str]:
        """Each hash the run read, once, in hash order: what its lock lists under ``reads``."""
        with self._lock:
            return sorted(self._hashes)


def output_hash(
    stage_name: str, key: str | None = None, *, root: pathlib.Path | None = None
) -> str:
    """The content hash the committed lock of ``stage_name`` records for its output ``key``:
    None for a stage's one unnamed output, a ``TypedDict`` field's name otherwise. A file's hash
    is the object ``read_by_hash`` reads; a directory's is its listing's. Read from the lock file
    of the project at ``root`` (None: the current project) alone: no pipeline is loaded and
    nothing is hashed, so one process can read several projects' locks.

    Raises:
        OutputMissingError: The stage has no lock Pivot can read, or its lock records no such
            output.
    """
    project_root = project.get_project_root() if root is None else root
    stages_dir = lock.get_stages_dir(state_dir_mod.locate_state_dir(project_root))
    lock_data = lock.StageLock(stage_name, stages_dir).read(warn_old_format=False)
    if lock_data is None:
        raise exceptions.OutputMissingError(f"Stage '{stage_name}' has no lock in {stages_dir}")
    identity = types.ArtifactIdentity(stage_name, key)
    if identity not in lock_data["output_hashes"]:
        recorded = ", ".join(repr(other.key) for other in lock_data["output_hashes"])
        raise exceptions.OutputMissingError(
            f"The lock of stage '{stage_name}' records no output {key!r}; it records {recorded}"
        )
    return lock_data["output_hashes"][identity]["hash"]


class _Active:
    """This process's recording, while a stage runs."""

    recording: Recording | None = None
    guard: threading.Lock = threading.Lock()


@contextlib.contextmanager
def recording(cache_dir: pathlib.Path | None) -> Generator[Recording]:
    """Record the objects the stage this process runs reads by hash, from any of its threads.
    ``cache_dir`` is the cache object root (``<cache>/files``); None for a run without a cache,
    whose reads fail."""
    current = Recording(cache_dir)
    with _Active.guard:
        if _Active.recording is not None:
            raise RuntimeError("pivot is already recording a stage run's reads in this process")
        _Active.recording = current
    try:
        yield current
    finally:
        with _Active.guard:
            _Active.recording = None


def read_by_hash(content_hash: str) -> pathlib.Path:
    """The path of the object whose content hash is ``content_hash``, in a stage Pivot runs.

    Pass a hash one of the stage's declared inputs names, such as the text of a pin file: a
    change to that input reruns the stage, and the stage's lock lists every hash its run read
    under ``reads``. Pivot looks the object up in the cache, else fetches it from the project's
    default remote into the cache, and checks that its bytes hash to ``content_hash`` before
    returning. The path is Pivot's read-only cache object: read it, never change it. Any thread
    of the stage's own process may call this.

    Raises:
        ObjectReadError: Called outside a stage run; ``content_hash`` is not a content hash; the
            object is in neither the cache nor the remote, or the remote cannot be used; or its
            bytes hash to another hash.
    """
    current = _Active.recording
    if current is None:
        raise exceptions.ObjectReadError(
            "pivot.read_by_hash works only inside a stage that Pivot runs (`pivot repro`, "
            + "`pivot run`), which records what the stage reads"
        )
    with stage_io.PIVOT_IO:
        path = _read(current.cache_dir, content_hash)
    current.add(content_hash)
    return path


def _read(cache_dir: pathlib.Path | None, content_hash: str) -> pathlib.Path:
    if not cache.is_valid_hash(content_hash):
        raise exceptions.ObjectReadError(
            f"{content_hash!r} is not a content hash: expected 16 lowercase hex characters"
        )
    if cache_dir is None:
        raise exceptions.ObjectReadError(
            f"Object {content_hash} cannot be read: this run has no cache"
        )
    path = cache.resolve_cache_object_path(cache_dir, content_hash)
    if path is None:
        path = _fetch(cache_dir, content_hash)
    actual, _ = cache.hash_file(path)
    if actual != content_hash:
        raise exceptions.ObjectReadError(
            f"Object {content_hash} in the cache ({path}) hashes to {actual}: the cache object "
            + "is corrupt; remove it, and the next read fetches it from the remote"
        )
    return path


def _fetch(cache_dir: pathlib.Path, content_hash: str) -> pathlib.Path:
    """Download the object at ``content_hash`` from the default remote into the cache; the
    remote layer checks the bytes before it publishes them (``remote.keys.atomic_download``)."""
    remote = _default_remote()
    if isinstance(remote, str):
        raise exceptions.ObjectReadError(f"Object {content_hash} is not in the cache, and {remote}")
    name, storage = remote
    destination = cache.get_cache_path(cache_dir, content_hash)
    try:
        (result,) = asyncio.run(
            storage.download_batch([(content_hash, destination)], readonly=True)
        )
    except exceptions.RemoteError as error:
        result = None
        reason = str(error)
    else:
        reason = result["error"] if "error" in result else "the download failed"
    if result is None or not result["success"]:
        raise exceptions.ObjectReadError(
            f"Object {content_hash} is not in the cache, and remote '{name}' ({storage.url}) "
            + f"did not give it: {reason}"
        )
    return destination


def _default_remote() -> tuple[str, protocol.RemoteStorage] | str:
    """The project's default remote and its name (``sync.create_remote_from_name``), or why
    there is no remote to use."""
    if not remote_config.list_remotes():
        return "no remote is configured"
    try:
        storage, name = sync.create_remote_from_name(None)
    except exceptions.RemoteError as error:
        return f"no remote can be used ({error})"
    return name, storage


def lock_unavailable(
    lock_data: types.LockData | None, cache_dir: pathlib.Path | None, *, remember: bool
) -> str | None:
    """``unavailable`` for the objects ``lock_data`` records its run read; None for no lock, or
    one that records none. Every skip decision passes this to ``skip.check_stage``."""
    if lock_data is None or "reads" not in lock_data:
        return None
    return unavailable(lock_data["reads"], cache_dir, remember=remember)


def unavailable(
    hashes: Collection[str], cache_dir: pathlib.Path | None, *, remember: bool
) -> str | None:
    """Why some of ``hashes``, objects a lock or run-cache entry records its run read, are in
    neither the cache at ``cache_dir`` nor the project's default remote; None when each is in one
    of them.

    The cache is looked up first, one ``lstat`` per object. Only objects the cache lacks reach the
    remote: an object the project's remote index (StateDB ``remote:``, which ``pivot push`` and
    ``pivot pull`` fill) records for that remote's URL counts as there, as ``pivot push`` counts
    it; the rest are asked of the remote in one batch. With ``remember``, those it holds are added
    to the index, best-effort, so the next check asks nothing; a worker passes False, since it
    writes the state database only through the coordinator (``DeferredWrites``). A remote that
    cannot be reached leaves them unavailable, with the reason.
    """
    missing = sorted(
        content_hash
        for content_hash in set(hashes)
        if cache_dir is None
        or cache.resolve_cache_object_path(cache_dir, content_hash, notice=False) is None
    )
    if not missing:
        return None
    remote = _default_remote()
    if isinstance(remote, str):
        return _missing_reason(missing, remote)
    name, storage = remote
    state_dir = state_dir_mod.locate_state_dir(project.get_project_root())
    with state.open_readonly(state_dir, create=False) as state_db:
        known = (
            state_db.remote_hashes_intersection(name, set(missing))
            if state_db is not None and state_db.remote_get_url(name) == storage.url
            else set[str]()
        )
    unknown = [content_hash for content_hash in missing if content_hash not in known]
    if not unknown:
        return None
    try:
        existence = asyncio.run(storage.bulk_exists(unknown))
    except exceptions.RemoteError as error:
        return _missing_reason(unknown, f"remote '{name}' could not be reached: {error}")
    found = {content_hash for content_hash, exists in existence.items() if exists}
    if found and remember:

        def record(state_db: state.StateDB) -> None:
            sync.check_remote_url(state_db, name, storage)
            state_db.remote_hashes_add(name, found)

        _ = state.save_best_effort(state_dir, record, what="objects found on the remote")
    absent = [content_hash for content_hash in unknown if content_hash not in found]
    if not absent:
        return None
    return _missing_reason(absent, f"remote '{name}' does not hold them")


def _missing_reason(hashes: list[str], why: str) -> str:
    return f"Read objects missing from the cache: {', '.join(hashes)} ({why})"
