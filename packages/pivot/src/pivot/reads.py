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
object that is not in the cache and not known to be on the remote (``unavailable``, decided from
local state as for outputs) counts as changed in every skip decision (``skip.check_stage``):
``status`` reports it stale and ``repro`` reruns it, where the read fetches the object or fails
naming it. The objects a run's reads fetch go back to the coordinator with the run's other
StateDB writes (``DeferredWrites.remote_hashes``), which records them in the remote index, as
``pivot pull`` records what it downloads, so the cache losing one later does not make it unknown.

The run's recording is process state, not a ``ContextVar``: a stage may read from threads it
starts (a thread pool, joblib's threading backend), and on Python 3.13 a new thread does not
inherit its parent's context. A worker runs one stage at a time, so the one recording of its
process is that stage's.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os
import threading
from typing import TYPE_CHECKING, final

from pivot import exceptions, project, stage_io, types
from pivot import state_dir as state_dir_mod
from pivot.remote import config as remote_config
from pivot.remote import sync
from pivot.storage import cache, lock, state

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Collection, Coroutine, Generator

    from pivot.remote import protocol


@final
class Recording:
    """The objects one stage run reads by hash from the cache at ``cache_dir``, in the process
    ``pid``: a forked child holds a copy of the recording that no lock ever reads. It also keeps
    the objects its reads fetched, by remote, for the coordinator to record in the remote index
    as ``pivot pull`` records what it downloads (``StateDB.apply_deferred_writes``)."""

    cache_dir: pathlib.Path | None
    pid: int
    _hashes: set[str]
    _fetched: dict[tuple[str, str], set[str]]
    _lock: threading.Lock

    def __init__(self, cache_dir: pathlib.Path | None) -> None:
        self.cache_dir = cache_dir
        self.pid = os.getpid()
        self._hashes = set[str]()
        self._fetched = dict[tuple[str, str], set[str]]()
        self._lock = threading.Lock()

    def add(self, content_hash: str) -> None:
        with self._lock:
            self._hashes.add(content_hash)

    def add_fetched(self, remote_name: str, url: str, content_hash: str) -> None:
        """Record that a read downloaded ``content_hash`` from remote ``remote_name`` at ``url``."""
        with self._lock:
            self._fetched.setdefault((remote_name, url), set[str]()).add(content_hash)

    def hashes(self) -> list[str]:
        """Each hash the run read, once, in hash order: what its lock lists under ``reads``."""
        with self._lock:
            return sorted(self._hashes)

    def fetched(self) -> list[types.RemoteHashes]:
        """The objects the run's reads fetched, per remote: what its ``DeferredWrites`` hand the
        coordinator under ``remote_hashes``."""
        with self._lock:
            return [
                types.RemoteHashes(name=name, url=url, hashes=sorted(hashes))
                for (name, url), hashes in sorted(self._fetched.items())
            ]


def _run_async[T](make: Callable[[], Coroutine[object, object, T]]) -> T:
    """Run the coroutine ``make`` returns to completion from synchronous code. A read may come
    from a coroutine (a stage's helper under ``asyncio.run``), where this thread already runs an
    event loop that a second loop may not join, so the coroutine then runs on a thread of its
    own while this one waits."""
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make())).result()


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


_STAGE_PROCESS_ENV = "PIVOT_READ_BY_HASH_STAGE_PID"
"""Set to the stage's process id while it runs, so a process the stage starts (a spawned or
forked child, joblib's processes backend) is told why it cannot read rather than that it is
outside a stage."""


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
    previous = os.environ.get(_STAGE_PROCESS_ENV)
    os.environ[_STAGE_PROCESS_ENV] = str(current.pid)
    try:
        yield current
    finally:
        if previous is None:
            _ = os.environ.pop(_STAGE_PROCESS_ENV, None)
        else:
            os.environ[_STAGE_PROCESS_ENV] = previous
        with _Active.guard:
            _Active.recording = None


def read_by_hash(content_hash: str) -> pathlib.Path:
    """The path of the object whose content hash is ``content_hash``, in a stage Pivot runs.

    Pass a hash one of the stage's declared inputs names, such as the text of a pin file: a
    change to that input reruns the stage, and the stage's lock lists every hash its run read
    under ``reads``. Pivot looks the object up in the cache, else fetches it from the project's
    default remote into the cache, and checks that its bytes hash to ``content_hash`` before
    returning. The path is Pivot's read-only cache object: read it, never change it. Any thread
    of the stage's own process may call this, a coroutine's included; a process the stage starts
    may not, since only the stage's own process records what it reads.

    Raises:
        ObjectReadError: Called outside a stage run or in a process the stage started;
            ``content_hash`` is not a content hash; the object is in neither the cache nor the
            remote, or the remote cannot be used; or its bytes hash to another hash.
    """
    current = _Active.recording
    if current is None or current.pid != os.getpid():
        if current is not None or _STAGE_PROCESS_ENV in os.environ:
            raise exceptions.ObjectReadError(
                "pivot.read_by_hash was called in a process a stage started: reads are recorded "
                + "only in the stage's own process and its threads, so read the object there "
                + "(a thread, or joblib's default threading backend) and pass its path or bytes "
                + "to the process"
            )
        raise exceptions.ObjectReadError(
            "pivot.read_by_hash works only inside a stage that Pivot runs (`pivot repro`, "
            + "`pivot run`), which records what the stage reads"
        )
    with stage_io.PIVOT_IO:
        path = _read(current, content_hash)
    current.add(content_hash)
    return path


def _read(current: Recording, content_hash: str) -> pathlib.Path:
    if not cache.is_valid_hash(content_hash):
        raise exceptions.ObjectReadError(
            f"{content_hash!r} is not a content hash: expected 16 lowercase hex characters"
        )
    cache_dir = current.cache_dir
    if cache_dir is None:
        raise exceptions.ObjectReadError(
            f"Object {content_hash} cannot be read: this run has no cache"
        )
    path = cache.resolve_cache_object_path(cache_dir, content_hash)
    if path is None:
        path = _fetch(current, cache_dir, content_hash)
    actual, _ = cache.hash_file(path)
    if actual != content_hash:
        raise exceptions.ObjectReadError(
            f"Object {content_hash} in the cache ({path}) hashes to {actual}: the cache object "
            + "is corrupt; remove it, and the next read fetches it from the remote"
        )
    return path


def _fetch(current: Recording, cache_dir: pathlib.Path, content_hash: str) -> pathlib.Path:
    """Download the object at ``content_hash`` from the default remote into the cache, and
    record in ``current`` where it came from; the remote layer checks the bytes before it
    publishes them (``remote.keys.atomic_download``)."""
    remote = _default_remote()
    if isinstance(remote, str):
        raise exceptions.ObjectReadError(f"Object {content_hash} is not in the cache, and {remote}")
    name, storage = remote
    destination = cache.get_cache_path(cache_dir, content_hash)
    try:
        (result,) = _run_async(
            lambda: storage.download_batch([(content_hash, destination)], readonly=True)
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
    current.add_fetched(name, storage.url, content_hash)
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
    lock_data: types.LockData | None, cache_dir: pathlib.Path | None
) -> str | None:
    """``unavailable`` for the objects ``lock_data`` records its run read; None for no lock, or
    one that records none. Every skip decision passes this to ``skip.check_stage``."""
    if lock_data is None or "reads" not in lock_data:
        return None
    return unavailable(lock_data["reads"], cache_dir)


def unavailable(hashes: Collection[str], cache_dir: pathlib.Path | None) -> str | None:
    """Why some of ``hashes``, objects a lock or run-cache entry records its run read, are
    neither in the cache at ``cache_dir`` nor known to be on the project's default remote; None
    when each is in one of them.

    The check is local, as for a stage's outputs, so plain ``status`` and ``repro`` never wait on
    the network: the cache is looked up first, one ``lstat`` per object, and an object it lacks
    counts as on the remote only when the project's remote index (StateDB ``remote:``, which
    ``pivot push``, ``pull`` and ``fetch`` fill, and a run with what its reads fetched) records it
    for that remote's URL, as ``pivot push`` counts it. No remote is contacted.
    ``pivot pull <stage>`` fetches the objects a stage's lock records, and a run's read fetches
    what it needs itself. When such objects are the reason, ``pull_fetches`` says so.
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
    absent = [content_hash for content_hash in missing if content_hash not in known]
    if not absent:
        return None
    return _missing_reason(absent, f"pivot has not seen them on remote '{name}'; {_PULL_FETCHES}")


_MISSING_PREFIX = "Read objects missing from the cache: "
_PULL_FETCHES = "`pivot pull` fetches them"


def _missing_reason(hashes: list[str], why: str) -> str:
    return f"{_MISSING_PREFIX}{', '.join(hashes)} ({why})"


def pull_fetches(reason: str) -> bool:
    """Whether ``reason``, why a stage must run, is ``unavailable``'s for objects a usable remote
    may hold but the remote index does not record, which ``pivot pull`` fetches; not its reason
    when no remote can be used. ``pivot status`` then suggests ``pivot pull``."""
    return reason.startswith(_MISSING_PREFIX) and reason.endswith(f"; {_PULL_FETCHES})")
