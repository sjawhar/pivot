"""Read or copy a file only when it is a regular file.

Every file Pivot hashes or copies into the cache (``storage.cache``), or shows as a metric
(``show.metrics``), is opened by ``open_regular_file``, which states the rule. ``entry_kind``
names what stands at a path that is neither a regular file nor a directory, for a refusal
(``exceptions.NotAFileError``) and for a warning (``track.occupant``).
"""

from __future__ import annotations

import errno
import io
import os
import shutil
import stat
import sys

from pivot import exceptions


def entry_kind(mode: int) -> str:
    """What an entry of ``stat`` mode ``mode`` that is neither a regular file nor a directory
    is, named for a message: "a socket", "a FIFO", "a device" or "an entry that is neither a file
    nor a directory". The one naming of these kinds: a read that refuses one
    (``exceptions.NotAFileError``) and what stands at a missing tracked path
    (``track.occupant``) both take it."""
    if stat.S_ISSOCK(mode):
        return "a socket"
    if stat.S_ISFIFO(mode):
        return "a FIFO"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return "a device"
    return "an entry that is neither a file nor a directory"


def not_a_file_error(path: str | os.PathLike[str], mode: int) -> OSError:
    """The error a read of ``path`` raises when what stands there, of ``stat`` mode ``mode``, is
    not a regular file: ``IsADirectoryError`` for a directory, as ``open`` raises it, else
    ``exceptions.NotAFileError`` naming its kind."""
    if stat.S_ISDIR(mode):
        return IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), os.fspath(path))
    return exceptions.NotAFileError(os.fspath(path), entry_kind(mode))


def open_regular_file(path: str | os.PathLike[str]) -> int:
    """Open the regular file at ``path``, symlinks followed, for reading, and return its
    descriptor for the caller to close (``open(descriptor, ...)`` takes it over). Every file
    Pivot hashes (``cache.hash_file``, ``cache.hash_directory``) or copies into the cache
    (``copy_regular_file``) is opened here, as is a metric output it shows (``open_text``).

    Pivot never reads a FIFO, a socket or a device: an open of a FIFO waits for a writer, and a
    device such as ``/dev/zero`` never ends. This open keeps that rule, and waits only where
    ``open`` waits on a regular file. It takes ``O_NONBLOCK``, so on a FIFO it returns at once
    instead of waiting for a writer, and the check is made on the descriptor itself (``fstat``),
    so an entry swapped in after any earlier ``stat`` of the path is refused too: a directory
    raises ``IsADirectoryError``, as ``open`` does, and a FIFO or a device
    ``exceptions.NotAFileError`` naming its kind, the descriptor closed (a socket's open fails
    first, with ``ENXIO``). It takes ``O_NOCTTY``, so a terminal swapped in never becomes the
    controlling terminal of a session leader that has none before the check refuses it. A
    regular file's descriptor is made blocking again before it is returned, so every read of it
    waits for its bytes, as a read of a file ``open`` opened does. Where the ``O_NONBLOCK`` open
    fails with ``EAGAIN``, the file is opened again as ``_open_leased`` says.
    """
    flags = os.O_RDONLY | os.O_NOCTTY
    try:
        descriptor = os.open(path, flags | os.O_NONBLOCK)
    except BlockingIOError as leased:
        return _open_leased(path, flags, leased)
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise not_a_file_error(path, mode)
        os.set_blocking(descriptor, True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_leased(path: str | os.PathLike[str], flags: int, leased: BlockingIOError) -> int:
    """Open with ``flags`` the file at ``path`` whose ``O_NONBLOCK`` open failed with
    ``leased`` (``EAGAIN``): a regular file under a conflicting lease, such as a Samba oplock or
    an nfsd write delegation on this host, which a blocking open waits to see broken. The path
    is opened with ``O_PATH``, which reads nothing and never waits, and the ``fstat`` of that
    descriptor decides as in ``open_regular_file``; a regular file is then opened through
    ``/proc/self/fd`` without ``O_NONBLOCK``, so the open waits as ``open`` waits, on the same
    inode, so nothing swapped in since is read. Where ``O_PATH`` or ``/proc`` is missing,
    ``leased`` is raised."""
    if not hasattr(os, "O_PATH"):
        raise leased
    located = os.open(path, os.O_PATH)
    try:
        mode = os.fstat(located).st_mode
        if not stat.S_ISREG(mode):
            raise not_a_file_error(path, mode)
        try:
            return os.open(f"/proc/self/fd/{located}", flags)
        except FileNotFoundError:
            # No /proc: the path's own error, never "no such file" for a file that is there
            raise leased from None
    finally:
        os.close(located)


def open_text(path: str | os.PathLike[str], *, newline: str | None = None) -> io.TextIOWrapper:
    """``open(path, newline=newline)``, a text stream, for the regular file at ``path``, opened
    by ``open_regular_file``. Its ``name`` is ``path``, as is the name of the stream ``open``
    returns, so a parser's error names the file (YAML's ``in "<path>", line 1``), not a
    descriptor number."""
    raw = io.FileIO(open_regular_file(path))
    raw.name = os.fspath(path)
    return io.TextIOWrapper(io.BufferedReader(raw), newline=newline)


# The least one sendfile call is asked to copy, as shutil.copyfile asks: a larger file is asked
# for whole, but on a 32-bit build never more than _SENDFILE_32BIT_BLOCK (bpo-38319).
_SENDFILE_MIN_BLOCK = 8 * 1024 * 1024
_SENDFILE_32BIT_BLOCK = 1024 * 1024 * 1024


def _send_file(
    source: int, target: int, src: str | os.PathLike[str], dst: str | os.PathLike[str]
) -> bool:
    """Copy every byte of the regular file open at ``source`` to ``target``, open at its start,
    in the kernel (``os.sendfile``), and return True, as ``shutil.copyfile`` copies on Linux.
    Return False, having written nothing, where ``shutil.copyfile`` copies through userspace
    instead: where it calls no sendfile (``shutil._USE_CP_SENDFILE`` false), and where the
    first call fails with any error but ENOSPC. ENOSPC, or an error once a byte was written, is
    raised, naming ``src`` and ``dst``."""
    # shutil.copyfile calls sendfile only while this flag is true: it is false off Linux, and a
    # program sets it false where a filesystem's sendfile is broken (cpython#87909). Typeshed
    # leaves the private flag out.
    if not shutil._USE_CP_SENDFILE:  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType] - see above
        return False
    block = max(os.fstat(source).st_size, _SENDFILE_MIN_BLOCK)
    if sys.maxsize < 2**32:
        block = min(block, _SENDFILE_32BIT_BLOCK)
    offset = 0
    while True:
        try:
            sent = os.sendfile(target, source, offset, block)
        except OSError as error:
            # As in shutil.copyfile, a first call that fails having written nothing gives way to
            # a copy through userspace, unless the filesystem is full
            if offset == 0 and error.errno != errno.ENOSPC:
                return False
            error.filename, error.filename2 = os.fspath(src), os.fspath(dst)
            raise
        if sent == 0:
            return True
        offset += sent


def copy_regular_file(
    src: str | os.PathLike[str], dst: str | os.PathLike[str], *, chunk_size: int
) -> None:
    """``shutil.copy2(src, dst)`` for the regular file at ``src``, whose bytes are read through
    ``open_regular_file``. ``shutil.copy2`` refuses a FIFO only when its ``stat`` finds one, so
    it waits on a FIFO swapped in after that ``stat``, and it reads a device (``/dev/zero`` for
    ever); this copy refuses both, naming the kind. The bytes are copied in the kernel
    (``_send_file``), or through userspace, ``chunk_size`` bytes at a time, where it does not;
    then the metadata, as ``shutil.copy2`` copies it (``shutil.copystat``)."""
    with open(open_regular_file(src), "rb", buffering=0) as source, open(dst, "wb") as target:
        if not _send_file(source.fileno(), target.fileno(), src, dst):
            shutil.copyfileobj(source, target, chunk_size)
    shutil.copystat(src, dst)
