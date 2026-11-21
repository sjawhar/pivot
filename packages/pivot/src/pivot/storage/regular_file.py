"""Read or copy a file only when it is a regular file.

Every file ``storage.cache`` reads to hash it or to save it into the cache, and every metric
file ``show.metrics.parse_metric_file`` reads, is opened by ``open_regular_file``, which states
the rule. ``entry_kind`` names what stands at a path that is neither a regular file nor a
directory.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import sys
from typing import TYPE_CHECKING

from pivot import exceptions

if TYPE_CHECKING:
    import io

# How a file is opened for reading; open_regular_file says why it takes O_NOCTTY
_READ_FLAGS = os.O_RDONLY | os.O_NOCTTY


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


def require_regular_file(path: str | os.PathLike[str], mode: int) -> None:
    """Refuse a read of ``path`` unless what stands there, of ``stat`` mode ``mode``, is a
    regular file: raise ``IsADirectoryError`` for a directory, as ``open`` raises it, else
    ``exceptions.NotAFileError`` naming its kind."""
    if stat.S_ISDIR(mode):
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), os.fspath(path))
    if not stat.S_ISREG(mode):
        raise exceptions.NotAFileError(os.fspath(path), entry_kind(mode))


def open_regular_file(path: str | os.PathLike[str]) -> int:
    """Open the regular file at ``path``, symlinks followed, for reading, and return its
    descriptor for the caller to close (``open(descriptor, ...)`` takes it over).

    Pivot never reads a FIFO, a socket or a device: an open of a FIFO waits for a writer, and a
    device such as ``/dev/zero`` never ends. This open keeps that rule. It takes ``O_NONBLOCK``,
    so on a FIFO it returns at once instead of waiting for a writer, and
    ``require_regular_file`` checks the descriptor itself (``fstat``), so an entry swapped in
    after any earlier ``stat`` of the path is refused too, the descriptor closed (a socket's
    open fails first, with ``ENXIO``). It takes ``O_NOCTTY``, so a terminal swapped in never
    becomes the controlling terminal of a session leader that has none before the check refuses
    it. A regular file's descriptor is made blocking again before it is returned, so every read
    of it waits for its bytes, as a read of a file ``open`` opened does.

    Where the ``O_NONBLOCK`` open fails with ``EAGAIN``, the file is opened again without
    ``O_NONBLOCK``, waiting as ``open`` waits, as ``_open_leased`` says. So the open waits only
    where ``open`` waits on a regular file, but for one race: where ``O_PATH`` or ``/proc`` is
    missing, and only there, an entry swapped in before that blocking open, after the failed
    ``O_NONBLOCK`` open (after the ``O_PATH`` check, where there is one), can make the open
    wait (a FIFO, for a writer) before the check refuses it.
    """
    try:
        descriptor = os.open(path, _READ_FLAGS | os.O_NONBLOCK)
    except BlockingIOError:
        return _open_leased(path)
    return _checked(path, descriptor)


def _checked(path: str | os.PathLike[str], descriptor: int) -> int:
    """``descriptor``, open for reading at ``path``, once ``require_regular_file`` passes on
    what it holds (``fstat``), made blocking (one opened without ``O_NONBLOCK`` already is).
    Where the check refuses, or making it blocking fails, the descriptor is closed and the
    error raised."""
    try:
        require_regular_file(path, os.fstat(descriptor).st_mode)
        os.set_blocking(descriptor, True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_leased(path: str | os.PathLike[str]) -> int:
    """Open for reading the file at ``path`` whose ``O_NONBLOCK`` open failed with ``EAGAIN``:
    a regular file under a conflicting lease, such as a Samba oplock or an nfsd write delegation
    on this host, which a blocking open waits to see broken. The path is opened with
    ``O_PATH``, which reads nothing and never waits, and ``require_regular_file`` checks that
    descriptor as in ``open_regular_file``; a regular file is then opened through
    ``/proc/self/fd`` without ``O_NONBLOCK``, so the open waits as ``open`` waits, on the same
    inode, so nothing swapped in since is read. Any other error of that open names ``path``, as
    ``open``'s would.

    Where ``O_PATH`` (Linux-only) is missing, or ``/proc`` (the open through it fails with
    ``ENOENT``), the ``O_PATH`` descriptor, where there is one, is closed and the path itself
    is opened without ``O_NONBLOCK``, waiting for the lease to break as 0.10.4's ``open`` did;
    the descriptor of that open is checked as ``open_regular_file`` checks its own
    (``_checked``), closed where the check refuses."""
    if hasattr(os, "O_PATH"):
        located = os.open(path, os.O_PATH)
        try:
            require_regular_file(path, os.fstat(located).st_mode)
            try:
                return os.open(f"/proc/self/fd/{located}", _READ_FLAGS)
            except FileNotFoundError:
                # No /proc: the path itself is opened below, never "no such file" for a file there
                pass
            except OSError as error:
                # EMFILE or ESTALE names the file, as open's would, never its /proc/self/fd link
                error.filename = os.fspath(path)
                raise
        finally:
            os.close(located)
    return _checked(path, os.open(path, _READ_FLAGS))


def open_text(path: str | os.PathLike[str], *, newline: str | None = None) -> io.TextIOWrapper:
    """``open(path, newline=newline)``, a text stream, for the regular file at ``path``:
    ``open`` builds it on the descriptor ``open_regular_file`` returns, so its ``name`` is
    ``path`` and a parser's error names the file (YAML's ``in "<path>", line 1``)."""
    return open(path, newline=newline, opener=lambda name, _flags: open_regular_file(name))


# The least one sendfile call is asked to copy, as shutil.copyfile asks: a larger file is asked
# for whole, but on a 32-bit build never more than _SENDFILE_32BIT_BLOCK (bpo-38319).
_SENDFILE_MIN_BLOCK = 8 * 1024 * 1024
_SENDFILE_32BIT_BLOCK = 1024 * 1024 * 1024


def _send_file(
    source: int, target: int, src: str | os.PathLike[str], dst: str | os.PathLike[str]
) -> bool:
    """Copy every byte of the regular file open at ``source`` to ``target``, open at its start,
    in the kernel (``os.sendfile``), and return True, as ``shutil.copyfile`` copies on Linux.
    Return False, having written nothing, where ``shutil.copyfile`` calls no sendfile
    (``shutil._USE_CP_SENDFILE`` false), and where the first call fails with any error but
    ENOSPC, after which ``shutil.copyfile`` copies through userspace. ENOSPC, or an error once a
    byte was written, is raised, naming ``src`` and ``dst``."""
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


def copy_regular_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """``shutil.copy2(src, dst)`` for the regular file at ``src``, whose bytes are read through
    ``open_regular_file``. ``shutil.copy2`` refuses a FIFO only when its ``stat`` finds one, so
    it waits on a FIFO swapped in after that ``stat``, and it reads a device (``/dev/zero`` for
    ever); this copy refuses both, naming the kind. On Linux the bytes are copied in the kernel
    (``_send_file``), as ``shutil.copyfile`` copies them, or, where ``_send_file`` declines,
    through userspace by ``shutil.copyfileobj``, as ``shutil.copyfile`` then copies them. Off
    Linux (where ``shutil.copyfile`` copies with ``fcopyfile`` on macOS) this copy always goes
    through userspace, which gives the same bytes. Then the metadata is copied as
    ``shutil.copy2`` copies it (``shutil.copystat``)."""
    with open(open_regular_file(src), "rb", buffering=0) as source, open(dst, "wb") as target:
        if not _send_file(source.fileno(), target.fileno(), src, dst):
            shutil.copyfileobj(source, target)
    shutil.copystat(src, dst)
