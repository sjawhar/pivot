from __future__ import annotations

import errno
import os
import shutil
import stat
import sys
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions
from pivot.storage import regular_file
from tests import helpers

if TYPE_CHECKING:
    import pathlib

_needs_o_path_and_proc = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="O_PATH and /proc are Linux-only"
)
_needs_sendfile = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="shutil copies with sendfile only on Linux"
)
# Linux fails a socket's open with ENXIO; other kernels give their own error number
_needs_linux_socket_open = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="a socket's open fails with ENXIO on Linux"
)

# What a copy through userspace reads at a time (shutil.copyfileobj); typeshed leaves it out
_COPY_BUFSIZE: int = shutil.COPY_BUFSIZE  # pyright: ignore[reportAttributeAccessIssue] - not in typeshed


def _helper_record_opens(
    monkeypatch: pytest.MonkeyPatch, *, leased: bool = False, proc_error: int | None = None
) -> list[tuple[str, int]]:
    """Record each ``os.open`` as (path, flags). With ``leased``, an open that takes
    ``O_NONBLOCK`` fails with ``EWOULDBLOCK``, as on a file under a conflicting lease (a Samba
    oplock, an nfsd write delegation); with ``proc_error``, an open under ``/proc`` fails with
    that errno, naming its ``/proc`` path: ``ENOENT`` where ``/proc`` is not mounted, ``EMFILE``
    where the process has no descriptor left."""
    real_open = os.open
    opened = list[tuple[str, int]]()

    def _open(
        path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        name = os.fspath(path)
        opened.append((name, flags))
        if leased and flags & os.O_NONBLOCK:
            raise OSError(errno.EWOULDBLOCK, os.strerror(errno.EWOULDBLOCK), name)
        if proc_error is not None and name.startswith("/proc/"):
            raise OSError(proc_error, os.strerror(proc_error), name)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", _open)
    return opened


@pytest.mark.parametrize(
    ("kind", "named"),
    [
        pytest.param("fifo", "a FIFO", id="fifo"),
        pytest.param("symlink_to_fifo", "a FIFO", id="symlink_to_fifo"),
        pytest.param("symlink_to_device", "a device", id="symlink_to_device"),
    ],
)
def test_open_regular_file_refuses_a_fifo_or_a_device_at_once(
    tmp_path: pathlib.Path, kind: str, named: str
) -> None:
    """With no ``stat`` before it, the open alone refuses a FIFO or a device, or a symlink to
    one: ``NotAFileError`` naming what the path leads to, at once, where an open of the FIFO
    without ``O_NONBLOCK`` would wait for a writer. Each call must return within the hard timeout
    of ``helpers.call_without_waiting``."""
    path = tmp_path / "input"
    helpers.occupy(path, kind)

    with pytest.raises(exceptions.NotAFileError) as raised:
        helpers.call_without_waiting(
            lambda: regular_file.open_regular_file(path),
            lambda: helpers.free_fifo(path),
            timeout=10,
        )

    assert (raised.value.kind, raised.value.filename) == (named, str(path))


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        pytest.param("directory", errno.EISDIR, id="directory"),
        pytest.param("socket", errno.ENXIO, id="socket", marks=_needs_linux_socket_open),
    ],
)
def test_open_regular_file_raises_for_a_directory_and_a_socket_as_open_does(
    tmp_path: pathlib.Path, kind: str, error: int
) -> None:
    """A directory raises ``IsADirectoryError``, as ``open`` raises it, and a socket's open
    fails with ``ENXIO`` on Linux, both naming the path."""
    path = tmp_path / "input"
    helpers.occupy(path, kind)

    with pytest.raises(OSError) as raised:
        regular_file.open_regular_file(path)

    assert (raised.value.errno, raised.value.filename) == (error, str(path))


def test_open_regular_file_returns_a_blocking_descriptor_of_a_regular_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One open, with ``O_NONBLOCK`` and with ``O_NOCTTY`` (a terminal swapped in never becomes
    the controlling terminal), gives a regular file's descriptor, blocking again once the check
    passes, as one ``open`` opened is, which reads the file's bytes."""
    path = tmp_path / "input"
    helpers.occupy(path, "regular_file")
    opened = _helper_record_opens(monkeypatch)

    with open(regular_file.open_regular_file(path), "rb") as f:
        blocking = os.get_blocking(f.fileno())
        data = f.read()

    assert (blocking, data) == (True, b"data")
    assert opened == [(str(path), os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)]


@_needs_o_path_and_proc
def test_open_regular_file_waits_as_open_does_on_a_leased_regular_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the ``O_NONBLOCK`` open fails with ``EWOULDBLOCK`` (a conflicting lease, which a
    blocking open waits to see broken), the path is opened with ``O_PATH``, which reads
    nothing, and the regular file it holds is opened again through ``/proc/self/fd`` without
    ``O_NONBLOCK``: the file is read as before, and the ``O_PATH`` descriptor is closed."""
    path = tmp_path / "input"
    helpers.occupy(path, "regular_file")
    opened = _helper_record_opens(monkeypatch, leased=True)

    with open(regular_file.open_regular_file(path), "rb") as f:
        data = f.read()
        nonblocking, located, (reopened, reopen_flags) = opened
        with pytest.raises(OSError) as closed:
            os.fstat(int(reopened.removeprefix("/proc/self/fd/")))

    assert data == b"data"
    assert (nonblocking, located) == (
        (str(path), os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK),
        (str(path), os.O_PATH),
    )
    assert (reopened.startswith("/proc/self/fd/"), reopen_flags) == (
        True,
        os.O_RDONLY | os.O_NOCTTY,
    )
    assert closed.value.errno == errno.EBADF


@_needs_o_path_and_proc
@pytest.mark.parametrize(
    ("kind", "named"),
    [
        pytest.param("fifo", "a FIFO", id="fifo"),
        pytest.param("symlink_to_device", "a device", id="symlink_to_device"),
    ],
)
def test_open_regular_file_refuses_a_fifo_or_a_device_where_the_open_is_leased(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, kind: str, named: str
) -> None:
    """The ``O_PATH`` open after an ``EWOULDBLOCK`` checks what the path holds before anything
    is opened for reading: a FIFO or a device is refused at once there too, within the hard
    timeout of ``helpers.call_without_waiting``."""
    path = tmp_path / "input"
    helpers.occupy(path, kind)
    opened = _helper_record_opens(monkeypatch, leased=True)

    with pytest.raises(exceptions.NotAFileError) as raised:
        helpers.call_without_waiting(
            lambda: regular_file.open_regular_file(path),
            lambda: helpers.free_fifo(path),
            timeout=10,
        )

    assert (raised.value.kind, raised.value.filename) == (named, str(path))
    assert [flags for _, flags in opened] == [
        os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK,
        os.O_PATH,
    ]


@pytest.mark.parametrize("missing", ["O_PATH", pytest.param("/proc", marks=_needs_o_path_and_proc)])
def test_open_regular_file_raises_the_lease_error_without_o_path_or_proc(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Where ``O_PATH`` (Linux-only) or ``/proc`` is missing, the ``O_NONBLOCK`` open's
    ``EWOULDBLOCK`` is raised, naming the path: never "no such file" for a file that is
    there."""
    path = tmp_path / "input"
    helpers.occupy(path, "regular_file")
    _helper_record_opens(
        monkeypatch, leased=True, proc_error=errno.ENOENT if missing == "/proc" else None
    )
    if missing == "O_PATH":
        monkeypatch.delattr(os, "O_PATH", raising=False)

    with pytest.raises(BlockingIOError) as raised:
        regular_file.open_regular_file(path)

    assert (raised.value.errno, raised.value.filename) == (errno.EWOULDBLOCK, str(path))


@_needs_o_path_and_proc
def test_open_regular_file_names_the_file_where_its_reopen_through_proc_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any other error of the reopen through ``/proc/self/fd`` after an ``EWOULDBLOCK``
    (``EMFILE``: no descriptor left) is raised naming the file, as ``open`` names it, never its
    ``/proc/self/fd`` link."""
    path = tmp_path / "input"
    helpers.occupy(path, "regular_file")
    _helper_record_opens(monkeypatch, leased=True, proc_error=errno.EMFILE)

    with pytest.raises(OSError) as raised:
        regular_file.open_regular_file(path)

    assert (raised.value.errno, raised.value.filename) == (errno.EMFILE, str(path))


def _helper_copy_source(path: pathlib.Path, size: int, mode: int) -> bytes:
    """Write ``size`` bytes, each other than its neighbours, at ``path``, with mode ``mode`` and
    times long past; return the bytes."""
    data = bytes(i % 251 for i in range(size))
    path.write_bytes(data)
    path.chmod(mode)
    os.utime(path, ns=(1_000_000_000, 2_000_000_000))
    return data


def _helper_assert_copied(src: pathlib.Path, dst: pathlib.Path, data: bytes) -> None:
    """``dst`` holds ``data`` and has ``src``'s mode and mtime, as ``shutil.copy2`` leaves it."""
    assert dst.read_bytes() == data
    src_stat, dst_stat = src.stat(), dst.stat()
    assert (stat.S_IMODE(dst_stat.st_mode), dst_stat.st_mtime_ns) == (
        stat.S_IMODE(src_stat.st_mode),
        src_stat.st_mtime_ns,
    )


@_needs_sendfile
@pytest.mark.parametrize(
    ("size", "mode"),
    [
        pytest.param(3 * 4096 + 7, 0o644, id="several_calls"),
        pytest.param(0, 0o644, id="empty"),
        pytest.param(4096, 0o755, id="executable"),
    ],
)
def test_copy_regular_file_copies_in_the_kernel_bytes_and_metadata_exactly(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, size: int, mode: int
) -> None:
    """The copy into the cache sends the bytes with ``os.sendfile``, call after call from where
    the last one stopped until one returns 0 (each call is held to 4096 bytes here), never
    through userspace, and gives the copy the source's mode and mtime, as ``shutil.copy2``
    does."""
    src = tmp_path / "src"
    data = _helper_copy_source(src, size, mode)
    dst = tmp_path / "dst"
    real_sendfile = os.sendfile
    offsets = list[int]()

    def _sendfile_4096(out_fd: int, in_fd: int, offset: int, count: int) -> int:
        offsets.append(offset)
        return real_sendfile(out_fd, in_fd, offset, min(count, 4096))

    monkeypatch.setattr(os, "sendfile", _sendfile_4096)
    monkeypatch.setattr(shutil, "copyfileobj", helpers.no_userspace_copy)

    regular_file.copy_regular_file(src, dst)

    _helper_assert_copied(src, dst, data)
    assert offsets == [*range(0, size, 4096), size]


_FIRST_CALL_ERRORS = [errno.ENODATA, errno.EIO, errno.ENOTSOCK, errno.EINVAL]


@_needs_sendfile
@pytest.mark.parametrize(
    "failure", _FIRST_CALL_ERRORS, ids=[errno.errorcode[e] for e in _FIRST_CALL_ERRORS]
)
def test_copy_regular_file_copies_through_userspace_where_the_first_sendfile_call_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: int
) -> None:
    """Where the first sendfile call fails before writing anything, with any error but ENOSPC
    (ENODATA from FSx for Lustre, EIO, ENOTSOCK where its target must be a socket, EINVAL), the
    bytes go through userspace instead, chunk by chunk, and the copy is the same, as
    ``shutil.copyfile`` falls back."""
    src = tmp_path / "src"
    data = _helper_copy_source(src, 2 * _COPY_BUFSIZE + 7, 0o755)
    dst = tmp_path / "dst"
    offsets = list[int]()

    def _fail(out_fd: int, in_fd: int, offset: int, count: int) -> int:
        offsets.append(offset)
        raise OSError(failure, os.strerror(failure))

    monkeypatch.setattr(os, "sendfile", _fail)

    regular_file.copy_regular_file(src, dst)

    _helper_assert_copied(src, dst, data)
    assert offsets == [0]


@_needs_sendfile
def test_copy_regular_file_raises_a_full_filesystem_on_the_first_sendfile_call(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENOSPC, a full filesystem, on the first sendfile call is raised, naming both files, and
    never gives way to a copy through userspace: the one first-call error ``shutil.copyfile``
    raises too."""
    src = tmp_path / "src"
    _helper_copy_source(src, 8192, 0o644)
    dst = tmp_path / "dst"

    def _full(out_fd: int, in_fd: int, offset: int, count: int) -> int:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(os, "sendfile", _full)
    monkeypatch.setattr(shutil, "copyfileobj", helpers.no_userspace_copy)

    with pytest.raises(OSError) as raised:
        regular_file.copy_regular_file(src, dst)

    assert (raised.value.errno, raised.value.filename, raised.value.filename2) == (
        errno.ENOSPC,
        str(src),
        str(dst),
    )


def test_copy_regular_file_copies_through_userspace_where_shutil_calls_no_sendfile(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``shutil._USE_CP_SENDFILE`` false (off Linux, or set so where a filesystem's
    sendfile is broken), the copy never calls sendfile, as ``shutil.copyfile`` does not, and
    copies the bytes and metadata exactly through userspace."""
    src = tmp_path / "src"
    data = _helper_copy_source(src, 2 * _COPY_BUFSIZE + 7, 0o644)
    dst = tmp_path / "dst"

    def _no_sendfile(out_fd: int, in_fd: int, offset: int, count: int) -> int:
        raise AssertionError("sendfile was called")

    monkeypatch.setattr(shutil, "_USE_CP_SENDFILE", False)
    monkeypatch.setattr(os, "sendfile", _no_sendfile)

    regular_file.copy_regular_file(src, dst)

    _helper_assert_copied(src, dst, data)


@_needs_sendfile
@pytest.mark.parametrize(
    "failure",
    [errno.EINVAL, errno.ENODATA],
    ids=[errno.errorcode[e] for e in (errno.EINVAL, errno.ENODATA)],
)
def test_copy_regular_file_raises_an_error_once_bytes_were_sent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: int
) -> None:
    """An error of a sendfile call after one wrote bytes is raised, naming both files: a copy
    through userspace then would not start where the sent bytes end."""
    src = tmp_path / "src"
    _helper_copy_source(src, 8192, 0o644)
    dst = tmp_path / "dst"
    real_sendfile = os.sendfile

    def _fail_after_4096(out_fd: int, in_fd: int, offset: int, count: int) -> int:
        if offset:
            raise OSError(failure, os.strerror(failure))
        return real_sendfile(out_fd, in_fd, offset, 4096)

    monkeypatch.setattr(os, "sendfile", _fail_after_4096)
    monkeypatch.setattr(shutil, "copyfileobj", helpers.no_userspace_copy)

    with pytest.raises(OSError) as raised:
        regular_file.copy_regular_file(src, dst)

    assert (raised.value.errno, raised.value.filename, raised.value.filename2) == (
        failure,
        str(src),
        str(dst),
    )
