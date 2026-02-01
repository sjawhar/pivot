---
tags: [python, filesystem, atomicity]
category: gotcha
module: storage
symptoms: ["ResourceWarning: unclosed file", "file descriptor leak", "atomic write fails silently"]
---

# Atomic Writes: Track File Descriptor Closure with `mkstemp()`

## Problem

When implementing atomic writes using the `tempfile.mkstemp()` + rename pattern, the file descriptor returned by `mkstemp()` must be explicitly closed before the rename. Failing to track fd closure leads to resource leaks:

```python
import os
import tempfile

def save_atomically(path, data):
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(data)
    os.rename(tmp_path, path)
```

This looks correct, but has a subtle bug: if an exception occurs between `mkstemp()` and `os.fdopen()`, the file descriptor leaks. The `os.fdopen()` call takes ownership of the fd and will close it when the file object is garbage collected, but if we never reach that line, the fd remains open.

Additionally, in exception handlers we need to avoid double-closing the fd:

```python
def save_atomically(path, data):
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.rename(tmp_path, path)
    except Exception:
        os.close(fd)  # Bug: fd already closed by os.fdopen()!
        os.unlink(tmp_path)
        raise
```

This double-closes the fd, which can close an unrelated file descriptor that was allocated in the meantime (file descriptors are reused).

## Solution

Track whether the fd has been closed to handle both cases correctly:

```python
import contextlib
import os
import pathlib
import tempfile

def atomic_write(dest: pathlib.Path, write_fn):
    """Atomically write using temp file + rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    tmp = pathlib.Path(tmp_path)
    fd_closed = False
    try:
        write_fn(fd)
        fd_closed = True  # write_fn took ownership via os.fdopen
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        # Only close if write_fn didn't (e.g., exception before os.fdopen)
        if not fd_closed:
            with contextlib.suppress(OSError):
                os.close(fd)
```

For simpler cases where you don't delegate to a callback, use a sentinel value:

```python
async def download_atomically(url: str, local_path: pathlib.Path) -> None:
    """Download file atomically."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=local_path.parent, prefix=".download_")
    move_succeeded = False
    try:
        await stream_to_fd(url, fd)
        os.close(fd)
        fd = -1  # Mark as closed
        shutil.move(tmp_path, local_path)
        move_succeeded = True
    finally:
        if fd >= 0:
            os.close(fd)
        if not move_succeeded and os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

For context manager patterns where `os.fdopen()` is called in a `with` statement:

```python
@contextlib.contextmanager
def atomic_write_ctx(output_path: pathlib.Path):
    """Context manager for atomic file writes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        yield fd
        os.rename(tmp_path, output_path)
    except BaseException:
        # Close fd if still open (fdopen may not have been called)
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

# Usage - os.fdopen takes ownership of fd
with atomic_write_ctx(output_path) as fd, os.fdopen(fd, "w") as f:
    yaml.dump(data, f)
```

For the special case where you just need a temp file path (not writing to fd directly):

```python
def copy_to_cache(src: pathlib.Path, cache_path: pathlib.Path) -> None:
    """Atomically copy file to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
    tmp = pathlib.Path(tmp_path)
    try:
        os.close(fd)  # Close immediately - we just needed the unique path
        shutil.copy2(src, tmp_path)
        os.chmod(tmp_path, 0o444)
        tmp.replace(cache_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
```

## Key Insight

`tempfile.mkstemp()` returns both a file descriptor AND a path, and you are responsible for closing the fd. The three ownership patterns are:

| Pattern | Who Closes fd | Tracking Needed |
|---------|---------------|-----------------|
| `os.fdopen(fd, ...)` | The file object | Yes - track if fdopen was reached |
| Direct `os.write(fd, ...)` / `os.close(fd)` | Your code | Yes - track if close was called |
| Immediate close (need path only) | Your code, immediately | No - close before try block |

The fd is a scarce resource. Each process has a limit (often 1024 by default), and leaked fds accumulate until you hit "Too many open files". The leak is silent until it causes mysterious failures elsewhere in the program.

Always ask: "If an exception occurs at any line, will the fd be closed exactly once?" If the answer is unclear, add explicit tracking.
