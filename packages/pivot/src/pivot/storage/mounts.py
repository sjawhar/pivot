"""The mount table and filesystem allowlist for persistent file metadata.

The file-hash identity index and the .pvt walk memo both rely on a filesystem keeping a live
file's ``(st_dev, st_ino)`` unique. They must make the same decision from one mount table.
"""

from __future__ import annotations

import os
import pathlib
import re
import typing

# Each type below has the identity and timestamp properties both persistent metadata users need.
# tmpfs is special: a 32-bit tmpfs inode counter wraps and can give live files one inode, so it is
# safe only when mountinfo's super options confirm the kernel's 64-bit counter.
_ALLOWED_FS_TYPES = frozenset({b"ext2", b"ext3", b"ext4", b"xfs", b"btrfs", b"f2fs", b"zfs"})
_MOUNTINFO = pathlib.Path("/proc/self/mountinfo")
_MOUNTINFO_ESCAPE = re.compile(rb"\\([0-7]{3})")


class Mounts(typing.TypedDict):
    """The identity/memo allowlist read from ``/proc/self/mountinfo``."""

    allowed: frozenset[int]  # devices mounted only with an allowlisted type and options
    refused: frozenset[int]  # devices mounted with any refused type or options
    points: list[tuple[bytes, bytes]]  # (mount point, type), deepest first


def _allowed_type(fstype: bytes, super_options: bytes) -> bool:
    """Whether this mount's type and options keep live inodes unique per device."""
    if fstype == b"tmpfs":
        return b"inode64" in super_options.split(b",")
    return fstype in _ALLOWED_FS_TYPES


def read_mounts() -> Mounts:
    """Read the mounts; without the file (not Linux) no device is allowed."""
    try:
        lines = _MOUNTINFO.read_bytes().splitlines()
    except OSError:
        return Mounts(allowed=frozenset(), refused=frozenset(), points=[])
    allowed = set[int]()
    refused = set[int]()
    points = list[tuple[bytes, bytes]]()
    for line in lines:
        # id parent major:minor root mount-point options [optional fields...] - type source opts
        fields = line.split()
        try:
            separator = fields.index(b"-", 6)
            fstype = fields[separator + 1]
            super_options = fields[separator + 3]
            major, minor = fields[2].split(b":")
            device = os.makedev(int(major), int(minor))
        except (ValueError, IndexError):
            continue
        (allowed if _allowed_type(fstype, super_options) else refused).add(device)
        point = _MOUNTINFO_ESCAPE.sub(lambda match: bytes([int(match[1], 8)]), fields[4])
        points.append((point, fstype))
    points.sort(key=lambda mount: len(mount[0]), reverse=True)
    return Mounts(allowed=frozenset(allowed - refused), refused=frozenset(refused), points=points)


def device_allowed(mount_table: Mounts, device: int, path: str | os.PathLike[str]) -> bool:
    """Whether a file on ``device`` may use persistent identity/timestamp metadata.

    Any device with a refused mount is refused. An unlisted device can be a btrfs subvolume,
    which reports its own device; it is allowed only when the deepest mount holding ``path`` is
    btrfs. ``path`` is resolved only for that fallback.
    """
    if device in mount_table["allowed"]:
        return True
    if device in mount_table["refused"]:
        return False
    real_path = os.fsencode(os.path.realpath(path))
    for point, fstype in mount_table["points"]:
        if real_path == point or real_path.startswith(point.rstrip(b"/") + b"/"):
            return fstype == b"btrfs"
    return False
