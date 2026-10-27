---
tags: [python, paths, database, caching]
category: design
module: storage
symptoms: ["duplicate cache entries", "generation lookup misses", "symlink path confusion", "slow warm hash lookups"]
---

# StateDB Key Strategies: the Absolute Path for Hashes, the Artifact Identity for Generations

## Problem

Pivot's StateDB stores two kinds of keyed data with different requirements:

1. **File hash cache** - maps a file to its content hash, so an unchanged file is not read again
2. **Generation counters** - count how many times each output has been produced

A generation key must not move when an output's bytes move. After a symlink checkout,
`output.csv` points into `.pivot/cache/...`; a key built from `resolve()` would change with
every new content hash, and each run would look like a new output.

A file-hash key must be cheap, because every warm lookup builds one. It must also be
correct when a path is a symlink that is later pointed elsewhere.

## Solution

```python
def _make_key_file_hash(path: pathlib.Path) -> bytes:
    return _HASH_PREFIX + os.fsencode(os.path.abspath(path))


def _make_key_output_generation(identity: str) -> bytes:
    return _GEN_PREFIX + identity.encode()
```

| Key | Built from | Why |
|-----|------------|-----|
| `hash:` | `os.fsencode(os.path.abspath(path))` | Cheap, and keeps names that are not UTF-8 |
| `gen:` | the artifact identity | Stays the same whatever the output's path points to |

The hash key does not follow symlinks. Correctness rests on the entry's value, not the key:
it holds the `(mtime_ns, size, device, inode)` of `path.stat()`, which follows symlinks. A
link pointed at another file on the same filesystem stats to another inode. A link pointed at
another filesystem can find a file with the same inode number, size and mtime_ns (two tmpfs
mounts, fresh images, block-level clones), so the device is part of the entry too: either way
the entry no longer matches and the file is read again. Two links to one file hold two
entries and the file is read once for each, which costs less than resolving every lookup.

The entry holds no ctime. It would catch a rewrite that puts mtime back, but a new hard link
changes the inode's ctime: every hardlink checkout, and every replica that hard-links pivot's
files, would invalidate the entries of every file it links.

Earlier keys called `path.resolve()`. That cost more per warm lookup (16.2 µs) than reading
and hashing a small file cold (8.8 µs), measured on main for BET-36 plan Task 1.10e, and
`str(...).encode()` failed on a file name that is not UTF-8. Moving to the lexical key re-keys
every path that passes through a symlink, so after an upgrade each such file is read once
more, until a command that writes the workspace records it.

## Replicas: finding an entry by the file's identity

A replica of a project made outside pivot holds the original's state database: BET's replica
hard-links pivot's read-only outputs and copies the rest, `.pivot/state.lmdb` included, and
`cp -al` hard-links even the database, so the two share it. Its `hash:` keys name the
original's paths. The replica's hard-linked outputs are the very inodes those entries
describe, yet a lookup by the replica's path misses, so on BET's 180,000-file replicas every
`pivot status` read every output again (Dispatch BET-263).

Every write of a `hash:` entry also writes `inode:<st_dev><st_ino>` → that entry's key, and a
lookup that misses by path follows it. The other path's entry is served only when it matches
the file's stat (mtime_ns, size, device, inode) and that path, stat'ed now, still names the
same device and inode with the same change time: then a lookup at that path would serve the
same hash, and both paths name one inode, so the identity lookup is never weaker than a path
lookup at the recorded path. The second check is what keeps a reused inode number from being
served: a copy that keeps mtimes (`cp -a`, `rsync -a`, `shutil.copy2`) can give a new file the
number, size and mtime of a deleted one, but the deleted file's path names nothing any more.
The entry itself has no ctime because a new hard link changes it, which would invalidate every
hard-linked replica; check (2) reads both change times at lookup time instead.

The argument needs one device and inode number to name one live file. Some filesystems break
that: mergerfs with `inodecalc=passthrough` reports each branch's inode number under one
device, and other FUSE, network and union filesystems make inode numbers up. The change time
cannot tell two such files apart: one command that changes both (`touch -d @T a b`,
`chmod -R`) gives them one change time to the nanosecond, and with it one size and mtime: on
mergerfs that let `pivot repro` keep a stale output. So
the identity lookup runs only on a device that `/proc/self/mountinfo` mounts with a type on
`state._IDENTITY_FS_TYPES` (ext2, ext3, ext4, xfs, btrfs, tmpfs, f2fs, zfs; a btrfs
subvolume's own device counts as btrfs). Any other filesystem, and macOS, which has no
mountinfo, gets the path lookup only: it loses the replica speedup, never correctness. The
`.pvt` walk memo (#513) reads the same list; whichever merges second moves both to one helper.

`pivot status`, every explanation, `pivot diff` and `pivot plots diff` open the database
read-only, then save the hashes they had to read with `state.write_back_file_hashes`,
best-effort, so a copied replica's second status reads nothing. A `--no-commit` run saves the
hashes its workers read the same way: a hash entry is a cache its stat check validates, not
part of the commit.

## Key Insight

A cache key only has to name the place a reader will look again; the value proves whether the
cached hash still applies. Put the validity check in the value (`stat` follows symlinks) and
keep the key lexical, so building it costs no system call.
