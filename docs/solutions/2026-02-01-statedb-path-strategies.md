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

## Key Insight

A cache key only has to name the place a reader will look again; the value proves whether the
cached hash still applies. Put the validity check in the value (`stat` follows symlinks) and
keep the key lexical, so building it costs no system call.
