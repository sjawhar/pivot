# Caching & Skip Detection

Pivot's cache ensures you never re-run a stage whose inputs haven't
changed. It is content-addressed (same inputs → same outputs), per-file
(not per-directory), and uses a three-tier skip algorithm to decide as
fast as possible.

## Content-Addressable Cache

Every output file is hashed with `xxhash64` and stored under
`.pivot/cache/files/<hash[:2]>/<hash[2:]>` (a two-level directory structure
using the first two hex characters as a prefix). The per-stage lock file
records each output's content hash.

```
.pivot/
  cache/
    files/
      a1/
        b2c3d4e5f6g7h8     # cached file (content-addressed)
        .pivot_cache.lock  # lock file pivot keeps in each prefix directory
      ...
    projects/
      3f2a1c0d9e8b7a65.json  # one project that uses this cache (see below)
    gc.lock                  # held while `pivot gc` collects this cache
  stages/
    demo/
      clean.lock           # lock file for stage "demo/clean"
      train.lock           # lock file for stage "demo/train"
  state.lmdb/              # LMDB database
```

Each prefix directory also holds a `.pivot_cache.lock` file, and while a file
or directory is being cached a `.pivot_tmp_*` entry; in `symlink` checkout
mode, each cached directory tree has a `<hash>.content-checked` marker and may
have a `<hash>.lock` file beside it. Pivot keeps the lock files and removes the
`.pivot_tmp_*` entries a killed process leaves (at most once an hour per
prefix directory). Do not delete these files while pivot runs. A tree that
pivot 0.5.11 or older marked `<hash>.verified` is checked by content again,
then marked anew. A user who cannot write the cache cannot mark it, so pivot
checks the tree on each of their symlink restores and commits, and their
symlink commit through a stale one fails. One
`pivot checkout --checkout-mode symlink` through the tree by a user who can
write the cache ends this.

A cache several machines share (a `cache.dir` on a network filesystem) needs
file locks that work across those machines: NFSv4, or NFSv3 with its lock
manager. With locks local to each machine (NFS mounted `nolock` or with
`local_lock`, Lustre with `localflock`), pivot on one machine can delete or
change files another machine is still writing into the cache.

Because the cache is content-addressed:

- Identical files are stored once, regardless of which stage produced them
- Reverting a parameter change can restore outputs from cache without
  re-executing
- The cache is shared across all [pipelines](pipelines.md) in a project

## What Keeps a Cache Object Alive

Nothing in the cache is ever removed by a pipeline run, so the cache only grows until you
run [`pivot gc`](../cli/index.md#pivot-gc). A collection reads what is on disk, never what
git records, and keeps every object that any of these names:

| Record | Where it lives | What it names |
|--------|----------------|---------------|
| Stage lock `outs` | `.pivot/stages/**/<stage>.lock` | each output's hash; a directory's tree hash **and** every hash in its manifest |
| Stage lock `deps` | the same file | each input's hash, which may outlive the stage that produced it |
| `.pvt` file | anywhere under the project root | a tracked file's hash, or a tracked directory's hash and members |
| Run-cache row | `.pivot/state.lmdb/` | the output hashes a previous run can restore without executing |
| Workspace link | `data/**` and any tracked path | the object a `symlink` checkout points at, file or whole directory |
| Cache ref | `.pivot/cache/files/refs/**` | the object a `kind: cache` store published |

Two more rules cover objects no record names yet:

- **A grace period.** A stage publishes its objects before it writes the lock that names
  them, so an object younger than `--grace-hours` (24 by default) is kept. The age is the
  object's `st_ctime`: a cached file carries its source file's mtime, which says nothing
  about when it was cached. The grace covers objects a run newly publishes; an object a run
  re-uses is protected by the naming lock below, not by its age.
- **Hard links.** An object with more than one link is kept, because some directory entry
  outside the cache holds that inode. `--remove-linked` overrides this, for a worktree that
  was deleted without unlinking its outputs; the plan then says how many objects that is.

A collection is safe to run while pipelines use the same cache, by mutual exclusion rather
than timing. Every pivot command that names a cache object — a stage committing its outputs
or restoring them from the run cache, `pivot commit`, `pivot track`, `pivot get --rev`, and a
project recording itself in a shared cache — holds `naming.lock` in the cache root shared
while it names: from its last check that the objects it needs are in the cache to the record
that names them. The lock lives in the cache root (the directory that holds `files/`), so it is
one file however the cache path is spelled, even when `files/` is a symlink to another disk.
Every writer hashes and copies its outputs (and `pivot get --rev` reads git and downloads)
before taking it, then re-checks the objects under it, publishing again any a collection removed
meanwhile, so a large output never holds the lock. `pivot gc` holds the same lock exclusive while it
re-reads the register, prunes run history, re-reads every record and run-cache row and
deletes, so no object can be named between that re-read and its removal. The shared lock
never covers a stage's execution, and a turnstile keeps a stream of writers from starving a
collection. The plan and the prompt run before the lock is taken, so a dry run or a waiting
prompt blocks nobody; nothing read before the lock decides what is deleted.
A collection also takes `<cache>/gc.lock`, so two collections never run on one cache, and lists
each prefix directory under its `.pivot_cache.lock`, waiting there for a publish in progress
before it takes the naming lock; under the naming lock it waits for nothing. It removes a
directory tree with its `.content-checked` marker (and the
`.layout-checked` and `.verified` markers older pivot versions wrote), markers first, and
never removes a lock file or a `.pivot_tmp_*` entry another process owns.

A symlinked directory inside the project whose target lies outside it — `data -> /mnt/big/data`,
or a plain `.venv -> /opt/venvs/x` — hides whatever records are beneath it, since pivot does not
follow a link out of the project. A collection that finds one keeps every object and removes
nothing, and names the link: point it at a directory inside the project (or remove it) to
collect. A symlinked directory that stays inside the project is walked. A link into the cache's
own `files/` is a checkout, read as one; a link anywhere else under the cache root (which may be a
broad scratch directory holding your data too) leaves the project like any other.

What this does and does not cover: there is no timing window between pivot processes that take
the lock. It holds when every pivot that uses the cache is this version or newer, the
filesystem honours `flock` across every host that mounts the cache, and every project that
shares the cache is in its register. A pivot older than this release that re-uses a cached
output during a collection can have that output removed (the grace still protects what it
newly publishes). Where the filesystem refuses `flock` (NFS mounted `nolock`, some Lustre and
network filesystems), `pivot gc` refuses to collect at all; collect such a cache with every
pivot process stopped, or move it to a filesystem with working locks. `pivot gc` cannot detect
a mount that keeps locks local to each host (NFS `local_lock`), and must not be run across
hosts on one.

### A Cache Several Projects Share

A `cache.dir` outside the project root (or a `PIVOT_CACHE_DIR` pointing elsewhere) can hold
the outputs of projects whose locks this one cannot read. Each project records itself in the
cache under `projects/<digest>.json`:

```json
{"root": "/home/me/experiment", "state_dir": "/home/me/experiment/.pivot",
 "host": "workstation", "pivot": "0.5.12", "last_write": "2026-10-27T06:43:11Z"}
```

Every pivot process that names a cache object registers its project first, inside the naming
lock described above: a stage commit, a run-cache restore, `pivot commit`, `pivot track` and
`pivot get --rev`, whatever the command and whether or not the cache existed when it started
(the project whose first run creates a shared cache is registered by that run). A command that
discovers the pipeline (`repro`, `run`, `status`, `checkout` and the others that load
`pipeline.py`) also registers when it starts. While a project's entry is under an hour old,
registering costs one `stat`. A collection reads every project the register names. An entry
whose root this host cannot see is never taken as deleted, because the project may be on a
disk that is not mounted, in a container's own mount namespace, or on another host: the
collection refuses and names the entry. Make the root visible and re-run, or, if the project is
gone for good, delete the entry it names. `pivot gc` **refuses to remove anything** from an
external cache that records no project, or whose records it cannot read; `pivot gc --projects
PATH` names the projects that share it and records them, so later runs need no flag. A
`--projects` path must be a real pivot project — a directory with a readable `.pivot` state
directory. A path that is not one (the cache itself, a parent directory, a worktree that was
deleted and re-created, a typo) names no records, so pivot refuses the whole run rather than
collect this cache on one project's records alone.

## Per-Stage Lock Files

Each stage has a `.lock` file that records everything about its last
successful run:

| Lock file field | Contents |
|-----------------|----------|
| `code_manifest` | [Fingerprint](fingerprinting.md) dict (key → hash) |
| `params` | Serialised [parameters](parameters.md) dict |
| `deps` | One entry per input: producer, key, content hash |
| `outs` | One entry per output: key, project-relative POSIX path, content hash |

This is the ground truth for "what did the stage look like last time it
ran?" The skip algorithm compares current state against these fields.

## Three-Tier Skip Algorithm

When a stage is about to execute, the worker decides whether to skip it.
The algorithm is designed to answer as quickly as possible, falling through
to more expensive checks only when needed.

### Tier 1: Generation Tracking — O(1)

StateDB maintains a monotonic **generation counter** for every output
file. When a stage runs and produces `data/demo/clean.jsonl`, the generation
for that file is incremented. The stage also records the generation of each
dependency at the time it ran.

On the next run, the worker checks: "are my dependency generations the
same as when I last ran?" This is a handful of integer comparisons — no
file I/O at all.

If generations match **and** the code fingerprint and params haven't
changed, the stage is skipped immediately.

### Tier 2: Lock File Comparison — O(n)

If generation tracking can't confirm a skip (first run, StateDB cleared,
generations diverged), the worker falls through to a full comparison:

1. Recompute the code fingerprint manifest, including call-site input wiring
   (parameter bindings, collection kind and order, and every input's declared path)
2. Serialise current parameters
3. Hash all dependency files (xxhash64, with StateDB-cached results)
4. Compare all three against the lock file

If everything matches, the stage is skipped. The worker also restores any
missing output files from the cache at this point.

### Tier 3: Run Cache — O(1) lookup

If tier 2 says the stage *should* run (something changed), there's one
more chance to skip. Pivot computes an **input hash** from the
combination of:

- Code fingerprint (including call-site input wiring)
- Current parameters
- Dependency hashes
- Output specs (identities and formats)

This input hash is looked up in the run cache
(`runcache:<stage>:<input_hash>` in StateDB). If a previous run with
identical inputs exists, Pivot restores its outputs from cache and writes
a new lock file — without executing the stage function.

This handles cases like: "I changed a parameter, ran the pipeline, then
changed it back." The original outputs are still in cache and can be
restored.

### Decision Flow

```
┌─────────────────────────┐
│ Tier 1: Generation check│──match──→ SKIP
│ (O(1), no file I/O)     │
└────────┬────────────────┘
         │ miss
┌────────▼────────────────┐
│ Tier 2: Lock comparison │──match──→ SKIP
│ (O(n), hash deps)       │
└────────┬────────────────┘
         │ changed
┌────────▼────────────────┐
│ Tier 3: Run cache       │──hit────→ SKIP (restore outputs)
│ (O(1), StateDB lookup)  │
└────────┬────────────────┘
         │ miss
         ▼
       EXECUTE
```

## File Change Detection

Pivot uses a two-level strategy for detecting file changes in dependencies and
outputs Pivot committed:

1. **Stat check** — compare `(mtime_ns, size, inode)` against the
   StateDB cache. If all three match, reuse the cached hash (no I/O).
2. **Content hash** — if stat differs, read the file and compute
   `xxhash64`. Store the new stat + hash in StateDB for next time.

This gives O(1) skip detection for unchanged files while remaining
correct when files do change (content hash is the source of truth).
The fast path cannot detect an edit that preserves the size, inode, and
mtime, such as a tool that restores the original mtime. Pivot detects the
file after its metadata changes. Run `pivot repro --force <stage>` to rewrite
the stage's output.

## Checkout Modes

When restoring outputs from cache, Pivot supports three strategies,
tried in order of preference. The same order links each stage output to
its cache entry after the stage runs, and when a skipped stage restores it:

| Mode | How it works | Trade-off |
|------|-------------|-----------|
| `hardlink` | Hard link to cache file | Zero copy, but editing the file modifies the cache |
| `symlink` | Symbolic link to cache file | Zero copy, visibly a link |
| `copy` | Full file copy | Safe but uses disk space |

The default order is `hardlink → symlink → copy`. A hard link refused because
the cache object already has as many links as the filesystem allows (65,000 on
ext4) makes that file a copy, even when `symlink` comes next or `copy` is not
in the order. Configure the order in `.pivot/config.yaml`:

```yaml
cache:
  checkout_mode: "copy"              # single mode
  checkout_mode: "hardlink,copy"     # fallback chain
```

Or override per-command:

```
$ pivot checkout --checkout-mode copy
```

## CLI Commands

### `pivot status --explain`

Shows *why* each stage would run or skip, breaking down the three-tier
decision:

```
$ pivot status --explain
train: stale (params changed: learning_rate 0.01 → 0.005)
clean: up to date (generation match)
```

### `pivot checkout`

Restore tracked files and stage outputs from cache:

```
$ pivot checkout                   # restore all
$ pivot checkout demo/train        # restore specific stage outputs
$ pivot checkout --only-missing    # restore only missing files; never overwrite edits
$ pivot checkout --force           # overwrite existing files
```

With `--force`, Pivot checks the required local cache objects before replacing an
existing file. If any object is missing, it names the missing hash and leaves
the workspace file unchanged.

### `pivot commit`

Record current outputs in lock files and cache without re-running:

```
$ pivot commit demo/train
```

Outputs a `pivot repro --no-commit` run wrote, unchanged since, are recorded
against the code that run ran, so a code edit made during or after it still
reruns the stage. Other outputs (edited by hand, or written another way) are
recorded against the code on disk now.

### `pivot gc`

Remove the cache objects no record names (see
[What Keeps a Cache Object Alive](#what-keeps-a-cache-object-alive)):

```
$ pivot gc --dry-run               # what would go, and how much space it holds
$ pivot gc                         # collect, asking first on a terminal
$ pivot gc --yes                   # collect from a script or a CI job
$ pivot gc --projects ../other     # a cache this project and ../other share
```

A collection also prunes this project's run history to `core.run_history_retention`, the same
policy every pipeline run applies, and the run-cache rows that prune drops are counted in the
plan. A `--dry-run` prunes nothing and reports exactly what a real run would remove; a run
refused at the prompt (or without a terminal and without `--yes`) prunes nothing and removes
nothing. The first `pivot gc` of a project that has never cached anything finds no cache
directory, reports nothing to remove and exits 0.

## Relationship to Other Concepts

The skip algorithm combines three input signals:

- **Code** — tracked by [fingerprinting](fingerprinting.md)
- **Parameters** — tracked via [parameters](parameters.md)
- **Data** — tracked via dependency hashes (see
  [dependencies](dependencies.md))

A stage is skipped only when all three match. Any single change is enough
to trigger re-execution.
