# Caching & Skip Detection

Pivot's cache ensures you never re-run a stage whose inputs haven't
changed. It is content-addressed (same inputs → same outputs), per-file
(not per-directory), and uses a three-tier skip algorithm to decide as
fast as possible.

## Content-Addressable Cache

Every output file is hashed with `xxhash64` and stored under
`<state dir>/cache/files/<hash[:2]>/<hash[2:]>` (a two-level directory structure
using the first two hex characters as a prefix). A directory has one cache
object at its directory hash too: its text listing. The directory's whole-tree
symlink checkout, when configured, is separate under
`<state dir>/cache/trees/<hash[:2]>/<hash[2:]>`; it cannot collide with a file or
listing object. The per-stage lock records a file's hash, or a directory's
listing hash, total size and file count. `<state dir>` is `core.state_dir`,
which defaults to `.pivot`.

```
<state dir>/
  cache/
    files/
      a1/
        b2c3d4e5f6g7h8     # cached file or a directory listing
        .pivot_cache.lock  # lock file pivot keeps in each prefix directory
      ...
    trees/
      a1/
        b2c3d4e5f6g7h8     # a complete directory for symlink checkout
    projects/
      3f2a1c0d9e8b7a65.json  # one view of a project that uses this cache (see below)
    mounts/
      9e1b2c3d4f5a6b7c.json  # mount points below the cache root that one view saw
    gc.lock                  # held while `pivot gc` collects this cache
  stages/
    demo/
      clean.lock           # lock file for stage "demo/clean"
      clean=.listing       # listing for clean's directory output, if any
      train.lock           # lock file for stage "demo/train"
  state.lmdb/              # LMDB database and this project's pivot-project-id
```

Each prefix directory also holds a `.pivot_cache.lock` file. A file or directory cache operation
uses a `.pivot_tmp_*` entry. In `symlink` mode, each cached directory tree has a
`<hash>.content-checked` marker and can have a `<hash>.lock` file beside it. The marker contains
the canonical listing bytes after Pivot verifies the tree. Pivot uses a tree without a check only
when the listing bytes match that marker. It checks an unmarked tree by content first, and it does
not use a tree marked for other listing bytes. A restore can mark a matching tree, but it never
repairs a tree. Only a commit can repair a tree from its file objects. Pivot keeps lock files and
removes stale `.pivot_tmp_*` entries at most once each hour per prefix directory. Do not delete
these files while Pivot runs. A tree that Pivot 0.5.11 or older marked `<hash>.verified` is
checked by content again. A user who cannot write the cache cannot mark or repair a tree. Pivot
then checks the tree on each of their symlink restores and commits, and their symlink commit
through a stale tree fails. A `pivot checkout --checkout-mode symlink` by a user who can write the
cache marks a tree that matches. Only a symlink commit by such a user repairs a stale tree.

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
| Stage lock `outs` | `.pivot/stages/**/<stage>.lock` | each output's hash; for a directory, its listing's hash (the listing object and the whole-directory tree) **and** every file its listing names |
| Stage lock `deps` | the same file | each input's hash, which may outlive the stage that produced it |
| `.pvt` file | anywhere under the project root | a tracked file's hash, or a tracked directory's listing hash and every file its listing names |
| Run-cache row | `.pivot/state.lmdb/` | the output hashes a previous run can restore without executing, a directory's members included |
| Workspace link | `data/**` and any tracked path | the object a `symlink` checkout points at: a file, or a whole directory's tree under `trees/` |
| Cache ref | `.pivot/cache/files/refs/**` | the object a `kind: cache` store published |

A directory's members are read from its listing: the cache's object at the directory's hash,
or, when the cache has lost it (a fresh clone), the listing file committed beside the lock or
`.pvt`. A recorded directory whose listing is in neither place refuses the collection, naming
the record: its files are unknown, so none of them may be removed. A `.pvt` that a stage
wrote into its own directory output is the stage's data, not a record; its objects are kept,
but a listing it lacks does not refuse. A lock an older pivot wrote lists each directory's
files inline, and those, with the directory hash it recorded, are kept as they are. An entry a
killed run left beside an output (`.pivot_backup_<token>.*`) is walked like any other, so the
links it still holds keep their objects until a later restore of that path removes it.

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
before it takes the naming lock; under the naming lock it waits for nothing. It lists both
stores, `files/` and `trees/`, and removes a directory tree with its `.content-checked` marker
(and the `.layout-checked` and `.verified` markers older pivot versions wrote), markers first,
and never removes a lock file or a `.pivot_tmp_*` entry another process owns.

A symlinked directory inside the project whose target lies outside it — `data -> /mnt/big/data`,
or a plain `.venv -> /opt/venvs/x` — hides whatever records are beneath it, since pivot does not
follow a link out of the project. A collection that finds one keeps every object and removes
nothing, and names the link: point it at a directory inside the project (or remove it) to
collect. A symlinked directory that stays inside the project is walked. A link into the cache's
own `files/` or `trees/` is a checkout, read as one; a link anywhere else under the cache root
(which may be a broad scratch directory holding your data too) leaves the project like any other.

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
the outputs of projects whose locks this one cannot read. Each view of a project records itself
in the cache under `projects/<digest>.json`. A view is one hostname, machine id and mount
namespace, so the key holds the root, the project id and the view:

```json
{"root": "/home/me/experiment", "state_dir": "/home/me/experiment/.pivot",
 "host": "workstation", "machine": "4f1c...", "mount_ns": 4026531841, "pivot": "0.5.12",
 "last_write": "2026-10-27T06:43:11Z", "project_id": "0f8c...",
 "mounts": ["/home/me/experiment/data"]}
```

The project id is a random value in `<state dir>/state.lmdb/pivot-project-id`, so git never
copies it to another checkout. A copy of the project (`cp -a`, a container built from an image
that holds it, a VM cloned from its disk) keeps the id and can keep its directories' inode
numbers, so neither proves which project a view reads. Every naming step therefore writes one
fresh nonce to `<state dir>/state.lmdb/gc-views/<view>` and to the entry's sibling
`projects/<digest>.nonce`, so a clone taken while a writer runs holds a nonce the writer's next
step replaces. A project that sets a new `core.state_dir` copies the id from its `.pivot` state.
`mounts` lists the mount points at or below the project root and state directory that this view
saw.

Every pivot process that names a cache object registers its project first, inside the naming
lock described above: a stage commit, a run-cache restore, `pivot commit`, `pivot track` and
`pivot get --rev`, whatever the command and whether or not the cache existed when it started
(the project whose first run creates a shared cache is registered by that run). Every restore
that links a cache object into a workspace registers first too: `checkout`, `pull`, a skipped
stage's restore in `repro` and `run`, and `--checkout-missing`. Read-only commands such as
`status`, `diff`, `list` and `dag` do not register. If the register cannot be written, the
command refuses before it names an object. A project id is best-effort through #503's state
resolver. If the state directory cannot store it, the entry has no id, the writer still
follows 0.5.2 behavior, and gc refuses that entry with recovery instructions. A read-only cache
is the one exception to a register-write failure, because no collection can delete from it.

A collection reads every entry, and an entry is live only after the collection finds its
project id in the project's current state directory and reads there, in its own view, the
nonce in the entry's `.nonce` file. A project shared over NFS shows another host's latest nonce, so it
stays collectable; a container's overlay or a cloned disk holds at most an older nonce, so the
collection refuses rather than read that copy as the project. An absent or unreadable root,
state directory, id or nonce refuses the collection and names the entry: the project may be on
a disk that is not mounted, in another mount namespace, or on another host, and a different
project may hold the same local path there. For an entry from this view,
every recorded mount must still be mounted. If you removed a mount on purpose, run
`pivot checkout` in that project; its naming step records the mounts it has now. An entry from
another host or mount namespace may record only the project root and state directory as
mounts: any other mount (a disk mounted under the project there) may hold records this host
cannot see, so the collection refuses. The cache's own mounts are recorded per view in
`mounts/` and checked the same way. If a project or a view of it is gone for good, or its state
database was reset, delete the entry the refusal names. `pivot gc` **refuses to remove
anything** from an external cache that records no project, or whose records it cannot read;
`pivot gc --projects` PATH names the projects that share it and records them, so later runs
need no flag. A `--projects` path must be a real pivot project — a directory with a readable
`.pivot` state directory. A path that is not one (the cache itself, a parent directory, a
worktree that was deleted and re-created, a typo) names no records, so pivot refuses the whole
run rather than collect this cache on one project's records alone.

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

1. **Stat check** — compare `(mtime_ns, size, device, inode)` against the
   StateDB cache. If all four match, reuse the cached hash (no I/O).
2. **Content hash** — if stat differs, read the file and compute
   `xxhash64`. Store the new stat + hash in StateDB for next time.

This gives O(1) skip detection for unchanged files while remaining
correct when files do change (content hash is the source of truth).
The fast path cannot detect an edit that preserves the size, inode, device and
mtime, such as a tool that restores the original mtime. Pivot detects the
file after its metadata changes. Run `pivot repro --force <stage>` to rewrite
the stage's output.

The StateDB entry is keyed by the file's path. A copy of a project made outside
Pivot holds the original's `<state dir>/state.lmdb`, whose entries name the original's
paths: a copy that hard-links Pivot's read-only outputs and copies the database,
or `cp -al`, which hard-links the database too, so the two share it. Pivot also
finds an entry by the file's device and inode: a hard-linked output reuses the
original's entry, after Pivot checks that the original path still names the same
file, so `pivot status` in such a copy reads no output. It does so only on a filesystem that gives
each live file its own inode number on its device: ext2, ext3, ext4, XFS, Btrfs, F2FS and ZFS,
or tmpfs only when mountinfo says `inode64`. On any other filesystem (FUSE mounts such as mergerfs
or sshfs, NFS, overlayfs, tmpfs without `inode64`), and on systems without
`/proc/self/mountinfo` such as macOS, a copy's outputs are read once instead. A copied output is a
new file and is read once: `pivot status` saves the hashes it computed, so the next `pivot status`
reads none of them. Every command that hashes a dependency or an output saves its hash the same
way, also those that write no lock file: `pivot repro --no-commit`, `--dry-run` and `--explain`,
`pivot verify`, `pivot diff` and `pivot plots diff`. The tracked-file pass of `pivot status`, and
commands that materialize or track files (`pivot checkout`, `get`, `import`, `update`, `track` and
restore), may create `state.lmdb` when they have hashes to save. The read-only comparison passes
of explanations, diff and plots diff never create it. A save is skipped, never an error, when the
database cannot be written or another Pivot command holds its write lock for more than 2 seconds.

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
