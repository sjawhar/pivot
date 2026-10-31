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
| Stage lock `reads` | the same file | each object the run read by hash ([`pivot.read_by_hash`](dependencies.md#objects-named-by-hash)) |
| `.pvt` file | anywhere under the project root | a tracked file's hash, or a tracked directory's listing hash and every file its listing names |
| Run-cache row | `.pivot/state.lmdb/` | the output hashes a previous run can restore without executing, a directory's members included, and the objects that run read by hash |
| `--no-commit` record | `.pivot/state.lmdb/` | the objects an uncommitted run read by hash, which `pivot commit` writes into the lock |
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
links it still holds keep their objects until a later restore of that path removes it. Each
instance of a mapped stage (`<stage>@<key>`, from `pipeline.map`) records its
outputs in a lock of its own, and a collection reads every lock on disk: an instance the last
run no longer expanded, because its key file is gone, keeps its objects like any other stage.

A record the collection cannot fully read refuses it, naming the record: a stage lock, `.pvt`
or run-cache row whose text does not parse (a git merge left conflict markers in it, a
truncated file), one that is not that kind of record, one that lacks a key or holds a key, a
schema version or a field this pivot does not know (a newer pivot may record objects there),
and a `.pvt` whose hash is not a cache key. Since a directory's `.pvt` and lock name its files
only through its listing, no reading of such a record's text could say which objects it
keeps. Finish the merge (or fix the file) and collect again. A `.pvt` a stage wrote into its
own output that does not parse is that stage's data, and names nothing.

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
project recording itself in a shared cache — holds `naming.lock` shared while it names: from
its last check that the objects it needs are in the cache to the record that names them. The
lock lives beside the cache's register: in the cache root (the directory that holds `files/`),
or, when `files/` is a link, in the directory that holds the store it links to (see
[A Cache Several Projects Share](#a-cache-several-projects-share)). So it is one file however
the cache path is spelled, and one
file for every cache root that uses the same object store.
Every writer hashes and copies its outputs (and `pivot get --rev` reads git and downloads)
before taking it, then re-checks the objects under it, publishing again any a collection removed
meanwhile, so a large output never holds the lock. `pivot gc` holds the same lock exclusive while it
re-reads the register, prunes run history, re-reads every record and run-cache row and
deletes, so no object can be named between that re-read and its removal. The shared lock
never covers a stage's execution, and a turnstile keeps a stream of writers from starving a
collection. The plan and the prompt run before the lock is taken, so a dry run or a waiting
prompt blocks nobody; nothing read before the lock decides what is deleted.
A collection also takes `gc.lock`, beside the naming lock, so two collections never run on one
object store, and lists
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
 "host": "workstation", "machine": "4f1c...", "mount_ns": 4026531841,
 "last_write": "2026-10-27T06:43:11Z", "project_id": "0f8c...",
 "mounts": ["/home/me/experiment/data"]}
```

The project id is a random value in `<state dir>/state.lmdb/pivot-project-id`, so git never
copies it to another checkout. A copy of the project (`cp -a`, a container built from an image
that holds it, a VM cloned from its disk) keeps the id and can keep its directories' inode
numbers, so neither proves which project a view reads. Every naming step therefore writes one
fresh nonce to `<state dir>/state.lmdb/gc-views/<view>-<cache id>` and to the entry's sibling
`projects/<digest>.nonce`, so a clone taken while a writer runs holds a nonce the writer's next
step replaces. The cache id is a random value in `projects/cache-id` of the cache's register
(the cache root, or beside the object store when `files/` is a link; see below), so a project
that uses two caches keeps one nonce for each, and registering in one never invalidates the
other's entry.
A project that sets a new `core.state_dir` copies the id from its `.pivot` state.
`mounts` lists the mount points at or below the project root and state directory that this view
saw.

Every pivot process that names a cache object registers its project first, inside the naming
lock described above: a stage commit, a run-cache restore, `pivot commit`, `pivot track` and
`pivot get --rev`, whatever the command and whether or not the cache existed when it started
(the project whose first run creates a shared cache is registered by that run). `repro` and
`run` also register once before any stage runs, so a project that is not in the register yet
(last run by an older pivot) is in it while its stages run, skip or restore; a run started
while a collection deletes waits for it there. `checkout` and `pull` register before they link
anything. Read-only commands such as
`status`, `diff`, `list` and `dag` do not register. If the register cannot be written, the
command refuses before it names an object. A project id is best-effort through #503's state
resolver. If the state directory cannot store it, the entry has no id, the writer still
follows 0.5.2 behavior, and gc refuses that entry with recovery instructions. A read-only cache
is the one exception to a register-write failure, because no collection can delete from it
there. A view (a host or mount namespace) that sees a cache read-only, while another view
writes it, does not list that cache in the project's state directory either: it can never write
an entry there, and a collection of the project's other caches would otherwise wait for one.
Such a view links nothing from that cache either: a collection run where the cache is writable
cannot see the unregistered project, so it could remove an object the link names. In symlink or
hardlink checkout the command refuses, saying to restore copies (`cache.checkout_mode: [copy]`)
or to mount the cache read-write; in copy checkout it restores copies, which are the project's
own data whatever a collection later removes.

A collection reads every entry, and an entry is live only after the collection finds its
project id in the project's current state directory and reads there, in its own view, the
nonce in the entry's `.nonce` file. A project shared over NFS shows another host's latest nonce,
so it stays collectable. A container's overlay or a cloned disk holds the nonce of the moment it
was copied: once the original takes another naming step, the copy's nonce is older, and a
collection where the copy is visible refuses rather than read that copy as the project. Until
then the copy and the original are the same bytes (see
[A Copy That Writes First](#a-copy-that-writes-first)). An absent or unreadable root,
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

The register (`projects/`), with the cache's id, its mount records (`mounts/`) and its lock
files (`naming.lock`, `naming.turnstile`, `gc.lock`), lives beside the object store: in the
cache root, or, when the cache root's `files/` is a link, in the directory that holds the
directory it links to. So every cache root that uses one object store (several roots whose
`files/` link one directory, or a root whose `files/` links another root's `files/`) shares one
register and one lock, and a collection through any of them reads every project that names
objects in the store. Whole-directory trees (`trees/`) live in two places in a linked layout:
a stage's commit through any root writes its tree beside the object store (`trees/` next to the
register), which every root of the store shares, and `pivot track` or a restore through a root
writes it under that root's own `trees/`. A collection through a root lists both of its tree
stores, removes a dead tree from either, and keeps a tree under another root of the store when
a project links it. The directory that holds the store must be writable by every user who names
objects in the store.
When the object store is the root of a mount of its own (a mount point or a bind mount), the
directory that holds it is on another filesystem, which other hosts and mount namespaces that
mount the store need not reach, so the register cannot follow the store: it stays at the cache
root. That is so when `files/` links to such a mount, and when `files/` is itself a mount point
(another device than the cache root, or a bind mount the mount table lists). Pivot refuses both
layouts: every command that names objects (a run, `pivot commit`, `pivot track`, `pivot
checkout`) refuses before it names anything, and `pivot gc` refuses, each naming the layout. So
no host or container writes into a store whose register is not where every user of the store
looks (a container given only another root's `files/` would register in its own cache root, out
of sight of that root's collection). Two layouts are supported instead: mount the filesystem at
the cache root (or point `cache.dir` at a directory inside it), so `files/` is a plain directory
of that filesystem; or keep the objects in a directory inside the mounted filesystem and link
`files/` to it. A collection resolves where the register and the store are once, when it
starts, and works through that for every step; if `files/` is re-pointed, or a store directory
or one of its two-character prefix directories is replaced (or covered by a mount) while it
runs, it refuses before deleting anything more, naming what moved. It opens one prefix
directory at a time, so it needs only a few open files whatever `ulimit -n` allows. Races that
need root or FUSE, such as a mount made inside a dead tree while the collection removes it, are
not covered: whoever can make them can delete the cache's objects directly.

### A Project That Uses More Than One Cache

A project registers in every cache it names objects in: a changed `cache.dir`, or a library run
given another `cache_dir`. Its state directory lists each one, with where the view that used it
found it, in `<state dir>/state.lmdb/gc-views/<view>-<cache id>.cache`, written before that
cache's first nonce. A copy of the project (a cloned disk) keeps the nonce of every cache as it
was when it was taken, and only a write the original makes to shared storage after that tells
the two apart. Pivot makes that write and checks it in two places:

- **Every naming step, in any cache, rotates the project's nonce in its entry in every other
  cache the project lists**, in any view: a cache another mount namespace or host of the project
  used counts, and the step registers its own view there first. A step writes only its own
  view's nonces and entries, never another view's, so a copy's own steps never make the
  original's entries match the copy. Each other cache costs one nonce rotation per step (its
  first step in a process registers there in full); a project that uses one cache lists only it,
  and nothing more is written.
- **A collection checks every cache any view of the project lists**, besides its own entries:
  each must be readable where that view recorded it, hold the same cache, and hold that view's
  entry with the nonce the state directory holds for it. A mismatch says the path may hold a
  copy and never suggests `pivot forget-cache`. (A cache root copied with the disk whose
  `files/` links to shared storage keeps its register beside the shared store, not on the disk,
  so the original's steps there rotate the nonce the copy's collection reads.)

A cache this view lists that a naming step cannot reach (it cannot be read where it is recorded,
another cache is at that path now, or its register cannot be written) does not fail the step.
Pivot marks that cache `unconfirmed_since` in the view's record, and the command warns once,
naming the cache and `pivot forget-cache`; a cache only another view lists (a host cache a
container does not mount) is passed over without a word. Until a step reaches that cache again,
**a copy of the project that can reach a cache the original cannot reach may collect that
cache's copies of the original's objects**: the original then reports those outputs missing and
gets a stage output back by fetching it or running its stage again, never using a stale result;
a tracked file with no remote copy is lost (see
[A Copy That Writes First](#a-copy-that-writes-first)). Pivot keeps the run going rather than
refuse it for a cache it does not need. A collection, though, refuses while a listed cache
cannot be reached.

If this view of the project can no longer reach a cache it lists and no longer uses it (deleted,
or on a disk that is gone), remove it with
[`pivot forget-cache`](../cli/index.md#pivot-forget-cache) CACHE, the path or id the warning or
refusal names, in the project itself: never in a copy of it. It removes only this view's record
(another view removes its own; if that view is gone for good, delete the record file the
refusal names), refuses a cache this view reaches whose entry for the project is current, and
deletes nothing from any cache. Any pivot command that uses the removed cache lists it again.

### A Copy That Writes First

A copy of a project made by copying its disk (a VM snapshot, a cloned volume, `cp -a` at the
same path) holds the original's project id and nonces. Nothing in shared storage tells the two
apart until one of them takes a naming step. When the original writes first, the copy's nonces
are stale and every collection where the copy is visible refuses, as above. **When the copy
writes first** (it re-tracks a file or runs a stage) while the original stays idle, its step
rotates the nonces the original's entries share, and a collection where the copy is visible
then reads the copy's records as the original's. It can remove objects that only the
original's records name. The original then reports those outputs missing. A stage output comes
back with `pivot pull` (when a remote holds it) or a rerun, and is never used stale. **A tracked
file or directory may be lost for good**: its object exists only in the cache unless it was
pushed, no stage produces it, and in symlink checkout the workspace holds only a link to the
removed object, so with no remote copy it cannot be restored at all. No design
without an identity a disk copy cannot carry can catch this case, so pivot documents it as a
limit rather than refuse every collection. Keying entries on something a copy does not carry,
such as the boot id, does not close it either: the copy still reads its own records under the
original's entry whichever view keys it, and a reboot of the original looks the same as a clone.

Avoid it in either of two ways:

- Run `pivot gc` on a shared cache from one designated machine that holds the original
  projects and never a copy of them, and give copies their own cache (`cache.dir`, or
  `PIVOT_CACHE_DIR`) before they run.
- Before a copy runs any pivot command, give it its own identity: delete
  `<state dir>/state.lmdb/pivot-project-id` in the copy (and `.pivot/state.lmdb/pivot-project-id`
  too when `core.state_dir` names another directory, since a new state directory takes its id
  from `.pivot`). Its next command records a new project id. A collection where the copy is
  visible then finds that the original's entry names a different project at that path, and
  refuses instead of reading the copy as the original. Do not delete that entry to make the
  collection proceed; run the collection where the original is visible.

## Per-Stage Lock Files

Each stage has a `.lock` file that records everything about its last
successful run:

| Lock file field | Contents |
|-----------------|----------|
| `code_manifest` | [Fingerprint](fingerprinting.md) dict (key → hash) |
| `params` | Serialised [parameters](parameters.md) dict |
| `deps` | One entry per input: producer, key, content hash |
| `outs` | One entry per output: key, project-relative POSIX path, content hash |
| `reads` | The content hashes of the objects the run read by hash ([`pivot.read_by_hash`](dependencies.md#objects-named-by-hash)), sorted, each once; absent when it read none |

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
