# Caching & Skip Detection

Pivot's cache ensures you never re-run a stage whose inputs haven't
changed. It is content-addressed (same inputs → same outputs), per-file
(not per-directory), and uses a three-tier skip algorithm to decide as
fast as possible.

## Content-Addressable Cache

Every output file is hashed with `xxhash64` and stored under
`.pivot/cache/<hash[:2]>/<hash[2:]>` (a two-level directory structure using
the first two hex characters as a prefix). The per-stage lock file records a
manifest mapping each output path to its content hash.

```
.pivot/
  cache/
    a1/
      b2c3d4e5f6g7h8    # cached file (content-addressed)
    ...
  stages/
    clean.lock             # lock file for "clean" stage
    train.lock             # lock file for "train" stage
  state.db                 # LMDB database
```

Because the cache is content-addressed:

- Identical files are stored once, regardless of which stage produced them
- Reverting a parameter change can restore outputs from cache without
  re-executing
- The cache is shared across all [pipelines](pipelines.md) in a project

## Per-Stage Lock Files

Each stage has a `.lock` file that records everything about its last
successful run:

| Lock file field | Contents |
|-----------------|----------|
| `code_manifest` | [Fingerprint](fingerprinting.md) dict (key → hash) |
| `params` | Serialised [parameters](parameters.md) dict |
| `dep_hashes` | Input file hashes (path → xxhash64) |
| `output_hashes` | Output file hashes (path → xxhash64) |

This is the ground truth for "what did the stage look like last time it
ran?" The skip algorithm compares current state against these fields.

## Three-Tier Skip Algorithm

When a stage is about to execute, the worker decides whether to skip it.
The algorithm is designed to answer as quickly as possible, falling through
to more expensive checks only when needed.

### Tier 1: Generation Tracking — O(1)

StateDB maintains a monotonic **generation counter** for every output
file. When a stage runs and produces `output.csv`, the generation for
that path is incremented. The stage also records the generation of each
dependency at the time it ran.

On the next run, the worker checks: "are my dependency generations the
same as when I last ran?" This is a handful of integer comparisons — no
file I/O at all.

If generations match **and** the code fingerprint and params haven't
changed, the stage is skipped immediately.

### Tier 2: Lock File Comparison — O(n)

If generation tracking can't confirm a skip (first run, StateDB cleared,
generations diverged), the worker falls through to a full comparison:

1. Recompute the code fingerprint manifest
2. Serialise current parameters
3. Hash all dependency files (xxhash64, with StateDB-cached results)
4. Compare all three against the lock file

If everything matches, the stage is skipped. The worker also restores any
missing output files from the cache at this point.

### Tier 3: Run Cache — O(1) lookup

If tier 2 says the stage *should* run (something changed), there's one
more chance to skip. Pivot computes an **input hash** from the
combination of:

- Code fingerprint
- Current parameters
- Dependency hashes
- Output path specs

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

Pivot uses a two-level strategy for detecting file changes:

1. **Stat check** — compare `(mtime_ns, size, inode)` against the
   StateDB cache. If all three match, reuse the cached hash (no I/O).
2. **Content hash** — if stat differs, read the file and compute
   `xxhash64`. Store the new stat + hash in StateDB for next time.

This gives O(1) skip detection for unchanged files while remaining
correct when files do change (content hash is the source of truth).

## Checkout Modes

When restoring outputs from cache, Pivot supports three strategies,
tried in order of preference:

| Mode | How it works | Trade-off |
|------|-------------|-----------|
| `hardlink` | Hard link to cache file | Zero copy, but editing the file modifies the cache |
| `symlink` | Symbolic link to cache file | Zero copy, visibly a link |
| `copy` | Full file copy | Safe but uses disk space |

The default order is `hardlink → symlink → copy`. Configure it in
`.pivot/config.yaml`:

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
$ pivot checkout train             # restore specific stage outputs
$ pivot checkout --only-missing    # skip files that already exist
$ pivot checkout --force           # overwrite existing files
```

### `pivot commit`

Record current outputs in lock files and cache without re-running:

```
$ pivot commit train
```

## Relationship to Other Concepts

The skip algorithm combines three input signals:

- **Code** — tracked by [fingerprinting](fingerprinting.md)
- **Parameters** — tracked via [parameters](parameters.md)
- **Data** — tracked via dependency hashes (see
  [dependencies](dependencies.md))

A stage is skipped only when all three match. Any single change is enough
to trigger re-execution.
