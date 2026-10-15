---
title: Performance Bottlenecks & Measurement Plan (StateDB/CAS/Materialization)
date: 2026-02-14
status: draft
---

# Performance Bottlenecks & Measurement Plan (StateDB/CAS/Materialization)

## Summary
Identify likely performance bottlenecks in Pivot's artifact-first DAG execution and
propose minimal mitigations plus concrete measurement points. The scope spans
StateDB (LMDB), content-addressed cache under `.pivot/cache/`, and workspace
materialization via hardlink/symlink/copy.

## Context
- StateDB is LMDB-backed with a single-writer constraint; workers open readonly and
  return deferred writes for coordinator commit.
- Skip detection relies on generation counters (O(1)) and hash caching keyed by
  `(mtime_ns, size, inode)` to avoid re-hashing unchanged files.
- Cache restoration supports `hardlink`, `symlink`, and `copy` modes; incremental
  outputs intentionally use `copy` to avoid cache corruption.
- CAS uses a 2-level directory fanout from hash prefixes.

References:
- `packages/pivot/src/pivot/storage/AGENTS.md`
- `packages/pivot/src/pivot/storage/state.py`
- `docs/architecture/execution.md`
- `docs/concepts/caching.md`
- `docs/solutions/2026-02-01-incremental-out-uses-copy-mode.md`
- `docs/solutions/2026-02-01-statedb-path-strategies.md`

## Constraints
- Keep hot paths O(1) when possible.
- Avoid extra syscalls (`exists`, redundant `stat`) in tight loops.
- Prefer mitigations that do not change correctness semantics or storage formats.

## Approaches Considered
1. **Taxonomy-first analysis (recommended)**: Enumerate bottlenecks by subsystem
   (LMDB, hashing, materialization, filesystem scaling) and attach minimal
   mitigations + measurements.
2. **Environment-specific deep dive**: Tailor analysis to local SSD vs shared FS
   vs hybrid remote cache; higher precision but less reusable.
3. **Measurement-first only**: Instrument broadly before hypotheses; lower
   speculation but slower time-to-insight.

## Recommended Design (Approach 1)

### 1) LMDB/StateDB contention
**Bottleneck:** single-writer serialization and potential lock wait time.
**Minimal mitigations:** keep write transactions short; batch writes via existing
`DeferredWrites` and coordinator commit; avoid long-lived read transactions.
**Measure:** write-lock wait time, write txn duration, read txn lifespan, StateDB
file growth vs entry count.

### 2) Hashing I/O amplification
**Bottleneck:** full content hashing when metadata diverges or cache misses.
**Minimal mitigations:** maximize metadata cache hit rate; favor generation
tracking for O(1) skips; avoid redundant hashing by batching lookups.
**Measure:** hash cache hit rate, bytes hashed per run, count of metadata mismatches
by cause (`mtime`, `size`, `inode`).

### 3) CAS materialization costs
**Bottleneck:** file-by-file restore in `copy`/`hardlink` modes; symlink mode is
fast but has operational trade-offs.
**Minimal mitigations:** default to hardlink/symlink for standard outputs; keep
`copy` for incremental outputs where correctness requires it.
**Measure:** restore mode distribution, bytes copied vs linked, restore latency per
file and per directory output.

### 4) Filesystem scaling & inode pressure
**Bottleneck:** many small files in CAS and workspace increase inode/dentry load
and metadata I/O.
**Minimal mitigations:** preserve hash-prefix fanout; avoid creating redundant
files when a link is sufficient; document filesystem expectations for large caches.
**Measure:** inode usage (`df -i`), directory entry counts in cache subdirs, `stat`
latency percentiles on cache and workspace trees.

### 5) Worker cold-start latency
**Bottleneck:** process spawn and import overhead.
**Minimal mitigations:** rely on reusable worker pools; avoid frequent teardown in
watch mode where feasible.
**Measure:** time-to-first-task, per-stage dispatch latency, warm vs cold worker
timings.

### 6) Remote cache interactions (if enabled)
**Bottleneck:** network latency and per-object overhead.
**Minimal mitigations:** prefer batch operations; separate fetch from checkout to
pre-warm local cache.
**Measure:** remote hit rate, batch size distributions, end-to-end fetch latency.

## Measurement Plan (Minimal Instrumentation)
- **StateDB:** capture write-lock wait time and txn duration around coordinator
  commits; log read txn lifetimes for hotspots.
- **Hashing:** track bytes hashed and cache hit/miss ratio per run.
- **Restore:** record checkout mode and bytes copied/linked per output.
- **Filesystem:** periodically sample inode usage and directory size for
  `.pivot/cache` and workspace outputs.
- **Workers:** record cold vs warm start latency and per-stage dispatch time.

## Success Criteria
- Identify top 2 bottlenecks by time or I/O volume in representative runs.
- Demonstrate measurable improvement or confirm current design is not the
  limiting factor for the dominant workload.
- Produce stable, low-overhead metrics that can be enabled in CI or benchmarks.

## Open Questions
- Primary deployment environment (local SSD, shared FS, hybrid remote cache)?
- Expected scale of cached files (10^5 vs 10^7) and average file size?
- Filesystem type and mount options for `.pivot/cache`?
