---
title: Minimal Integrity & Security Checks (Store/CAS/Lockfiles)
date: 2026-02-14
status: draft
---

# Minimal Integrity & Security Checks (Store/CAS/Lockfiles)

## Summary
Define low-overhead integrity and security checks for store I/O, CAS, lockfiles, and
presentation-layer materialization, prioritizing hot-path performance while preserving
fail-safe behavior (corruption → treat as missing → re-run/restore).

## Context
- Store protocol: `read`/`write`/`exists` across processes
- CacheStore: content-addressed storage under `.pivot/cache/files/{hash}`
- WorkspaceStore: user-configurable path function
- Engine/workers pass `ArtifactRef`s + `Store` across processes
- Lockfile v2: deps `{producer,key,hash}`, outs `{key,hash}`, plus `merkle_id`
- Presentation layer: materialize CAS via symlink/hardlink
- Current lockfiles: YAML in `.pivot/stages`

Constraints: keep hot paths O(1) when possible; avoid extra syscalls in
`exists/read/write` and hashing loops.

## Threat Model (default assumption)
Single-user local filesystem with accidental corruption and TOCTOU races. If the
threat model is stronger (multi-user or adversarial tampering), we can enable the
optional integrity markers below (lockfile checksum/merkle, stronger hashes) with
minimal incremental overhead.

## Approaches Considered
1. **Boundary validation only (recommended)**
   - Validate external inputs and serialized metadata at read/write boundaries;
     keep hot paths clean. Fail-safe on invalid data.
   - Trade-off: does not detect deliberate tampering when attacker can modify
     both data and metadata.
2. **Boundary validation + cheap invariants**
   - Add constant-time invariants (hash format/length, schema version,
     path traversal rejection, ownership/permission sanity).
   - Trade-off: small extra checks per read; still avoids full re-hash.
3. **Lightweight integrity markers (optional)**
   - Add a lockfile checksum or per-entry checksum to detect edits; verify
     checksum on read.
   - Trade-off: minimal CPU, but adds extra hashing on lockfile read.

## Recommended Design (Approach 1 + selective invariants)

### Store Protocol (`read/write/exists`)
- **Boundary validation** on paths and keys (path traversal, absolute path rejection)
  before invoking file I/O.
- **Fail-safe behavior:** treat invalid/corrupt reads as missing; avoid partial writes.
- **Atomic writes** for lockfiles and manifest-like files.

### CacheStore / CAS (`.pivot/cache/files/{hash}`)
- **Hash format guard** (`len==16`, hex-only) before constructing cache paths.
- **Symlink safety:** resolve symlinks only when verifying a cache target; ensure
  resolved target is inside cache root.
- **Manifest restore pre-check:** verify all referenced hashes exist before any writes.

### WorkspaceStore
- **Path function output validation** once per call boundary (relative, no traversal).
- **Cache directory exclusion**: prevent writing into `.pivot/cache/` unless explicit.

### Cross-Process ArtifactRefs + Store
- **Schema version check** for lockfile v2 (`schema_version`/`merkle_id` fields);
  on mismatch, treat as missing and re-run.

### Lockfiles (YAML in `.pivot/stages`)
- **Structural validation** (`code_manifest`, `params`, `deps`, `outs`) and non-null
  hashes; fail-safe to “missing.”
- **Optional lockfile checksum** (cheap) for tamper detection; failure → re-run.

### Presentation Layer (symlink/hardlink materialization)
- **No chmod on hardlinks** (protect read-only cache invariants).
- **Atomic swap** on restore; lock per-target path to avoid concurrent races.

## Risk Analysis & Minimal Mitigations (Plan Highlights)

### CAS at `.pivot/cache/files/{hash}`
**Risks:**
- **Path traversal / unsafe path construction** if `hash` is not strictly validated.
- **Symlink swap** (cache entry is a symlink to outside cache root).
- **Hash collision abuse** (different content sharing a hash) for non-cryptographic hashes.
- **Concurrent writers / partial writes** leading to corrupted cache entries.
- **TOCTOU** if `exists()` is used before `open()`.

**Minimal mitigations:**
- **Hash format guard** (length + hex-only) before any path construction.
- **Resolve + `is_relative_to(cache_root)`** when following symlink targets; treat
  non-cache targets as misses.
- **Atomic write** (temp path + rename) for any cache write operations.
- **Open-then-handle** (`open()` + handle `FileNotFoundError`) instead of exists→open.
- **Optional stronger hash** (SHA-256) if adversarial tampering is in scope.

**Minimal tests:**
- Hash format rejects invalid length/charset.
- Cache entry symlink to outside cache root treated as miss.
- Corrupted entry (partial write) treated as miss and re-run.
- Concurrent write attempts do not corrupt cache (lock + atomic rename).

### Workspace store (path function) and path safety
**Risks:**
- **Path traversal** (`..`), absolute paths, or writing into `.pivot/cache/`.
- **Symlink escape** (path resolves outside project root at execution time).
- **Permission/ownership** errors causing partial writes.
- **TOCTOU** between path validation and write.

**Minimal mitigations:**
- Validate path function output: **relative only**, no `..`, reject absolute paths.
- **Cache directory exclusion** unless explicitly allowed.
- **Execution-time symlink escape validation** using `resolve()` + `is_relative_to`.
- **Atomic write** (temp + rename) and open-then-handle for file creation.

**Minimal tests:**
- Path function returning `../x` or absolute path is rejected.
- Symlink escape at execution time raises (OUT) / warns (DEP).
- Unwritable path fails cleanly with no partial output.

### Passing stores across processes (StateDB/Store protocol)
**Risks:**
- **Concurrent writers** corrupting shared state or deadlocking.
- **Write starvation** if writer lock is held indefinitely.
- **Stale read assumptions** across worker processes.

**Minimal mitigations:**
- **Readonly workers + deferred writes** to coordinator (single writer).
- **Write-time timeout** with outer flock (fail fast on lock contention).
- **MVCC** semantics via LMDB (readers do not block writers).

**Minimal tests:**
- Concurrent readers + writer do not deadlock; writer timeout triggers.
- Worker cannot write in readonly mode (raises).

### Lockfile v2 and `merkle_id`
**Risks:**
- **Tampering** (edit lockfile to fake hashes).
- **Schema drift** (missing/unknown fields) causing undefined behavior.
- **Partial writes** (crash during write).
- **Hash collision** if merkle uses non-cryptographic hash.

**Minimal mitigations:**
- **Schema/version enforcement** (`schema_version`, required fields, non-null hashes).
- **Atomic write + fsync** for lockfile writes; treat parse failures as missing.
- **Optional checksum/merkle** over lockfile payload for tamper detection;
  if adversarial, use SHA-256 for merkle.

**Minimal tests:**
- Missing `schema_version` or required fields → treated as missing.
- Edited lockfile fails checksum/merkle and re-runs.
- Partial write (truncated YAML) treated as missing.

### Symlink/hardlink tree build (presentation layer)
**Risks:**
- **Symlink race** (swap target during build).
- **Hardlink mutation** corrupting cache if outputs are writable.
- **Path traversal** in manifest/relpaths.
- **Permission issues** and cross-filesystem (EXDEV) failures.

**Minimal mitigations:**
- **Build in temp dir + atomic rename** for tree materialization.
- **Use lstat / follow_symlinks=False** when scanning directories.
- **Never chmod hardlinked files**; keep cache immutable.
- **Fallback chain** (hardlink→symlink→copy) with explicit EXDEV handling.

**Minimal tests:**
- Manifest with `../` path is rejected.
- Symlink swap during restore is prevented (temp symlink + atomic swap).
- Hardlink outputs are read-only; cache permissions preserved.

## Hot-Path Rules
- Prefer O(1) checks: hash format, schema version, path traversal regex.
- Avoid extra `stat()` or `exists()` in tight loops; use cached metadata when present.
- Only hash content when required by correctness (skip detection/restore).

## Cheap vs Expensive Checks (Performance Lens)

### Cheap (O(1), low syscall/CPU)
- **Hash format guards** (length + hex) before cache path construction.
- **Schema/version checks** for lockfiles and protocol structs.
- **Path traversal rejection** (relative-only, no `..`, no absolute, no drive/root).
- **Structural validation** of YAML/TypedDict shape (required keys, type-ish checks).
- **Atomic write pattern** (temp + rename) for lockfiles and manifests.

### Expensive (avoid in hot paths)
- **Re-hashing file contents** on every read (CPU + disk I/O).
- **Extra `exists()`/`stat()` calls** before `open()` (adds syscalls, TOCTOU).
- **Recursive directory walks** for verification outside skip/restore paths.
- **Full CAS integrity scans** (walk + hash) outside on-demand restore.

## Minimal Mitigations (Low-Overhead, Correctness-Preserving)

### exists/read TOCTOU
- **Prefer `open()` + handle `FileNotFoundError`** instead of `exists()` then `open()`.
- When correctness requires a check, **read via the same file descriptor** used for
  verification (no separate pre-check).
- **Fail-safe**: treat unexpected read errors or invalid data as missing and re-run.

### Hash verification
- **Verify content hash only at correctness boundaries** (skip detection, restore),
  not on every `Store.read()`.
- **Cache hash results** in StateDB to avoid repeat hashing within the same run.
- **Optional lockfile checksum** for tamper/corruption detection on read;
  on mismatch, treat lockfile as missing and re-run.

### Lockfile validation
- **Strict schema validation** (required fields, non-null hash strings, version).
- **Path constraints**: lockfile paths must be relative; reject traversal/absolute.
- **Atomic write + fsync** for lockfiles to avoid partial state after crashes.

### CAS/presentation layer
- **Hash format guard** before any CAS path access.
- **Materialize via temp dir then atomic rename** to avoid partial trees.
- **Avoid chmod on hardlinks**; preserve cache immutability.

## Open Questions
- Confirm target threat model: single-user local vs multi-user/adversarial.
