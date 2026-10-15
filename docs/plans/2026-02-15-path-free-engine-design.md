# Path-Free Engine Design

**Goal:** Remove file paths as identity throughout the pipeline stack; treat paths as late-bound storage/presentation details resolved by Store.

## Scope

- Finish remaining path-free-engine tasks (presentation layer, CLI/TUI identity display, E2E verification).
- Resolve 4 failing integration tests caused by identity/path mismatches and deferred write gaps.
- No backward compatibility (pre-alpha); lockfiles and RPC are allowed to break.

## Non‑Goals

- No migration code for older lock schemas.
- No redesign of CLI/TUI layout beyond identity display updates.
- No compose API redesign beyond already‑landed changes.

## Approach Options

1) **Identity-first everywhere (recommended)**
   - Public types use structured `ArtifactIdentity` objects; RPC/JSON serializes as objects.
   - Lockfiles store identity objects (no string keys).
   - StateDB uses a single canonical identity key encoding for internal map keys.
   - Pros: eliminates identity/path drift; fixes skip‑detection edge cases.
   - Cons: schema change; requires updates across CLI/TUI/tests.

2) **Hybrid (structured surface, string storage)**
   - Keep lockfiles/StateDB keyed by identity strings.
   - Pros: smaller schema change.
   - Cons: preserves mismatch risk that caused current failures.

3) **Path‑bridged**
   - Keep path‑shaped identities and resolve paths in explain/worker.
   - Pros: fewer changes.
   - Cons: weakens identity semantics; ambiguity between path and logical identity.

**Decision:** Option 1 (Identity-first everywhere).

## Architecture Summary

- **Canonical identity:** `ArtifactIdentity(producer, key)` is the only semantic identity.
- **Serialization:** RPC/JSON/lockfiles encode identity as `{producer, key}` objects.
- **Internal keys:** A single `identity_key()` encoder (e.g., `producer` or `producer:key`) is used only
  for internal dict keys and StateDB prefixes. No `str(identity)` usage.
- **Store as resolver:** `Store` is the sole identity→path resolver. Explain/worker hash paths
  via Store, not by directly interpreting identity strings as paths.
- **Presentation layer:** A pluggable symlink tree materializes CAS outputs into conventional
  workspace paths (data/metrics/plots) after a successful run.

## Components

1) **Types / Skip detection**
   - `DepChange` and `OutputChange` carry `ArtifactIdentity` (not path strings).
   - `skip.diff_dep_hashes()` compares identity‑keyed hashes only.

2) **Lockfile schema v2**
   - `dep_hashes` and `output_hashes` stored as lists of entries with `identity` objects and hash info.
   - Optional `display` field for derived presentation path (read‑ignored).

3) **StateDB keys**
   - Generation and dep generation keys use canonical `identity_key()` encoding.
   - No file paths in key prefixes.

4) **Worker hashing / explain**
   - Hashing uses Store‑resolved paths, ensuring consistent normalization.
   - Explain compares identity‑keyed lock entries and current hash entries computed via Store.

5) **CLI/TUI**
   - Display identity strings derived from `ArtifactIdentity` and (when available) presentation paths.
   - RPC contract returns structured identity objects.

6) **Presentation layer**
   - Builds a symlink tree from CAS into conventional workspace locations.
   - Group artifacts materialize as directories with per‑key files.

## Data Flow (Run)

1) Engine builds graph and schedules stages with `ArtifactRef`.
2) Worker reconstructs `Store` and resolves identities to paths for hashing/IO.
3) Lockfile v2 written with identity entries and Merkle IDs.
4) StateDB updates generation counters using canonical identity keys.
5) Presentation layer materializes symlink tree on successful run.

## Error Handling

- Identity serialization/validation errors are raised at boundaries (CLI/RPC/lock parsing).
- Store resolution errors surface as stage failures; no silent fallback to path heuristics.

## Testing & Verification

- Existing failing integration tests act as TDD red cases; fix by unifying identity encoding and Store‑based hashing.
- Add tests for presentation tree layout and CLI/TUI identity rendering.
- Full verification: pytest (pivot + pivot‑tui), ruff format/check, basedpyright, eval‑pipeline repro.

## Risks & Mitigations

- **Schema churn:** pre‑alpha allows breaking changes; lockfiles regenerated.
- **Identity/key drift:** enforce single `identity_key()` helper and avoid `str(identity)`.
- **Path ambiguity:** store resolves paths; explain/worker never treat identity strings as paths.

## Status

Proceeding under autonomy directive; design is treated as approved.
