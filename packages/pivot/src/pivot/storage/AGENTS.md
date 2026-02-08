# Pivot Storage - Development Guidelines

LMDB-backed state database for caching and skip detection.

## Skip Detection Algorithm

Three-tier algorithm for deciding whether to skip stage execution:

1. **O(1) generation tracking** - `worker.can_skip_via_generation()` checks if generation counter matches
2. **O(n) lock file comparison** - Compare fingerprint + params + dependency hashes
3. **Run cache lookup** - Check if input hash matches a cached run

## StateDB Key Prefixes

StateDB uses key prefixes for namespacing. Current prefixes in `state_db.py`:

| Prefix | Purpose |
|--------|---------|
| `hash:` | Content hashes |
| `gen:` | Generation counters |
| `dep:` | Dependency tracking |

**When adding new state types**, define a new prefix constant and document it here.

## LMDB Specifics

- Single writer, multiple readers
- Transactions are mandatory—use context managers
- Map size must be set upfront (we use 1GB default)
- Keys and values are bytes—use consistent encoding

## Path Storage

All paths stored in the database must be **relative** to ensure portability across machines and correct cache behavior.

See `docs/solutions/2026-02-01-statedb-path-strategies.md` for path handling patterns.
