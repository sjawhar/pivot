# Brainstorm: Split Loader into Reader/Writer

**Date:** 2026-01-31
**Issue:** [#237 - Asymmetric Loaders](https://github.com/sjawhar/pivot/issues/237)
**Status:** Ready for planning

## What We're Building

Split the `Loader[T]` base class into separate `Reader[T]` and `Writer[T]` ABCs, allowing asymmetric formats (write Figure → read ndarray).

## Why This Approach

The "Hybrid Loader Refactor" approach was chosen over alternatives because:

1. **Minimal complexity** - No AdapterRegistry, no Format classes, no runtime type extraction
2. **Type-safe** - `isinstance(loader, Reader)` enables type narrowing
3. **Fixes real pain points** - Eliminates phantom types and write-only hacks
4. **Backwards compatible pattern** - `Loader[T](Reader[T], Writer[T])` preserves existing API

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base classes | `Reader[T]`, `Writer[T]`, `Loader[T]` | Clean separation, `Loader` combines both |
| Phantom types | Remove for fixed-type loaders | `Csv(Loader[DataFrame])` is honest |
| Variable-type loaders | Keep generic, document limitation | `Json[T]` for type checking only |
| PEP8 naming | Rename (`CSV` → `Csv`, etc.) | Consistency |
| Validation | At registration time | Catch `Writer` used with `Dep` early |
| AdapterRegistry | Not doing | Adds complexity, explicit loaders are fine |

## Special Cases

### DirectoryOut
- Uses `Writer[dict[str, T]]` - no change needed
- Each value written via the loader's `save()` method

### IncrementalOut
- Requires full `Loader[T]` (needs both read and write)
- Must call `empty()` on first run
- Attribute stays as `loader`, not `reader`/`writer`

### Dep/Out Annotations
- `Dep.loader` → `Dep.reader: Reader[T]`
- `Out.loader` → `Out.writer: Writer[T]`
- `IncrementalOut.loader: Loader[T]` (unchanged)

## Open Questions

None - design is complete and ready for implementation.

## Next Steps

Run `/workflows:plan` to create implementation tasks.
