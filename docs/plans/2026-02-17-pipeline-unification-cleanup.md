# Pipeline Unification & Codebase Cleanup

**Issue:** [#449](https://github.com/sjawhar/pivot/issues/449)

**Goal:** Eliminate the dual-Pipeline-class architecture and remove vestigial
declarative-API code now that the compositional API is the only registration path.

**Approach:** Define a `PipelineLike` Protocol for the engine contract, make
`compose.Pipeline` the sole implementation, delete `pipeline.Pipeline` and all
supporting code.

---

## Context

The compositional API refactor introduced `compose.Pipeline` (user-facing: context
manager, `@stage` decorator, `ArtifactHandle` wiring) alongside the existing
`pipeline.Pipeline` (engine-facing: wraps `StageRegistry`, DAG construction, external
dep resolution). A `build()` bridge converts between them.

This dual-class architecture causes bugs:
- `build()` re-derives metadata by reflection (`get_type_hints`) that compose already knows
- Engine reaches into `pipeline._registry` (private) in 3 places with pyright ignores
- `resolve_external_dependencies()` mutates the registry after `build()`, invalidating
  compose's validation
- `build()` mutates global function objects (`func.__pivot_no_fingerprint__`)
- The identity-as-path assumption in external dep resolution is semantically wrong

Additionally, several modules exist only to support the old declarative API and have
zero production imports.

## Decisions

1. **Direction:** Protocol for engine contract + compose.Pipeline as sole implementation
2. **Dead code:** Delete `matrix.py`, `dvc_import.py`, and their tests
3. **External dep resolution:** Delete entirely (~300 lines). The compose API resolves
   all deps via handles (within pipeline), `include()` (cross-pipeline), and `p.input()`
   (external data). Filesystem traversal to find producers is vestigial.
4. **Unresolved deps:** Become a hard error at DAG build time with a clear message
   suggesting `pipeline.input()` for external data sources.

## Design

### 1. PipelineLike Protocol

Defined in `registry.py` (alongside `RegistryStageInfo`), based on an audit of every
engine, CLI, discovery, and executor call site:

```python
@runtime_checkable
class PipelineLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def root(self) -> pathlib.Path: ...

    @property
    def state_dir(self) -> pathlib.Path: ...

    @property
    def input_bindings(self) -> dict[str, str]: ...

    def list_stages(self) -> list[str]: ...
    def get_stage(self, name: str) -> RegistryStageInfo: ...
    def ensure_fingerprint(self, name: str) -> dict[str, str]: ...

    def build_dag(self) -> DiGraph[str]: ...
    def invalidate_dag_cache(self) -> None: ...

    def snapshot(self) -> dict[str, RegistryStageInfo]: ...
    def restore(self, snapshot: dict[str, RegistryStageInfo]) -> None: ...

    def include(self, other: PipelineLike) -> None: ...
```

This replaces all 3 private `._registry` accesses. `prepare_worker_info` calls
`pipeline.ensure_fingerprint(name)` instead of `pipeline._registry.ensure_fingerprint()`.

Note: `@runtime_checkable` only checks method existence, not signatures. Discovery
should also validate the `pipeline` variable is "materialized" (has stages) — not
just that it has the right attribute names.

### 2. compose.Pipeline Absorbs Engine Capabilities

`compose.Pipeline` gains an internal `StageRegistry` and implements `PipelineLike`:

- `__exit__()` validates then materializes all `_StageNode`s into `RegistryStageInfo`
  entries in the registry. After exit, the Pipeline is ready for the engine.
- `ArtifactHandle` still works during construction via transient `_StageNode` references.
- `include()` deep-copies stages from another pipeline (moved from `pipeline.Pipeline`).
- `snapshot()`/`restore()` delegate to registry (moved from `pipeline.Pipeline`).
- `build_dag()` delegates to `registry.build_dag()` directly (no external dep resolution).

**Lifecycle:** compose.Pipeline has two phases:
1. **Construction** (inside `with` block): `@stage` calls record `_StageNode`s,
   `ArtifactHandle`s wire dependencies, `_inputs` tracks external data.
2. **Materialized** (after `__exit__`): `_StageNode`s converted to `RegistryStageInfo`,
   Protocol methods are usable. Protocol methods called before materialization raise
   `RuntimeError`.

**Non-context-manager usage** (`_discover_all_pipelines` creates `Pipeline("all")`
and calls `include()` without a `with` block): `include()`, `snapshot()`, `restore()`,
and `list_stages()` must work on an empty-but-materialized pipeline. The constructor
starts in materialized state — the context manager is optional (used only when
registering stages via `@stage`). Entering the context manager transitions to
construction phase; exiting re-materializes.

**User migration:** Existing `pipeline.py` files that call `.build()` must remove
that call. Discovery now expects the `pipeline` variable to be the compose.Pipeline
itself (already materialized after the `with` block exits).

### 3. Discovery Changes

Discovery validates against the Protocol instead of `isinstance(Pipeline)`:

```python
def _validate_pipeline(obj: object, path: Path) -> PipelineLike:
    required = ["name", "list_stages", "get_stage", "build_dag", ...]
    missing = [attr for attr in required if not hasattr(obj, attr)]
    if missing:
        raise DiscoveryError(
            f"{path}: object is missing pipeline methods: {missing}"
        )
    return obj
```

User pipeline.py files export compose.Pipeline directly — no `.build()` call.

### 4. Dead Code Removal

| File | Lines | Reason |
|------|-------|--------|
| `pipeline/pipeline.py` | ~446 | Replaced by compose.Pipeline |
| `pipeline/yaml.py` | ~19 | Stub that raises "no longer supported" |
| `pipeline/__init__.py` | ~5 | Package becomes empty |
| `matrix.py` + `test_matrix.py` | ~200 | Zero production imports, YAML matrix expansion |
| `dvc_import.py` + `test_dvc_import.py` | ~400 | Zero production imports, DVC compat layer |
| External dep resolution functions | ~300 | Vestigial in compose API |
| `discovery.find_parent_pipeline_paths()` | ~28 | Only used by tier 1 resolution |
| `discovery.find_pipeline_paths_for_dependency()` | ~38 | Only used by tier 1 resolution |

`PipelineConfigError` relocates to `exceptions.py`.

### 5. Unresolved Dependency Handling

With no external resolution, unresolved deps become a hard error:

```
Error: Stage 'my_pipeline/process' depends on 'raw_data' which has no producer.
Use pipeline.input("raw_data") to declare external data sources.
```

### 6. Bug Fix: Function Object Mutation

`build()` currently does `func.__pivot_no_fingerprint__ = True` on the original
function object — a global side effect that leaks across pipelines. Fix: store the
flag on `RegistryStageInfo` (or a wrapper) instead of mutating the function.

## Phasing

Delete aggressively as each phase completes rather than accumulating deletions.

### Phase 1: Dead Code Removal
- Delete `matrix.py`, `dvc_import.py`, their tests
- Delete `pipeline/yaml.py`, relocate `PipelineConfigError` to `exceptions.py`
- Delete external dep resolution functions from `pipeline/pipeline.py`
- Delete `find_parent_pipeline_paths()` and `find_pipeline_paths_for_dependency()` from `discovery.py`
- Remove engine's `resolve_external_dependencies()` call in reload path
- Remove `_external_deps_resolved` flag and related logic

### Phase 2: Protocol + Engine Migration
- Define `PipelineLike` Protocol
- Add public methods to `pipeline.Pipeline` replacing private `._registry` access
- Switch engine, executor, CLI to use Protocol (not concrete class)
- Switch discovery to validate Protocol instead of `isinstance`
- Delete `pipeline.Pipeline`'s now-unused private access patterns
- All tests pass with `pipeline.Pipeline` still as the implementation

### Phase 3: compose.Pipeline Unification
- Add `StageRegistry` to `compose.Pipeline`
- Move `include()`, `snapshot()`/`restore()`, `build_dag()` to compose.Pipeline
- Materialize `RegistryStageInfo` in `__exit__()` (after all `_StageNode`s recorded)
- Fix `__pivot_no_fingerprint__` function mutation
- Add unresolved-dep error to `build_dag()`
- Delete `compose.Pipeline.build()` bridge method
- Delete `pipeline/pipeline.py` entirely
- Delete `pipeline/` package
- Update discovery to expect compose.Pipeline directly

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Stage naming changes break lockfiles | Widespread reruns | Verify `qualified_name` produces identical names; write comparison test |
| Watch-mode reload regression | Broken hot-reload | snapshot/restore semantics must match; end-to-end reload test |
| `--all` mode regression | Multi-pipeline projects break | Test two separate pipeline.py files with include() |
| Discovery type-check change | User confusion | Clear error messages listing missing methods |
| Function mutation leakage | Fingerprint bugs | Store flag on RegistryStageInfo, not function object |

## Scope Summary

| What | Action | ~Lines |
|------|--------|--------|
| Dead modules + tests | Delete | -800 |
| `pipeline/` package | Delete | -470 |
| External dep resolution | Delete | -300 |
| Discovery traversal helpers | Delete | -66 |
| `compose.Pipeline` | Absorb capabilities | +200 |
| `PipelineLike` Protocol | New | +30 |
| Engine/CLI/discovery updates | Modify | ~70 changed |
| **Net** | | **~1,300 deleted, ~300 added** |
