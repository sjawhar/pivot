# Known Root Causes

Catalog of fingerprinting gap patterns discovered through adversarial testing. Check before planning a new test — if your change exercises a cataloged root cause through a similar code path, skip it unless testing regression of a fix.

## Open Root Causes

### RC-1: `default_factory` Transitive Dependencies Not Followed

`_hash_pydantic_schema` uses `hash_function_ast` for `default_factory` callables instead of `_add_callable_to_manifest`. Names appearing in the factory's AST (classes, helper functions, module-level variables) are not resolved to their definitions.

**Where in code:** `fingerprint.py` `_hash_pydantic_schema()` — line `default_hashes[field_name] = hash_function_ast(default_factory)`.

**Example patterns:**
- Factory function body references a class: `default_factory=_make_agents` where `_make_agents` constructs `AgentConfig(...)` — change to `AgentConfig` class not detected
- Class-body lambda references helper: `default_factory=lambda: _make_configs(prefix)` — change to `_make_configs` body not detected
- Nested Pydantic model as factory: `default_factory=PlotParams` where `PlotParams` has its own fields with factories — inner factories not recursed into

**Severity:** High when combined with output-preserving changes that defeat the params hash backup (e.g., extracting literals into named variables).

**Fix:** Use `_add_callable_to_manifest` for callable `default_factory` values.

---

### RC-2: Primitive Collections in Closures Ignored

Collections (tuple, list, dict, set, frozenset) containing only primitives are routed to `_process_collection_dependency`, which only scans for callables. Primitive data values are silently discarded.

**Where in code:** `fingerprint.py` `_process_closure_values()` matches tuples/lists/etc on the collection branch → `_process_collection_dependency()` iterates elements checking only `callable(value) and is_user_code(value)`.

**Asymmetry:** The module attribute path (`_process_module_dependency`) checks `_is_primitive_collection()` and hashes the serialized value. The closure path has no equivalent — same data detected or ignored depending on import style.

**Example patterns:**
- `_FIGSIZE = (9, 5)` used in `plt.subplots(figsize=_FIGSIZE)` via direct reference
- Color tuples, threshold lists, dimension specs — any `tuple`/`list`/`dict` of numbers/strings used in function bodies via `LOAD_GLOBAL`

**Fix:** Check `_is_primitive_collection()` before routing to `_process_collection_dependency`. Hash serialized value for primitive-only collections.

---

### RC-3: Type Alias Transparency

Type aliases (`types.UnionType`, `typing.Union`, simple assignments) are transparent to `get_type_hints()` — the alias name is replaced with its resolved value. Changes that only add/remove builtin types produce identical manifest entries.

**Where in code:** Three independent failures: (1) Under `from __future__ import annotations`, aliases in annotation strings don't generate `LOAD_GLOBAL` bytecode. (2) `get_type_hints` resolves aliases before `_process_type_hint` sees them. (3) Function source AST is unchanged since annotation strings are unchanged.

**Scope:** Only affects aliases used exclusively in annotations (not in function bodies) in modules with `from __future__ import annotations`, where the change only adds/removes builtin or third-party types (not new user-defined types).

**Fix:** Hash full resolved type hint repr per parameter, not just extracted user-defined types.

---

### RC-4: Module-Level Annotations Blind Spot

No mechanism processes module-level type annotations (`module.__annotations__`). Types referenced ONLY in module-level variable annotations are invisible.

**Where in code:** `_process_type_hint_dependencies` calls `get_type_hints(func)` which only covers function parameter/return annotations. `_process_module_dependency` examines `module.attr` usage in function ASTs but not module annotations.

**Example:** `_WRANGLE_BASE: WrangleLogisticConfig = {...}` — the `WrangleLogisticConfig` TypedDict is never discovered. The dict DATA is tracked via params hash, but the TypedDict class definition is not.

**Scope:** Narrow — only affects types used exclusively as module-level variable annotations, never in function annotations or bodies. Changes to such types rarely affect runtime behavior since the annotation is documentation.

**Fix:** Process `module.__annotations__` for modules in the source map.

## Fixed Root Causes

### RC-5: Class Body Annotations Opaque (FIXED)

**Fixed by:** `_process_class_body_dependencies` (Design Decision #9)

Previously, classes tracked via `_add_callable_to_manifest` were opaque source blobs — field type annotations referencing other user-defined types were NOT followed transitively. `getclosurevars()` fails on classes (no `__code__`), so no transitive dependencies were discovered.

**Now:** When `getclosurevars` raises `TypeError` on a class, `_process_class_body_dependencies` parses the class source AST, extracts names from base classes and field annotations (including subscripts like `list[MyType]`, unions via `|`, and dotted refs), resolves them in the module namespace, and recursively tracks user-code types via `_add_callable_to_manifest`. Pydantic models found this way also get `_hash_pydantic_schema` called.

**Regression test value:** High — this was the most frequently rediscovered bug (8 instances across different class types, nesting depths, and discovery paths). Regression testing should verify TypedDict→TypedDict chains, Pydantic field types, and generic type arguments in class annotations are still tracked.

---

### RC-6: TYPE_CHECKING Import Poisoning (FIXED)

**Fixed by:** `_resolve_annotations_individually` (Design Decision #23)

Previously, `typing.get_type_hints()` was all-or-nothing: if ANY annotation string failed to resolve (because the name was only available under `TYPE_CHECKING`), ALL annotations were skipped.

**Now:** When `get_type_hints()` fails, each annotation string is evaluated individually via `eval()` in the function's global namespace. Resolvable annotations are still tracked; only unresolvable ones are skipped.

**Regression test value:** Medium — the fix is clean but worth verifying in modules with mixed TYPE_CHECKING and runtime-available types.

---

### RC-7: Third-Party Instance Catch-All Missing (FIXED)

**Fixed by:** `_hash_unrecognized_closure_value` with `repr()` (Design Decision #22)

Previously, third-party class instances (numpy arrays, etc.) in function closures fell through all `isinstance` checks in `_process_closure_values` and were silently ignored.

**Now:** The `else` branch at the end of `_process_closure_values` calls `_hash_unrecognized_closure_value`, which hashes via `repr()` if the repr is deterministic (no `0x` memory addresses) and small (< 10KB). Non-deterministic or oversized reprs fall back to `_check_mutable_capture`.

**Limitation:** Large numpy arrays may exceed the 10KB repr limit. Arrays with custom dtypes may have non-deterministic repr. These edge cases fall through to the mutable capture check rather than being hashed.

**Regression test value:** Medium — verify numpy arrays, datetime objects, regex patterns, and enum values are hashed correctly. Test the 10KB boundary.
