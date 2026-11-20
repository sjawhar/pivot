# Known root causes

Fingerprint gaps found by adversarial testing. Put the ID in the log's `root_cause` field.

## Open

### RC-1: pydantic `default_factory` dependencies not followed

`_compute_schema_parts` (called from `_process_model_schema`) hashes a pydantic field's `default_factory` with `hash_function_ast`, which covers only the factory's own source. Nothing walks the helpers, classes or module values the factory names. `StageParams` is a pydantic model, so its fields have this gap. Dataclass factories are walked (`_process_dataclass_defaults`).

- `default_factory=_make_agents`, where `_make_agents` builds `AgentConfig(...)`: an edit to `AgentConfig` is missed unless a field annotation names it (field classes are walked as `fc:` edges).
- `default_factory=lambda: _make_configs(prefix)`: an edit to `_make_configs` is missed.

Severity is highest when the change keeps the values the factory produces, so the lock's `params` stay the same too.

Fix: walk the factory as a stage helper, as `_process_dataclass_defaults` does.

### RC-3: type aliases used only in annotations

With `from __future__ import annotations`, an alias used only in annotations is a string. The function reads no global for it, `get_type_hints` resolves it before `_hint_classes` (which keeps only user classes) sees it, and the function's source does not change. So adding or removing a builtin or library type in the alias is missed. Adding a user-defined type to it is caught. Matrix row: "Type alias definition change" (⚠️).

Fix: hash each parameter's full resolved hint, not only the user classes in it.

### RC-4: module-level variable annotations

Nothing reads a module's `__annotations__`, so a type named only in a module-level variable annotation (`_BASE: RunConfig = {...}`) is never walked. This rarely changes output, because such an annotation is documentation.

Fix: walk `module.__annotations__` for the modules in the source map.

## Fixed (regression targets)

### RC-2: immutable module values

`_immutable_value_text` records exact tuples and frozensets of `bool`, `int`, `float`, `str`, `bytes` and `None`, and compiled regexes (pattern and flags), by value. Lists, dicts, sets, nested collections, enum members and class instances are refused.

Regression: `test_safe_fingerprinting.py::test_immutable_capture_is_tracked_by_value`, `test_determinism.py::test_frozenset_capture_is_deterministic_across_hash_seeds`, `test_manifest_cache_cli.py::test_editing_immutable_module_value_reruns_stage`.

### RC-5: class annotations opaque

`_expand_class` walks a user class's user bases and the user classes its statement's annotations name (`_process_class_body_dependencies`: `list[MyType]`, `A | B`, dotted names). A pydantic model also gets `_process_model_schema`. Check TypedDict-to-TypedDict chains, pydantic field types and generic arguments.

Regression: `test_fingerprint.py::test_a_class_walk_tracks_its_bases_and_annotations`, `::test_a_typed_dict_walks_the_base_its_class_statement_names`.

### RC-6: one `TYPE_CHECKING` import hides every hint

When `get_type_hints` fails, `_resolve_annotations_individually` evaluates each annotation alone, so an unresolvable name skips only itself. Check modules that mix `TYPE_CHECKING` and runtime types.

Regression: `test_fingerprint.py::test_resolve_annotations_individually_mixed`.

### RC-7: library instances in closures ignored

A global or closure value of a library type (a numpy array, a `datetime`) is refused unless unsafe fingerprinting is on (`_process_captured_value`, `_check_non_scalar_capture`). Inside a partial, a default or a loader config, `_form` records it by its state or refuses it.

Regression: `test_safe_fingerprinting.py::test_non_scalar_closure_capture_raises`, `test_fingerprint.py::test_unknown_library_value_inside_a_partial_is_refused`.
