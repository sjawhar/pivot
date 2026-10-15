# Fingerprint Tests

This directory contains all tests for Pivot's automatic code change detection (fingerprinting) system.

## Test Files

- **`test_fingerprint.py`** - Unit tests for fingerprinting functions
- **`test_integration.py`** - End-to-end tests with real Python files on disk
- **`test_change_detection.py`** - Comprehensive change detection behavior tests
- **`test_pydantic_defaults.py`** - Tests for Pydantic default data tracking
- **`test_functools.py`** - Tests for `functools.partial` and `functools.wraps` handling
- **`test_callback_vulnerabilities.py`** - Tests documenting callback detection edge cases
- **`test_determinism.py`** - Cross-process fingerprint stability tests (builtins, default_factory)
- **`test_safe_fingerprinting.py`** - Tests for safe fingerprinting error guards (data class methods, dynamic name access)
- **`test_no_fingerprint.py`** - Worker skip/re-run behavior for `@no_fingerprint()` stages (file-level hashing)
- **`test_fingerprint_properties.py`** - Hypothesis properties over generated stage bodies: fresh-process determinism, formatting insensitivity, one-token sensitivity, and `params.yaml` overrides vs. the run-cache key

Loader fingerprint collision tests live in `tests/test_compose.py`.

---

# Change Detection Matrix

This document exhaustively catalogs what code changes are and are not detected by Pivot's fingerprinting system, with test references.

**Legend:**

- ✅ **Detected** - Change triggers cache miss (correct behavior)
- ❌ **Not Detected** - Change does NOT trigger cache miss (limitation)
- 🚫 **Intentionally Ignored** - By design, change should NOT trigger cache miss
- ⚠️ **Partial** - Some cases detected, others not

---

## 1. Stage Function Itself

| Change Type                     | Detected? | Test Reference                                                        |
| ------------------------------- | --------- | --------------------------------------------------------------------- |
| Function body logic change      | ✅        | `test_change_detection.py::test_function_body_change_causes_miss`     |
| Function argument added/removed | ✅        | `test_change_detection.py::test_function_argument_change_causes_miss` |
| Default argument value change   | ✅        | `test_change_detection.py::test_default_value_change_causes_miss`     |
| Local variable rename           | ✅        | `test_change_detection.py::test_variable_rename_causes_miss`          |
| Parameter name change           | ✅        | `test_fingerprint.py::test_different_variable_names_different_hash`   |
| Type annotation change          | ✅        | `test_change_detection.py::test_type_annotation_change_causes_miss`   |
| Type alias definition change    | ⚠️        | Without `from __future__ import annotations`: alias is in globals, hashed by closure catch-all. With `from __future__`: alias only in string annotation, not captured. |
| Function name change            | 🚫        | `test_change_detection.py::test_function_rename_no_miss`              |
| Docstring change                | 🚫        | `test_change_detection.py::test_docstring_change_no_miss`             |
| Comment change                  | 🚫        | `test_change_detection.py::test_comment_change_no_miss`               |
| Whitespace change               | 🚫        | `test_change_detection.py::test_whitespace_change_no_miss`            |
| Decorator on stage function     | ❌        | **NO TEST** - decorators on the stage itself not tracked              |
| Async vs sync change            | ✅        | `test_change_detection.py::test_async_sync_change_causes_miss`        |

---

## 2. Same-Module Dependencies

| Change Type                                          | Detected? | Test Reference                                                                                                                      |
| ---------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Helper function body change                          | ✅        | `test_fingerprint.py::test_helper_function_captured`                                                                                |
| Helper function via alias (`f = helper; f()`)        | ✅        | `test_fingerprint.py::test_aliased_function_captured`                                                                               |
| Transitive helper (helper's helper)                  | ✅        | `test_fingerprint.py::test_transitive_dependencies_captured`                                                                        |
| Global primitive constant                            | ✅        | `test_fingerprint.py::test_constant_captured`, `test_fingerprint.py::test_multiple_constants_captured`                              |
| Global primitive constant change                     | ✅        | `test_change_detection.py::test_global_constant_change_causes_miss`                                                                 |
| Global tuple, frozenset of scalars, or compiled regex | ✅ | `test_safe_fingerprinting.py::test_immutable_capture_is_tracked_by_value`, `test_safe_fingerprinting.py::test_editing_immutable_capture_changes_fingerprint`, `test_manifest_cache_cli.py::test_editing_immutable_module_value_reruns_stage` |
| Global list, dict, set, nested tuple, enum, or class instance | ✅ (error) | `test_safe_fingerprinting.py::test_non_scalar_closure_capture_raises` |
| Non-scalar module attribute                          | ✅ (error) | `test_integration.py::test_non_scalar_module_attribute_raises`                                                                      |
| Method bound to an instance (`Scaler(3).apply`, `default_rng(0).normal`) as a global, cell, partial argument, loader field or `module.NAME` | ✅ (error) | `test_safe_fingerprinting.py::test_non_scalar_closure_capture_raises`, `test_integration.py::test_non_scalar_module_attribute_raises`, `test_manifest_cache_cli.py::test_method_bound_to_an_instance_is_an_error` |
| Method bound to a Logger, a scalar, a class or a library module's own state (`np.random.normal`) | 🚫 | `test_safe_fingerprinting.py::test_methods_bound_to_a_logger_scalar_or_library_are_allowed`, `test_manifest_cache_cli.py::test_method_bound_to_a_scalar_class_or_library_is_allowed` |
| Pydantic schema hash (model_json_schema)             | ✅        | `test_pydantic_defaults.py::test_pydantic_default_data_captured`                                                                    |
| Pydantic class in type hint                          | ✅        | `test_pydantic_defaults.py::test_pydantic_class_captured_from_type_hint`                                                           |
| Pydantic class read as a global (`Out().scale`), import-time field default | ✅ | `test_manifest_cache_cli.py::test_value_read_through_a_class_at_import_reruns_stage` |
| Class attribute read as `Config.SCALE` (scalar, Enum member, function) | ✅ | `test_fingerprint.py::test_class_attributes_a_stage_reads_are_tracked_one_by_one`, `test_manifest_cache_cli.py::test_value_read_through_a_class_at_import_reruns_stage` |
| Non-scalar class attribute read as `Config.COLUMNS` | ✅ (error) | `test_manifest_cache_cli.py::test_non_scalar_class_attribute_is_an_error` |
| Class attribute the language or a library defines (`NamedTuple._fields`, `Enum._member_names_`, an inherited stdlib dict, a property object) | 🚫 (class AST only) | `test_manifest_cache_cli.py::test_class_attribute_the_language_or_a_library_defines_is_not_refused` |
| `settings.Config.SCALE` (class reached through a user module) | ✅ | `test_manifest_cache_cli.py::test_class_attribute_read_through_a_module_at_import_reruns_stage`, `test_manifest_cache_cli.py::test_class_attribute_through_a_module_follows_the_class_rule`, `test_manifest_cache_cli.py::test_non_scalar_class_attribute_through_a_module_is_an_error` |
| `settings.log`, `settings.ROUND` (a partial), `settings.AGG` (`np.mean`) | ✅ (same rule as the global form) | `test_manifest_cache_cli.py::test_logger_partial_and_library_function_read_through_a_module_are_accepted`, `test_manifest_cache_cli.py::test_library_function_rebound_in_a_module_reruns_stage` |
| `pkg.sub.helper`, `pkg.sub.SCALE` (submodule through its package) | ✅ | `test_manifest_cache_cli.py::test_submodule_read_through_its_package_is_tracked` |
| `cls.SCALE` in a classmethod, `Outer.Inner.X`, `Config().SCALE`, a `StageParams` `ClassVar` | ❌ | Not tracked; documented in `docs/architecture/fingerprinting.md` (use `StageParams`) |
| Class definition change                              | ✅        | `test_change_detection.py::test_class_definition_change_causes_miss`                                                               |
| Class method change                                  | ✅        | `test_change_detection.py::test_class_definition_change_causes_miss` (class AST includes methods)                                  |
| Data class with user-defined methods                 | ✅ (error) | `test_fingerprint.py::test_check_data_class_methods_rejects_user_methods`                                                         |
| Class base/field annotation dependencies             | ✅        | `test_fingerprint.py::test_process_class_body_dependencies_tracks_bases_and_annotations`                                           |
| StageParams `@property` method change                | ✅        | `test_change_detection.py::test_stageparams_property_change_causes_miss`                                                           |
| StageParams regular method change                    | ✅        | `test_change_detection.py::test_stageparams_method_change_causes_miss`                                                             |
| StageParams `ClassVar` change                        | ✅        | `test_change_detection.py::test_stageparams_class_variable_change_causes_miss`                                                     |
| Nested function body change                          | ✅        | `test_fingerprint.py::test_nested_function_not_in_globals` (detected via parent AST hash)                                          |
| Globals referenced only by nested functions          | ✅        | `test_change_detection.py::test_nested_function_global_reference_detected`, `test_change_detection.py::test_nested_function_global_change_causes_miss` |
| Helper starting with `_` prefix                      | ✅        | `test_change_detection.py::test_underscore_helper_change_detected`                                                                 |
| Stdlib callable via its module (e.g., `typing.cast`) | 🚫        | `test_fingerprint.py::test_stdlib_callable_in_closure_does_not_raise`                                                              |
| Name rebound to another library function (`agg = np.mean` → `np.median`, from-import, partial) | ✅ | `test_manifest_cache_cli.py::test_rebinding_a_library_function_reruns_stage` |
| Library function chosen at import (`getattr(np, os.environ[...])`) | ✅ | `test_manifest_cache_cli.py::test_library_function_chosen_at_import_reruns_stage` |
| Library function in loader config stays cached | 🚫 | `test_manifest_cache_cli.py::test_library_function_in_loader_config_stays_cached` |
| `logging.Logger` in closure                          | 🚫        | Loggers are functionally irrelevant for cache invalidation                                                                          |
| Helper starting with `__` dunder                     | ❌        | `test_fingerprint.py::test_fingerprint_with_underscore_globals`                                                                     |

---

## 3. Cross-Module Dependencies (Module-Level Imports)

### 3.1 Direct Import (`from X import func`)

| Change Type                           | Detected? | Test Reference                                                               |
| ------------------------------------- | --------- | ---------------------------------------------------------------------------- |
| Imported function body change         | ✅        | `test_integration.py::test_direct_import_change_detected`                    |
| Imported function change (end-to-end) | ✅        | `test_change_detection.py::test_helper_via_direct_import_change_causes_miss` |
| Transitive dependency change          | ✅        | `test_integration.py::test_transitive_change_detected`                       |
| Imported constant                     | ✅        | `test_fingerprint.py::test_constant_captured` (same mechanism)               |
| Imported class                        | ✅        | `test_change_detection.py::test_class_definition_tracked_with_class_prefix`  |

### 3.2 Module Attribute (`import X; X.func()`)

| Change Type                         | Detected? | Test Reference                                                                          |
| ----------------------------------- | --------- | --------------------------------------------------------------------------------------- |
| Module function body change         | ✅        | `test_integration.py::test_module_attr_change_detected`                                 |
| Module function change (end-to-end) | ✅        | `test_change_detection.py::test_helper_via_module_attr_change_causes_miss`              |
| Module immutable tuple, frozenset, or compiled regex | ✅ | `test_manifest_cache_cli.py::test_editing_immutable_module_value_reruns_stage` |
| Module constant                     | ✅        | `test_integration.py::test_constant_via_module_attr_captured`                           |
| Module constant captured            | ✅        | `test_change_detection.py::test_module_constant_captured_via_module_attr`               |
| Nested attribute (`X.sub.func`)     | ⚠️        | `test_fingerprint.py::test_extract_nested_attr_access` (extracted but not fully tested) |
| Multiple attrs from same module     | ✅ | `test_fingerprint.py::test_extract_multiple_attrs_same_module` |
| Both import styles in same stage    | ✅        | `test_integration.py::test_both_import_styles_in_same_stage`                            |

---

## 4. Cross-Module Dependencies (Lazy Imports Inside Function Body)

| Change Type                          | Detected? | Test Reference                                                       |
| ------------------------------------ | --------- | -------------------------------------------------------------------- |
| `from X import func` inside function | ❌        | `test_change_detection.py::test_lazy_import_change_detected` (xfail) |
| `import X` inside function           | ❌        | **NO TEST** - same limitation                                        |
| Conditional import inside function   | ❌        | **NO TEST** - same limitation                                        |

---

## 5. Special Naming Patterns

| Change Type                  | Detected? | Test Reference                                                     |
| ---------------------------- | --------- | ------------------------------------------------------------------ |
| Names starting with `_`      | ✅        | `test_change_detection.py::test_underscore_helper_change_detected` |
| Names starting with `__`     | ❌        | `test_fingerprint.py::test_fingerprint_skips_underscore_globals`   |
| `__name__`, `__file__`, etc. | 🚫        | `test_fingerprint.py::test_fingerprint_with_underscore_globals`    |

---

## 6. Stdlib and Third-Party

| Change Type                                | Detected? | Test Reference                                                            |
| ------------------------------------------ | --------- | ------------------------------------------------------------------------- |
| Stdlib function (e.g., `json.dumps`)       | 🚫        | `test_integration.py::test_stdlib_module_attrs_not_tracked`               |
| Stdlib via module attr                     | 🚫        | `test_change_detection.py::test_stdlib_module_attrs_not_tracked`          |
| Third-party function (e.g., `numpy.array`) | 🚫        | `test_fingerprint.py::test_is_user_code_non_user` (pytest.fixture tested) |
| Third-party package version change         | ❌        | **NO TEST** - versions not tracked                                        |
| Pivot framework code (`StageParams`, etc.) | 🚫        | `test_fingerprint.py::test_is_user_code_pivot_is_framework`               |
| Builtin function (`len`, `print`)          | 🚫        | `test_fingerprint.py::test_fingerprint_builtin_function_skipped`          |

---

## 7. Dynamic/Runtime Patterns

| Change Type                              | Detected? | Test Reference                                                                                                                                |
| ---------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `getattr(module, "func")`                | ❌        | **NO TEST** - can't resolve statically                                                                                                        |
| `getattr(obj, name)` (non-literal)       | ✅ (error) | `test_fingerprint.py::test_check_dynamic_name_access_rejects_dynamic_getattr`                                                        |
| `globals()` / `locals()`                 | ✅ (error) | `test_fingerprint.py::test_check_dynamic_name_access_rejects_globals`                                                                  |
| `func_dict[key]()` dispatch              | ✅        | `test_change_detection.py::test_dynamic_dispatch_change_detected`, `test_change_detection.py::test_dispatch_dict_function_change_causes_miss` |
| `eval()` / `exec()`                      | ❌        | **NO TEST** - can't analyze dynamically                                                                                                       |
| `importlib.import_module(string)`        | ❌        | **NO TEST** - can't resolve statically                                                                                                        |
| `importlib.import_module(...)` (call)    | ✅ (error) | `test_fingerprint.py::test_check_dynamic_name_access_rejects_import_module`                                                           |
| Method call on instance (`obj.method()`) | ❌        | `test_change_detection.py::test_class_instance_method_change_detected` (xfail)                                                                |
| Callback/function passed as argument     | ❌        | **NO TEST** - not tracked                                                                                                                     |
| Closure capturing outer variable         | ⚠️        | `test_fingerprint.py::test_fingerprint_with_nonlocal` (value captured, not source)                                                            |

---

## 8. Closures and Nonlocals

| Change Type                       | Detected? | Test Reference                                                     |
| --------------------------------- | --------- | ------------------------------------------------------------------ |
| Nonlocal primitive constant       | ✅        | `test_fingerprint.py::test_fingerprint_with_nonlocal`              |
| Nonlocal callable (user function) | ✅        | `test_fingerprint.py::test_fingerprint_nonlocal_callable_function` |
| Closure variable value change     | ✅        | `test_fingerprint.py::test_fingerprint_callable_nonlocal`          |
| Closure value with deterministic repr | ✅        | `test_fingerprint.py::test_fingerprint_captures_unrecognized_closure_value` |
| Closure value with `0x` in repr (non-deterministic) | 🚫  | `test_fingerprint.py::test_hash_unrecognized_closure_value_memory_address_raises` |
| Closure value with repr > 10KB        | 🚫        | `test_fingerprint.py::test_hash_unrecognized_closure_value_large_repr_raises` |
| Two closures from one factory (`h2 = make(2)`, `h5 = make(5)`) | ✅ | `test_manifest_cache_cli.py::test_two_closures_from_one_factory_keep_their_values_in_every_process` |
| Secret in a helper default or loader field changes | ✅ | `test_manifest_cache_cli.py::test_changing_a_secret_read_by_a_stage_reruns_it`, `test_functools.py::test_secret_partial_argument_hashes_by_keyed_digest` |
| Factory-built stage under a same-named pipeline, run separately | ✅ | `test_manifest_cache_cli.py::test_same_named_pipelines_run_separately_keep_factory_closures_apart` |
| `include()` of two factory-built stages under one name | ✅ (error) | `test_manifest_cache_cli.py::test_include_refuses_two_factory_closures_under_one_stage_name` |
| `include()` of one stage read through two factory-built loaders | ✅ (error) | `test_manifest_cache_cli.py::test_include_refuses_one_stage_reading_through_two_factory_loaders` |

---

## 9. Edge Cases

| Change Type                               | Detected? | Test Reference                                                                  |
| ----------------------------------------- | --------- | ------------------------------------------------------------------------------- |
| Lambda function                           | ⚠️        | `test_fingerprint.py::test_lambda_function_fingerprinted` (hashed but unstable) |
| Function without source (builtin)         | ⚠️        | `test_fingerprint.py::test_hash_builtin_function` (fallback hash)               |
| Function without `__code__`               | ⚠️        | `test_fingerprint.py::test_hash_function_no_code_object` (id-based hash)        |
| Circular references                       | ✅        | `test_fingerprint.py::test_circular_reference_handled`                          |
| Recursive function                        | ✅        | Covered by circular reference handling                                          |
| `getclosurevars` raises exception         | ✅        | `test_fingerprint.py::test_fingerprint_getclosurevars_exception`                |
| Mixed resolvable/unresolvable type hints  | ✅        | `test_fingerprint.py::test_resolve_annotations_individually_mixed`              |
| Recursive Pydantic models (self-referential) | ✅     | `test_fingerprint.py::test_recursive_pydantic_model_no_infinite_recursion`      |
| Mutually recursive Pydantic models        | ✅        | `test_fingerprint.py::test_mutual_recursive_pydantic_models`                    |
| User-defined generic origin (`MyGeneric[int]`) | ✅   | `test_fingerprint.py::test_user_defined_generic_origin_tracked`                 |
| Dotted string forward ref in class body   | ✅        | `test_fingerprint.py::test_collect_annotation_names_dotted_string_forward_ref`  |
| Module not in sys.modules                 | ✅        | `test_fingerprint.py::test_is_user_code_module_not_in_sys_modules`              |
| Attribute doesn't exist on module         | ✅        | **NO TEST** - recorded as `"unknown"` in the manifest                           |
| Function with complex AST                 | ✅        | `test_fingerprint.py::test_hash_complex_ast`                                    |
| AST syntax error fallback                 | ✅        | `test_fingerprint.py::test_hash_function_syntax_error_fallback`                 |
| Child manifest merge (no duplicate self:) | ✅        | `test_fingerprint.py::test_fingerprint_merges_child_manifest`                   |

---

## 10. Hash Stability

| Property                                | Verified? | Test Reference                                                  |
| --------------------------------------- | --------- | --------------------------------------------------------------- |
| Same function → same hash               | ✅        | `test_fingerprint.py::test_unchanged_function_same_fingerprint` |
| Hash is deterministic                   | ✅        | `test_fingerprint.py::test_hash_is_stable`                      |
| Hash format (16-char hex)               | ✅        | `test_fingerprint.py::test_hash_format`                         |
| Multiple runs stable                    | ✅        | `test_change_detection.py::test_multiple_runs_stable`           |
| Same logic, different names → same hash | ✅        | `test_fingerprint.py::test_identical_functions_same_hash`       |
| Dict key order doesn't affect hash      | ✅        | `test_change_detection.py::test_fingerprint_ordering_stability` |
| Same source → same manifest in fresh processes (any hash seed) | ✅ | `test_fingerprint_properties.py::test_same_source_gives_same_manifest_in_fresh_processes` |
| Formatting that keeps the syntax tree → same manifest | ✅ | `test_fingerprint_properties.py::test_formatting_that_keeps_the_syntax_tree_keeps_the_manifest` |
| Any one-token body change → changed manifest | ✅ | `test_fingerprint_properties.py::test_changing_one_token_of_the_body_changes_the_manifest` |
| Param override changes run-cache key iff effective params change | ✅ | `test_fingerprint_properties.py::test_override_changes_run_cache_key_exactly_when_effective_params_change` |
| Override for one stage leaves other stages' keys | ✅ | `test_fingerprint_properties.py::test_override_for_one_stage_leaves_other_stages_run_cache_keys` |

---

## 11. Loader Fingerprinting

| Change Type                     | Detected? | Test Reference                                                              |
| ------------------------------- | --------- | --------------------------------------------------------------------------- |
| Loader load() method change     | ✅        | `test_fingerprint.py::test_loader_fingerprint_includes_load_method`         |
| Loader save() method change     | ✅        | `test_fingerprint.py::test_loader_fingerprint_includes_save_method`         |
| Loader config field change      | ✅        | `test_fingerprint.py::test_loader_config_change_changes_fingerprint`        |
| Different loader types          | ✅        | `test_fingerprint.py::test_different_loader_types_different_fingerprint`    |
| Custom loader code change       | ✅        | `test_fingerprint.py::test_custom_loader_code_change_detected`              |
| Custom loader fingerprinting    | ✅        | `test_fingerprint.py::test_custom_loader_fingerprint`                       |
| Fingerprint stability           | ✅        | `test_fingerprint.py::test_loader_fingerprint_stable`                       |
| Config collision between outputs| ✅        | `test_compose.py::test_output_loader_change_detected[same-class-different-config]`     |
| YAML→CSV with existing CSV out  | ✅        | `test_compose.py::test_output_loader_change_detected[yaml-to-csv-with-existing-csv]`   |
| @no_fingerprint loader collision| ✅        | `test_compose.py::test_no_fingerprint_output_loader_collision`                          |
| Dep config collision            | ✅        | `test_compose.py::test_dep_loader_config_collision`                                     |
| Scalar read in custom loader's own `save()` | ✅ | `test_manifest_cache_cli.py::test_editing_scalar_read_by_custom_loader_method_reruns_stage` |
| Non-scalar read in custom loader's own `save()` | ✅ (error) | `test_manifest_cache_cli.py::test_non_scalar_read_by_custom_loader_method_is_an_error` |
| Scalar read by a function in loader config | ✅ | `test_manifest_cache_cli.py::test_value_read_by_loader_config_function_reruns_stage` |
| Pydantic model class in loader config edited | ✅ | `test_manifest_cache_cli.py::test_editing_model_class_used_by_loader_reruns_stage` |

---

## 12. Callback/Function Argument Detection

Tests in `test_callback_vulnerabilities.py` document edge cases where callback changes may not be detected.

| Change Type                                        | Detected? | Test Reference                                                                          |
| -------------------------------------------------- | --------- | --------------------------------------------------------------------------------------- |
| Module-level callback variable change              | ✅        | `test_callback_vulnerabilities.py::test_module_variable_callback_change_detected`       |
| Dict callback in closure                           | ✅        | `test_callback_vulnerabilities.py::test_dict_callback_in_closure_detected`              |
| Multi-layer wrapped callbacks via closure          | ✅        | `test_callback_vulnerabilities.py::test_multilayer_closure_callbacks_detected`          |
| Async/generator callback change                    | ✅        | `test_callback_vulnerabilities.py::test_async_callback_change_detected`                 |
| `@functools.wraps` decorator logic change          | ✅        | `test_functools.py::test_decorator_change_triggers_fingerprint_change`                  |
| Manual `__wrapped__` attribute detection           | ✅        | `test_callback_vulnerabilities.py::test_manual_wrapped_attribute_hides_code`            |
| `functools.partial` arguments tracked              | ✅        | `test_functools.py::test_partial_args_change_triggers_fingerprint_change`               |
| `functools.partial` wrapped function tracked       | ✅        | `test_functools.py::test_partial_underlying_func_change_triggers_fingerprint_change`    |
| Instance/container state callbacks not tracked     | ❌        | `test_callback_vulnerabilities.py::test_container_callback_change_detected`             |
| Class attribute callbacks (runtime modified)       | ✅        | `test_callback_vulnerabilities.py::test_class_attribute_callback_change_detected`       |

---

## 13. `@no_fingerprint` Decorator (File-Level Hashing)

When a stage uses `@pivot.no_fingerprint()`, AST fingerprinting is bypassed entirely. Change detection uses whole-file hashing instead.

| Change Type                                    | Detected? | Test Reference                                                                  |
| ---------------------------------------------- | --------- | ------------------------------------------------------------------------------- |
| Stage source file content change               | ✅        | `test_no_fingerprint.py::test_no_fingerprint_reruns_on_source_file_change`      |
| `code_deps` file content change                | ✅        | `test_no_fingerprint.py::test_no_fingerprint_reruns_on_code_deps_change`        |
| `code_deps` via `Pipeline`, `@no_fingerprint` inside `@stage` | ✅ | `test_no_fingerprint.py::test_no_fingerprint_code_deps_tracked_through_pipeline[no_fingerprint-inside-stage]` |
| `code_deps` via `Pipeline`, `@no_fingerprint` outside `@stage` | ❌ | `test_no_fingerprint.py::test_no_fingerprint_code_deps_tracked_through_pipeline[no_fingerprint-outside-stage]` (strict xfail, red test) |
| Unrelated file change (no re-run)              | ✅        | `test_no_fingerprint.py::test_no_fingerprint_skip_unrelated_change`             |
| Comment/whitespace change in source file       | ✅        | Not separately tested — file hash changes on any content change                 |
| Helper function in different file              | ❌        | Not detected unless listed in `code_deps`                                       |
| Missing `code_deps` file raises error          | ✅        | `test_no_fingerprint.py::test_no_fingerprint_missing_code_deps_file`            |
| Mixed pipeline (AST + no_fingerprint)          | ✅        | `test_no_fingerprint.py::test_no_fingerprint_mixed_pipeline`                    |
| Skip on unchanged (second run)                 | ✅        | `test_no_fingerprint.py::test_no_fingerprint_stage_skips_when_unchanged`        |

---

## Summary: Remaining Gaps

| Gap                                              | Priority | Difficulty                    |
| ------------------------------------------------ | -------- | ----------------------------- |
| Decorator on stage function not tracked          | Medium   | Easy                          |
| Nested module attribute (`X.sub.func`) full test | Low      | Easy                          |
| `import X` inside function not tracked           | Low      | Easy (already documented)     |
| `eval()`/`exec()` limitation                     | Low      | Easy                          |
| Third-party package version tracking             | Medium   | Hard (would need new feature) |
| Type alias change with `from __future__`         | Low      | Hard (alias only in string annotation, not in bytecode) |

---

## Design Decisions

1. **Dunder name filtering**: Names starting with `__` are skipped to filter `__name__`, `__file__`, `__doc__`, etc. Single-underscore names like `_private_helper()` are tracked normally.

2. **Stdlib/third-party marked "callable"**: These are not hashed because we don't want package version changes to trigger rebuilds (too sensitive).

3. **Function name normalization**: Function names are normalized to `"func"` in AST to allow renaming without cache miss.

4. **Docstrings/comments ignored**: These don't affect behavior, so they're stripped from AST.

5. **Lazy imports not tracked**: Imports inside function bodies are not detected. Recommended pattern: use module-level imports.

6. **Non-scalar module values are refused**: A stage may capture module-level `bool`, `int`,
   `float`, `str`, `bytes`, and `None` values, a `tuple` or `frozenset` whose members are all
   such scalars (one level deep), and a compiled regex (recorded by full pattern and flags).
   Any other module-level value—including a list, dict, set, a tuple holding a collection or
   object, enum member, frozen model, or class instance—raises `StageDefinitionError`.

7. **Portable bytecode fallback**: When source is unavailable or a function is wrapped,
   `marshal.dumps(__code__)` captures bytecode and constants. Pivot replaces every
   `co_filename` (including nested code objects) before marshalling, so the result is stable
   across checkout paths.

8. **Module-qualified manifest keys**: `func:`, `class:`, `const:`, `schema:`, `partial:`,
   `ext:` and `loader:` keys use the definition or binding module. A value read as
   `settings.NAME` gets the key it would get as a global (`const:settings.FACTOR`,
   `func:settings.helper`, `ext:settings.AGG`). A manifest can therefore
   contain `func:project.alpha.helper` and `func:project.beta.helper` without one overwriting
   the other. A closure cell is keyed under the function that holds it: the stage function's
   own cells under its definition (`const:project.make_stage.<locals>.make.factor`), a local
   helper's cells under the binding it was reached through (`const:project.h2.factor`, with the
   helper at `func:project.h2`), so two closures from one factory keep their own values.

9. **Class body dependency tracking**: When `getclosurevars()` fails on a class, the class body
   is parsed to capture base classes and field annotations. Resolved user-code classes are
   fingerprinted transitively; third-party/stdlib bases are ignored.

10. **Pydantic schemas ARE tracked**: Pydantic model classes used in type hints are automatically
    detected and their JSON schemas are hashed via `model_json_schema()` (with titles/descriptions
    stripped to avoid false invalidation from docstring changes). The fingerprint includes a
    module-qualified class-AST key and schema key, e.g.
    `class:project.params.Model` and `schema:project.params.Model`.

    Nested model types are discovered by walking `model_fields` annotations, so changes to nested
    models propagate. A placeholder is inserted before field walking to prevent infinite recursion
    on self-referential models.

    Tests: `test_pydantic_defaults.py::test_pydantic_default_data_captured`,
    `test_pydantic_defaults.py::test_pydantic_class_captured_from_type_hint`,
    `test_pydantic_defaults.py::test_pydantic_default_change_triggers_different_hash`,
    `test_fingerprint.py::test_recursive_pydantic_model_no_infinite_recursion`

11. **`functools.wraps` / `__wrapped__` ARE tracked**: Functions decorated with
    `@functools.wraps` are fingerprinted using portable bytecode hashing. Since
    `inspect.getsource()` follows the `__wrapped__` chain and returns the original function's
    source, Pivot hashes the wrapper's bytecode and still tracks the wrapped function through its
    closure.

12. **`functools.partial` IS tracked**: Partial objects include module-qualified bound-argument
    keys and the underlying user function's definition key. A user function or class bound as
    an argument, directly, inside a nested partial, or inside a list, tuple, set or dict, gets
    the stage-helper walk: its globals, closure values and defaults are tracked, and a
    non-scalar it reads is refused. Loader config fields get the same walk.

    Tests: `test_functools.py::test_function_in_partial_argument_collection_is_walked`,
    `test_manifest_cache_cli.py::test_value_read_by_partial_argument_function_reruns_stage`,
    `test_manifest_cache_cli.py::test_value_read_by_loader_config_function_reruns_stage`,
    `test_manifest_cache_cli.py::test_non_scalar_read_by_bound_function_is_an_error`

13. **StageParams classes fully tracked**: `StageParams` subclasses used in type hints are tracked
    via `_process_type_hint_dependencies()`. The entire class definition is hashed, including
    properties, regular methods, and `ClassVar` declarations.

    Tests: `test_change_detection.py::test_stageparams_property_change_causes_miss`,
    `test_change_detection.py::test_stageparams_method_change_causes_miss`,
    `test_change_detection.py::test_stageparams_class_variable_change_causes_miss`

14. **Builtin types ARE deterministic**: Builtin types (`list`, `dict`, `set`, `tuple`, etc.)
    used as `default_factory` in Pydantic fields are hashed using their qualified name (for
    example, `builtin:list`) rather than `id()`.

    Tests: `test_determinism.py::test_builtin_default_factory_deterministic_across_processes`,
    `test_determinism.py::test_builtin_type_deterministic`

    Set and frozenset values are ordered by their canonical form, not by iteration order,
    which follows `PYTHONHASHSEED`. This covers `StageParams` schema defaults, `default:`
    values, loader config, `functools.partial` arguments, and the params recorded in the
    lock file.

    Tests: `test_determinism.py::test_set_valued_params_are_deterministic_across_hash_seeds`,
    `test_manifest_cache_cli.py::test_set_valued_params_stay_cached_across_processes`

15. **Mutable and nested module collections are refused**: A stage reading a list, dict,
    set, or a tuple or frozenset containing a non-scalar raises `StageDefinitionError` unless
    unsafe fingerprinting is explicitly enabled, in which case Pivot warns instead of
    pretending to track its contents. Exact tuples and frozensets of scalars and compiled
    regexes are recorded by canonical value.

    A method bound to an instance is refused the same way (`apply = Scaler(3).apply`,
    `normal = np.random.default_rng(SEED).normal`): the instance it carries is state no code
    hash records. A method bound to a module, a class, a scalar or a `logging.Logger`, or a
    library module's own export such as `np.random.normal`, is recorded by name instead.

    Tests: `test_integration.py::test_non_scalar_module_attribute_raises`,
    `test_safe_fingerprinting.py::test_immutable_capture_is_tracked_by_value`,
    `test_safe_fingerprinting.py::test_non_scalar_closure_capture_raises`

16. **Manifest cache invalidation is path-scoped**: Watch-mode reloads invalidate only the cached manifests whose source maps include changed paths, leaving unaffected stage manifests intact. Affected stages are recomputed on next fingerprint access and re-cached.

    Tests: `test_fingerprint.py::test_invalidate_manifests_for_paths_selective`, `test_fingerprint.py::test_invalidate_manifests_for_paths_recomputes_affected_stage`, `test_fingerprint.py::test_invalidate_manifests_for_paths_cold_start`

17. **Pivot framework excluded from `is_user_code`**: Modules in the `pivot` package are treated as framework code, not user code. This prevents framework internals (e.g., `StageParams`, `stage`, `Pipeline`) from appearing in manifests when using editable installs where pivot isn't in `site-packages`.

    Tests: `test_fingerprint.py::test_is_user_code_pivot_is_framework`

18. **Non-scalar closure values are refused**: Values outside the permitted scalar,
    one-level immutable collection, and compiled-regex set raise `StageDefinitionError`.

19. **Per-annotation type hint fallback**: When `typing.get_type_hints()` fails (e.g., one unresolvable `TYPE_CHECKING` import kills resolution for all annotations), each annotation string is evaluated individually via `eval()` in the function's global namespace. Resolvable annotations are still tracked; unresolvable ones are skipped.

    Tests: `test_fingerprint.py::test_resolve_annotations_individually_mixed`

20. **Library callables are recorded by name**: A stdlib or third-party function or class read
    through a global, closure cell, default or partial (`agg = np.mean`,
    `from numpy import median as agg`) gets an `ext:<binding>` entry holding its definition
    name (`numpy.mean`); its code is not hashed. Rebinding the name reruns the stage, and a
    cache hit re-reads the name. In loader config and partial arguments a library function's
    canonical form is `["external", name]`, since its `repr()` can hold a memory address.
    Reached through its module (`typing.cast`), a library callable adds no entry.

    Tests: `test_fingerprint.py::test_stdlib_callable_in_closure_does_not_raise`,
    `test_manifest_cache_cli.py::test_rebinding_a_library_function_reruns_stage`

21. **Loggers silently skipped**: `logging.Logger` instances captured in closures are skipped. Module-level loggers (`_logger = logging.getLogger(__name__)`) are functionally irrelevant for cache invalidation — their repr can change based on logging configuration.

22. **User-defined generic origins tracked**: For generic types like `MyContainer[int]`, the origin class (`MyContainer`) is now fingerprinted if it's user code. Previously only the type arguments were walked.
