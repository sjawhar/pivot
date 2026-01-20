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
| Global constant (int, float, str, bool, bytes, None) | ✅        | `test_fingerprint.py::test_constant_captured`, `test_fingerprint.py::test_multiple_constants_captured`                              |
| Global constant change                               | ✅        | `test_change_detection.py::test_global_constant_change_causes_miss`                                                                 |
| Global collection (list, dict, set) with callables   | ✅        | `test_change_detection.py::test_list_callable_tracking`, `test_change_detection.py::test_dispatch_dict_function_change_causes_miss` |
| Global collection DATA (non-callable values)          | ❌        | `test_pydantic_defaults.py::test_list_constants_not_captured_as_const`                                                               |
| Pydantic field default data                           | ✅        | `test_pydantic_defaults.py::test_pydantic_default_data_captured`                                                                     |
| Pydantic class in type hint                           | ✅        | `test_pydantic_defaults.py::test_pydantic_class_captured_from_type_hint`                                                             |
| Global class instance                                | ✅        | `test_change_detection.py::test_class_instance_tracked`, `test_change_detection.py::test_class_instance_change_causes_miss`         |
| Class definition change                              | ✅        | `test_change_detection.py::test_class_definition_change_causes_miss`                                                                |
| Class method change                                  | ✅        | `test_change_detection.py::test_class_definition_change_causes_miss` (class AST includes methods)                                   |
| StageParams `@property` method change                | ✅        | `test_change_detection.py::test_stageparams_property_change_causes_miss`                                                            |
| StageParams regular method change                    | ✅        | `test_change_detection.py::test_stageparams_method_change_causes_miss`                                                              |
| StageParams `ClassVar` change                        | ✅        | `test_change_detection.py::test_stageparams_class_variable_change_causes_miss`                                                      |
| Nested function (defined inside stage)               | 🚫        | `test_fingerprint.py::test_nested_function_not_in_globals` (part of stage body)                                                     |
| Helper starting with `_` prefix                      | ✅        | `test_change_detection.py::test_underscore_helper_change_detected`                                                                  |
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
| Transitive dependency change        | ✅        | `test_change_detection.py::test_transitive_dependency_change_causes_miss`               |
| Module constant                     | ✅        | `test_integration.py::test_constant_via_module_attr_captured`                           |
| Module constant captured            | ✅        | `test_change_detection.py::test_module_constant_captured_via_module_attr`               |
| Nested attribute (`X.sub.func`)     | ⚠️        | `test_fingerprint.py::test_extract_nested_attr_access` (extracted but not fully tested) |
| Multiple attrs from same module     | ✅        | `test_fingerprint.py::test_multiple_module_attrs_detected`                              |
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
| Stdlib function (e.g., `json.dumps`)       | 🚫        | `test_integration.py::test_stdlib_marked_callable_not_hashed`             |
| Stdlib via module attr                     | 🚫        | `test_change_detection.py::test_stdlib_function_marked_callable`          |
| Third-party function (e.g., `numpy.array`) | 🚫        | `test_fingerprint.py::test_is_user_code_non_user` (pytest.fixture tested) |
| Third-party package version change         | ❌        | **NO TEST** - versions not tracked                                        |
| Builtin function (`len`, `print`)          | 🚫        | `test_fingerprint.py::test_fingerprint_builtin_function_skipped`          |

---

## 7. Dynamic/Runtime Patterns

| Change Type                              | Detected? | Test Reference                                                                                                                                |
| ---------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `getattr(module, "func")`                | ❌        | **NO TEST** - can't resolve statically                                                                                                        |
| `func_dict[key]()` dispatch              | ✅        | `test_change_detection.py::test_dynamic_dispatch_change_detected`, `test_change_detection.py::test_dispatch_dict_function_change_causes_miss` |
| `eval()` / `exec()`                      | ❌        | **NO TEST** - can't analyze dynamically                                                                                                       |
| `importlib.import_module(string)`        | ❌        | **NO TEST** - can't resolve statically                                                                                                        |
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
| Module not in sys.modules                 | ✅        | `test_fingerprint.py::test_is_user_code_module_not_in_sys_modules`              |
| Attribute doesn't exist on module         | ✅        | `test_fingerprint.py::test_module_attr_with_attribute_error`                    |
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
| Class attribute callbacks (runtime modified)       | ❌        | `test_callback_vulnerabilities.py::test_class_attribute_callback_change_detected`       |

---

## Summary: Gaps Requiring New Tests

| Gap                                              | Priority | Difficulty                    |
| ------------------------------------------------ | -------- | ----------------------------- |
| Decorator on stage function not tracked          | Medium   | Easy                          |
| Nested module attribute (`X.sub.func`) full test | Low      | Easy                          |
| `import X` inside function not tracked           | Low      | Easy (already documented)     |
| `getattr()` dynamic lookup                       | Low      | Easy                          |
| `eval()`/`exec()` limitation                     | Low      | Easy                          |
| `importlib.import_module()` limitation           | Low      | Easy                          |
| Third-party package version tracking             | Medium   | Hard (would need new feature) |

---

## Design Decisions

1. **Dunder name filtering**: Names starting with `__` are skipped to filter `__name__`, `__file__`, `__doc__`, etc. Single-underscore names like `_private_helper()` are tracked normally.

2. **Stdlib/third-party marked "callable"**: These are not hashed because we don't want package version changes to trigger rebuilds (too sensitive).

3. **Function name normalization**: Function names are normalized to `"func"` in AST to allow renaming without cache miss.

4. **Docstrings/comments ignored**: These don't affect behavior, so they're stripped from AST.

5. **Lazy imports not tracked**: Imports inside function bodies are not detected. Recommended pattern: use module-level imports.

6. **Collection callable tracking**: Callables inside global collections (list, dict, set, tuple, frozenset) are detected and hashed. Only callables are tracked (not data values) to avoid sensitivity to mutable runtime state. Dict keys are sorted alphabetically for deterministic ordering; sets are also sorted.

7. **Bytecode fallback uses marshal**: When source code is unavailable, `marshal.dumps(func.__code__)` captures the full code object including constants. This ensures that `return x + 1` and `return x + 999` produce different hashes (raw bytecode alone doesn't include constants).

8. **Class definition tracking**: Classes are tracked using the `class:` prefix (e.g., `class:MyProcessor`). The entire class definition is hashed including all methods, class variables, and decorators. Module-level class instances (e.g., `processor = Processor()`) have their class type tracked via `class:varname.__class__`.

9. **No inheritance tracking**: Base classes are not recursively tracked. Most user classes inherit from `object` or third-party framework classes (both intentionally ignored). If users have deep inheritance hierarchies of user-defined classes, they can factor shared logic into helper functions.

10. **Callable instances tracked as functions**: Objects with `__call__` methods are matched by the `callable()` check before the instance check. This means changes to non-`__call__` methods on such classes may not trigger cache invalidation. Workaround: use the class directly instead of a pre-instantiated callable.

11. **NamedTuple instances tracked as tuples**: `NamedTuple` instances inherit from `tuple` and are matched by the collection check. Changes to the `NamedTuple` class definition may not be detected. Workaround: use regular classes or dataclasses instead.

12. **Pydantic field defaults ARE tracked**: Pydantic model classes used in type hints are automatically detected and their field default values are hashed. The fingerprint includes:
    - `class:ModelName` - Hash of the class AST (captures structural changes)
    - `pydantic:ModelName.field` - Hash of each field's default value (captures data changes)

    This allows Pivot to detect when Pydantic configuration changes without requiring explicit `deps`.

    Tests: `test_pydantic_defaults.py::test_pydantic_default_data_captured`, `test_pydantic_defaults.py::test_pydantic_class_captured_from_type_hint`, `test_pydantic_defaults.py::test_pydantic_default_change_triggers_different_hash`

13. **`functools.wraps` / `__wrapped__` ARE tracked**: Functions decorated with `@functools.wraps` are properly fingerprinted using bytecode hashing. Since `inspect.getsource()` follows the `__wrapped__` chain and returns the original function's source, we detect `__wrapped__` and use `marshal.dumps(__code__)` to hash the wrapper's bytecode instead. This correctly captures decorator logic changes while closure analysis still tracks the wrapped function.

    Tests: `test_functools.py::test_decorator_change_triggers_fingerprint_change`, `test_functools.py::test_wrapped_uses_bytecode_not_source`

14. **`functools.partial` IS tracked**: Partial objects are specially handled before the `is_user_code()` check. The fingerprint includes:
    - `partial:<name>.args` - Hash of bound positional arguments
    - `partial:<name>.kwargs` - Hash of bound keyword arguments
    - `func:<name>.func` - Hash of the underlying function (if user code)

    This allows Pivot to detect changes to both the partial's bound arguments and the underlying function.

    Tests: `test_functools.py::test_partial_is_detected`, `test_functools.py::test_partial_args_change_triggers_fingerprint_change`

15. **Instance state not tracked**: For user-defined class instances, only the class definition is hashed, not instance state (attributes, dict contents). Runtime-assigned callbacks on instances or in containers are not detected. Workaround: use module-level variables or explicit deps for mutable configuration.

16. **Class attributes modified at runtime not tracked**: Class attributes set after class definition (e.g., `Config.callback = func`) are not detected because only the original class source is hashed. Workaround: use module-level variables instead of class attributes for runtime configuration.

17. **StageParams classes fully tracked**: `StageParams` subclasses used in type hints are tracked via `_process_type_hint_dependencies()`. The entire class definition is hashed using `inspect.getsource()`, which captures:
    - `@property` methods
    - Regular methods
    - `ClassVar` declarations

    Changes to any of these trigger a cache miss.

    Tests: `test_change_detection.py::test_stageparams_property_change_causes_miss`, `test_change_detection.py::test_stageparams_method_change_causes_miss`, `test_change_detection.py::test_stageparams_class_variable_change_causes_miss`

18. **Builtin types ARE deterministic**: Builtin types (`list`, `dict`, `set`, `tuple`, etc.) used as `default_factory` in Pydantic fields are hashed using their qualified name (e.g., `builtin:list`) rather than `id()`. This ensures fingerprints are stable across Python sessions. Previously, `id()` was used as a fallback for objects without source code, causing spurious "Code changed" invalidations.

    Tests: `test_determinism.py::test_builtin_default_factory_deterministic_across_processes`, `test_determinism.py::test_builtin_type_deterministic`
