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
- **`test_safe_fingerprinting.py`** - Tests for safe fingerprinting error guards (dynamic name access, unsafe fingerprinting and the run cache)
- **`test_fingerprint_gaps.py`** - CLI regressions for callable class members, dynamic-import spellings, and body imports
- **`test_no_fingerprint.py`** - Worker skip/re-run behavior for `@no_fingerprint()` stages (file-level hashing)
- **`test_fingerprint_properties.py`** - Hypothesis properties over generated stage bodies: fresh-process determinism, formatting insensitivity, one-token sensitivity, and `params.yaml` overrides vs. the run-cache key
- **`test_worker_imports.py`** - A module the program that runs Pivot holds must be the module a stage's worker imports by name (`loaded_source.check_worker_imports`): prediction, confirmation in a new interpreter, the CLI cases that must not fail

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
| Unsafe fingerprinting (refusal downgraded to a warning): result kept out of the run cache, by `pivot repro` and `pivot commit` | ✅ | `test_safe_fingerprinting.py::test_unsafe_result_is_kept_out_of_the_run_cache`, control `test_safe_fingerprinting.py::test_safe_result_is_filed_in_the_run_cache` |
| `pivot config set core.unsafe_fingerprinting true` | ✅ | `test_safe_fingerprinting.py::test_config_set_accepts_unsafe_fingerprinting` |
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
| Global read only inside a method of a user class (`Scaler().apply` reads `SCALE`) | ✅ | `test_change_detection.py::test_method_global_change_causes_miss` |
| Compiled regex read only inside a method, class imported from another module (`ScopeWalk().step`) | ✅ | `test_change_detection.py::test_method_regex_change_causes_miss` |
| Helper called only from a classmethod (`Runner.go` calls `bump`) | ✅ | `test_change_detection.py::test_method_helper_change_causes_miss` |
| Callable class member (`partial`, `partialmethod`, nested class) | ✅ | `test_fingerprint_gaps.py::test_class_member_edit_reruns_stage` |
| Callable instance class member | ✅ (error) | `test_fingerprint_gaps.py::test_dataclass_callable_member_is_refused` |
| Decorator wrapper body on a class method | ✅ | `test_fingerprint_gaps.py::test_decorated_method_wrapper_edit_reruns_stage` |
| Data class method reading a global (method walked, not refused) | ✅ | `test_change_detection.py::test_dataclass_method_reading_global_is_walked` |
| Data class default from a module value | ✅ | `test_fingerprint_gaps.py::test_class_member_edit_reruns_stage[dataclass-field-default]` |
| In-body `from X import` of a mutable loaded module value | ✅ (error) | `test_change_detection.py::test_in_body_from_import_of_non_scalar_is_refused` |
| Relative in-body import (`from .sibling import value`) | ✅ | `test_change_detection.py::test_relative_in_body_import_change_causes_miss` |
| Unloaded body import refused without running it | ✅ (error) | `test_fingerprint_gaps.py::test_guarded_unloaded_body_import_is_refused_without_running_it`, `test_change_detection.py::test_unloaded_body_import_is_refused_without_running_it` |
| Static body import of a loaded submodule (`import pkg.sub`, `from pkg import sub`) | ✅ | `test_fingerprint_gaps.py::test_static_body_import_of_loaded_submodule_is_reused` |
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
| Loaded `from X import func` inside function | ✅ | `test_change_detection.py::test_loaded_body_import_is_tracked_as_a_helper` |
| Unloaded user module (`import m`, `from m import f`, `import pkg.sub`, `pkg.sub` read through `import pkg`) | ✅ (error) | `test_fingerprint_gaps.py::test_unloaded_user_body_import_is_refused_at_definition`, `test_fingerprint_gaps.py::test_body_import_of_unloaded_submodule_through_loaded_package_is_refused`, `test_fingerprint_gaps.py::test_preloaded_package_attribute_of_unloaded_package_is_refused` |
| Library module imported in the body | ✅ | `test_fingerprint_gaps.py::test_body_import_of_a_library_module_is_not_refused` |
| User module loaded only at run time (partial, `vars`, `exec`, manual loader) | ✅ (not reused) | `test_fingerprint_gaps.py::test_runtime_dynamic_import_makes_result_not_reusable` |
| Import from a thread an earlier stage left running | ✅ (not reused) | `test_fingerprint_gaps.py::test_import_from_a_thread_outliving_its_stage_does_not_fail_the_next_stage` |
| `@no_fingerprint(code_deps=...)` stage importing a user module | ✅ (not checked; `code_deps` tracked) | `test_runtime_import_contract.py::test_d1_no_fingerprint_stage_with_body_import_is_reused` |
| User loader `load`/`save` importing a loaded module (`dep:`/`out:` module keys); an output field named `module` (tag `out:<stage>:module:tag`) covers no module `tag`, nor does any dependency, field or stage named like an entry kind | ✅ | `test_runtime_import_contract.py::test_d2_user_loader_body_import_of_loaded_module_is_covered`, `::test_d6_an_output_named_module_does_not_cover_a_module_its_tag_names`, `test_fingerprint.py::test_covered_user_modules_reads_every_module_key_form`, `test_fingerprint_properties.py::test_no_dependency_output_or_stage_name_is_read_as_a_module_or_by_name_entry` |
| `pivot commit` after a flagged `repro` keeps the stage stale | ✅ | `test_runtime_import_contract.py::test_d3_commit_after_a_flagged_committing_repro_is_described`, `core/test_skip.py::test_carry_rerun_marks_keeps_marker_when_static_fingerprint_is_unchanged` |
| Following the warning's fix clears it | ✅ | `test_fingerprint_gaps.py::test_following_the_uncovered_import_warning_fix_clears_it`, `test_runtime_import_contract.py::test_d4_uncovered_import_warning_names_a_fix_that_applies` |

### 4.1 Code Edited After Discovery

| Change Type | Detected? | Test Reference |
| ----------- | --------- | -------------- |
| In-root helper edited by an upstream stage (worker runs the loaded helper; lock records it) | ✅ (reruns next run) | `test_code_edited_during_run.py::test_in_root_helper_edited_by_an_upstream_stage_is_not_cached_stale`, `test_discovery.py::test_stage_fingerprint_is_of_the_helper_source_discovery_loaded` |
| A stage's own source edited by an upstream stage, AST and `@no_fingerprint` | ✅ (reruns next run) | `test_code_edited_during_run.py::test_stage_edited_by_an_upstream_stage_is_not_cached_stale` |
| Out-of-root helper a warm worker imported before the edit | ✅ (reruns next run) | `test_code_edited_during_run.py::test_out_of_root_helper_a_warm_worker_holds_is_not_cached_stale` |
| Out-of-root helper a worker imports after the edit | ✅ (warns, not reused) | `test_code_edited_during_run.py::test_out_of_root_helper_imported_after_its_edit_is_not_reused` |
| No edit: nothing warns, every stage cached | ✅ | `test_code_edited_during_run.py::test_unedited_project_is_cached_on_the_next_run` |
| Loaded source survives `linecache` clearing and matches `linecache`'s line split | ✅ | `test_loaded_source.py` |
| A stage that never used the edited module is not marked (one worker; CLI and a long-lived engine) | ✅ (cached next run) | `test_code_edited_during_run.py::test_stage_that_never_touches_the_edited_module_is_not_marked`, `integration/test_warm_workers.py::test_server_runs_do_not_rerun_a_stage_that_never_used_an_edited_helper`, `test_loaded_source.py::test_a_stage_uses_modules_its_manifest_hashes_or_it_imported` |
| A helper loaded by a meta-path finder, `spec_from_file_location` or a zip importer | ✅ (reruns next run) | `test_code_edited_during_run.py::test_helper_from_a_meta_path_finder_edited_mid_run_is_not_cached_stale`, `::test_in_root_helper_loaded_by_spec_from_file_location_is_not_cached_stale`, `::test_helper_in_a_zip_edited_mid_run_is_not_cached_stale` |
| A sourceless module | ✅ (warns, reruns every run, no run cache) | `test_code_edited_during_run.py::test_stage_using_a_sourceless_module_is_never_reused`, `test_loaded_source.py::test_a_sourceless_module_is_recorded_as_unreadable` |
| A helper an import hook serves after the recorder with a loader Pivot does not trust (a `SourceFileLoader` subclass, another `SourceLoader`, a proxy around CPython's loader, a loader that runs CPython's inside it, CPython's `SourceFileLoader` definition on its private bases, a `FileLoader` with its own bytecode cache), whose bytecode is stale: edited mid-run with its mtime put back, or left stale before the run, in or out of the root | ✅ (unreadable, with a reason naming the loader: warns, reruns every run, no run cache) | `test_code_edited_during_run.py::test_helper_a_loader_pivot_cannot_vouch_for_serves_is_never_cached_stale`, `test_loaded_source.py::test_a_module_a_loader_pivot_cannot_vouch_for_loads_through_the_recorder_is_unreadable`, `::test_a_reload_an_untrusted_loader_runs_does_not_keep_the_earlier_record` |
| A helper in a zip that also holds a legacy `.pyc` whose header matches the `.py` member, which zipimport's own finder picks over the source | ✅ (unreadable: reruns every run) | `test_code_edited_during_run.py::test_helper_a_zip_runs_from_its_bytecode_member_is_never_cached_stale`, `test_loaded_source.py::test_a_zip_module_run_from_its_bytecode_member_is_unreadable` |
| The same zip, served by a finder that builds the spec itself with the `.py` member as origin (zipimport's `exec_module` would still run the `.pyc`), or rewritten between Pivot's read and the import | ✅ (Pivot runs the `.py` bytes it read: readable, cached with the current output) | `test_code_edited_during_run.py::test_helper_a_finder_specs_from_its_zip_py_member_runs_the_source_pivot_recorded`, `test_loaded_source.py::test_a_zip_module_whose_spec_names_its_py_member_runs_the_source_pivot_recorded`, `::test_a_zip_rewritten_after_pivot_read_its_member_runs_the_bytes_pivot_recorded` |
| A wrapper around a loader Pivot does not trust, and a module whose body imports a submodule while its untrusted loader runs it | ✅ (unreadable; the reason names the real loader) | `test_loaded_source.py::test_a_wrapper_around_a_loader_pivot_does_not_trust_names_the_inner_loader`, `::test_an_unreadable_record_made_while_the_module_ran_does_not_stand` |
| A helper a finder modelled on dd-trace-py's ModuleWatchdog serves by wrapping the loader the rest of `sys.meta_path` (Pivot's recorder among it) returns | ✅ (readable: reruns next run, then cached) | `test_code_edited_during_run.py::test_helper_a_loader_wrapping_pivots_own_serves_stays_readable`, `test_loaded_source.py::test_a_loader_that_wraps_pivots_own_keeps_the_source_pivot_recorded` |
| A helper an import hook serves with a loader that has only `load_module` (PEP 302) still imports with the recorder installed, as on main | ✅ (unreadable) | `test_loaded_source.py::test_a_legacy_load_module_only_loader_still_imports_with_the_recorder_installed` |
| A new package shadowing the module mid-run (same name, other file) | ✅ (warns, not reused) | `test_code_edited_during_run.py::test_new_package_shadowing_the_helper_mid_run_is_compared_by_name` |
| A marked run files no run-cache entry | ✅ | `test_code_edited_during_run.py::test_marked_run_files_no_run_cache_entry` |
| `--no-commit`, then `pivot commit` of unchanged outputs, records the code the run ran (marks included) | ✅ (reruns next run) | `test_code_edited_during_run.py::test_no_commit_then_commit_does_not_cache_the_loaded_helper_output`, `::test_no_commit_then_commit_keeps_the_edited_source_mark`, `::test_edited_source_warning_under_no_commit_warns_that_commit_keeps_the_mark`, `test_fingerprint_gaps.py::test_runtime_dynamic_import_under_no_commit_warns_that_commit_keeps_the_mark` |
| `pivot commit` of outputs edited by hand after a `--no-commit` run trusts the user | ✅ (cached next run) | `test_code_edited_during_run.py::test_commit_of_outputs_edited_after_a_no_commit_run_trusts_the_user` |
| A later discovery drops the earlier project's records and `linecache` pins | ✅ | `test_discovery.py::test_a_later_discovery_forgets_the_earlier_projects_loaded_source`, `test_loaded_source.py::test_prune_forgets_modules_no_longer_loaded_and_their_linecache_lines` |
| Workers get the records once; a pool whose workers hold other records is replaced | ✅ | `test_loaded_source.py::test_a_stage_carries_digests_only_after_the_record_changed`, `engine/test_worker_pool.py::test_start_replaces_workers_that_started_with_other_loaded_source_records` |
| A helper `sitecustomize` or a `.pth` import line imports before discovery, edited mid-run (coordinator and workers) | ✅ (not cached stale) | `test_code_edited_during_run.py::test_helper_imported_before_discovery_edited_mid_run_is_not_cached_stale`, `::test_helper_a_pth_line_imports_at_startup_edited_mid_run_is_not_cached_stale` |
| A helper imported before discovery and unchanged for more than the 2 s margin when the process started: recorded from its file, reused | ✅ (cached next run) | `test_code_edited_during_run.py::test_helper_imported_before_discovery_and_left_unchanged_is_reused`, `test_loaded_source.py::test_a_module_imported_before_the_recorder_is_recorded_from_its_unchanged_file` |
| A helper imported before discovery whose file or bytecode cannot be shown to hold its code (file changed after the process started or less than 2 s before, another loader, stale or replaced bytecode, a forked process), in the coordinator or a worker | ✅ (warns why once per process, reruns every run, no run cache) | `test_code_edited_during_run.py::test_helper_imported_before_discovery_and_edited_before_it_is_never_reused`, `test_loaded_source.py::test_a_module_whose_file_changed_since_the_process_started_is_unreadable`, `::test_a_module_another_loader_loaded_before_the_recorder_is_unreadable`, `::test_stale_bytecode_the_import_ran_makes_the_module_unreadable`, `::test_bytecode_written_after_the_process_started`, `::test_a_module_imported_without_bytecode_is_recorded_unless_bytecode_was_removed`, `::test_a_forked_process_cannot_vouch_for_modules_its_parent_loaded`, `::test_a_worker_judges_the_modules_it_imported_before_its_initializer`, `::test_a_worker_logs_the_coordinators_reason_for_an_unseen_module` |
| A finished discovery (returned, failed or interrupted) leaves no import hook installed; tests pass in any order in one process | ✅ | `test_discovery.py::test_a_finished_discovery_leaves_no_import_hook_installed`, `::test_session_end_gives_user_code_cpythons_loader_back`, `test_suite_order.py` |
| A stage's own module, and a module it imported while it ran (already loaded in the worker), count as used | ✅ (marked) | `test_loaded_source.py::test_a_stage_checks_the_module_that_defines_it`, `::test_a_stage_checks_the_modules_it_imported_while_it_ran`, `test_code_edited_during_run.py::test_module_a_stage_imports_while_it_runs_edited_mid_run_is_marked`, `::test_stage_defined_out_of_root_edited_mid_run_is_never_cached_stale` |
| A module that replaces itself in `sys.modules` while it loads keeps discovery's record | ✅ | `test_discovery.py::test_a_module_that_replaces_itself_in_sys_modules_keeps_its_record` |
| A lock a later run writes retires the `--no-commit` record | ✅ | `test_code_edited_during_run.py::test_a_lock_a_later_run_writes_retires_the_no_commit_record` |
| A change time in whole seconds (ext4 with 128-byte inodes) or a forward step of the wall clock of up to 2 s cannot show an edit made after the start came before it; a step after Pivot's import never moves the start later | ✅ (unreadable, not cached stale) | `test_unseen_modules.py::test_a_change_time_in_whole_seconds_cannot_show_an_edit_came_before_the_start`, `test_code_edited_during_run.py::test_helper_edited_in_the_second_its_process_started_on_a_whole_second_file_system` (needs `CI` and sudo), `test_loaded_source.py::test_a_forward_clock_step_after_an_edit_cannot_show_the_edit_came_before_the_start`, `::test_a_step_of_the_wall_clock_after_import_never_moves_the_start_later` |
| A module a stage's module loads with a loader it calls itself (`load_module`, `spec_from_loader`), edited mid-run: the worker judges it as discovery does | ✅ (marked, not cached stale after the undo) | `test_unseen_modules.py::test_helper_a_stage_module_loads_with_its_own_loader_edited_mid_run_is_not_cached_stale`, `test_loaded_source.py::test_a_worker_judges_a_module_a_loader_called_directly_loaded`, `::test_a_process_without_the_coordinators_records_judges_no_module_after_a_stage` |
| Every production worker that checks a stage holds the coordinator's records, so the worker's judgement after each stage always runs (one worker and many, `pivot run`, `--watch`, the RPC server's warm pool) | ✅ | `test_code_edited_during_run.py::test_every_worker_that_checks_a_stage_holds_the_coordinators_records` |
| A `--no-commit` record holding only another feature's key (`pvt_members`) reads as present, with that key as stored, and `pivot commit` treats the stage as one with no record; applying a run's record replaces only its own keys | ✅ | `storage/test_state.py::test_a_no_commit_record_reads_each_valid_field_it_knows`, `::test_applying_a_no_commit_record_replaces_its_keys_and_keeps_every_other`, `::test_a_written_lock_removes_the_whole_no_commit_record`, `test_code_edited_during_run.py::test_commit_reads_a_no_commit_record_without_code_as_no_record` |

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
| `pivot_tui` installed editable by its own backend (`uv_build`) | 🚫 | `test_fingerprint.py::test_pivot_tui_installed_from_a_checkout_is_framework` |
| A user module or package named `pivot_tui` | ✅        | `test_fingerprint.py::test_pivot_tui_is_pivot_s_package_only_where_the_pivot_tui_distribution_installed_it`, `cli/test_cli_acceptance.py::test_editing_a_project_package_named_pivot_tui_reruns_its_stage` |
| The user's part of a namespace package shared with a package pip checked out under the interpreter's prefix (`pip install -e git+...`) | ✅ | `test_fingerprint.py::test_is_user_code_namespace_package_split_with_a_directory_under_the_prefix`, `cli/test_cli_acceptance.py::test_editing_the_user_s_part_of_a_namespace_shared_with_a_pip_checkout_reruns_its_stage` |
| Builtin function (`len`, `print`)          | 🚫        | `test_fingerprint.py::test_fingerprint_builtin_function_skipped`          |
| Library module read as a global or imported in a body, and the module that defines a library callable, recorded by name (`byname:`) by the walk, outside the lock | ✅ | `test_worker_imports.py::test_a_library_module_read_as_a_global_is_recorded_by_name_outside_the_lock`, `::test_a_built_in_module_a_root_file_names_needs_no_new_interpreter`, `::test_a_frozen_module_needs_no_new_interpreter`, `test_change_detection.py::test_stdlib_module_attrs_not_tracked`, `test_fingerprint_by_name.py::test_an_unmarked_project_gets_mains_locks_and_an_upgrade_reruns_nothing` |
| A module the program that runs Pivot holds from another file than a worker would import (outside the root, a notebook under the root, a root namespace directory, a package and its submodule, a library read as a module, as a callable or imported in a body, a namespace child of a library, a loader's module) | ✅ (error) | `test_worker_imports.py::test_a_module_the_program_holds_from_another_file_fails_the_stages_fingerprint`, `::test_a_submodule_the_program_holds_from_another_file_fails_the_stages_fingerprint`, `::test_a_library_the_program_holds_with_a_project_modules_name_fails_the_stages_fingerprint`, `::test_a_library_the_program_holds_and_a_stage_imports_in_its_body_fails_the_fingerprint`, `::test_a_namespace_child_the_program_holds_from_a_library_fails_the_stages_fingerprint`, `::test_a_loaders_module_named_by_name_is_compared`, `::test_a_module_from_another_file_fails_the_run_before_a_worker_runs_it` |
| A root file named like a built-in module (`time.py`), a module every interpreter imports at startup (`sitecustomize`) or a startup finder serves, the project's own file, a symlinked path to the same file, a held submodule a project package sends by value | ✅ (runs, no error) | `test_worker_imports.py::test_a_root_file_named_like_a_built_in_module_runs_the_built_in_one`, `::test_a_sitecustomize_module_with_a_root_files_name_runs_in_the_cli`, `::test_a_module_every_interpreter_imports_at_startup_is_the_one_its_worker_runs`, `::test_a_module_a_startup_finder_before_path_finder_serves_needs_no_new_interpreter`, `::test_a_module_the_program_holds_from_the_projects_file_fingerprints`, `::test_a_module_the_program_reaches_through_a_symlink_is_its_real_file`, `::test_a_submodule_the_program_holds_under_a_project_package_goes_to_the_worker_by_value` |
| A library submodule (`email.mime.text`) a worker imports from the same file, found in its package's directories | ✅ (no interpreter) | `test_worker_imports.py::test_a_library_submodule_a_worker_imports_from_the_same_file_needs_no_new_interpreter` |
| A pipeline fingerprinted after a later discovery found nothing (its session ended): still compared against what the program held | ✅ (error) | `test_worker_imports.py::test_a_pipeline_fingerprinted_after_a_later_discovery_found_nothing_is_still_compared` |
| A project file that takes over a library's name (vendored `helpers.py`): the manifest cache re-checks where each by-name module resolves | ✅ | `test_manifest_cache_cli.py::test_a_project_module_that_takes_over_a_librarys_name_is_not_hidden_by_the_cache` |
| The confirming interpreter: startup output on stdout, a `pathlib.Path` entry in `sys.path` (skipped, as imports skip it), one interpreter per session for every suspect, kept per `sys.path`, started in the working directory (a relative `PYTHONPATH`), after loky's imports, and a failure that leaves the prediction to decide; the check's error reported with the stage's other refusals; a partial over a library callable and a library submodule of a project package recorded by name; an input's or output's loader reading a held module; a lock leaving out the `byname:` entries but keeping every entry of a dependency or output named `byname`, so an edit to its loader reruns the stage | ✅ | `test_worker_imports.py::test_a_sitecustomize_module_with_a_root_files_name_runs_in_the_cli`, `::test_a_path_object_in_sys_path_is_skipped_as_the_import_system_skips_it`, `::test_the_suspects_of_a_session_are_confirmed_in_one_interpreter`, `::test_a_confirmation_is_kept_per_sys_path`, `::test_a_relative_startup_path_names_a_directory_under_the_working_directory`, `::test_a_module_a_worker_imports_before_it_gets_its_path_is_the_one_it_uses`, `::test_a_confirmation_that_cannot_run_fails_on_the_prediction`, `::test_the_check_is_reported_with_the_stages_other_refusals`, `::test_a_partial_over_a_library_callable_the_program_holds_fails_the_fingerprint`, `::test_a_library_submodule_a_stage_imports_from_a_project_package_is_recorded_by_name`, `::test_a_loaders_module_named_by_name_is_compared`, `::test_a_lock_keeps_every_entry_of_a_dependency_or_output_named_byname`, `::test_a_dependency_and_an_output_named_byname_rerun_when_their_loader_changes`, `test_fingerprint_properties.py::test_no_dependency_output_or_stage_name_is_read_as_a_module_or_by_name_entry` |
| A module the program loads by file location on another thread while discovery runs; a `sys.meta_path` finder it inserted after its interpreter started; a package whose `__init__` changes its `__path__`; a pipeline object run again after `sys.path` changed; a relative `sys.path` entry with a `chdir` inside one session | ❌ (stated limit) | Documented in `docs/architecture/fingerprinting.md` |

---

## 7. Dynamic/Runtime Patterns

| Change Type                              | Detected? | Test Reference                                                                                                                                |
| ---------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `getattr(module, "func")`                | ❌        | **NO TEST** - can't resolve statically                                                                                                        |
| `getattr(obj, name)` (non-literal)       | ✅ (error) | `test_fingerprint.py::test_check_dynamic_name_access_rejects_dynamic_getattr` |
| `globals()` / `locals()`                 | ✅ (error) | `test_fingerprint.py::test_check_dynamic_name_access_rejects_globals` |
| `func_dict[key]()` dispatch              | ✅        | `test_change_detection.py::test_dynamic_dispatch_change_detected`, `test_change_detection.py::test_dispatch_dict_function_change_causes_miss` |
| `eval()` / `exec()` of code reading other names | ❌        | **NO TEST** - can't analyze dynamically |
| Lambda created by `eval()` (no source) | ✅        | `test_fingerprint_gaps.py::test_eval_lambda_is_hashed_by_its_code_object` |
| `x.import_module(...)` call | ✅ (error) | `test_fingerprint.py::test_check_dynamic_name_access_rejects_import_module` (main's refusal) |
| `functools.wraps` wrapper whose own body uses a computed `getattr`, `import_module` or `globals()` (decorator in `pipeline.py` or in a helper module) | ✅ (runs, as on main: the refusals read the wrapped function's source) | `test_fingerprint_gaps.py::test_dynamic_name_refusals_read_the_wrapped_function_source` (4 cases) |
| Computed `getattr` in a function a wrapper reaches only through `__wrapped__` | ✅ (error, as on main) | `test_fingerprint_gaps.py::test_dynamic_name_refusal_reads_a_function_reached_only_through_wrapped` |
| Other dynamic spelling of a user module (alias, `__import__`, partial, default, literal `getattr`) | ✅ (not reused) | `test_fingerprint_gaps.py::test_dynamic_import_spelling_of_a_user_module_makes_result_not_reusable` |
| Relative `builtins.__import__` with package from `__name__`, `__spec__.parent` or `ModuleType` fallback | ✅ (not reused, cold or warm) | `test_fingerprint_gaps.py::test_relative_builtins_import_uses_cpythons_package_fallback` |
| Dynamic import of a library (`__import__("json")`, an `import_module` alias) | ✅ | `test_fingerprint_gaps.py::test_dynamic_import_of_a_library_runs_and_caches` |
| A resolved helper receiving `"import_module"` | ✅ | `test_fingerprint_gaps.py::test_resolved_helper_accepts_import_module_argument_name` |
| Library that creates modules without a source file (`pyexpat.errors`) | ✅ | `test_fingerprint_gaps.py::test_library_that_creates_fileless_modules_caches` |
| Body import of a loaded module that imports other user modules | ✅ | `test_fingerprint_gaps.py::test_body_import_of_loaded_module_that_imports_user_modules_is_reused`, `test_fingerprint_gaps.py::test_uncovered_dynamic_import_after_a_covered_import_still_warns` |
| Body import of a loaded module that registers a module without the hooks (`exec_module`, `EntryPoint.load()`, `importlib.__import__`) | ✅ (cached, as on main) | `test_fingerprint_gaps.py::test_body_import_of_module_that_registers_a_module_without_the_hooks_is_reused` |
| Thread the stage starts after its covered import returned | ✅ (not reused) | `test_fingerprint_gaps.py::test_thread_import_after_the_covered_import_returned_still_warns`, `test_covered_threads.py::test_d1_control_a_thread_the_stage_starts_after_the_covered_import_still_warns` |
| Body import of a loaded module whose import starts a thread that imports a user module, joined inside the import (`__import__`, `ThreadPoolExecutor` + `import_module`) | ✅ (cached, as on main) | `test_fingerprint_gaps.py::test_body_import_of_module_whose_import_thread_imports_a_user_module_is_reused` |
| Loader thread a covered import starts, importing after the import returned and joined by the stage | ✅ (cached; lib edit reruns) | `test_covered_threads.py::test_d1_body_import_of_module_whose_loader_thread_imports_after_it_returns_is_reused`, `test_covered_threads.py::test_d1_loader_thread_stage_reruns_on_a_lib_edit` |
| Covered import's thread importing a module an upstream stage loaded in the same worker | ✅ (cached, warm or cold) | `test_covered_threads.py::test_d2_covered_import_whose_joined_thread_imports_a_module_an_upstream_stage_loaded_caches`, `test_covered_threads.py::test_d2_control_without_the_upstream_import_the_same_layout_caches` |
| Hand-off shapes of a covered import: straddle (joined or working), nested thread, `threading.Timer`, `ThreadPoolExecutor`, asyncio loop on a thread, `concurrent.futures.wait`, daemon thread | ✅ (cached; lib edit reruns) | `test_covered_threads.py::test_thread_a_covered_import_hands_work_to_is_covered`, `execution/test_worker_contracts.py::test_thread_a_covered_import_starts_is_covered_after_the_import_returns` |
| Pool that existed before the covered import (stated limit) | ✅ (not reused; warning names it) | `test_covered_threads.py::test_import_handed_to_a_thread_that_existed_before_the_covered_import_is_counted`, `execution/test_worker_contracts.py::test_hooked_import_on_a_thread_no_covered_thread_started_is_counted` |
| Thread native code starts outside `_thread` (a C extension, `pthread_create` through `ctypes`) | ❌ (stated limit: not covered, so not reused) | Documented in `docs/architecture/fingerprinting.md` and `docs/concepts/fingerprinting.md`; import the module statically |
| Covered thread of one stage importing during the next stage | ✅ (counted for the next stage) | `test_covered_threads.py::test_covered_thread_of_one_stage_is_not_covered_in_the_next`, `execution/test_worker_contracts.py::test_covered_thread_of_an_earlier_stage_is_not_covered_in_a_later_stage` |
| Unhooked registration on another thread while a covered thread runs (stated limit) | ✅ (excused) | `execution/test_worker_contracts.py::test_unhooked_registration_on_another_thread_while_a_covered_thread_runs_is_excused` |
| Pipe reader, parent watcher or stop-grace timer Pivot starts inside a covered import | ✅ (not reused: later manual load is counted) | `test_fingerprint_gaps.py::test_pivot_thread_started_inside_a_covered_import_does_not_excuse_a_later_manual_load`, `execution/test_worker_contracts.py::test_pivot_owned_worker_threads_start_through_the_bypass_helper` |
| Stage registers a module without the hooks after every covered thread ended | ✅ (not reused) | `execution/test_worker_contracts.py::test_module_the_stage_registers_after_every_covered_thread_ended_is_counted` |
| Covered thread still running when the stage returns, having loaded a module by either route | ✅ (excused) | `execution/test_worker_contracts.py::test_what_a_covered_thread_loads_is_excused_when_the_thread_outlives_the_stage` |
| Every `_thread` start function hooked (`start_new_thread`, `start_new`, `start_joinable_thread`) | ✅ | `execution/test_worker_contracts.py::test_every_thread_start_function_is_hooked` |
| Stage loads a module by hand, then a covered import imports it | ✅ (not reused) | `test_covered_threads.py::test_module_the_stage_loads_by_hand_stays_counted_when_a_covered_import_then_imports_it`, `execution/test_worker_contracts.py::test_module_the_stage_loads_outside_every_window_stays_counted_when_a_covered_import_uses_it` |
| Covered import of a dependency an earlier stage loaded (warm worker) | ✅ (cached, warm or cold) | `test_fingerprint_gaps.py::test_covered_import_of_a_dependency_an_earlier_stage_loaded_is_reused_warm_or_cold`, `execution/test_worker_contracts.py::test_covered_import_of_an_already_resident_dependency_is_excused_on_its_own_thread` |
| Module a covered module loads with `exec_module`, `EntryPoint.load()` or `importlib.__import__`: excused, not fingerprinted (stated limit) | ✅ (cached, as on main) | `test_fingerprint_gaps.py::test_body_import_of_module_that_registers_a_module_without_the_hooks_is_reused` |
| Marked lock with a run-cache entry at its input hash or another (0.5.2, `pivot commit`) | ✅ (runs; entry deleted) | `test_run_cache_after_flagged_run.py::test_d1_forced_run_after_an_edit_is_not_undone_by_a_0_5_2_run_cache_entry`, `test_run_cache_after_flagged_run.py::test_d1_stage_a_forced_run_marked_reruns_on_the_next_edit`, `test_run_cache_after_flagged_run.py::test_d1_forced_run_after_no_commit_and_commit_is_not_undone`, `test_run_cache_after_flagged_run.py::test_flagged_run_deletes_the_run_cache_entry_at_its_input_hash`, `test_run_cache_after_flagged_run.py::test_status_explain_reports_a_marked_stage_as_will_run_not_run_cache`, `test_run_cache_after_flagged_run.py::test_marked_lock_is_not_satisfied_from_an_entry_at_another_input_hash`, `test_run_cache_after_flagged_run.py::test_commit_carrying_the_mark_deletes_the_entry_at_its_input_hash` |
| Body import a raising `sys.meta_path` finder rejects | ✅ (runs and caches, as on main) | `test_fingerprint_gaps.py::test_body_import_a_raising_meta_path_finder_rejects_runs_and_caches` |
| User class reached only through a pickled input | ✅ (not reused; annotation covers it) | `test_fingerprint_gaps.py::test_user_class_reached_only_through_a_loaded_input` |
| Unloaded package located only by a `sys.meta_path` finder | ✅ (error) | `test_fingerprint_gaps.py::test_unloaded_package_found_by_a_meta_path_finder_is_refused` |
| `runpy.run_path` / `runpy.run_module` of a top-level module | ❌ (stated limit) | `test_runtime_import_limits.py::test_d2_runpy_load_of_a_user_file_is_caught_or_listed_as_a_limit` |
| Result cached by 0.5.2 for a dynamic-import stage | ❌ until `repro --force` (stated limit) | `test_runtime_import_limits.py::test_result_cached_by_0_5_2_is_reused_until_a_forced_run_marks_it` |
| Upgrade where the static walk gains a user class method or user module | ✅ (one `Code changed` rerun, then reuse) | `test_runtime_import_limits.py::test_result_cached_by_0_5_2_is_reused_until_a_forced_run_marks_it`; audit upgrade probes |
| Method call on instance (`obj.method()`) | ✅        | `test_change_detection.py::test_method_global_change_causes_miss`, `test_change_detection.py::test_method_regex_change_causes_miss` |
| Callback/function passed as argument     | ❌        | **NO TEST** - not tracked |
| Closure capturing outer variable         | ⚠️        | `test_fingerprint.py::test_fingerprint_with_nonlocal` (value captured, not source) |

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
| Lambda function                           | ✅        | `test_fingerprint.py::test_lambda_function_fingerprinted`; without source, its `__code__` is hashed |
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
| `eval()`/`exec()` limitation                     | Low      | Easy                          |
| Third-party package version tracking             | Medium   | Hard (would need new feature) |
| Type alias change with `from __future__`         | Low      | Hard (alias only in string annotation, not in bytecode) |

---

## Design Decisions

1. **Dunder name filtering**: Names starting with `__` are skipped to filter `__name__`, `__file__`, `__doc__`, etc. Single-underscore names like `_private_helper()` are tracked normally.

2. **Stdlib/third-party marked "callable"**: These are not hashed because we don't want package version changes to trigger rebuilds (too sensitive).

3. **Function name normalization**: Function names are normalized to `"func"` in AST to allow renaming without cache miss.

4. **Docstrings/comments ignored**: These don't affect behavior, so they're stripped from AST.

5. **Imports inside function bodies are walked**: each `import` and `from ... import` statement in a user function's body is resolved (a relative one against the function's package) and what it binds is walked as a module-level import would be. A call `x.import_module(...)` is refused, as on main; other dynamic spellings are left to the worker's stage-import check.

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

17. **Pivot framework excluded from `is_user_code`**: `fingerprint.is_library_path` decides what is library code rather than user code: a path under the interpreter's prefix (the stdlib, or a virtualenv), in a `site-packages` or `dist-packages` directory, or in one of pivot's own packages. `pivot`'s directory is the one `fingerprint` itself is loaded from. `pivot_tui`'s are the directories the installed pivot-tui distribution records that hold it: its site directory, and each directory a path line of its `.pth` file adds to `sys.path`. They never come from looking the name up on `sys.path`, so a user's own module or package named `pivot_tui` is user code, and an editable install of pivot-tui that records neither (a `.pth` file that installs an import hook or adds a link tree) leaves `pivot_tui` user code. This keeps framework internals (e.g., `StageParams`, `stage`, `Pipeline`, the TUI) out of manifests under an editable install, where neither package is in `site-packages`. A package installed from a source checkout outside the interpreter's prefix is user code. A namespace package is library code when one of its directories is in `site-packages` or `dist-packages` or in one of pivot's own packages; a directory under the interpreter's prefix alone, as pip's checkout of a `pip install -e git+...` package in `<venv>/src`, leaves a namespace the user shares with it user code.

    Tests: `test_fingerprint.py::test_is_user_code_pivot_is_framework`, `test_fingerprint.py::test_pivot_tui_installed_from_a_checkout_is_framework`, `test_fingerprint.py::test_pivot_tui_is_pivot_s_package_only_where_the_pivot_tui_distribution_installed_it`, `test_fingerprint.py::test_pivot_tui_pth_file_python_cannot_read_adds_no_directory`, `test_fingerprint.py::test_pivot_tui_pth_file_that_is_not_utf_8_is_read_in_the_locale_s_encoding`, `test_fingerprint.py::test_is_user_code_namespace_package_in_a_library_directory`, `test_fingerprint.py::test_is_user_code_namespace_package_split_with_a_directory_under_the_prefix`, `cli/test_cli_acceptance.py::test_editing_the_user_s_part_of_a_namespace_shared_with_a_pip_checkout_reruns_its_stage`

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
