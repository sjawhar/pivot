# Solutions

Documented learnings from bugs, design decisions, and implementation patterns. Check here before implementing features touching these areas.

## Patterns

| Document | Summary |
|----------|---------|
| [Critical Patterns](patterns/critical-patterns.md) | Required-reading patterns from repeated bugs across modules |

## Type System & Generics

| Document | Date | Summary |
|----------|------|---------|
| [Dataclass Inheritance Field Ordering](2026-01-31-dataclass-inheritance-field-ordering.md) | 2026-01-31 | Non-default fields cannot follow fields with defaults |
| [Dict Invariance vs Protocol Covariance](2026-01-31-dict-invariance-protocol-typing.md) | 2026-01-31 | `dict` invariance vs `Protocol` covariance in the type system |
| [Dict Invariance with Class Hierarchies](2026-01-31-dict-invariance-typing.md) | 2026-01-31 | `dict[str, BaseClass]` not assignable to `dict[str, SubClass]` |
| [DirectoryOut Generic Type Mismatch](2026-01-31-directoryout-type-consistency.md) | 2026-01-31 | Generic type parameter mismatch in DirectoryOut |
| [DirectoryOut Inheritance Creates Type Lie](2026-01-31-directoryout-type-lie.md) | 2026-01-31 | Loader type doesn't match Out generic after inheritance |
| [isinstance Checks After Hierarchy Refactor](2026-01-31-isinstance-checks-after-hierarchy-refactor.md) | 2026-01-31 | Update isinstance checks when splitting inheritance hierarchies |
| [isinstance Sibling Class Gotcha](2026-01-31-isinstance-sibling-class-gotcha.md) | 2026-01-31 | isinstance checks break when refactoring subclass to sibling |
| [PEP 696 Type Defaults with ABC Runtime](2026-01-31-pep696-type-defaults-abc-runtime.md) | 2026-01-31 | Type parameter defaults don't work with ABC at runtime |
| [PEP 696 Type Parameter Defaults](2026-01-31-pep696-type-parameter-defaults.md) | 2026-01-31 | Asymmetric generics with type parameter defaults |
| [List/Dict Invariance with Protocols](2026-01-31-python-list-dict-invariance-with-protocols.md) | 2026-01-31 | `list[Subclass]` not assignable to `list[BaseClass]` |
| [Type Narrowing with isinstance](2026-01-31-type-narrowing-with-isinstance.md) | 2026-01-31 | Type narrowing for mixed base classes |

## Fingerprinting & Code Detection

| Document | Date | Summary |
|----------|------|---------|
| [MRO Method Override Detection](2026-01-31-mro-method-override-detection.md) | 2026-01-31 | MRO-based method override detection for fingerprinting |
| [AST Function Bodies Need Statement](2026-02-01-ast-function-bodies-need-statement.md) | 2026-02-01 | Function bodies require at least one statement in AST manipulation |
| [Lambda Fingerprinting Is Non-Deterministic](2026-02-01-lambda-fingerprinting-non-deterministic.md) | 2026-02-01 | Lambdas cause fingerprint changes without code changes |
| [Single Underscore Functions ARE Tracked](2026-02-01-single-underscore-functions-are-tracked.md) | 2026-02-01 | `_private` functions are included in fingerprints |
| [Test Helpers Must Be Module-Level](2026-02-01-test-helpers-must-be-module-level.md) | 2026-02-01 | Fingerprinting requires module-level test helpers |
| [Skip Detection Invariants](2026-02-08-skip-detection-invariants.md) | 2026-02-08 | Three-tier skip detection algorithm invariants |

## Execution & Workers (loky)

| Document | Date | Summary |
|----------|------|---------|
| [loky Cannot Pickle mp.Queue](2026-02-01-loky-cannot-pickle-mp-queue.md) | 2026-02-01 | Use Manager queues instead of mp.Queue with loky |
| [loky cpu_count Respects cgroups](2026-02-01-loky-cpu-count-respects-cgroups.md) | 2026-02-01 | Container-aware worker limits via loky.cpu_count() |
| [loky Reusable Executor Warm Workers](2026-02-01-loky-reusable-executor-warm-workers.md) | 2026-02-01 | Warm worker pools with reusable executor |
| [Stage Functions Must Be Module-Level](2026-02-01-stage-functions-must-be-module-level.md) | 2026-02-01 | Pickling requires module-level functions and TypedDicts |

## Storage & State

| Document | Date | Summary |
|----------|------|---------|
| [Atomic Writes: Track FD Closure](2026-02-01-atomic-writes-track-fd-closure.md) | 2026-02-01 | File descriptor leak with mkstemp() atomic writes |
| [LMDB: Extend StateDB with Prefixes](2026-02-01-lmdb-extend-with-prefixes.md) | 2026-02-01 | Consolidate state into LMDB with key prefixes |
| [StateDB Path Strategies](2026-02-01-statedb-path-strategies.md) | 2026-02-01 | `resolve()` for hashes, `normpath()` for generations |
| [IncrementalOut Uses COPY Mode](2026-02-01-incremental-out-uses-copy-mode.md) | 2026-02-01 | COPY mode prevents cache corruption for incremental outputs |

## Architecture & Design

| Document | Date | Summary |
|----------|------|---------|
| [Circular Imports: Extract Shared Types](2026-02-01-circular-imports-extract-shared-types.md) | 2026-02-01 | Break circular imports by extracting types to shared module |
| [Engine Is a Context Manager](2026-02-01-engine-is-context-manager.md) | 2026-02-01 | Engine must be used as context manager for resource cleanup |
| [Path Overlap Detection Use Trie](2026-02-01-path-overlap-detection-use-trie.md) | 2026-02-01 | Trie-based path overlap detection, not string matching |
| [YAML Library Choice: ruamel vs PyYAML](2026-02-01-yaml-library-choice-ruamel-vs-pyyaml.md) | 2026-02-01 | Comment-preserving YAML with ruamel.yaml |
| [Path Resolution Design Patterns](2026-02-02-path-resolution-design-patterns.md) | 2026-02-02 | normpath vs resolve, symlinks, registration vs execution |
| [Engine Dispatcher Drain Race](2026-02-05-engine-dispatcher-drain-race.md) | 2026-02-05 | Drain dispatcher before task group cancellation |

## Testing

| Document | Date | Summary |
|----------|------|---------|
| [Cross-Process Tests: Use File State](2026-02-01-cross-process-tests-use-file-state.md) | 2026-02-01 | File-based state instead of shared lists for cross-process tests |

## Logic Errors

| Document | Date | Summary |
|----------|------|---------|
| [Cascade Failure Redundant Traversal](logic-errors/cascade-failure-redundant-traversal-20260203.md) | 2026-02-03 | Redundant stack-based traversal in `_cascade_failure` |
| [Dry Run Bypasses Resolution](logic-errors/dry-run-bypasses-resolution-20260206.md) | 2026-02-06 | `--dry-run` bypasses external dependency resolution |
| [Exists Check Prevents Re-Resolution](logic-errors/exists-check-prevents-reresolution-20260206.md) | 2026-02-06 | `exists()` check prevents re-resolution of external deps |
| [File Output Verification Asymmetry](logic-errors/file-output-verification-asymmetry-cache-20260131.md) | 2026-01-31 | Output verification only checked existence, not content |
| [Output Index Shared Root](logic-errors/output-index-state-dir-shared-root-20260206.md) | 2026-02-06 | Wrong pipeline directory for shared-root pipelines |

## Runtime Errors

| Document | Date | Summary |
|----------|------|---------|
| [Signal Handler Thread in Textual TUI](runtime-errors/signal-handler-thread-textual-tui-20260203.md) | 2026-02-03 | TUI crashes with signal handler called from wrong thread |

## Integration Issues

| Document | Date | Summary |
|----------|------|---------|
| [Missing E2E Test for CLI Serve Mode](integration-issues/missing-e2e-test-cli-serve-mode-20260201.md) | 2026-02-01 | Components work individually but fail when wired together |
