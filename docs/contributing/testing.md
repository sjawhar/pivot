# Testing Guide

Guidelines for writing tests in Pivot.

## Philosophy

**Real assurance over passing tests.** Tests prove correctness, not just exercise code paths. A test that mocks internal logic proves nothing - it just confirms mocks return what you told them to.

## Structure

- **No `class Test*`** - Use flat `def test_*` functions; group with comment separators if needed
- **No `@pytest.mark.skip`** - If a test isn't ready, don't write it yet
- **No lazy imports** - All imports at module level, never inside test functions
- **No duplicate library code** - Import and test the real library, don't reimplement helpers

## Naming

- Files: `test_<module>.py`
- Functions: `test_<behavior>` - e.g., `test_helper_function_change_triggers_rerun` not `test_fingerprint`

## Module-Level Helpers

Inline functions inside tests do NOT capture module imports in closures - `getclosurevars()` can't see them. **Always define helpers at module level with `_helper_` prefix:**

```python
# Module level - works
def _helper_uses_math():
    return math.pi

# Inline in test - FAILS fingerprinting
def test_it():
    def uses_math():  # math won't be in closure!
        return math.pi
```

## Parametrization

- **Consolidate repetitive tests** with `@pytest.mark.parametrize`
- **Put data in parameters, not logic** - No if/else in test bodies based on parameters
- **Use `pytest.param(id="...")`** for readable test names
- **Consolidate when:** Same test logic, different input data
- **Keep separate when:** Different behaviors, complex assertions, unique edge cases

## Fixtures

### Global State (autouse)

`conftest.py` has autouse fixtures that reset state between tests: `clean_registry`, `reset_pivot_state`.

**Never** manually reset these in individual tests or create duplicate fixtures.

### usefixtures

When a test needs a fixture's side effect but not its return value:

```python
@pytest.mark.usefixtures("set_project_root")
def test_something(tmp_path: pathlib.Path) -> None:
    ...
```

### Available Fixtures

- `tmp_pipeline_dir` - Temporary directory for pipeline tests
- `sample_data_file` - Create sample CSV
- `set_project_root` - Set project root to `tmp_path`
- `git_repo` - Create a git repo with commit function

## Mocking

- **Use `mocker` or `monkeypatch`** - Never manual assignment (they auto-restore)
- **Always `autospec=True`** when mocking functions/methods (catches signature mismatches)
- **Exception:** No autospec when patching to a literal value (None, {}, etc.)

### Mock Boundaries Only

Mock external boundaries (network, filesystem in unit tests, time, randomness). Never mock internal functions to control return values.

**Signs of circular mock testing:**

- Mock returns X, test asserts X is returned
- Mocking the function you're trying to test
- Mock setup mirrors the assertion exactly

## Test Behavior, Not Implementation

- **No private attribute access** in assertions - Use public interfaces
- **No position-based CLI output parsing** - Use simple containment checks or `--json` output

## Assertions

Use assertion messages that appear in failure output:

```python
assert x, "Should have y"  # Good - message in failure output

# Don't use inline comments
assert x  # Should have y  # Bad - comment not shown
```

## CLI Integration Tests

Every CLI command needs an integration test that:

1. Creates real filesystem (`tmp_path` or `runner.isolated_filesystem()`)
2. Writes actual files (Python stages, data, `.git`)
3. Runs actual CLI via `runner.invoke()`
4. Verifies both output AND filesystem state

Required test cases:

- Success paths
- Error paths
- Output formats (`--json`, `--md`)

## Coverage

- Minimum: 90%
- Critical files (100%): `fingerprint.py`, `lock.py`, `dag.py`, `scheduler.py`
- CLI/explain: 80-85% acceptable

## Fingerprint Tests

All fingerprint tests live in `tests/fingerprint/`. Before modifying fingerprinting behavior, consult `tests/fingerprint/README.md` for the change detection matrix. Update it when adding tests.

## Debugging

```bash
pytest -x       # Stop on first failure
pytest -s       # Show print statements
pytest --pdb    # Debugger on failure
pytest --lf     # Re-run failed tests
```

## Cross-Process Tests

When testing multiprocessing behavior, use file-based state instead of shared memory:

```python
# Bad - shared mutable state silently fails in multiprocessing
execution_log = list[str]()

def my_stage():
    execution_log.append("ran")  # Each process has its own copy!

# Good - file-based logging for cross-process communication
def my_stage():
    with open("log.txt", "a") as f:
        f.write("ran\n")
```

## TypedDict Test Patterns

When tests construct TypedDict instances (like `PipelineReloaded`, `StageCompleted`), they must include all required fields:

```python
# Good - includes all required fields
event: types.PipelineReloaded = {
    "type": "pipeline_reloaded",
    "stages": ["stage_a", "stage_b"],  # Required field
    "stages_added": ["stage_b"],
    "stages_removed": [],
    "stages_modified": [],
    "error": None,
}

# Bad - missing required "stages" field (basedpyright will catch this)
event: types.PipelineReloaded = {
    "type": "pipeline_reloaded",
    "stages_added": ["stage_b"],
    ...
}
```

**When adding required fields to TypedDicts:**

1. Search for all test usages: `rg "TypedDictName" tests/`
2. Update all instances with the new field
3. Run `uv run basedpyright .` locally before pushing

## See Also

- [Code Style](style.md) - Coding conventions
- [Common Gotchas](gotchas.md) - Pitfalls to avoid
