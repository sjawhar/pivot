# Pivot - Testing Rules

**Framework:** pytest | **Coverage Target:** 90%+

---

## Test Structure

- **NEVER use `class Test*` test classes.** All tests must be flat `def test_*` functions at module level—use comment separators to group related tests if needed.
- No `@pytest.mark.skip` markers; if test isn't ready, don't write it yet.
- File naming: `test_<module>.py`; function naming: `test_<behavior>`.

## Imports (Critical)

- Import modules not functions; use qualified names.
- **NEVER import inside test functions**—all imports must be at module level.
- **NO LAZY IMPORTS.** This includes `from x import y` inside functions, even for "local" imports. Move ALL imports to the top of the file.

```python
# Bad - import inside test function (lazy import)
def test_queue_writer_fileno():
    import io  # WRONG - move to module level
    with pytest.raises(io.UnsupportedOperation):
        writer.fileno()

# Bad - lazy import of project module
def test_init_creates_valid_project():
    from pivot import project  # WRONG - this is a lazy import!
    project._project_root_cache = None

# Good - import at module level
import io  # At top of file with other imports

from pivot import project  # At top of file

def test_queue_writer_fileno():
    with pytest.raises(io.UnsupportedOperation):
        writer.fileno()

def test_init_creates_valid_project():
    project._project_root_cache = None  # Module already imported at top
```

## Test the Library, Not Duplicates (Critical)

**NEVER duplicate library code in test files.** Tests must import and use actual library functions.

```python
# WRONG - duplicate implementation that doesn't test the library!
def get_recursive_fingerprint(func):  # Don't do this!
    # ... reimplementation of library code ...
    pass

def test_fingerprint():
    result = get_recursive_fingerprint(my_func)  # Testing duplicate, not library!

# CORRECT - import and test the actual library
from pivot import fingerprint

def test_fingerprint():
    result = fingerprint.get_stage_fingerprint(my_func)  # Tests real code
```

If you find yourself writing helper functions that duplicate library functionality, stop and import from the library instead.

## Fingerprint Tests (Critical)

All fingerprint-related tests are in the `tests/fingerprint/` directory.

**Before modifying fingerprinting behavior, consult `tests/fingerprint/README.md`** which contains:

- Complete change detection matrix (what is/isn't detected)
- Test coverage for each behavior
- Known limitations and design decisions

**When adding or modifying fingerprint tests, ALWAYS update `tests/fingerprint/README.md`** to keep the change detection matrix in sync. This includes:

- Adding new test references when adding tests
- Updating "NO TEST" entries when gaps are filled
- Documenting new limitations or behaviors discovered

## Module-Level Helpers (Critical for Fingerprinting)

- Inline functions inside tests do NOT capture module imports in closures.
- Define helpers at module level with `_helper_` prefix.

```python
# Good - module level
import math

def _helper_uses_math():
    return math.pi

def test_it():
    fp = fingerprint.get_stage_fingerprint(_helper_uses_math)
    assert "mod:math.pi" in fp  # Works!

# Bad - inline (FAILS!)
def test_it():
    def uses_math():
        return math.pi
    fp = fingerprint.get_stage_fingerprint(uses_math)
    assert "mod:math.pi" in fp  # Fails! math not in closure
```

## No Double Docstrings

```python
# Bad
def test_foo():
    """Test description."""

    """This is implementation."""  # Don't do this
```

## Assertion Messages (Not Comments)

```python
# Good - appears in failure output
assert "func:helper" in fp, "Should capture helper function"

# Bad - doesn't help when test fails
# Should capture helper
assert "func:helper" in fp
```

## Parametrization (Critical)

**Always consolidate repetitive tests with `@pytest.mark.parametrize`.**

### Put Test Data in Parameters, Not Logic

```python
# Bad - hidden logic, boolean flags
@pytest.mark.parametrize(
    ("marker", "create_subdirs"),
    [(".git", False), (".git", True)]
)
def test_find_root(tmp_path, marker, create_subdirs):
    if create_subdirs:
        subdir = tmp_path / "src" / "nested"
        subdir.mkdir(parents=True)
        work_dir = subdir
    else:
        work_dir = tmp_path

# Good - explicit paths in data
@pytest.mark.parametrize(
    ("marker", "work_dir"),
    [(".git", "."), (".git", "src/nested")]
)
def test_find_root(tmp_path, marker, work_dir):
    work_dir = tmp_path / work_dir
```

### Use Lists for Complex Setup

```python
# Excellent - directories list describes entire scenario
@pytest.mark.parametrize(
    ("directories", "work_dir", "expected_root"),
    [
        # Find marker from project root
        ([".git"], ".", None),
        # Find marker from subdirectory
        ([".git", "src/nested"], "src/nested", None),
        # Stop at closer marker
        ([".git", "sub/.pivot"], "sub", "sub"),
        # No markers - fallback
        (["no_markers"], "no_markers", "no_markers"),
    ],
)
def test_find_root(tmp_path, directories, work_dir, expected_root):
    for d in directories:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(tmp_path / work_dir):
        root = find_project_root()
        expected = tmp_path if expected_root is None else tmp_path / expected_root
        assert root == expected
```

### When to Consolidate

- **Consolidate:** Same test logic, different input data
- **Separate:** Different behaviors, complex assertions, unique edge cases

## Test Quality

- Fast (<100ms unit, <5s integration), independent, repeatable.
- Descriptive names: `test_helper_function_change_triggers_rerun` not `test_fingerprint`.
- Inline test data for clarity; avoid hidden fixtures for simple cases.

## Coverage Goals

- Minimum 90%; 100% required for critical files: `fingerprint.py`, `lock.py`, `dag.py`, `scheduler.py`.
- CLI/explain acceptable at 80-85%.

## CLI Integration Tests (Critical)

**Every CLI command MUST have an integration test that runs in a real pipeline.** Non-negotiable for new commands.

An integration test must:
- Create real filesystem (use `runner.isolated_filesystem()` or `tmp_path`)
- Write actual files (Python stages, data files, `.git` directory)
- Run actual CLI commands via `runner.invoke()`
- Verify both CLI output AND filesystem state

**Reference Examples:**
- Pipeline execution: `tests/cli/test_cli.py::test_cli_run_prints_results`
- Git revision testing: `tests/cli/test_cli_metrics.py::test_metrics_diff_integration`
- Remote config: `tests/remote/test_cli_remote.py::test_remote_add_creates_config`
- Tracked files: `tests/test_cli_track.py::test_track_file_creates_pvt_and_caches`

**Required Coverage:** Every command needs tests for success paths, error paths, and output formats (`--json`, `--md`). See existing `tests/cli/test_cli_*.py` files for patterns.

## Fixtures (conftest.py)

### Global State Reset (Critical)

The global `conftest.py` has **autouse fixtures** that automatically reset state between tests:

- `clean_registry` - clears `REGISTRY._stages` via `mocker.patch.dict`
- `reset_pivot_state` - resets `project._project_root_cache`, `config._config_cache`, `console._console`

**NEVER:**
- Define duplicate `clean_registry` fixtures in individual test files
- Manually call `REGISTRY.clear()` inside test functions
- Manually assign `project._project_root_cache = None` inside test functions
- Create your own state reset fixtures for these globals

**If you add new global state to the codebase**, update `reset_pivot_state` in `conftest.py` instead of creating a new fixture.

### Other Fixtures

- `tmp_pipeline_dir` - temporary directory for pipeline tests
- `sample_data_file` - create sample CSV
- `set_project_root` - explicitly set project root to `tmp_path` (for tests that need a specific root)
- `git_repo` - create a git repo with commit function `(path, commit_fn)`

## Mocking (Critical)

**Use `mocker` or `monkeypatch` fixtures for overriding state—never manual assignment.**

**ALWAYS use `autospec=True`** with `mocker.patch()` and `mocker.patch.object()` when mocking functions or methods. This ensures mocks have the same signature as the original, catching typos and incorrect arguments at test time.

**Exception:** Don't use `autospec=True` when patching to a literal value (None, {}, etc.). autospec creates a mock with the original's signature, but literals aren't mocks.

```python
# Good - autospec for functions/methods
mocker.patch("module.func", autospec=True, return_value=42)
mocker.patch.object(obj, "method", autospec=True)

# Good - no autospec when patching to a literal value
mocker.patch.object(module, "_cache", None)  # Replacing with None, not a mock

# Bad - autospec with literal value (will fail)
mocker.patch.object(module, "_cache", None, autospec=True)  # Wrong!
```

```python
# Good - mocker automatically restores after test
@pytest.fixture(autouse=True)
def reset_state(mocker: MockerFixture) -> Generator[None]:
    mocker.patch.object(module, "_cache", None)
    yield

# Good - monkeypatch also auto-restores
def test_something(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_cache", {})
    ...

# Bad - manual assignment requires cleanup and can leak between tests
@pytest.fixture(autouse=True)
def reset_state() -> Generator[None]:
    old_value = module._cache
    module._cache = None
    yield
    module._cache = old_value  # Error-prone, can forget to restore
```

### Mock Boundaries, Not Internal Logic (Critical)

**Only mock external boundaries:** filesystem I/O, network calls, time, randomness. Never mock internal functions just to control their return values.

```python
# Bad - circular mock testing (mocks function, asserts mocked return value)
def test_get_stages(mocker):
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=["stage1", "stage2"])
    result = completion._get_stages_full()
    assert set(result) == {"stage1", "stage2"}  # Just testing the mock!

# Good - use real objects, test real behavior
def test_get_stages():
    registry.REGISTRY.register(lambda: None, name="stage1", deps=[], outs=[])
    registry.REGISTRY.register(lambda: None, name="stage2", deps=[], outs=[])
    result = completion._get_stages_full()
    assert set(result) == {"stage1", "stage2"}  # Tests real registration flow
```

**Signs of circular mock testing:**
- Mock returns X, test asserts X is returned
- Mocking the function you're trying to test
- Mock setup mirrors the assertion exactly

**When mocking IS appropriate:**
- External HTTP calls (`requests.get`, `httpx.Client`)
- Filesystem operations in fast unit tests
- Time-dependent code (`time.time`, `datetime.now`)
- Random number generation for determinism

## Test Behavior, Not Implementation (Critical)

**Test what the code does, not how it's built internally.**

### No Private Attribute Access

Never access `_private` attributes in assertions. Use public interfaces or type checks.

```python
# Bad - tests internal implementation details
def test_diff_panel():
    result = DataDiffResult(...)
    panel = DiffSummaryPanel(result)
    assert panel._result == result  # Exposes internal attribute

# Good - tests observable behavior
def test_diff_panel():
    result = DataDiffResult(...)
    panel = DiffSummaryPanel(result)
    assert isinstance(panel, DiffSummaryPanel)  # Verifies construction succeeds
```

**Why?** Private attributes can change without breaking functionality. Tests tied to implementation break during refactoring even when behavior is preserved.

### Test Public Interfaces

```python
# Bad - testing internal state
def test_app_stores_entries():
    app = DataDiffApp(entries, key_cols=None, max_rows=1000)
    assert app._diff_entries == entries
    assert app._key_cols is None

# Good - test through public methods or observable effects
def test_app_initialization():
    app = DataDiffApp(entries, key_cols=None, max_rows=1000)
    # Test that app is usable, not its internal state
    assert isinstance(app, DataDiffApp)
```

## CLI Output Testing (Critical)

**Never parse CLI output with position-based string manipulation.**

```python
# Bad - fragile, breaks if section order or headers change
def test_cli_help_shows_commands(runner):
    result = runner.invoke(cli.cli, ["--help"])
    pipeline_idx = output.find("Pipeline Commands:")
    inspection_idx = output.find("Inspection Commands:")
    pipeline_section = output[pipeline_idx:inspection_idx]
    assert "run" in pipeline_section

# Good - simple containment checks with clear messages
def test_cli_help_shows_commands(runner):
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output, "Should show 'run' command"
    assert "Pipeline Commands:" in result.output
```

**For strict section testing**, use structured output (JSON) instead of parsing human-readable text:

```python
# Best - use --json flag when available
def test_cli_list_stages(runner):
    result = runner.invoke(cli.cli, ["list", "--json"])
    data = json.loads(result.output)
    assert "my_stage" in data["stages"]
```

## Debugging

```bash
pytest -x          # Stop on first failure
pytest -s          # Show print statements
pytest --pdb       # Debugger on failure
pytest -l          # Show local variables
pytest --lf        # Re-run failed tests
```
