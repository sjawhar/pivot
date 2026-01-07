# Pivot - Testing Rules

**Framework:** pytest | **Coverage Target:** 90%+

---

## Test Structure

- **NEVER use `class Test*` test classes.** All tests must be flat `def test_*` functions at module level—use comment separators to group related tests if needed.
- No `@pytest.mark.skip` markers; if test isn't ready, don't write it yet.
- File naming: `test_<module>.py`; function naming: `test_<behavior>`.

## Imports

- Import modules not functions; use qualified names.
- No imports inside test functions—import at module level.

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

## Fixtures (conftest.py)

- `clean_registry` - reset registry before each test.
- `tmp_pipeline_dir` - temporary directory for pipeline tests.
- `sample_data_file` - create sample CSV.

## Mocking (Critical)

**Use `mocker` or `monkeypatch` fixtures for overriding state—never manual assignment.**

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

## Debugging

```bash
pytest -x          # Stop on first failure
pytest -s          # Show print statements
pytest --pdb       # Debugger on failure
pytest -l          # Show local variables
pytest --lf        # Re-run failed tests
```
