# Pivot Testing Strategy

**Directory:** `/workspaces/treeverse/pivot/tests/`
**Updated:** 2026-01-04
**Framework:** pytest
**Coverage Target:** >90%

---

## Testing Philosophy

### Test-Driven Development (TDD)

1. **Write test first** - Define expected behavior
2. **Run test (should fail)** - Verify test catches the problem
3. **Implement minimal code** - Make test pass
4. **Refactor** - Improve code while keeping tests green
5. **Repeat** - For each feature/bug fix

### Test Organization

```
tests/
├── CLAUDE.md                     # This file
├── conftest.py                   # Shared fixtures
├── test_<module>.py              # Unit tests (mirror src/)
│   ├── test_fingerprint.py
│   ├── test_ast_utils.py
│   ├── test_registry.py
│   └── ... (one per source module)
│
├── integration/                  # End-to-end tests
│   ├── test_real_pipeline.py
│   ├── test_change_detection.py
│   ├── test_dvc_compatibility.py
│   └── test_parallel_performance.py
│
└── fixtures/                     # Test data
    ├── sample_pipelines/
    ├── sample_data/
    └── expected_outputs/
```

---

## Unit Test Structure

### File Naming Convention

- Test file: `test_<module>.py`
- Test function: `test_<behavior>`
- **NO test classes** - Use flat function structure

### Test Writing Standards

**1. Flat Structure (No Test Classes)**

```python
# Good
def test_simple_function_fingerprinted():
    ...

# Bad - Don't use classes
class TestGetStageFingerprint:
    def test_simple_function_fingerprinted(self):
        ...
```

**2. No Skip Markers**

- All tests should run (fail initially in TDD)
- Remove `@pytest.mark.skip` markers
- If a test isn't ready, don't write it yet

**3. Global Imports Only**

```python
# Good - imports at top of file
import math
from pivot import fingerprint

# Good - module-level helper (CRITICAL for fingerprinting tests!)
def _helper_uses_math():
    return math.pi * 2.0

def test_use_math():
    result = fingerprint.get_stage_fingerprint(_helper_uses_math)
    assert "mod:math.pi" in result

# Bad - imports inside test functions
def test_use_math():
    import math  # Don't do this
    ...
```

**4. No Module Docstrings in Test Functions**

```python
# Good
def test_simple_function():
    """Test simple function with no dependencies."""
    ...

# Bad - No docstring at top of function body
def test_simple_function():
    """Test simple function with no dependencies."""

    """This is the test implementation."""  # Don't do this
    ...
```

**5. Assertion Messages Instead of Inline Comments**
Use assertion error messages to explain what's being tested instead of inline comments:

```python
# Good - assertion messages explain the check
def test_stage_captures_dependencies():
    fp = fingerprint.get_stage_fingerprint(my_func)

    assert "func:helper" in fp, "Should capture helper function dependency"
    assert "mod:math.pi" in fp, "Should capture math.pi module attribute"

# Bad - inline comments that should be assertion messages
def test_stage_captures_dependencies():
    fp = fingerprint.get_stage_fingerprint(my_func)

    # Should capture helper function
    assert "func:helper" in fp

    # Should capture module attributes
    assert "mod:math.pi" in fp
```

**Why:** Assertion messages appear in test failure output, making debugging easier. Inline comments don't help when a test fails.

**6. Use Parametrization to Avoid Repetition**

```python
# Good - parametrized
@pytest.mark.parametrize(
    "obj,expected_is_user_code",
    [
        (len, False),           # builtin
        (print, False),         # builtin
        (lambda x: x, True),    # local lambda
    ],
)
def test_is_user_code(obj, expected_is_user_code):
    result = fingerprint.is_user_code(obj)
    assert result == expected_is_user_code

# Bad - repetitive tests
def test_len_not_user_code():
    assert not fingerprint.is_user_code(len)

def test_print_not_user_code():
    assert not fingerprint.is_user_code(print)

def test_lambda_is_user_code():
    assert fingerprint.is_user_code(lambda x: x)
```

**6. Module-Level Helpers for Fingerprinting Tests (CRITICAL)**

**Why:** Inline function definitions inside test functions do NOT properly capture module-level imports in their closures. This causes fingerprinting tests to fail because `inspect.getclosurevars()` won't see the imported modules.

```python
# Good - module-level helpers at top of file
import math

def _helper_uses_math():
    """Helper that uses math.pi."""
    return math.pi * 2.0

def _helper_uses_constant():
    """Helper that uses a constant."""
    MY_CONSTANT = 100  # Module-level constant
    return MY_CONSTANT * 2

def test_math_attr_detected():
    """Should detect math.pi usage."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_math)
    assert "mod:math.pi" in fp  # ✓ Works!

# Bad - inline function definition
def test_math_attr_detected():
    """Should detect math.pi usage."""
    def uses_math():  # ❌ Won't capture math in closure!
        return math.pi * 2.0

    fp = fingerprint.get_stage_fingerprint(uses_math)
    assert "mod:math.pi" in fp  # ✗ Fails! math not in closure
```

**Naming Convention:**

- Module-level helpers: `_helper_<description>`
- Module-level test constants: `_TEST_<NAME>`

**Example Organization:**

```python
# tests/test_fingerprint.py

import math
import os
from pivot import fingerprint

# --- Module-level helper functions ---
# These capture imports properly in their closures

def _helper_uses_math_pi():
    """Helper that uses math.pi."""
    return math.pi * 2.0

def _helper_uses_multiple_modules():
    """Helper that uses multiple modules."""
    return len(os.path.join("a", "b")) + int(math.pi)

# --- Module-level test constants ---

_TEST_STRING = "Hello, World!"
_TEST_INT = 42

def _helper_uses_constants():
    """Helper that uses test constants."""
    return _TEST_STRING + str(_TEST_INT)

# --- Test functions ---

def test_module_attr_usage():
    """Should detect module attribute usage."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_math_pi)
    assert "mod:math.pi" in fp
```

**7. Explicit Parametrization (No Conditional Logic in Test Body)**

```python
# Good - explicit expected result in parameters
@pytest.mark.parametrize(
    "func1_code,func2_code,should_match",
    [
        ("return 42", "return 42", True),      # identical
        ("return 42", "return 43", False),     # different logic
    ],
)
def test_hash_matching(func1_code, func2_code, should_match):
    func1 = eval(f"lambda: {func1_code}")
    func2 = eval(f"lambda: {func2_code}")

    h1 = fingerprint.hash_function_ast(func1)
    h2 = fingerprint.hash_function_ast(func2)

    if should_match:
        assert h1 == h2
    else:
        assert h1 != h2

# Better - separate tests or explicit comparison
@pytest.mark.parametrize(
    "code1,code2",
    [
        ("return 42", "return 42"),           # identical
        ("return 42", "return 42  # comment"), # whitespace
    ],
)
def test_identical_functions_same_hash(code1, code2):
    func1 = eval(f"lambda: {code1}")
    func2 = eval(f"lambda: {code2}")
    assert fingerprint.hash_function_ast(func1) == fingerprint.hash_function_ast(func2)

@pytest.mark.parametrize(
    "code1,code2",
    [
        ("return 42", "return 43"),
        ("x * 2", "x * 3"),
    ],
)
def test_different_functions_different_hash(code1, code2):
    func1 = eval(f"lambda: {code1}")
    func2 = eval(f"lambda: {code2}")
    assert fingerprint.hash_function_ast(func1) != fingerprint.hash_function_ast(func2)
```

### Example Structure

```python
# tests/test_fingerprint.py
import math

import pytest

from pivot import fingerprint


def test_simple_function_fingerprinted():
    """Should hash simple function with no dependencies."""
    def simple():
        return 42

    fp = fingerprint.get_stage_fingerprint(simple)

    assert "self:simple" in fp
    assert len(fp) == 1


def test_helper_function_captured():
    """Should capture referenced helper function in manifest."""
    def helper(x):
        return x * 2

    def main(x):
        return helper(x) + 1

    fp = fingerprint.get_stage_fingerprint(main)

    assert "self:main" in fp
    assert "func:helper" in fp


@pytest.mark.parametrize(
    "obj,expected",
    [
        (len, False),      # builtin function
        (print, False),    # builtin function
        (int, False),      # builtin type
        (None, False),     # None
    ],
)
def test_builtin_not_user_code(obj, expected):
    """Should identify builtins as not user code."""
    assert fingerprint.is_user_code(obj) == expected


def test_module_attr_usage_detected():
    """Should detect module.attr patterns via AST scan."""
    def use_math():
            return math.pi * 2

        fp = get_stage_fingerprint(use_math)

        assert "mod:math.pi" in fp

    def test_transitive_dependencies_captured(self):
        """Should recursively fingerprint helper functions."""
        def leaf(x):
            return x + 1

        def middle(x):
            return leaf(x) * 2

        def top(x):
            return middle(x) + 10

        fp = get_stage_fingerprint(top)

        assert "self:top" in fp
        assert "func:middle" in fp
        assert "func:leaf" in fp

    def test_unchanged_function_same_fingerprint(self):
        """Should produce identical fingerprint for unchanged function."""
        def func():
            return 42

        fp1 = get_stage_fingerprint(func)
        fp2 = get_stage_fingerprint(func)

        assert fp1 == fp2

    def test_changed_function_different_fingerprint(self):
        """Should produce different fingerprint when function changes."""
        def func_v1():
            return 42

        def func_v2():
            return 43  # Changed!

        fp1 = get_stage_fingerprint(func_v1)
        fp2 = get_stage_fingerprint(func_v2)

        assert fp1 != fp2


class TestIsUserCode:
    """Tests for is_user_code function."""

    def test_stdlib_not_user_code(self):
        """Should identify stdlib modules as not user code."""
        import os
        assert not is_user_code(os.path.join)

    def test_site_packages_not_user_code(self):
        """Should identify site-packages as not user code."""
        import numpy as np
        assert not is_user_code(np.array)

    def test_local_function_is_user_code(self):
        """Should identify functions in project as user code."""
        def local_func():
            pass
        assert is_user_code(local_func)
```

---

## Integration Test Strategy

### test_real_pipeline.py

**Purpose:** Validate end-to-end pipeline execution

```python
def test_simple_ml_pipeline_runs(tmp_path):
    """Complete ML pipeline: load → preprocess → train."""
    # Setup
    # - Create sample data file
    # - Define 3 stages with @stage decorator
    # - Run pipeline

    # Assert
    # - All stages complete
    # - Output files exist
    # - Re-run skips unchanged stages
    # - Changing input re-runs downstream
```

### test_change_detection.py

**Purpose:** Validate automatic code change detection

**Test Scenarios:**

```python
def test_helper_function_change_detected()
def test_constant_change_detected()
def test_transitive_dependency_change_detected()
def test_unused_function_change_ignored()
def test_module_attr_change_detected()
def test_aliased_function_change_detected()
```

### test_dvc_compatibility.py

**Purpose:** Ensure Pivot behavior matches DVC

```python
def test_pivot_and_dvc_produce_identical_outputs():
    """Export to dvc.yaml, run both, compare outputs."""
    # 1. Define pipeline in Pivot
    # 2. Run Pivot, collect output hashes
    # 3. Export to dvc.yaml
    # 4. Run DVC repro
    # 5. Compare output file hashes
    assert pivot_hashes == dvc_hashes

def test_pivot_faster_than_dvc():
    """Benchmark on identical pipeline."""
    # Create 50+ stage pipeline
    # Time Pivot execution
    # Export and time DVC execution
    assert pivot_time < dvc_time * 0.7  # At least 30% faster
```

### test_parallel_performance.py

**Purpose:** Validate parallel speedup

```python
def test_parallel_faster_than_sequential():
    """Independent stages should benefit from parallelism."""
    # Create 10 independent stages (1s each)
    # Run sequential: ~10s
    # Run parallel (4 workers): ~3s
    assert parallel_time < sequential_time / 2.5
```

---

## Fixtures

### conftest.py - Shared Fixtures

```python
import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_pipeline_dir():
    """Temporary directory for pipeline tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_file(tmp_pipeline_dir):
    """Create sample CSV data file."""
    data_file = tmp_pipeline_dir / "data.csv"
    data_file.write_text("id,value\n1,10\n2,20\n3,30\n")
    return data_file


@pytest.fixture
def clean_registry():
    """Reset stage registry before each test."""
    from pivot.registry import REGISTRY
    original = REGISTRY._stages.copy()
    REGISTRY._stages.clear()
    yield REGISTRY
    REGISTRY._stages = original
```

---

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### Unit Tests Only

```bash
pytest tests/test_*.py -v
```

### Integration Tests Only

```bash
pytest tests/integration/ -v
```

### With Coverage

```bash
pytest tests/ --cov=src/pivot --cov-report=html --cov-report=term
```

### Specific Test

```bash
pytest tests/test_fingerprint.py::TestGetStageFingerprint::test_helper_function_captured -v
```

### Watch Mode (Re-run on Changes)

```bash
pytest-watch tests/ -- -v
```

---

## Test Quality Standards

### Good Test Characteristics

1. **Fast** - Unit tests < 100ms, integration < 5s
2. **Independent** - Can run in any order
3. **Repeatable** - Same result every time
4. **Self-validating** - Clear pass/fail
5. **Timely** - Written before/with implementation

### Test Naming

```python
# Good: Describes behavior
def test_helper_function_change_triggers_rerun():
    ...

# Bad: Generic or unclear
def test_fingerprint():
    ...
```

### Assertions

```python
# Good: Specific assertion with message
assert stage_result.status == StageStatus.COMPLETED, \
    f"Expected COMPLETED, got {stage_result.status}"

# Good: Multiple specific assertions
assert "self:train" in fingerprint
assert "func:helper" in fingerprint
assert len(fingerprint) == 2

# Bad: Generic assertion
assert fingerprint  # What are we checking?
```

### Test Data

```python
# Good: Inline test data for clarity
def test_parse_stage_info():
    stage_data = {
        'name': 'train',
        'deps': ['data.csv'],
        'outs': ['model.pkl']
    }
    result = parse_stage(stage_data)
    assert result.name == 'train'

# Bad: Hidden test data in fixtures (for simple cases)
def test_parse_stage_info(complex_stage_fixture):
    # Where is the test data defined?
    result = parse_stage(complex_stage_fixture)
    ...
```

---

## Coverage Goals

### Minimum Coverage: 90%

**Priority Files (Must Have 100% Coverage):**

- `fingerprint.py` - Critical for correctness
- `lock.py` - Critical for correctness
- `dag.py` - Critical for correctness
- `scheduler.py` - Critical for correctness

**Acceptable Lower Coverage (<90%):**

- `cli.py` - UI code harder to test (80% acceptable)
- `explain.py` - Output formatting (85% acceptable)

### Measuring Coverage

```bash
pytest --cov=src/pivot --cov-report=term-missing
```

### Coverage Reports

- HTML: `htmlcov/index.html` (detailed line-by-line)
- Terminal: Shows missing lines immediately

---

## Continuous Integration (Future)

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
- pytest tests/test_*.py  # Unit tests only (fast)
- ruff check src/ tests/
- black --check src/ tests/
- mypy src/
```

### CI Pipeline (GitHub Actions)

```yaml
# Run on every PR
- pytest tests/ --cov=src/pivot --cov-fail-under=90
- ruff check src/ tests/
- black --check src/ tests/
- mypy src/ --strict
```

---

## Debugging Failed Tests

### Useful pytest Flags

```bash
# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Enter debugger on failure
pytest --pdb

# Verbose output
pytest -vv

# Show local variables on failure
pytest -l
```

### Common Issues

**Issue:** Test fails only when run with other tests

- **Cause:** Shared state (registry, filesystem)
- **Fix:** Use fixtures to isolate state

**Issue:** Test passes locally, fails in CI

- **Cause:** Timing issues, different Python version, missing dependencies
- **Fix:** Pin Python version, check CI logs, add retries for flaky tests

**Issue:** Integration test too slow

- **Cause:** Not using tmp_path fixture, large data files
- **Fix:** Use tmp_path, reduce test data size

---

## Next Steps

1. **Week 1:** Write tests for fingerprint.py, ast_utils.py, registry.py
2. **Run tests continuously** - pytest-watch during development
3. **Maintain coverage** - Check coverage after each module
4. **Review tests** - Tests are documentation too!

**Remember:** Tests are production code. Keep them clean, readable, and well-documented.
