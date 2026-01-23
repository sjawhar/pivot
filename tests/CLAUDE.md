# Pivot - Testing Rules

**Framework:** pytest | **Coverage Target:** 90%+

---

## Philosophy

**Real assurance over passing tests.** Tests should prove correctness, not just exercise code paths. Prefer real execution over mocks — mock external boundaries (network, filesystem in unit tests), not internal logic.

---

## Structure

- **No `class Test*`** — use flat `def test_*` functions; group with comment separators if needed
- **No `@pytest.mark.skip`** — if a test isn't ready, don't write it yet
- **No lazy imports** — all imports at module level, never inside test functions
- **No duplicate library code** — import and test the real library, don't reimplement helpers

## Naming

- Files: `test_<module>.py`
- Functions: `test_<behavior>` — e.g., `test_helper_function_change_triggers_rerun` not `test_fingerprint`

---

## Module-Level Helpers (Critical for Fingerprinting)

Inline functions inside tests do NOT capture module imports in closures — `getclosurevars()` can't see them. Always define helpers at module level with `_helper_` prefix.

```python
# Module level — works
def _helper_uses_math():
    return math.pi

# Inline in test — FAILS fingerprinting
def test_it():
    def uses_math():  # math won't be in closure!
        return math.pi
```

---

## Parametrization

- **Consolidate repetitive tests** with `@pytest.mark.parametrize`
- **Put data in parameters, not logic** — no if/else in test bodies based on parameters
- **Use `pytest.param(id="...")`** for readable test names instead of cryptic auto-generated IDs
- **Consolidate when:** same test logic, different input data
- **Keep separate when:** different behaviors, complex assertions, unique edge cases

---

## Fixtures

### Global State (autouse)

`conftest.py` has autouse fixtures that reset state between tests: `clean_registry`, `reset_pivot_state`.

**Never** manually reset these in individual tests or create duplicate fixtures.

### usefixtures

When a test needs a fixture's side effect but not its return value, use `@pytest.mark.usefixtures("name")` instead of an unused parameter.

### Available Fixtures

- `tmp_pipeline_dir` — temporary directory for pipeline tests
- `sample_data_file` — create sample CSV
- `set_project_root` — set project root to `tmp_path`
- `git_repo` — create a git repo with commit function

---

## Mocking

- **Use `mocker` or `monkeypatch`** — never manual assignment (they auto-restore)
- **Always `autospec=True`** when mocking functions/methods (catches signature mismatches)
- **Exception:** no autospec when patching to a literal value (None, {}, etc.)

### Mock Boundaries Only

Mock external boundaries (network, filesystem in unit tests, time, randomness). Never mock internal functions to control return values.

**Signs of circular mock testing:**
- Mock returns X, test asserts X is returned
- Mocking the function you're trying to test
- Mock setup mirrors the assertion exactly

---

## Test Behavior, Not Implementation

- **No private attribute access** in assertions — use public interfaces
- **No position-based CLI output parsing** — use simple containment checks or `--json` output

---

## Assertions

Use assertion messages that appear in failure output: `assert x, "Should have y"` — not inline comments.

---

## CLI Integration Tests

Every CLI command needs an integration test that:
1. Creates real filesystem (`tmp_path` or `runner.isolated_filesystem()`)
2. Writes actual files (Python stages, data, `.git`)
3. Runs actual CLI via `runner.invoke()`
4. Verifies both output AND filesystem state

Required: success paths, error paths, output formats (`--json`, `--md`).

---

## Coverage

- Minimum: 90%
- Critical files (100%): `fingerprint.py`, `lock.py`, `dag.py`, `scheduler.py`
- CLI/explain: 80-85% acceptable

---

## Fingerprint Tests

All fingerprint tests live in `tests/fingerprint/`. Before modifying fingerprinting behavior, consult `tests/fingerprint/README.md` for the change detection matrix. Update it when adding tests.

---

## Debugging

```bash
pytest -x       # Stop on first failure
pytest -s       # Show print statements
pytest --pdb    # Debugger on failure
pytest --lf     # Re-run failed tests
```