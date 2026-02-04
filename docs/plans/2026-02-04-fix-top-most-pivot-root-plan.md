---
title: fix: Use top-most .pivot as project root
type: fix
date: 2026-02-04
---

# Use Top-Most .pivot as Project Root - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Pivot resolve project root by selecting the top-most `.pivot/` directory when walking upward from CWD, and error when none exists.

**Architecture:** Change `find_project_root()` to traverse upward remembering all `.pivot/` directories found, return the highest one. Remove `.git` fallback. Add minimal `ProjectNotInitializedError` when no `.pivot/` exists above CWD.

**Tech Stack:** Python 3.13+, pytest, Click CLI

---

## Task 1: Implement Top-Most .pivot Discovery

**Files:**
- Modify: `src/pivot/exceptions.py` (add ~4 lines after line 374)
- Modify: `src/pivot/project.py:10-19`
- Test: `tests/utils/test_project_root.py`

**Step 1: Add minimal exception to `src/pivot/exceptions.py`**

Add after `AlreadyInitializedError` (around line 374):

```python
class ProjectNotInitializedError(InitError):
    """Raised when no .pivot directory exists above the current directory."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot init' in your project root to initialize Pivot"
```

**Step 2: Write the failing tests in `tests/utils/test_project_root.py`**

Add import at top:
```python
from pivot import exceptions, project
```

Replace the existing parametrized test and add new tests:

```python
@pytest.mark.parametrize(
    ("directories", "work_dir", "expected_root", "expect_error"),
    [
        pytest.param([".pivot"], ".", None, False, id="pivot_at_root"),
        pytest.param([".pivot", "src/nested"], "src/nested", None, False, id="pivot_from_subdir"),
        pytest.param([".pivot", "child/.pivot"], "child", None, False, id="topmost_over_nearest"),
        pytest.param(
            [".pivot", "mid/.pivot", "mid/deep/.pivot"],
            "mid/deep",
            None,
            False,
            id="topmost_over_nearest_and_mid",
        ),
        pytest.param([".git"], ".", None, True, id="git_only_raises"),
        pytest.param(["no_markers"], "no_markers", None, True, id="no_markers_raises"),
    ],
)
def test_find_project_root(
    tmp_path: Path,
    directories: list[str],
    work_dir: str,
    expected_root: str | None,
    expect_error: bool,
) -> None:
    """Should find project root by walking up to top-most .pivot directory."""
    for directory in directories:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)

    with contextlib.chdir(tmp_path / work_dir):
        if expect_error:
            with pytest.raises(exceptions.ProjectNotInitializedError):
                project.find_project_root()
        else:
            root = project.find_project_root()
            expected = tmp_path if expected_root is None else tmp_path / expected_root
            assert root == expected


def test_find_project_root_ignores_pivot_file(tmp_path: Path) -> None:
    """Should ignore .pivot that is a file, not a directory."""
    (tmp_path / ".pivot").write_text("not a directory")
    subdir = tmp_path / "project"
    subdir.mkdir()
    (subdir / ".pivot").mkdir()

    with contextlib.chdir(subdir):
        root = project.find_project_root()
        assert root == subdir, "Should use .pivot directory, not file"


def test_find_project_root_follows_pivot_symlink(tmp_path: Path) -> None:
    """Should recognize .pivot even when it's a symlink to a directory."""
    real_pivot = tmp_path / ".real_pivot"
    real_pivot.mkdir()
    (tmp_path / ".pivot").symlink_to(real_pivot)

    with contextlib.chdir(tmp_path):
        root = project.find_project_root()
        assert root == tmp_path


def test_find_project_root_at_filesystem_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should raise error when reaching filesystem root without finding .pivot."""
    monkeypatch.setattr(Path, "cwd", lambda: Path("/"))

    with pytest.raises(exceptions.ProjectNotInitializedError):
        project.find_project_root()
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/utils/test_project_root.py -v -x`
Expected: FAIL (current implementation returns nearest marker and uses .git fallback)

**Step 4: Implement the new `find_project_root()` in `src/pivot/project.py`**

Replace lines 10-19:

```python
def find_project_root() -> pathlib.Path:
    """Walk up from cwd to find the top-most .pivot directory.

    Raises:
        ProjectNotInitializedError: If no .pivot directory exists above cwd.
    """
    from pivot import exceptions

    current = pathlib.Path.cwd().resolve()
    topmost_pivot: pathlib.Path | None = None

    for parent in [current, *current.parents]:
        if (parent / ".pivot").is_dir():
            topmost_pivot = parent

    if topmost_pivot is None:
        raise exceptions.ProjectNotInitializedError(
            f"No .pivot directory found above '{current}'. "
            "Run 'pivot init' to initialize a Pivot project."
        )

    logger.debug(f"Project root: {topmost_pivot}")
    return topmost_pivot
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/utils/test_project_root.py -v`
Expected: PASS

**Step 6: Commit**

```bash
jj describe -m "fix(project): use top-most .pivot as project root, remove .git fallback"
```

---

## Task 2: Fix All Broken Tests

**Files:**
- Modify: `tests/conftest.py:136-143` (update `set_project_root` fixture)
- Modify: Various test files that create `.git` without `.pivot`

**Step 1: Update `set_project_root` fixture in `tests/conftest.py`**

Change lines 136-143:

```python
@pytest.fixture
def set_project_root(tmp_path: pathlib.Path, mocker: MockerFixture) -> Generator[pathlib.Path]:
    """Set project root to tmp_path for tests that register stages with temp paths.

    Creates .pivot directory so project root discovery works.
    Also creates .git for fingerprinting and other git-dependent operations.
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    yield tmp_path
```

**Step 2: Run full test suite to find remaining failures**

Run: `uv run pytest tests/ -x -v --tb=short`

**Step 3: For each failing test, add `.pivot` directory creation**

Pattern to fix:
```python
# Before (creates .git only)
(tmp_path / ".git").mkdir()

# After (add .pivot)
(tmp_path / ".pivot").mkdir()
(tmp_path / ".git").mkdir()  # Keep if git operations needed
```

Common locations to check:
- Tests using `git_repo` fixture may need `.pivot` added
- Tests manually creating `.git` directories

**Step 4: Run full test suite until green**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
jj describe -m "fix(tests): add .pivot markers for project root discovery"
```

---

## Task 3: Add CLI Integration Tests

**Files:**
- Modify: `tests/cli/test_cli_init.py` (add tests to existing file)

**Step 1: Add tests to `tests/cli/test_cli_init.py`**

```python
def test_repro_fails_without_pivot_init(tmp_path: pathlib.Path) -> None:
    """Running pivot repro without .pivot should error with helpful message."""
    runner = click.testing.CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["repro"])

        assert result.exit_code == 1
        assert "No .pivot directory found" in result.output
        assert "pivot init" in result.output


def test_list_fails_without_pivot_init(tmp_path: pathlib.Path) -> None:
    """Running pivot list without .pivot should error with helpful message."""
    runner = click.testing.CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 1
        assert "No .pivot directory found" in result.output
```

**Step 2: Run the new tests**

Run: `uv run pytest tests/cli/test_cli_init.py -v -k "without_pivot"`
Expected: PASS (CLI should propagate ProjectNotInitializedError)

**Step 3: Commit**

```bash
jj describe -m "test(cli): verify commands error without pivot init"
```

---

## Task 4: Final Verification

**Step 1: Run full test suite with coverage**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 2: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: No errors

**Step 3: Manual smoke test**

```bash
cd /tmp
mkdir -p test_pivot/subdir
cd test_pivot/subdir
pivot repro  # Should error: "No .pivot directory found"
cd ..
pivot init
cd subdir
pivot list   # Should work, finding .pivot in parent
cd ..
rm -rf /tmp/test_pivot
```

**Step 4: Final commit**

```bash
jj describe -m "fix(project): use top-most .pivot as project root

BREAKING CHANGE: Repos without .pivot will now error instead of using .git or CWD fallback.

- find_project_root() traverses upward and selects top-most .pivot
- Raises ProjectNotInitializedError when no .pivot exists
- Removes .git fallback - only .pivot directories are recognized
- Uses .is_dir() to ignore .pivot files"
```

---

## Summary

| Task | Description | Est. Changes |
|------|-------------|--------------|
| 1 | Implement top-most .pivot discovery + exception | ~25 lines code, ~50 lines test |
| 2 | Fix all broken tests | Variable |
| 3 | Add CLI integration tests | ~20 lines test |
| 4 | Final verification | 0 lines |

**Key tests added:**
- 3-level nesting (topmost beats nearest AND middle)
- `.pivot` file vs directory (use `.is_dir()`)
- `.pivot` symlink (should work)
- CLI error messages (specific assertions)

**Breaking changes:**
- Repos relying on `.git` as project marker must run `pivot init`
- Repos with nested `.pivot` directories will now use the top-most one
