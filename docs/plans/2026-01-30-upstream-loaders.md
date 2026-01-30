# Upstream Loaders Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three new loaders to pivot from eval-pipeline: `Text`, `JSONL`, and `DataFrameJSONL`.

**Architecture:** Each loader follows the existing `Loader[T]` pattern with frozen dataclasses, atomic writes via temp file + rename, and `empty()` for IncrementalOut support.

**Tech Stack:** Python stdlib (json, tempfile, os, pathlib), pandas for DataFrameJSONL.

---

## Task 1: Text Loader

**Files:**
- Modify: `src/pivot/loaders.py` (add after `YAML` class, ~line 137)
- Modify: `tests/test_loaders.py` (add new test section)

**Step 1: Write the failing tests**

Add to `tests/test_loaders.py` after the YAML loader tests section:

```python
# ==============================================================================
# Text loader tests
# ==============================================================================


def test_text_loader_load(tmp_path: pathlib.Path) -> None:
    """Text loader reads string from file."""
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("hello world\nline two")

    loader = loaders.Text()
    data = loader.load(txt_file)

    assert data == "hello world\nline two"


def test_text_loader_save(tmp_path: pathlib.Path) -> None:
    """Text loader writes string to file."""
    txt_file = tmp_path / "output.txt"
    data = "some text content"

    loader = loaders.Text()
    loader.save(data, txt_file)

    assert txt_file.exists()
    assert txt_file.read_text() == "some text content"


def test_text_loader_save_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """Text loader creates parent directories if needed."""
    txt_file = tmp_path / "nested" / "dir" / "output.txt"

    loader = loaders.Text()
    loader.save("content", txt_file)

    assert txt_file.exists()
    assert txt_file.read_text() == "content"


def test_text_loader_save_type_error() -> None:
    """Text loader raises TypeError for non-string data."""
    loader = loaders.Text()
    with pytest.raises(TypeError, match="Text save expects str"):
        loader.save(123, pathlib.Path("test.txt"))  # type: ignore[arg-type]


def test_text_loader_empty() -> None:
    """Text loader returns empty string for empty()."""
    loader = loaders.Text()
    assert loader.empty() == ""


def test_text_loader_is_picklable() -> None:
    """Text loader can be pickled and unpickled."""
    loader = loaders.Text()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.Text)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sami/pivot/default && uv run pytest tests/test_loaders.py -k "text" -v`
Expected: FAIL with "module 'pivot.loaders' has no attribute 'Text'"

**Step 3: Write the implementation**

Add to `src/pivot/loaders.py` after the `YAML` class (around line 137):

```python
@dataclasses.dataclass(frozen=True)
class Text(Loader[str]):
    """Plain text file loader.

    Saves atomically via temp file + rename to prevent corruption.
    """

    @override
    def load(self, path: pathlib.Path) -> str:
        return path.read_text()

    @override
    def save(self, data: str, path: pathlib.Path) -> None:
        if not isinstance(data, str):
            raise TypeError(f"Text save expects str, got {type(data).__name__}")

        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".txt.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.rename(tmp_path_str, path)
        except Exception:
            tmp_path = pathlib.Path(tmp_path_str)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @override
    def empty(self) -> str:
        return ""
```

Also add imports at the top of the file (after existing imports):

```python
import os
import tempfile
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sami/pivot/default && uv run pytest tests/test_loaders.py -k "text" -v`
Expected: PASS (6 tests)

**Step 5: Run quality checks**

Run: `cd /home/sami/pivot/default && uv run ruff format src/pivot/loaders.py tests/test_loaders.py && uv run ruff check src/pivot/loaders.py tests/test_loaders.py && uv run basedpyright src/pivot/loaders.py tests/test_loaders.py`
Expected: No errors or warnings

**Step 6: Commit**

```bash
cd /home/sami/pivot/default && jj describe -m "feat(loaders): add Text loader for plain text files

- Atomic saves via temp file + rename
- Supports empty() for IncrementalOut"
```

---

## Task 2: JSONL Loader

**Files:**
- Modify: `src/pivot/loaders.py` (add after `Text` class)
- Modify: `tests/test_loaders.py` (add new test section)

**Step 1: Write the failing tests**

Add to `tests/test_loaders.py` after the Text loader tests section:

```python
# ==============================================================================
# JSONL loader tests
# ==============================================================================


def test_jsonl_loader_load(tmp_path: pathlib.Path) -> None:
    """JSONL loader reads list of dicts from file."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"a": 1}\n{"b": 2}\n')

    loader = loaders.JSONL()
    data = loader.load(jsonl_file)

    assert data == [{"a": 1}, {"b": 2}]


def test_jsonl_loader_load_skips_blank_lines(tmp_path: pathlib.Path) -> None:
    """JSONL loader skips blank lines."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"a": 1}\n\n{"b": 2}\n   \n')

    loader = loaders.JSONL()
    data = loader.load(jsonl_file)

    assert data == [{"a": 1}, {"b": 2}]


def test_jsonl_loader_load_invalid_json(tmp_path: pathlib.Path) -> None:
    """JSONL loader reports line number on invalid JSON."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"a": 1}\n{invalid}\n')

    loader = loaders.JSONL()
    with pytest.raises(ValueError, match=r"Invalid JSON at .+:2"):
        loader.load(jsonl_file)


def test_jsonl_loader_save(tmp_path: pathlib.Path) -> None:
    """JSONL loader writes list of dicts to file."""
    jsonl_file = tmp_path / "output.jsonl"
    data = [{"x": 1}, {"y": 2}]

    loader = loaders.JSONL()
    loader.save(data, jsonl_file)

    assert jsonl_file.exists()
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"x": 1}
    assert json.loads(lines[1]) == {"y": 2}


def test_jsonl_loader_save_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """JSONL loader creates parent directories if needed."""
    jsonl_file = tmp_path / "nested" / "output.jsonl"

    loader = loaders.JSONL()
    loader.save([{"a": 1}], jsonl_file)

    assert jsonl_file.exists()


def test_jsonl_loader_save_type_error() -> None:
    """JSONL loader raises TypeError for non-list data."""
    loader = loaders.JSONL()
    with pytest.raises(TypeError, match="JSONL save expects list"):
        loader.save({"a": 1}, pathlib.Path("test.jsonl"))  # type: ignore[arg-type]


def test_jsonl_loader_empty() -> None:
    """JSONL loader returns empty list for empty()."""
    loader = loaders.JSONL()
    assert loader.empty() == []


def test_jsonl_loader_is_picklable() -> None:
    """JSONL loader can be pickled and unpickled."""
    loader = loaders.JSONL()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.JSONL)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sami/pivot/default && uv run pytest tests/test_loaders.py -k "jsonl" -v`
Expected: FAIL with "module 'pivot.loaders' has no attribute 'JSONL'"

**Step 3: Write the implementation**

Add to `src/pivot/loaders.py` after the `Text` class:

```python
@dataclasses.dataclass(frozen=True)
class JSONL(Loader[list[dict[str, Any]]]):
    """JSONL (JSON Lines) file loader - one JSON object per line.

    Saves atomically via temp file + rename. Reports line numbers on parse errors.
    """

    @override
    def load(self, path: pathlib.Path) -> list[dict[str, Any]]:
        results = list[dict[str, Any]]()
        with path.open() as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
        return results

    @override
    def save(self, data: list[dict[str, Any]], path: pathlib.Path) -> None:
        if not isinstance(data, list):
            raise TypeError(f"JSONL save expects list, got {type(data).__name__}")

        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            os.rename(tmp_path_str, path)
        except Exception:
            tmp_path = pathlib.Path(tmp_path_str)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @override
    def empty(self) -> list[dict[str, Any]]:
        return []
```

Also add to TYPE_CHECKING imports at top of file:

```python
from typing import TYPE_CHECKING, Any, Literal, override
```

Wait - `Any` should be imported directly per CLAUDE.md. Check the existing imports and add `Any` to the typing imports if not present.

**Step 4: Run tests to verify they pass**

Run: `cd /home/sami/pivot/default && uv run pytest tests/test_loaders.py -k "jsonl" -v`
Expected: PASS (8 tests)

**Step 5: Run quality checks**

Run: `cd /home/sami/pivot/default && uv run ruff format src/pivot/loaders.py tests/test_loaders.py && uv run ruff check src/pivot/loaders.py tests/test_loaders.py && uv run basedpyright src/pivot/loaders.py tests/test_loaders.py`
Expected: No errors or warnings

**Step 6: Commit**

```bash
cd /home/sami/pivot/default && jj describe -m "feat(loaders): add JSONL loader for JSON Lines files

- Line-by-line parsing with line number error reporting
- Skips blank lines
- Atomic saves via temp file + rename
- Supports empty() for IncrementalOut"
```

---

## Task 3: DataFrameJSONL Loader

**Files:**
- Modify: `src/pivot/loaders.py` (add after `JSONL` class)
- Modify: `tests/test_loaders.py` (add new test section)

**Step 1: Write the failing tests**

Add to `tests/test_loaders.py` after the JSONL loader tests section:

```python
# ==============================================================================
# DataFrameJSONL loader tests
# ==============================================================================


def test_dataframe_jsonl_loader_load(tmp_path: pathlib.Path) -> None:
    """DataFrameJSONL loader reads DataFrame from file."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')

    loader = loaders.DataFrameJSONL()
    df = loader.load(jsonl_file)

    assert isinstance(df, pandas.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
    assert df["a"].tolist() == [1, 3]


def test_dataframe_jsonl_loader_save(tmp_path: pathlib.Path) -> None:
    """DataFrameJSONL loader writes DataFrame to file."""
    jsonl_file = tmp_path / "output.jsonl"
    df = pandas.DataFrame({"x": [1, 2], "y": [3, 4]})

    loader = loaders.DataFrameJSONL()
    loader.save(df, jsonl_file)

    assert jsonl_file.exists()
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 2


def test_dataframe_jsonl_loader_roundtrip(tmp_path: pathlib.Path) -> None:
    """DataFrameJSONL loader roundtrips data correctly."""
    jsonl_file = tmp_path / "data.jsonl"
    df = pandas.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    loader = loaders.DataFrameJSONL()
    loader.save(df, jsonl_file)
    loaded = loader.load(jsonl_file)

    pandas.testing.assert_frame_equal(df, loaded)


def test_dataframe_jsonl_loader_save_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """DataFrameJSONL loader creates parent directories if needed."""
    jsonl_file = tmp_path / "nested" / "output.jsonl"
    df = pandas.DataFrame({"a": [1]})

    loader = loaders.DataFrameJSONL()
    loader.save(df, jsonl_file)

    assert jsonl_file.exists()


def test_dataframe_jsonl_loader_save_type_error() -> None:
    """DataFrameJSONL loader raises TypeError for non-DataFrame data."""
    loader = loaders.DataFrameJSONL()
    with pytest.raises(TypeError, match="DataFrameJSONL save expects DataFrame"):
        loader.save([{"a": 1}], pathlib.Path("test.jsonl"))  # type: ignore[arg-type]


def test_dataframe_jsonl_loader_empty() -> None:
    """DataFrameJSONL loader returns empty DataFrame for empty()."""
    loader = loaders.DataFrameJSONL()
    result = loader.empty()
    assert isinstance(result, pandas.DataFrame)
    assert len(result) == 0


def test_dataframe_jsonl_loader_is_picklable() -> None:
    """DataFrameJSONL loader can be pickled and unpickled."""
    loader = loaders.DataFrameJSONL()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.DataFrameJSONL)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sami/pivot/default && uv run pytest tests/test_loaders.py -k "dataframe_jsonl" -v`
Expected: FAIL with "module 'pivot.loaders' has no attribute 'DataFrameJSONL'"

**Step 3: Write the implementation**

Add to `src/pivot/loaders.py` after the `JSONL` class:

```python
@dataclasses.dataclass(frozen=True)
class DataFrameJSONL(Loader[pandas.DataFrame]):
    """JSONL (JSON Lines) file loader that returns a pandas DataFrame.

    Uses pandas.read_json with lines=True for efficient loading.
    Saves atomically via temp file + rename.
    """

    @override
    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        return pandas.read_json(  # pyright: ignore[reportUnknownMemberType] - pandas has complex overloads
            path, lines=True, orient="records", convert_dates=False
        )

    @override
    def save(self, data: pandas.DataFrame, path: pathlib.Path) -> None:
        if not isinstance(data, pandas.DataFrame):
            raise TypeError(f"DataFrameJSONL save expects DataFrame, got {type(data).__name__}")

        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                data.to_json(f, lines=True, orient="records")  # pyright: ignore[reportUnknownMemberType]
            os.rename(tmp_path_str, path)
        except Exception:
            tmp_path = pathlib.Path(tmp_path_str)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @override
    def empty(self) -> pandas.DataFrame:
        return pandas.DataFrame()  # pyright: ignore[reportUnknownMemberType]
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sami/pivot/default && uv run pytest tests/test_loaders.py -k "dataframe_jsonl" -v`
Expected: PASS (7 tests)

**Step 5: Run quality checks**

Run: `cd /home/sami/pivot/default && uv run ruff format src/pivot/loaders.py tests/test_loaders.py && uv run ruff check src/pivot/loaders.py tests/test_loaders.py && uv run basedpyright src/pivot/loaders.py tests/test_loaders.py`
Expected: No errors or warnings

**Step 6: Commit**

```bash
cd /home/sami/pivot/default && jj describe -m "feat(loaders): add DataFrameJSONL loader for pandas DataFrames

- Uses pandas.read_json(lines=True) for efficient loading
- Atomic saves via temp file + rename
- Supports empty() for IncrementalOut"
```

---

## Task 4: Final Verification

**Step 1: Run all loader tests**

Run: `cd /home/sami/pivot/default && uv run pytest tests/test_loaders.py -v`
Expected: All tests pass

**Step 2: Run full quality checks**

Run: `cd /home/sami/pivot/default && uv run ruff format . && uv run ruff check . && uv run basedpyright .`
Expected: No errors or warnings

**Step 3: Run full test suite**

Run: `cd /home/sami/pivot/default && uv run pytest tests/ -n auto`
Expected: All tests pass

**Step 4: Squash commits**

```bash
cd /home/sami/pivot/default && jj squash
```

**Step 5: Final commit message**

```bash
cd /home/sami/pivot/default && jj describe -m "feat(loaders): add Text, JSONL, and DataFrameJSONL loaders

Upstream loaders from eval-pipeline:

- Text: Plain text files with atomic saves
- JSONL: JSON Lines as list[dict] with line-number error reporting
- DataFrameJSONL: JSON Lines as pandas DataFrame

All loaders support:
- Atomic writes via temp file + rename
- empty() for IncrementalOut support
- Pickling for ProcessPoolExecutor"
```
