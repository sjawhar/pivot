# Dependencies

A stage's dependencies are its function parameters. Pivot loads each one before
the stage runs, passes the loaded value in, and uses the dependency to build the
[DAG](artifacts-and-dag.md) and to decide when the stage must re-run.

A parameter can receive one of four values at the call site:

| Call-site value | The function receives | Creates a DAG edge |
|-----------------|-----------------------|--------------------|
| A handle (from `pipeline.input()` or another stage call) | The loaded artifact | Yes |
| A `list` or `tuple` of handles | A `list` or `tuple` of loaded artifacts | One per handle |
| A `StageParams` instance (at most one) | That instance, with `params.yaml` overrides applied | No (see [Parameters](parameters.md)) |
| `pipeline.previous()` or `pipeline.previous("key")` | The stage's prior tagged incremental output | No |

Anything else (an `int`, a string, a path) is rejected when the pipeline is
built: `parameter 'factor' has unsupported type 'int'. Use ArtifactHandle
(dependency), StageParams (config), or list/tuple of ArtifactHandle.` Put such
values in the stage's `StageParams`.

## Previous incremental outputs

`pipeline.previous()` is valid only as an argument to the stage call whose
return type has one output tagged `incremental`. For a `TypedDict` result, use
`pipeline.previous("key")` to select the tagged field. Each marker can bind one
parameter only; it is not a handle and does not add a self-edge to the DAG.

## Inputs: `pipeline.input()`

Files that no stage produces are declared once, by name:

```python
raw = pipeline.input("scores.csv")            # data/raw/scores.csv
```

| Argument | Effect |
|----------|--------|
| `name` | Input name; also the file name under `data/raw/` |
| `external=True` | Read `data/external/<name>` instead |
| `path="dir/file.csv"` | Read this path (relative to the project root) |
| `t=pathlib.Path` | Pass the stage a `pathlib.Path` instead of loading the file |
| `t=str` | Pass the stage the file's text |
| `format=loaders.CSV(sep=";")` | Load with this [reader](loaders.md) |

Without `t` or `format`, the extension picks the loader:

| Extension | Stage receives |
|-----------|----------------|
| `.csv` | `pandas.DataFrame` (`CSV()`) |
| `.jsonl` | `pandas.DataFrame` (`DataFrameJSONL()`) |
| `.json` | parsed JSON (`JSON()`) |
| `.yaml`, `.yml` | parsed YAML (`YAML()`) |
| `.txt`, `.sql`, `.jinja` | `str` (`Text()`) |
| `.pkl`, `.pickle` | unpickled object (`Pickle()`) |
| anything else | `pathlib.Path` (`PathOnly()`) |

An input whose file does not exist fails before anything runs:
`Stage 'demo/keep' depends on 'missing.csv' which is not produced by any stage
and does not exist on disk`.

### Directory inputs and `.pivotignore`

A directory input (`t=pathlib.Path` on a directory) is hashed file by file, and
any change to it makes its consumers stale, except for the files and
directories the project's `.pivotignore` names. `pivot init` writes a starter
`.pivotignore` (`pivot check-ignore --show-defaults` lists it): compiled Python
(`*.pyc`, `__pycache__/`), virtual environments, version-control and editor
files, build directories. So a Python process that imports from an input
directory and writes `__pycache__/*.pyc` there leaves the stage cached.

- Patterns are read and matched as git 2.52 reads and matches a root
  `.gitignore`, with a port of git's own matcher: the same syntax, the same
  `**` rules (`logs/**` names what is inside `logs`, not `logs` itself, so
  `!logs/.gitkeep` brings that file back; `foo**/bar` names `foo/bar` and
  `foox/bar` but not `foobar`, which git before 2.52 also named), and the same
  bytes (a line that is not valid UTF-8 matches the same bytes in a name; a NUL
  ends a line). Matching is case-sensitive, as git's is with `core.ignorecase`
  off. Each file is
  matched by its path inside the project (the input's path plus the file's path
  under it), never by where a symlink points, so a directory linked in from
  outside the project is treated as a local one. A file under an ignored
  directory stays ignored, as in git: a later `!` pattern cannot bring it back.
- `.pivotignore` decides two things: which files of a directory input count
  for its hash (so whether its consumers rerun), and which changes inside a
  directory input `pivot repro --watch` skips. `pivot check-ignore` answers by
  the same rules. For an input directory that `.pivotignore` ignores (below),
  `check-ignore` reports its files ignored, as git would, but the input is
  hashed whole. `pipeline.py` and `.pivot/` at the project root are never
  ignored, whatever the patterns say, and `check-ignore` takes a relative path
  from the project root, not the current directory.
- A line may hold at most 256 runs of `*` that are tried at more than one
  place. A run at the end of the line does not count, nor a lone `*` right
  before a `/` (it matches the rest of one name), and consecutive `**/` count
  as one. Pivot matches a longer line differently from git, with a warning
  that quotes it, and always toward hashing more. For a *member* (a file or
  directory inside an input) a negation (`!...`) is taken to match it and any
  other such line not to, so every file git might keep is kept. For an *input
  directory* (the input's own directory, or one above it) it is the other way
  round: a negation matches none, so an earlier `build/` rule still makes an
  input at `build` hash whole, and any other such line matches every input
  directory, so an input is hashed whole unless a later line unignores it and
  every directory above it. At worst a stage reruns when git's rule would not
  count the file.
- It does not apply to stage outputs, to data you `pivot track`, or to the
  cache: every file in a directory output or a tracked directory is hashed,
  cached and restored, a `.pyc` included. A stage that reads another stage's
  directory output depends on that output's whole hash.
- The listing committed for a directory input
  ([Listing files](outputs.md#listing-files)) holds the files its hash counts,
  so it leaves out what `.pivotignore` names too. A tracked directory's
  `.pvt.listing` keeps every file, and `pivot pull <stage>` fetches every one
  of them, so `pivot checkout` restores the directory as it was.
- An input file you declare by name is always hashed, whatever `.pivotignore`
  says. An input directory that `.pivotignore` ignores itself (or that lies in
  a directory it ignores, such as an input under `build/`) is hashed whole,
  with a warning: leaving out every file in it would hide every change to it.
- Only the project's `.pivotignore` counts: not one in your home directory,
  since lock files carry input hashes to every machine, and not one in a
  subdirectory, which is data like any other file.
- Changing `.pivotignore` changes the hash of every directory input holding a
  file whose status it changes, so those stages run once. `pivot status
  --explain` shows the input's hash change; `git diff` on the input's committed
  listing (`.pivot/stages/=inputs/`) after the run shows which files entered or
  left its hash.

## Example

```python
# pipeline.py
import pathlib

import pandas as pd

from pivot import Pipeline, StageParams, loaders, stage


class ShiftParams(StageParams):
    offset: int = 0


@stage
def shift(raw: pd.DataFrame, params: ShiftParams) -> pd.DataFrame:
    return raw.assign(value=raw["value"] + params.offset)


@stage
def stack(tables: tuple[pd.DataFrame, pd.DataFrame], lookup: dict) -> pd.DataFrame:
    left, right = tables
    combined = pd.concat([left, right], ignore_index=True)
    return combined.assign(value=combined["value"] + lookup["bonus"])


@stage
def describe(query: str, notes: pathlib.Path, rows: pd.DataFrame) -> dict:
    return {"query": query.strip(), "notes_bytes": notes.stat().st_size, "rows": len(rows)}


with Pipeline("deps") as pipeline:
    raw = pipeline.input("values.csv")
    lookup = pipeline.input("lookup.yaml", external=True)
    query = pipeline.input("query.sql")
    notes = pipeline.input("notes.md", t=pathlib.Path)
    semi = pipeline.input("semi.csv", format=loaders.CSV(sep=";"))

    low = shift(raw, params=ShiftParams(offset=1))
    high = shift(semi, params=ShiftParams(offset=100))
    stacked = stack((low, high), lookup)
    describe(query, notes, stacked)
```

with these files:

```bash
mkdir -p data/raw data/external
printf 'id,value\n1,10\n2,20\n' > data/raw/values.csv
printf 'id;value\n3;30\n' > data/raw/semi.csv
printf 'bonus: 5\n' > data/external/lookup.yaml
printf 'select 1\n' > data/raw/query.sql
printf '# notes\n' > data/raw/notes.md
```

The two calls to `shift` become the stages `deps/shift` and `deps/shift@1`
(see [variants](pipelines.md#variants)). `stack` receives a tuple of two
DataFrames because it was called with a tuple of two handles.

## Outputs of other stages

A stage call returns a handle to the stage's output. For a stage that returns a
`TypedDict` (several outputs), select one field before passing it on, with
attribute or item syntax:

```python
split = split_rows(raw)        # returns a TypedDict with fields "train" and "test"
fit(split.train)               # or split["train"]
```

Passing the whole handle fails: `Stage 'demo/fit' received handle from
multi-output stage 'demo/split_rows' without selecting an output. Use
handle.key or handle['key']. Available: ['train', 'test']`.

A [directory output](outputs.md#directory-outputs) is passed whole, as a
`dict` from each file's path to its value, or one file at a time by indexing
its handle with that path:

```python
parts = split(raw)             # returns Annotated[dict[str, pd.DataFrame], directory]
count(parts)                   # every file: reruns when any of them changes
fit(parts["train.csv"])        # one file: reruns only when train.csv changes
```

## Testing is natural

Outside a `with Pipeline` block a `@stage` function is an ordinary function, so
tests pass data directly:

```python
def test_shift() -> None:
    df = pd.DataFrame({"value": [1, 2]})
    assert shift(df, ShiftParams(offset=1))["value"].tolist() == [2, 3]
```

## Files tracked outside the pipeline

`pivot track` caches files that are not stage outputs (large raw data, for
example) and writes a `.pvt` sidecar to commit in their place:

```bash
pivot track data/raw/values.csv
```

The sidecar sits beside its data, and its `path:` is the data's name
(`values.csv` in `data/raw/values.csv.pvt`). `pivot checkout`, `pivot pull` and
`pivot repro --checkout-missing` restore the file that `path:` names, under the
directory holding the `.pvt`.

**A `.pvt` restores its file only inside the project.** A sidecar whose `path:`
names no file (`.`, `./` or empty), holds `..`, is absolute, or names a file
whose directory really lies outside the project through a symlink, is skipped
with a warning that names it; the command keeps going and exits 0. Move such a
sidecar beside its data inside the project and set `path:` to the data's name.

`pivot track` refuses a path whose directory really lies outside the project, as
from a `data -> /mnt/external` symlink: the `.pvt` would be written beside the
data, outside the project, where no Pivot command reads it again.

The check reads "inside the project" from the resolved path, so it does not see
a bind mount inside the project that points outside: `resolve()` cannot detect a
bind mount, and a `.pvt` restored under it lands outside the project (git and DVC
share this limit, and setting a bind mount up needs `CAP_SYS_ADMIN`). The check
and the write are separate steps, so a local process that can already write in
the project can replace a checked directory with a symlink pointing outside
between the two and land the file there.

---

**See also:** [Artifacts & the DAG](artifacts-and-dag.md) | [Outputs](outputs.md) | [Loaders](loaders.md)
