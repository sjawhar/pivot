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

## Objects named by hash

A declared input can name another object by its content hash instead of holding
it: a pin file holding the hash of a snapshot another project pushed, an index
naming the hashes of records. The stage reads each such object with
`pivot.read_by_hash`:

```python
import pivot
from pivot import Pipeline, stage


@stage
def plan(pin: str) -> dict:
    snapshot = pivot.read_by_hash(pin.strip())  # a pathlib.Path to the object
    return {"taken": snapshot.read_text().splitlines()}


with Pipeline("plan") as pipeline:
    plan(pipeline.input("counted.pin", t=str))
```

- **Where the object comes from.** Pivot looks the object up in the cache, else
  downloads it from the project's default remote into the cache, and checks that
  its bytes hash to the hash asked for. The path is the read-only cache object:
  read it, never change it. Any file object in the cache can be read this way: a
  stage's file output, one file of a directory output, or a tracked file, once
  `pivot push` has put it on the remote.
- **Finding a hash to pin.** `pivot.output_hash("pipeline/stage", key)` returns the
  content hash a stage's committed lock records for its output `key` (`None` for
  a stage's one unnamed output, a `TypedDict` field's name otherwise), read from
  the lock file alone, so a command that writes a pin can name the output it
  pins. It raises `pivot.exceptions.OutputMissingError` when no lock records it.
- **What reruns the stage.** The input that names the hash, as any input does:
  a new hash in `counted.pin` reruns `plan`.
- **What the lock records.** Each hash the run read, once, sorted, under
  `reads`, beside the stage's declared dependencies. A run-cache restore records
  what the restored run read, and `pivot commit` after `pivot repro --no-commit`
  records what that run read. A stage that reads nothing by hash has no `reads`.
  A pivot older than this one drops `reads` when it writes the lock: from a
  run-cache restore, or from `pivot commit` after this pivot's `--no-commit` run.
- **A recorded object that is gone.** When an object the lock records is in
  neither the cache nor the remote, the stage cannot be run again as it ran:
  `pivot status` reports it stale, naming the objects and why, and `pivot repro`
  runs it, where the read fails. An object the cache lacks costs one check of the
  remote, which Pivot then remembers in its remote index, as `pivot push`,
  `pivot pull` and `pivot status --remote` do: once the index records an object
  on the remote, its later loss from the remote is found by the read, not by
  `pivot status`.
- **Pulling.** `pivot pull <stage>` fetches the objects its lock records under
  `reads`, as it fetches the stage's dependencies, so the stage can run again in
  the clone.
- **A read that fails** raises `pivot.exceptions.ObjectReadError` naming the hash:
  a value that is not a content hash (16 lowercase hex characters), an object
  neither the cache nor the remote holds, or bytes that hash to something else.

`pivot.read_by_hash` works only inside a stage Pivot runs, from any thread of the
stage's own process, a coroutine's included. A process the stage starts (a forked
child, joblib's processes backend) is refused: only the stage's own process
records what it reads, so read there and pass the path or bytes on.

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
