# Outputs

A stage's outputs are its return value. Pivot saves whatever the stage returns,
chooses the file path, and makes the saved artifact available to downstream
stages through the handle the stage call returned.

## The return type decides the format

The return annotation is required (`-> None` means the stage has no outputs).
Pivot picks a writer from it:

| Return type | Saved as | Extension |
|-------------|----------|-----------|
| `pandas.DataFrame` | JSON Lines (`DataFrameJSONL()`) | `.jsonl` |
| `dict`, `list` (including `dict[str, int]` etc.) | YAML (`YAML()`) | `.yaml` |
| `str` | text (`Text()`) | `.txt` |
| `pathlib.Path` | a file or directory created outside the project, moved by `PathOnly()` | `.dat` (or a directory) |
| `Annotated[T, <loader>]` | that loader | the loader's extension |
| `Annotated[dict, metric]` | YAML, stored under `metrics/` | `.yaml` |
| `Annotated[Figure, plot]` | PNG via `MatplotlibFigure()`, stored under `plots/` | `.png` |
| `Annotated[T, incremental]` | the type's default `Loader`, restored before the stage runs | the loader's extension |
| `Annotated[dict[str, T], directory]` | a directory, one file per key, each written by `T`'s writer (or a loader given beside the tag) | none on the directory; each key names its file |
| a `TypedDict` | one output per field, each field typed as above | per field |

Any other type fails when the pipeline is built:
`Cannot infer serialization format for type int. Use Annotated[T, format] on
the return type to specify explicitly.`

`Annotated` appears only on return types and `TypedDict` fields. Stage
parameters are never annotated with loaders; inputs take their format from
the producing stage or from `pipeline.input()`.

## Where outputs are stored

| Output | Path |
|--------|------|
| Single output of stage `s` in pipeline `p` | `data/p/s.<ext>` |
| Field `f` of a `TypedDict` output | `data/p/s/f.<ext>` |
| Tagged `directory` | `data/p/s/` (or `data/p/s/f/` for a field), one file per key inside it |
| Tagged `metric` | `metrics/p/...` instead of `data/p/...` |
| Tagged `plot` | `plots/p/...` instead of `data/p/...` |

Extensions come from the writer: `jsonl`, `csv`, `yaml`, `json`, `txt`, `pkl`,
`png`, and `dat` for custom writers. Variant stages keep their `@` suffix in the
file name (`data/p/scale@x2.csv`).

For a `pathlib.Path` return, the stage must create a file or directory outside
the project and return its path. Pivot moves it to the managed output path. A
returned path inside the project is rejected because Pivot must not take
ownership of a file it did not create.

## Example

```python
# pipeline.py
from typing import Annotated, TypedDict

import matplotlib.figure
import pandas as pd

from pivot import Pipeline, loaders, metric, plot, stage


class SplitResult(TypedDict):
    train: pd.DataFrame
    test: Annotated[pd.DataFrame, loaders.CSV()]
    sizes: Annotated[dict, metric]


@stage
def split(raw: pd.DataFrame) -> SplitResult:
    cut = len(raw) // 2
    return {
        "train": raw.iloc[:cut],
        "test": raw.iloc[cut:],
        "sizes": {"train": cut, "test": len(raw) - cut},
    }


@stage
def chart(train: pd.DataFrame) -> Annotated[matplotlib.figure.Figure, plot]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(train["id"], train["value"])
    return fig


@stage
def report(test: pd.DataFrame) -> str:
    return f"{len(test)} test rows\n"


with Pipeline("out") as pipeline:
    parts = split(pipeline.input("values.csv"))
    chart(parts.train)
    report(parts["test"])
```

After `pivot repro`:

```
data/out/split/train.jsonl
data/out/split/test.csv
metrics/out/split/sizes.yaml
plots/out/chart.png
data/out/report.txt
```

The plot needs matplotlib: `pip install "pivot[matplotlib]"`.

## Multi-output stages

A `TypedDict` return type declares several outputs. The stage returns a dict
with exactly those keys (`SplitResult(train=..., ...)` works too). Downstream,
select one field from the handle with `parts.train` or `parts["train"]`; each
field is a separate artifact with its own hash, so a change that only affects
`sizes` does not re-run stages that consume `train`.

Define the `TypedDict` at module level. The field name `_single` is reserved.

## Directory artifacts

A stage can return a `pathlib.Path` to a directory it created outside the
project. `PathOnly()` moves that directory to Pivot's managed output path and
Pivot caches it as an artifact. A stage that returns a `dict[str, T]` tagged
`directory` also makes a directory artifact; see
[Directory outputs](#directory-outputs).

A directory artifact, input or output, is hashed and cached whole. Every file
under it counts, including names that code fingerprinting or watch mode may
ignore, such as `__pycache__/`, `build/`, `.git/` or `*.pyc`. Editing any of
them changes the directory hash, and all of them return on restore.

Empty directories are part of a directory artifact too: adding or removing one
changes the directory hash, and every one the hash records comes back on
restore, so an artifact restored from a hash Pivot 0.3.4 or later recorded
holds the same directories as the one the stage built, in every checkout mode.

A directory hash Pivot 0.3.3 or earlier recorded lists no empty directories,
and neither does any hash of a directory that had none. The two look the same,
so Pivot cannot tell whether such a directory had any. Restoring an output from
such a lock where none is on disk (in a fresh clone, with `pivot checkout`, or
with `pivot repro` after the output was deleted) leaves its empty directories
out, and the stage stays cached.

When an output on disk holds the files such a lock records plus empty
directories it does not, `pivot repro` does not restore the output to the lock,
which would delete them. `pivot status` shows the stage stale ("Output holds
empty directories its lock does not record"), and `pivot repro` runs it, or
takes its outputs from a run-cache entry that records the directories, so its
lock records them from then on. An empty directory you add inside such an
output yourself makes the stage run once, and the output the stage builds
replaces yours, without that directory. `pivot checkout` leaves such an output
as it is; `pivot checkout --force` is the exception that restores it to the
lock, deleting its empty directories, and the stage stays cached.

An output that lacks the empty directories its stage makes matches its old lock
and stays cached, whether it was restored from that lock as above or Pivot
0.3.4 or 0.3.5 deleted them (those versions restored such an output to its old
lock). To repair it, re-run the stages that write the directories (`pivot run
--force <stage>`, then `pivot repro`) so their locks record them. To re-record
a tracked directory, run `pivot track --force <path>`.

A symlink inside a directory artifact is not stored: it is left out of the
manifest and does not come back on restore. When a stage writes the artifact,
or `pivot track` records it, Pivot logs one warning naming each symlink it
left out, for example `Directory artifact data/p/bundle.dat leaves out symlink
link.txt`; replace the link with a real file or directory. A later release
refuses these symlinks. The one exception is a link Pivot made itself: a
symlink-mode checkout links each file to an object in the configured cache
(`cache.dir`), and that link counts as the file it points to. A link to a
hash-named file anywhere else is a user symlink. A symlink Pivot can see but
not read (reading it, or what it points to, fails, as on a faulty FUSE or NFS
mount) is an error, not a left-out link: Pivot cannot tell whether it made the
link, and leaving it out could record the directory without one of its files.
A command that hashes the directory reports an error that names the link, and
changes nothing: `pivot status`, `pivot verify` and `pivot repro --dry-run`
give it as the reason of the stage that writes the directory, and the other
commands stop with it. `pivot repro` fails that stage when the stage would
otherwise be cached; a stage that runs replaces its output. A tracked directory
holding such a link stops `pivot status`, `pivot run` and `pivot repro`. Make
the link and what it points to readable, or remove the link.

A lock records each file of a directory artifact by its path inside the
directory, and Pivot reads back every path it writes. Any character a file name
holds is recorded, spaces and format characters included (a macOS screenshot's
narrow no-break space, `Screenshot … 10.00.00 AM.png`). A name with a
backslash cannot be recorded: the stage, or `pivot commit`, fails and names the
file, and the previous lock is kept. A name with a control character (a TAB,
an escape, DEL) is recorded with a warning naming it; a later release refuses
these names, so rename such files.

## Directory outputs

Tag a `dict[str, T]` return value with `directory` when a stage writes a
variable set of files. Each key is a file's path inside the directory and each
value is written by `T`'s default writer, or by the loader given beside the
tag. The loader never comes from a key's extension, so give keys the extension
of the files the loader writes:

A directory handle is not a collection while `pipeline.py` is loading. Do not iterate it or test
membership; name one member (`parts["train.csv"]`) or pass the handle itself.

```python
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, loaders, stage


@stage
def split(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]:
    return {f"{name}.csv": frame for name, frame in raw.groupby("split")}


@stage
def count(parts: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: len(frame) for name, frame in parts.items()}


@stage
def fit(train: pd.DataFrame) -> dict[str, int]:
    return {"rows": len(train)}


with Pipeline("dirs") as pipeline:
    parts = split(pipeline.input("values.csv"))
    count(parts)
    fit(parts["train.csv"])
```

After `pivot repro`, `data/dirs/split/` holds one CSV per split. The directory
is one artifact: it is cached, restored, pushed, pulled, checked out and
verified whole, and a rerun that returns fewer keys leaves only those files.
A `TypedDict` field can be tagged `directory` too (`data/p/s/f/`).

A downstream stage takes the directory in one of two ways:

- **The whole directory** (`count(parts)`): the parameter receives a `dict`
  from each file's path inside the directory to its value, read with the
  producer's loader. The stage reruns when any file changes.
- **One file** (`fit(parts["train.csv"])`, or `out.parts["train.csv"]` for a
  `TypedDict` field): the parameter receives that file's value, or its path
  when the producer's files are `pathlib.Path`. The consumer's lock records
  that file's hash alone, so it reruns when that file changes and not when
  another file in the directory does. A run where the producer did not write
  the file fails the consumer with `missing deps:` naming the producer and the
  file, and a path that names a directory inside the output fails it with
  `unreadable deps:`; `pivot status` reads the file's hash from the
  producer's lock when the directory is not on disk.

A key must be a relative path with an extension, with no `..`, backslash,
colon or null byte (each file is named `stage:path` as a dependency). Two keys
that normalize to one path, two that differ only by case, and a key inside
another key's path (`a.json` beside `a.json/b.json`) are refused too. Every
output of the stage is checked before any is written, so a refused key fails
the stage in one line naming it, the output and the key, and leaves the
previous run's outputs, cache and lock as they were. A file named by a consumer
is checked by the same rules when the pipeline is built, naming the consumer,
the producer and the key, and a `TypedDict` field named like a file of a
directory field (`parts/a.json` beside `parts`) is refused when the stage is
defined. A directory of files whose loader only writes, such as
`MatplotlibFigure()`, is a valid output, but no stage can depend on it or on
one of its files.

## Incremental outputs

Tag a return value with `incremental` when the stage needs its own prior output.
Bind the matching parameter to `pipeline.previous()`, or use
`pipeline.previous("key")` for a field of a `TypedDict` result. The first run
receives the loader's `empty()` value; later runs restore the cached output as a
writable copy before calling the stage. Previous-output parameters do not create
DAG edges.

```python
from typing import Annotated

from pivot import Pipeline, incremental, stage


@stage
def update_cache(previous: dict[str, int]) -> Annotated[dict[str, int], incremental]:
    return {"count": previous.get("count", 0) + 1}


with Pipeline("state") as pipeline:
    update_cache(previous=pipeline.previous())
```

An incremental output requires a full `Loader` with `load()`, `save()`, and a
concrete `empty()` implementation; a writer-only format such as
`MatplotlibFigure()` and loaders that inherit the default raising `empty()`
(including `PathOnly()` and `Pickle()`) are rejected. An existing incremental
output without a lock-file cache entry is refused unless
`pivot repro --allow-uncached-incremental` is explicit.

## Metrics and plots

`metric` and `plot` (imported from `pivot`) are tags, not paths:

- `Annotated[dict, metric]` saves YAML under `metrics/`. `pivot metrics show`
  and `pivot metrics diff` read every metric output. Metric files are not
  copied into the cache; commit them to git.
- `Annotated[Figure, plot]` saves a PNG under `plots/` and closes the figure.
  `pivot plots show` renders all plot outputs as an HTML gallery.

A tag can be combined with an explicit writer: `Annotated[pd.DataFrame, metric,
loaders.CSV()]` saves a CSV under `metrics/`.

A `dict` or `list` output is written as YAML. Numpy scalars and arrays are
converted to plain Python numbers and lists, so `{"mean": df["v"].mean()}`
works. A DataFrame, a dataclass, or any other object inside it is written with
Python-specific YAML tags that `pivot metrics show` and downstream stages cannot
read, and the `pivot repro` report prints a warning under the stage naming the
output, the type and where it sits. Convert it first (`df.to_dict()`,
`dataclasses.asdict(x)`); a later release will fail the stage instead.

## How Pivot writes outputs

After a stage returns, Pivot:

1. Checks that a `TypedDict` return value contains every declared key, and
   checks every output's value (a directory output's keys, and that a returned
   `pathlib.Path` lies outside the project) before writing any of them
2. Replaces each previous output with the new value, written by its writer to
   the path above, creating directories
3. Hashes each file and copies it into `<state dir>/cache/` (except metrics)
4. Records the hashes in the stage's lock file under `<state dir>/stages/`

The previous output stays on disk while the stage runs, so a stage that reads
its own output path sees the last run's file. Read a previous output through
`pipeline.previous()` on an `incremental` output instead. If a check in step 1
fails, the stage fails and the previous outputs, cache and lock stay as they
were.

---

**See also:** [Artifacts & the DAG](artifacts-and-dag.md) | [Dependencies](dependencies.md) | [Loaders](loaders.md)
