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

A directory output is hashed and cached whole. Every file under it counts,
including names `.pivotignore` lists, such as `__pycache__/`, `build/`, `.git/`
or `*.pyc`: editing any of them changes the directory hash, and all of them
return on restore. A directory input leaves out the files `.pivotignore` names
from its hash; see
[Directory inputs and `.pivotignore`](dependencies.md#directory-inputs-and-pivotignore).

Empty directories are part of a directory artifact too: adding or removing one
changes the directory hash, and every one the artifact held comes back on
restore, in every checkout mode.

### Listing files

A lock file records a directory by its listing's hash, its total size and its
file count (`listing`, `size`, `nfiles`), and a file by its `hash`. The
directory's members are in its *listing*, a text file committed beside the
lock: one line per file (`<hash> <size> <x|-> <path>`, `x` when its owner may
execute it) and one per deepest empty directory (`- - - <path>/`), sorted by
path. The directory hash is the hash of the listing's bytes, so a change to one
file of a 1,000-file output changes one line of its listing and the `listing`
line of its lock, and the lock's `size` line too when the file's size changes.
A stage that reads the whole directory records it the same way, so its lock
changes the same lines. Each listing is also kept in the cache, at the
directory's hash, and pushed and pulled with it.

A lock that records a directory keeps its entries under `records`, and leaves
its top-level `deps` and `outs` empty; a lock that records only files keeps them
at the top level. Released versions of Pivot, up to and including 0.5.20, read
the top-level lists alone, and would read a directory record that lists no files
inside it as one file's hash. Finding empty lists instead, they treat the lock as
one that records nothing: they run an ordinary stage again from its inputs, and
stop at a stage with an incremental output ("Cache missing for incremental
output", exit 1) rather than restart that output from empty. Run-cache entries
are kept apart from theirs for the same reason.

| Directory | Listing file |
|---|---|
| A stage output | `.pivot/stages/<stage>=<key>.listing`, beside the stage's lock (`<stage>=.listing` for a stage's single output) |
| A directory the pipeline declares as an input | `.pivot/stages/=inputs/<path>.listing`, one per project |
| A tracked directory | `<path>.pvt.listing`, beside its `.pvt` |

An input directory's listing holds the members its dependency hash counts, so
it leaves out the files `.pivotignore` names, exactly as the hash in the lock
beside it does; writing a `.pyc` into an input changes neither file. A tracked
directory's `.pvt.listing` holds every file, because `pivot checkout` restores
the whole directory from it (see
[Directory inputs and `.pivotignore`](dependencies.md#directory-inputs-and-pivotignore)).

In a file name, every byte of the key or path other than a letter, digit, `_`,
`.`, `@` or `-` is written `%XX`; a name that would then be longer than 255
bytes is written `~` and the xxhash64 of the key or path instead (for a tracked
directory, `.~<hash>.pvt.listing` beside its `.pvt`). Commit listing files with
the lock files. `pivot init` writes `!/**` to the `.gitignore` in the stages
directory, so a root `.gitignore` pattern such as `lib/` or `build/` cannot
hide them; `pivot repro` warns, naming each one, when git ignores a lock or
listing file it just wrote, and `pivot track` does the same for a `.pvt` and
its listing.

A listing file whose bytes do not hash to the directory hash its lock or `.pvt`
records (hand-edited, left behind by a lock put back without it, merged line by
line, or a shared input listing another stage has since rewritten), or that
cannot be read, is passed over for the cache's copy of that listing, with one
warning naming the file. `pivot verify` refuses such a file, naming it, since a
fresh clone has no cache to fall back on, and so does a command that needs the
listing's content when the cache lacks that copy (`checkout` of that output,
`get --rev`); `pivot status` marks the stages that need it, and `pivot pull`
fetches the copy a remote holds. `pivot repro --force <stage>` or `pivot track
--force <path>` rewrites the file, and `git checkout <the revision its lock or
.pvt came from> -- <file>` restores the one committed with it; `git checkout --
<file>` alone changes nothing when git itself holds the stale file.

`pivot init` writes `*.listing -merge` into `.gitattributes`, so git never
merges a listing line by line: when both branches of a merge changed a
directory, its listing conflicts beside its conflicted lock. Take both from one
side (`git checkout --ours -- .pivot/stages`, or `--theirs`, then `git add`), or
settle the inputs and run `pivot repro`, which writes the lock and listings
afresh; then commit. In a project made before this, add the line to
`.gitattributes` yourself.

A lock file an earlier Pivot wrote (any `schema_version` below 3; one that
records a directory lists the directory's files inside it) is read as it is,
with one warning naming it: each directory it lists stands for its listing, and
that listing goes into the cache, so nothing reruns or restores differently
because of the lock's format. The next time its stage's lock is written, it is
written in this format, with its listing files. A lock a newer Pivot wrote is
read as missing. A directory `.pvt` an earlier Pivot wrote is read as it is, its
inline file list standing in for the listing, with one warning; `pivot track
--force <path>` converts it.

**A team upgrades together.** Everyone who works on a project upgrades Pivot
together. A release before this
format, given a project this Pivot wrote, restores no output of a stage whose
lock records a directory and no tracked directory (it skips a directory `.pvt`
it cannot read): in a fresh clone, `pivot checkout` and `pivot pull` leave them
absent, and `pivot repro` reruns those stages from their inputs and stops at
each stage with an incremental output; in an existing workspace, `pivot repro`
stops at an incremental output already on disk ("incremental output files exist
but are not in cache"). Its `pivot track` of a tracked directory writes its own
`.pvt` there without `--force`.

This Pivot then reads what it left. A lock it rewrote is read as it is (above),
and the listing this Pivot wrote beside that lock is not read. A `.pvt` it
rewrote is read from its own file list; the `<path>.pvt.listing` beside it is
named in one warning, and `git rm <path>.pvt.listing` removes it. One case is
refused: when an older release started a stage with an incremental output from
empty (its lock removed by hand, as its message offers), a file this Pivot wrote
beside the lock is still there — the restart proof it commits for every
incremental stage, and for a directory output its listing too — and `pivot
status` reports the stage, `pivot verify` fails it and `pivot repro` refuses it,
each naming the lock and that file. Restore the output from the commit before
the older release's (`git checkout <commit> -- <lock> <file>`, then `pivot
checkout --force <stage>`), or rerun the stage from what that lock records, on
purpose, with `pivot repro --force <stage>`. This holds whether the incremental
output is a directory or a file: the restart proof stands beside the lock for
both.

A symlink inside a directory artifact is not stored: it is left out of the
listing and does not come back on restore. When a stage writes the artifact,
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

### One stage per file: `pipeline.map`

`pipeline.map(handle)` makes a stage run once per file of a directory output,
for a set of files no one knows while `pipeline.py` loads:

```python
from typing import Annotated

from pivot import Pipeline, directory, loaders, stage


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def cut(entry: dict[str, int]) -> dict[str, int]:
    return {"cut": entry["value"] * 2}


@stage
def build(cut: dict[str, int]) -> dict[str, int]:
    return {"built": cut["cut"] + 1000}


with Pipeline("p") as pipeline:
    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))
    build(cut(pipeline.map(planned)))
```

`cut` and `build` are *mapped stages*. When a `pivot repro` finds `p/plan`
current (it ran, or was skipped with its output present), it adds one instance
of each per file of `data/p/plan/`, named `<stage>@<key>`, where the key is the
file's path without its extension: `p/cut@a`, `p/build@a`, and so on. It runs
them in the same run. Each instance is an ordinary stage with its own lock and
outputs (`data/p/cut@a.yaml`); `p/cut@a` depends on `data/p/plan/a.yaml` alone,
and `p/build@a` on `p/cut@a`. So an edit that changes one file reruns that
key's chain only, and a new file runs the producer and the new chain only.

- **Keys.** The keys come from the listing the producer's lock records for the
  directory (see "Listing files" above), the same bytes restore, push and pull
  read; a `--no-commit` run, which records no listing, reads the directory it wrote. A
  key follows the variant-name rule (letters, digits, `_`, `.`, `-`), so a file
  in a subdirectory, or two files with one key (`a.yaml`, `a.json`), fail the
  mapped stage in the run, naming the files; so does an instance whose name or
  output path another stage holds. A failed expansion adds no instance. A run and
  every command give such a refusal in one wording, `cannot expand over the files
  of 'p/plan': ...`. A producer whose listing neither its lock's directory nor the
  cache holds leaves its mapped stages unexpanded with `the listing of 'p/plan' is
  missing`: `pivot pull` fetches it from a remote that holds it, and
  `pivot repro --force p/plan` writes it again.
- **What expands.** `pivot repro <producer>` runs the producer and its
  upstream only, and expands nothing. `pivot repro p/cut` runs `p/plan`, then
  every instance of `p/cut`. An instance name (`pivot repro p/build@a`, or
  `build@a`) names that instance and its chain only, as `pivot status p/build@a`
  reads it: the run expands `p/cut` and `p/build` over key `a` alone, so it runs
  `p/plan`, `p/cut@a` and `p/build@a`, and `pivot repro --force p/cut@a` forces
  `p/plan` and `p/cut@a`, no other instance. When `p/plan` is current, a key it has
  no file for is refused as `pivot status` refuses it (`Unknown stage(s):
  p/cut@zzz`); when it is not, the run refuses the key once `p/plan` has run and
  has no file for it, failing `p/cut` with `Unknown stage(s): p/cut@zzz, needed by
  the target p/build@zzz` for a target further down the chain. `pivot run p/cut`
  is refused: a mapped stage expands only in a run that finds its producer current.
  Nothing about an expansion is saved: each
  run expands afresh. A watch run (`pivot repro --watch`) expands as any run does,
  including the run that follows a reload after an edit to the code or to
  `params.yaml`.
- **The other commands** (`status`, `list`, `dag`, `checkout`, `verify`,
  `push`, `pull`, `commit`, `params show`, `params diff`, `repro --dry-run` and
  `--explain`) expand each mapped stage by the same rule, from the producer's lock
  and listing, when the producer is current: its lock matches, and its output is
  in the cache or, with a remote configured, missing from it and restorable by
  `pivot pull`. A fresh clone therefore lists every instance before any pull.
  Until then a mapped stage is *not yet expanded*: `pivot status` shows it
  (`not_expanded` in `--json`) with what it waits for, `pivot list` lists it under
  "Mapped stages, not yet expanded" (`templates` in `--json`), and `pivot dag`
  labels its node. `params show` and `params diff` without targets list each such
  mapped stage that takes parameters under the same heading (on stderr beside
  `--json`). `checkout`, `push`, `pull` and `params` refuse a target naming
  it, and `verify` fails it. `pull` first fetches the listing of a current
  producer that the clone lacks, then expands. `pivot commit` commits the
  producer first, then expands and commits each stale instance, or the instances
  its targets name (`pivot commit p/plan p/cut@a`).
- **Mapping rules.** `pipeline.map` takes a stage's directory output, or a
  directory field of a multi-output stage (`out.parts`). It refuses an input, one
  file of a directory, any other output, and a handle that is already mapped. A
  stage called with a mapped stage's output is mapped over the same files; one
  stage cannot be mapped over two directories, and no stage can take every
  instance's output at once (a stage that receives a mapped handle is mapped).
- **Parameters.** `params.yaml` key `p/cut` applies to every instance, and
  `p/cut@a` to one, whether or not a run has added it; each is checked against
  the mapped stage's `StageParams`. A mapped variant's own name
  (`p/cut@x`) is refused: its instances read `p/cut` and `p/cut@x@<key>`.
- **Files left behind.** A key that leaves the plan has no instance in the next
  run; its lock, listing and outputs stay, and `pivot status` does not count them
  as orphaned.

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
