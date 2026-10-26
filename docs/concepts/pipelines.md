# Pipelines

A pipeline is a named group of stages. You build one with a
`with Pipeline("name") as pipeline:` block in a file called `pipeline.py`;
Pivot runs that file to discover the stages and their dependencies.

## Creating a Pipeline

```python
# pipeline.py
import pandas as pd

from pivot import Pipeline, StageParams, stage


class TrimParams(StageParams):
    keep: int = 2


@stage
def trim(raw: pd.DataFrame, params: TrimParams) -> pd.DataFrame:
    return raw.head(params.keep)


with Pipeline("demo") as pipeline:
    trim(pipeline.input("values.csv"), params=TrimParams())
```

Rules for `pipeline.py`:

- The pipeline must be bound to a module-level variable named `pipeline`.
  Otherwise discovery fails with `does not define a 'pipeline' variable`.
- The `with` block is ordinary Python: loops, conditionals, helper functions and
  dict comprehensions all work, because Pivot executes the block to build the DAG.
- Inside the block, a stage call records a stage and returns a handle; the stage
  runs later, in a worker process, when you run `pivot repro` or `pivot run`.
- Errors in the block (unsupported argument types, wrong arity) are collected
  and reported together when the block exits.

`Pipeline(name, root=None)` takes the pipeline name. A pipeline keeps its lock files and state in
the project's configured state directory and its outputs in the project's `data/`, so `root` may
only be the project root (the top-most directory that contains `.pivot/`) or omitted; discovery
refuses any other `root`.

### Where Pivot looks

Pivot uses the `pipeline.py` in the current directory if you are inside the
project, otherwise the one at the project root. `--all` (on `repro`, `run`,
`status`, `list`, `dag` and others) loads every `pipeline.py` under the project
root, skipping `.venv`, `.git`, `__pycache__` and similar directories.

## Stage Names

A stage's name is `<pipeline>/<function>`. The CLI takes qualified names:

```bash
pivot run demo/trim
pivot repro demo/trim
```

## Variants

Variants are Python loops. Calling the same stage more than once in one
pipeline gives the later calls a numeric suffix (`trim`, `trim@1`, `trim@2`).
For readable names, wrap calls in `pipeline.variant()`:

```python
# pipeline.py
import pandas as pd

from pivot import Pipeline, StageParams, stage


class TrimParams(StageParams):
    keep: int = 2


@stage
def trim(raw: pd.DataFrame, params: TrimParams) -> pd.DataFrame:
    return raw.head(params.keep)


@stage
def sizes(tables: list[pd.DataFrame]) -> dict:
    return {"rows": [len(t) for t in tables]}


with Pipeline("demo") as pipeline:
    raw = pipeline.input("values.csv")
    trimmed = []
    for keep in [1, 3]:
        with pipeline.variant(f"keep{keep}"):
            trimmed.append(trim(raw, params=TrimParams(keep=keep)))
    sizes(trimmed)
```

This defines `demo/trim@keep1`, `demo/trim@keep3` and `demo/sizes`. Nested
variant blocks append both names (`trim@outer@inner`). Variant names may
contain letters, digits, `_`, `.` and `-`; `/` and `@` separate the parts of
a stage name, so `pipeline.variant()` rejects them.

Each variant has its own lock file, outputs (`data/demo/trim@keep1.jsonl`) and
parameters. In `params.yaml`, a key without the `@` suffix applies to every
variant and a full name overrides one (see [Parameters](parameters.md)).

## Several Pipelines

Pipeline names must be unique within a project. A pipeline's name is the directory of its lock
files (`<state dir>/stages/<pipeline>/`) and outputs (`data/<pipeline>/`), so two pipelines with
one name would share both.
`pivot repro --all` refuses two `pipeline.py` files that declare the same
name, and names both files.

Imports resolve from the project root (the directory holding `.pivot/`): a root
`pipeline.py` imports sibling modules directly (`from stages import clean`); a
pipeline in a subdirectory imports with the full path
(`from report.helpers import x`, `from prep.pipeline import kept`).

A handle is an ordinary Python object, so one pipeline can consume another's
output by importing it. Pivot adds the upstream stages it needs and runs them
first. See [Multi-Pipeline Projects](../guides/multi-pipeline.md) for a full
example.

### `include()`

`include()` copies every stage of another pipeline into this one, which is how
you make one root pipeline from several. It can be called inside the `with`
block or after it:

```python
# pipeline.py
from pivot import Pipeline
from prep.pipeline import pipeline as prep
from report.pipeline import pipeline as report

pipeline = Pipeline("all")
pipeline.include(prep)
pipeline.include(report)
```

Included stages keep their names (`prep/...`, `report/...`) and their own state
directories. A stage reached twice (two included pipelines that share an
upstream stage, or one pipeline included twice) is added once. Two stages under
one name are refused unless they are one function (same file, name, code, closure
values and default values) called with the same inputs and input formats, outputs
and output formats, params, resources, variant and fingerprint settings; the
error names the first difference and the source files. A pipeline cannot include
itself.

A pipeline resolves `core.state_dir` when it loads, and its stages, runs and
caches all use that directory. That includes the fallback to `.pivot` when the
value, or the path it names, cannot be used: a pipeline loaded while the path
was a regular file keeps `.pivot` after the file is removed. So `include()`
refuses a pipeline loaded before an edit of `core.state_dir` (or of a symbolic
link it names, or of whether the path it names can be used) into one loaded
after it: the error names both directories. Load both pipelines after the
change. The pipelines of one `pivot` command, or of one discovery, always share
one directory.

## State and Cache Layout

`core.state_dir` selects the state directory and defaults to `.pivot`. The `.pivot/` marker stays
at the project root and holds the project configuration.

A state directory belongs to one project. Set `core.state_dir` in the project's own
`.pivot/config.yaml`: a value in the global config is ignored, with a warning. Never point two
projects at the same directory, including with an absolute path. They would share lock files
and overwrite each other's records of what ran, so each would see the other's stages as stale.

`config get core.state_dir` shows the stored configuration value. It does not resolve the state
directory, create anything, or warn when that stored value is unusable. `config list` likewise
does not resolve state; it displays a validated merged configuration. As in Pivot 0.5.2, a
non-string `core.state_dir` makes `config list` fail in one line that names the key and its
expected string type. A command that reads or writes state does resolve it. If
`.pivot/config.yaml` cannot be read, or its `core.state_dir` cannot be used or created as a
writable, searchable directory (including a read-only directory), that command prints one
warning naming the file and the problem. Pivot creates a configured state directory on its first
write. Every state-using command that Pivot 0.5.2 ran keeps its exit code and uses the default
`.pivot`; a state-using command that 0.5.2 failed still fails in one error line. The exception
is a hand-edited empty value: `show` returns 0 because it reads the retained `.pivot` run
history, and `history` reports that same history. `--quiet` silences the warning.

When `core.state_dir` names another directory but `.pivot` holds stage locks or cache objects,
Pivot reads a stage lock missing from the configured directory, and its cached objects, from
`.pivot`. Pivot 0.5.2 kept them there, and so does a run while `core.state_dir` cannot be used
(an empty value, or a path that is not a writable directory). Pivot warns once per command and
directs new writes to the configured state directory. Move the tracked locks with
`git mv .pivot/stages "<state dir>/stages"`, or delete them if such a run made them.

```
<project_root>/
  .pivot/
    config.yaml
  <state dir>/
    cache/files/       # Content-addressable cache (shared across pipelines)
    stages/<pipeline>/ # Per-stage lock files (commit these)
    state.lmdb/        # LMDB database (hashes, generations, run cache)
    locks/             # Artifact locks for concurrent execution
  data/raw/            # Inputs you provide
  data/<pipeline>/     # Stage outputs
  metrics/<pipeline>/  # Outputs tagged metric
  plots/<pipeline>/    # Outputs tagged plot
```

## Relationship to Other Concepts

- Stages receive [dependencies](dependencies.md) and return [outputs](outputs.md)
- Stage code is tracked via [fingerprinting](fingerprinting.md)
- Stage configuration is tracked via [parameters](parameters.md)
- Execution and skip logic use the [caching](caching.md) system
