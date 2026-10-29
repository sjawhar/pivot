# CLI Reference

Complete reference for all Pivot command-line commands.

## Quick Reference

| Task | Command |
|------|---------|
| Run pipeline | `pivot repro` |
| Run specific stages + deps | `pivot repro pipeline/stage1 pipeline/stage2` |
| Run single stage (no deps) | `pivot run pipeline/stage` |
| Run with live output | `pivot repro --show-output` |
| See what would run | `pivot repro -n` |
| Understand why stage runs | `pivot status --explain stage` |
| List all stages | `pivot list` |
| Show stage status | `pivot status` |
| Visualize DAG | `pivot dag` |
| Compare data files | `pivot diff data/demo/clean.jsonl` |
| Verify reproducibility | `pivot verify` |
| Import from remote repo | `pivot import REPO_URL ARTIFACT --out PATH` |
| Update imported artifacts | `pivot update` |
| Push outputs to remote | `pivot push` |
| Fetch to local cache | `pivot fetch` |
| Pull outputs from remote | `pivot pull` |
| Reset fingerprint cache | `pivot fingerprint reset` |
| Watch for changes | `pivot repro --watch` |

---

## Global Options

All commands support:

| Option | Description |
|--------|-------------|
| `--verbose` / `-v` | Show detailed output |
| `--quiet` / `-q` | Suppress non-essential output |
| `--help` | Show help message |

**Output.** Three rules hold for every command:

- **URL credentials are hidden.** Any URL Pivot prints — an import's repository URL, a remote
  cache URL, a URL in a warning, a log line or an error — shows `***` in place of its user-info:
  `https://user:token@host/repo.git` prints as `https://***@host/repo.git`. The stored URL is
  unchanged, and a URL without credentials prints byte for byte. An scp-like `git@host:path` is
  not a URL and prints as typed.
- **Control characters are escaped in text output.** A value Pivot read from a file (a `.pvt`
  file, a lock file, `.pivot/config.yaml`, `params.yaml`, a remote listing) prints each control
  character as visible text (`\x1b`), tab and newline included, so one value stays on one line
  and a crafted repository cannot drive the terminal. `--json` output is unchanged: JSON escapes
  these characters itself.
- **`--json` and `--jsonl` are strict JSON.** A non-finite float prints as the string
  `"Infinity"`, `"-Infinity"` or `"NaN"`, so a strict RFC 8259 parser reads every document. See
  [CI Integration](../guides/ci-integration.md#pivot-status---json).

**Errors.** A failure pivot recognizes prints its own message, often with a tip. Any other failure
prints one line: an operating-system error as its message and the path it names (for example
`Error: Permission denied: /home/me/proj/.pivot/cache/files/c9`), anything else as
`Error: Unexpected <type>: <message>`. In `run` and `repro`, a stage prints that one line under
its name only when pivot's own file operation on the files it manages failed: storing the stage's
outputs in the cache, opening the state database, reading or writing the stage's lock file,
preparing the stage's output path, and creating the output file and renaming it into place in a
format pivot writes itself (text, JSON, JSON Lines, YAML, pickle, DataFrame JSON Lines). Every
other stage failure prints its traceback, as before: one in your stage function, in a loader or
writer (pivot's own included, with any code of yours it runs), and any failure pivot did not
classify. The `--json` record keeps the stage's full reason in both cases: its last line and your
own frames are what they were, and pivot may add a frame of its own above them.
`pivot --verbose <command>` also prints the traceback, after the TUI exits for `--tui`.

The same cause therefore reads differently by output format, because a format that encodes your
value as it writes it runs your value's own code inside the write:

- **A disk that fills while the file is written** is one line for a text, YAML or DataFrame JSON
  Lines output, where pivot writes text it has already encoded. It prints the traceback for a
  JSON, JSON Lines or pickle output, where `json` and `pickle` encode your value as they write
  it, so a `__getstate__`, a `__reduce__` or a container of yours may be what failed.
- **A read-only output directory** is one line for those six formats, which pivot opens itself.
  It prints the traceback for a CSV output and a plot, where pandas and matplotlib open the path
  while running your value's code, and for a `pathlib.Path` output, whose move pivot has not
  classified.
- **A directory pivot has to create for the output** is one line for every format, including CSV
  and a plot: pivot creates it before the writer runs.

The first two rules above hold for these lines too: a credential in a path or a message is
hidden, and a control character in either is escaped, in the one line and in the traceback alike.

**Path arguments.** `pivot track`, `push`, `fetch` and `pull` read a relative path from the
current directory. `diff`, `checkout`, `repro`, `dag`, `get` and `check-ignore` read it from the
project root, so from `sub/` they take the file `data/x.csv` as `data/x.csv`, not
`../data/x.csv`. `pivot get --output` reads its path from the current directory.

An absolute path that reaches the project through a symlink outside it counts as the place it
points to in `diff`, `checkout`, `repro`, `dag`, `push`, `fetch`, `pull` and `track`. The
symlink can point to the project root, to one of its ancestors, or to a directory inside the
project: from a `link -> project` symlink, `link/data/x.csv` names `data/x.csv`. A symlink
inside the project is not followed: from `project/dlink -> data`, `dlink/x.csv` matches no
output in `diff`, `checkout`, `repro`, `dag`, `push`, `fetch` and `pull`, so give the path it
points to, `data/x.csv`. `track` tracks a file through such a symlink, with a warning that the
file is inside a symlinked directory, unless the directory really lies outside the project:
then the `.pvt` would be written outside too, where no command finds it again, so `track`
refuses ("its .pvt would be written outside the project").

`pivot get` refuses every absolute path ("Path traversal not allowed in target").
`pivot check-ignore` keeps a path under a symlink as typed, so a pattern anchored to the project
root, such as `data/*.tmp`, does not match it. Give those two the path relative to the project
root.

A path that leaves the project, including `link/../x.csv`, is refused by `checkout`, `push`,
`fetch`, `pull` and `track` (`resolves outside project root`) and matches no output in
`diff`, `repro` and `dag`.

**A `.pvt` restores its file only inside the project.** `checkout`, `pull` and
`repro --checkout-missing` read every `.pvt` under the project root and restore the file its
`path:` names. A `.pvt` is skipped, with a warning naming it and exit code 0, when its `path:`
names no file (`.`, `./` or empty), holds `..`, is absolute, or names a file whose directory
really lies outside the project, through a symlink. `pivot track` and `pivot import` write
`path:` as the name of the file beside the `.pvt`, which is always restored; any other value
is warned about, because `checkout x.pvt`, `get`, `update` and `push` take the file beside the
`.pvt` while `checkout` with no target takes `path:`.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PIVOT_CACHE_DIR` | Override cache directory location |

**`PIVOT_CACHE_DIR`** takes precedence over the `cache.dir` config setting. Relative paths are
resolved against the project root. Empty or whitespace-only values are treated as unset, falling
back to the config file or `<state dir>/cache`. A cache directory that Pivot creates inside a
configured state directory, other than `<state dir>/cache`, gets its own `.gitignore` that
ignores everything in it. A cache directory that equals, contains, or is inside
`<state dir>/stages` instead warns once and uses `<state dir>/cache`, so it cannot hide committed
stage locks.

---

## Pipeline Execution

### `pivot repro`

Reproduce pipeline stages with full dependency resolution. This is the primary command for running pipelines.

```bash
pivot repro [STAGES...] [OPTIONS]
```

**Arguments:**

- `STAGES` - Stage names (`pipeline/stage`) or output identities to run (optional, runs all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--dry-run` / `-n` | Show what would run without executing |
| `--explain` / `-e` | Show detailed breakdown of why stages run |
| `--force` / `-f` | Force re-run of stages, ignoring cache (in --watch mode, first run only) |
| `--watch` / `-w` | Watch for file changes and re-run affected stages |
| `--debounce MS` | Debounce delay in milliseconds (default: 300, requires --watch) |
| `--tui` | Use interactive TUI display (default: plain text) |
| `--jsonl` / `--json` | Stream results as JSONL (one JSON object per line) |
| `--show-output` | Stream stage stdout/stderr to terminal |
| `--tui-log PATH` | Write TUI messages to JSONL file for monitoring |
| `--no-commit` | Run stages without writing locks or cache; StateDB keeps a record of the `*.pvt` files each stage wrote, and file hashes are still saved |
| `--resources NAME=CAPACITY` | Override a scheduler resource capacity; may be repeated |
| `--fail-fast` | Stop on first failure (default) |
| `--keep-going` / `-k` | Continue running stages after failures; skip only downstream dependents |
| `--serve` | Start RPC server for agent control (requires --watch) |
| `--allow-uncached-incremental` | Allow running stages with incremental output files not in cache |
| `--checkout-missing` | Restore tracked files from cache before running |
| `--allow-missing` | Allow missing dep files if tracked (only affects --dry-run) |
| `--all` | Run across all pipelines in the project |

**Examples:**

```bash
# Run entire pipeline
pivot repro

# Run specific stages and their dependencies
pivot repro demo/train demo/evaluate

# See what would run
pivot repro --dry-run

# Watch mode - re-run on file changes
pivot repro --watch

# Continue after failures
pivot repro --keep-going

# Allow two stages that request gpu to run together
pivot repro --resources gpu=2
```

---

### `pivot run`

Execute specified stages directly, without resolving dependencies. Use this when you want to run specific stages in a specific order.

```bash
pivot run STAGES... [OPTIONS]
```

**Arguments:**

- `STAGES` - Stage names to run (required, at least one), qualified by pipeline (`demo/train`)

**Options:**

| Option | Description |
|--------|-------------|
| `--force` / `-f` | Force re-run of stages, ignoring cache |
| `--tui` | Use interactive TUI display (default: plain text) |
| `--jsonl` / `--json` | Stream results as JSONL (one JSON object per line) |
| `--show-output` | Stream stage stdout/stderr to terminal |
| `--tui-log PATH` | Write TUI messages to JSONL file for monitoring |
| `--no-commit` | Run stages without writing locks or cache; StateDB keeps a record of the `*.pvt` files each stage wrote, and file hashes are still saved |
| `--resources NAME=CAPACITY` | Override a scheduler resource capacity; may be repeated |
| `--fail-fast` | Stop on first failure |
| `--keep-going` / `-k` | Continue running stages after failures |
| `--allow-uncached-incremental` | Allow running stages with incremental output files not in cache |
| `--checkout-missing` | Restore tracked files from cache before running |
| `--all` | Run across all pipelines in the project |

**Examples:**

```bash
# Run a single stage (no dependencies)
pivot run demo/train

# Run multiple stages in order
pivot run demo/preprocess demo/train

# Stop immediately on failure
pivot run demo/preprocess demo/train --fail-fast
```

---

## Pipeline Introspection

### `pivot list`

List all registered stages.

```bash
pivot list [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--deps` | Show stage dependencies |
| `--all` | List stages across all pipelines in the project |

---

### `pivot status`

Show pipeline, tracked files, and remote status.

```bash
pivot status [STAGES...] [OPTIONS]
```

**Arguments:**

- `STAGES` - Stages to check (optional, checks all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--verbose` / `-v` | Show all stages, not just stale |
| `--explain` / `-e` | Show detailed breakdown of why stages would run |
| `--json` | Output as JSON. Writes one document to stdout; notes go to stderr. |
| `--stages-only` | Show only pipeline status |
| `--tracked-only` | Show only tracked files |
| `--remote-only` | Show only remote status |
| `--remote` / `-r` | Include remote sync status |
| `--check-imports` | Check for import updates (requires network). JSON adds an `imports` field. |
| `--all` | Show status across all pipelines in the project |

---

### `pivot commit`

Commit current workspace state for stages. Hashes deps and outputs on disk, writes lock files, and updates cache.

```bash
pivot commit [STAGES...] [OPTIONS]
```

Without arguments, commits all stale stages. With stage names, unconditionally commits those stages.

A `.pvt` file inside an output is the stage's only when the stage's lock lists it there, or
Pivot recorded it there before writing it (a `pivot repro` run, `--no-commit` included, or
`pivot get` of the stage) and it is unchanged since. `pivot commit` refuses, naming it, any
other `.pvt` in an output, such as a user's sidecar or one this version of Pivot cannot read,
instead of recording it as the stage's output. A refusal fails that stage alone: the command
still commits the other stages, lists the refused one under `Failed`, and exits 1.

The code a lock records depends on where the outputs came from. If they are the outputs a `pivot repro --no-commit` run wrote, unchanged since, the lock records the code that run ran, so an edit made while it ran, or since, still reruns the stage. Otherwise (outputs edited by hand, or written another way) the lock records the code on disk now: the "trust me" path.

**Options:**

| Option | Description |
|--------|-------------|
| `--all` | Commit across all pipelines in the project |

---

### `pivot history`

List recent pipeline runs.

```bash
pivot history [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--limit` / `-n N` | Number of runs to show |
| `--json` | Output as JSON |

---

### `pivot show`

Show details of a specific run.

```bash
pivot show [RUN_ID] [OPTIONS]
```

**Arguments:**

- `RUN_ID` - Run ID to show (optional, shows most recent if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

---

### `pivot dag`

Visualize the pipeline DAG.

```bash
pivot dag [TARGETS...] [OPTIONS]
```

Shows the dependency graph of artifacts (default) or stages. Without targets, shows the entire graph. With targets, shows the subgraph containing those targets and their upstream dependencies.

**Arguments:**

- `TARGETS` - Stage names or artifact paths to filter (optional, shows entire graph if not specified). Stage names take precedence when a name matches both a stage and a file path.

**Options:**

| Option | Description |
|--------|-------------|
| `--dot` | Output Graphviz DOT format |
| `--mermaid` | Output Mermaid format |
| `--md` | Output Mermaid wrapped in markdown |
| `--stages` | Show stages as nodes (default: artifacts) |
| `--all` | Show the DAG across all pipelines in the project |

**Examples:**

```bash
# Show entire artifact DAG
pivot dag

# Show DAG for specific stage and its dependencies
pivot dag demo/train

# Output as Mermaid diagram
pivot dag --mermaid

# Show stage-level DAG instead of artifacts
pivot dag --stages

# Output Graphviz DOT for external rendering
pivot dag --dot > pipeline.dot
```

---

### `pivot verify`

Verify pipeline was reproduced and outputs are available.

```bash
pivot verify [STAGES...] [OPTIONS]
```

Checks that all stages are cached (code, params, deps match lock files) and output files exist locally or on remote. Use in CI pre-merge gates to ensure pipeline is reproducible.

For a directory output, `pivot verify` also checks that the workspace path is a directory. A file
at that path fails verification.

Every committed listing file a stage's lock records, its outputs' and its directory inputs', must
hash to its record; one that does not fails that stage, naming the file and how to put it right.
With no stage named, `pivot verify` also checks each tracked directory's `<path>.pvt.listing`
against its `.pvt`, whether or not a stage reads it, and fails naming that path.

**Arguments:**

- `STAGES` - Stage names to verify (optional, verifies all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--allow-missing` | Allow missing local files if on remote |
| `--json` | Output as JSON |
| `--all` | Verify all stages (ignore target filtering) |

**Exit Codes:**

- `0` - Verification passed
- `1` - Verification failed (stale stages or missing files)

**Examples:**

```bash
# Verify entire pipeline
pivot verify

# Verify specific stages
pivot verify demo/train demo/evaluate

# CI gate: allow missing local files if on remote
pivot verify --allow-missing

# JSON output for scripting
pivot verify --json
```

---

## File Tracking

### `pivot track`

Track files or directories for caching.

```bash
pivot track PATHS... [OPTIONS]
```

**Arguments:**

- `PATHS` - File or directory paths to track (required)

**Options:**

| Option | Description |
|--------|-------------|
| `--force` / `-f` | Overwrite existing `.pvt` files and a directory's `.pvt.listing` file |

Creates a `.pvt` sidecar for a file. For a directory it also creates
`<path>.pvt.listing`, the committed listing of its files and empty directories;
commit both files. `track` refuses a target that is the `.pvt` or listing file
of another tracked path, or whose own `.pvt` or listing file would be another
tracked path, naming that path; any other name can be tracked, one ending in
`.pvt` or `.pvt.listing` included. It warns when git ignores either sidecar.
`--force` rewrites or removes a `<path>.pvt.listing` only when Pivot wrote it:
the cache (also the `.pivot` cache from before `core.state_dir` was set) holds
exactly its bytes, or, with no cached copy, its bytes hash to the directory the
`.pvt` records. Any other file there is refused, naming it.

`pivot track` loads every pipeline in the project and refuses a new path that overlaps a
declared output of any of them, including pipelines this directory does not select, and names
the stage that writes that output. A `pipeline.py` that fails to load makes `pivot track` and
`pivot import` refuse until it is fixed; one that loads but defines no Pivot pipeline declares no
outputs. A file tracked or imported inside an output before this check stays where it is:
`pivot status` and `pivot repro` (with `--all` for another pipeline's output) warn once per run
naming it, `pivot track --force` updates it in place with the same warning, and any stage write,
restore, `pivot commit`, forced checkout or forced `pivot get` that would remove it or its `.pvt`
sidecar fails before changing anything. A tracked file and sidecar copied into a stage's output
while a run replaces that output fail the stage the same way, naming them: before the stage
writes, and when a symlink-mode commit links a directory output to its cache tree, Pivot renames
an old directory output aside and checks it again before deleting it, and puts it back, with both
files, when it finds them; it checks an old file output again in place, just before it removes
it. An incremental stage's previous output is restored over the old one, with nothing removed
before the restore's checks. Not covered: a file moved over a file output in the instant between
Pivot's last check and its removal (a stage write) or replacement (a restore) is removed; a file
moved over a file output while the stage's own writer writes it is replaced by the stage's
output: for an output Pivot caches, the commit then fails the stage, naming the sidecar, while a
metric output or a `--no-commit` run finishes and the next `pivot status` or `pivot repro` warns,
naming it; on a filesystem that cannot rename a directory aside (overlayfs with `redirect_dir`
off), Pivot checks the old directory in place, and a file copied in after that check is deleted
with it; a commit in hardlink mode (or symlink mode, for a file output) checks the output once,
before it links each file the stage wrote to the cache; and the links a run makes at output paths
for outputs kept only in the cache (the presentation layer) are checked once, before they replace
what is there. A forced restore of a tracked directory (or of a tracked file over a directory)
likewise fails before it would remove another tracked file, import or `.pvt` sidecar beneath it.
A restore requested through a stage name restores that stage's output, so a tracked file at that
output's path makes it fail too. Move the file and its `.pvt` sidecar outside the output (for
example under `data/raw/`). The check finds a sidecar beside its file, where Pivot writes it; a
hand-edited `.pvt` elsewhere whose `path` points into an output is not protected.

A stage may write a `*.pvt` file in a directory output: a member of a `directory` output, or a
file inside a directory a `PathOnly` output returns. Pivot treats such a file as the stage's
when the stage's lock records it at that path, or Pivot recorded it there before writing it and
it is unchanged since. A run, a run-cache restore (also under `pivot repro --no-commit`) and
`pivot get` of a stage record each such file before they write it; `pivot checkout` and
`pivot pull` restore only what the lock records. After the lock changes through git (for
example `git checkout` of an older commit), a file at a path the lock records whose content is
in the cache is the stage's too: the stage wrote it in a run Pivot cached. Every other `.pvt`
under an output is a user's sidecar, which a `.pvt` this version of Pivot cannot read may be.
`pivot commit` claims no other `.pvt`: it refuses, naming the file, rather than record it as the
stage's output, after which the next run would delete it.

---

### `pivot checkout`

Restore tracked files and stage outputs from cache.

```bash
pivot checkout [TARGETS...] [OPTIONS]
```

**Arguments:**

- `TARGETS` - Targets to restore (optional, restores all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--checkout-mode MODE` | `symlink`, `hardlink`, or `copy` |
| `--force` / `-f` | Overwrite existing files when every required cache object is available; otherwise refuse without changing the workspace file |
| `--only-missing` | Only restore files that don't exist on disk. For a directory, restores its missing files and empty directories; if a file differs from the target, or an extra file or directory exists, replaces the directory when every such file is pivot's own earlier output (its content is in the cache; an extra empty directory holds nothing to lose), and otherwise refuses, naming the local changes, so they are never overwritten; use `--force` to overwrite |
| `--all` | Checkout all targets (ignore target filtering) |

Checkout summaries count restored, skipped, and missing **files**. Each file in a directory output
counts separately.

A checkout that would replace an existing path holding another user's file refuses: a stage
output that holds a tracked file or `.pvt` sidecar, or a tracked path that holds another tracked
file, import or `.pvt` sidecar beneath it. A target that names a stage restores that stage's
outputs, so a tracked file or sidecar at one of their paths refuses it too. Only `--force` and an
`--only-missing` directory swap replace an existing path. `--force` checks every target before it
changes any: if it refuses one, it changes nothing and exits 1, naming every refused target and
the files that block it. It then checks each target again before it builds that target's
replacement, and once more at its swap, over the old output once it is moved aside, so a tracked
file and sidecar copied into an output while the command runs, even while that output's own
replacement is being built, refuse that target alone, named, and the command exits 1 with the
file kept. `pivot pull --force` restores the same way.
An `--only-missing` swap is checked when it is decided, and at its swap. A directory inside an
output that Pivot cannot read refuses the restore too, naming that directory: a tracked file may
hide inside where Pivot cannot look.

---

### `pivot get`

Retrieve files or stage outputs from a specific git revision.

Each output comes from the local cache, then git, then the remote: `get` downloads the cache objects the output is missing and restores it from the cache. An output that fails is reported with its reason; `get` still restores and reports the other targets. An object the remote does not serve (missing, corrupt, refused with an HTTP status, or a download that breaks off) fails only the outputs that need it. If no connection to the remote can be made (a DNS failure, a refused connection, a connect timeout, a TLS failure or a proxy that cannot be connected to, after `remote.retries` retries; on S3 also a connection reset on every retry), `get` stops using the remote for the rest of the command: each later output that needs it fails naming the remote, even if the remote would answer by then, so run `get` again. Missing or refused credentials do not stop it: each output that needs the remote tries again.

A target whose `.pvt` file at the revision has a field `get` cannot use fails naming the file and that field: a `path` or `hash` that is not a string, a `manifest` that is not a list, a manifest entry that is not a mapping, lacks `relpath`, `hash` or `isexec`, or whose `relpath` or `hash` is not a string, and a manifest `relpath` or an `empty_dirs` entry holding a NUL byte. A target already in the workspace is skipped first, as for a valid `.pvt` file. A `.pvt` file that is not the shape pivot writes (a mapping with `path`, `hash` and `size`, and only keys pivot writes) is not read as a `.pvt` file, nor is one whose fields `get` can use but whose `path` is absolute or holds `..`: `get` looks the target up in git, and stops before restoring anything if git does not track it.

```bash
pivot get TARGETS... --rev REVISION [OPTIONS]
```

**Arguments:**

- `TARGETS` - Stage names (`demo/train`, or `demo/train:key` for one output), or paths of tracked (`.pvt`) or git-tracked files (required). A stage output's workspace path, such as `data/demo/train.jsonl`, works only if git tracks that file; name the stage instead.

**Options:**

| Option | Description |
|--------|-------------|
| `--rev` / `-r REV` | Git revision (SHA, branch, tag) - required |
| `--remote NAME` | Remote name (uses default if not specified). An unknown name fails before anything is restored |
| `--output` / `-o PATH` | Output path (single file only) |
| `--checkout-mode MODE` | `symlink`, `hardlink`, or `copy` |
| `--force` / `-f` | Overwrite existing files. Refuses, before changing anything, when a replaced path holds another user's file: a stage output or an `--output` path that holds a tracked file or `.pvt` sidecar, or a tracked path that holds another tracked file, import or `.pvt` sidecar beneath it. Each target is checked again before its replacement is built and once more at its swap, over the old entry once it is moved aside: one that a user's file entered meanwhile, even while its replacement was being built, fails alone, named, and the others are still restored. A target whose `.pvt` is not valid is only reported, so it refuses nothing. A tracked file restored over a file is always overwritten |

A downloaded cache object has no executable bit, so an executable file in a directory output cannot be hard-linked or symlinked to it. With a mode order that has no `copy` (`--checkout-mode hardlink` or `symlink`, or a `cache.checkout_mode` without `copy`), that output fails with an error naming the file. The default order, hardlink, symlink, copy, ends with `copy`, which restores the bit.

**Examples:**

```bash
# Get a stage's outputs from a specific commit
pivot get demo/train --rev abc123

# Get stage output from branch
pivot get demo/train --rev feature-branch

# Get a stage's single output to a custom path (relative to the current directory)
pivot get demo/train --rev v1.0 --output old_train.jsonl

# Get from a specific remote
pivot get demo/train --rev abc123 --remote origin
```

---

## Metrics

### `pivot metrics show`

Display metric values.

```bash
pivot metrics show [TARGETS...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--md` | Output as Markdown table |
| `--recursive` / `-R` | Search directories recursively |
| `--precision N` | Decimal precision for floats (default: 5) |

---

### `pivot metrics diff`

Compare metrics against git HEAD.

```bash
pivot metrics diff [TARGETS...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--md` | Output as Markdown table |
| `--recursive` / `-R` | Search directories recursively |
| `--no-path` | Hide path column |
| `--precision N` | Decimal precision (default: 5) |

---

## Plots

### `pivot plots show`

Render plots as HTML gallery.

```bash
pivot plots show [TARGETS...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--output` / `-o PATH` | Output HTML path (default: `<state dir>/plots/index.html`, ignored by git). Image paths are relative to the HTML file. |
| `--open` | Open browser after rendering |

---

### `pivot plots diff`

Show which plots changed since last commit.

```bash
pivot plots diff [TARGETS...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--md` | Output as Markdown table |
| `--no-path` | Hide path column |

---

## Parameters

### `pivot params show`

Display current parameter values.

```bash
pivot params show [STAGES...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--md` | Output as Markdown table |
| `--precision N` | Decimal precision (0-10, default: 5) |

---

### `pivot params diff`

Compare parameters against git HEAD.

```bash
pivot params diff [STAGES...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--md` | Output as Markdown table |
| `--precision N` | Decimal precision (0-10, default: 5) |

---

## Data Comparison

### `pivot diff`

Compare data files against git HEAD.

```bash
pivot diff TARGETS... [OPTIONS]
```

**Arguments:**

- `TARGETS` - Stage outputs to compare (required): a stage name, or a path at, inside or above a
  stage output. `.` compares every output. A directory output compares its CSV, JSON, and JSONL
  files one by one, and a path inside it compares the files at or below that path. A target that
  names no CSV, JSON, or JSONL file is named in a warning and not compared. An output that moved
  to a new path since git HEAD is compared as removed at its old path and added at its new one; a
  path that holds a file at git HEAD and a directory in the workspace, or the reverse, is named
  in a warning with both kinds and not compared.

Each `--json` entry carries `change_type`: `added`, `removed` or `modified`. A kind change has
no entry, only its warning. When git HEAD's side of an entry is not in the local cache, as in a
fresh clone, the entry keeps its change type with that side's rows and columns `null`. A warning
names the path and says to run `pivot fetch`, which fills the cache without touching the
workspace, so the next diff compares that entry; every other entry is compared as usual.

**Options:**

| Option | Description |
|--------|-------------|
| `--key COLUMNS` | Comma-separated key columns for row matching |
| `--positional` | Use positional (row-by-row) matching |
| `--summary` | Show summary only (schema + counts) |
| `--no-tui` | Print to stdout instead of TUI |
| `--json` | Output as JSON (implies --no-tui) |
| `--md` | Output as Markdown (implies --no-tui) |
| `--max-rows N` | Max rows for comparison (default: 10000) |

**Examples:**

```bash
# Interactive TUI mode
pivot diff data/demo/clean.jsonl

# Key-based row matching
pivot diff data/demo/clean.jsonl --key id,timestamp

# JSON output for scripting
pivot diff data/demo/clean.jsonl --json
```

---

## Remote Storage

Remote storage is configured using `pivot config` commands. See the [Configuration](#configuration) section below for details.

### `pivot remote list`

List configured remote storage locations.

```bash
pivot remote list
```

Shows all remotes configured in the project, with the default marked.

---

### `pivot push`

Push cached outputs to remote storage.

```bash
pivot push [TARGETS...] [OPTIONS]
```

**Arguments:**

- `TARGETS` - Stage names or file paths to push (optional, pushes all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--remote` / `-r NAME` | Remote name (uses default if not specified) |
| `--dry-run` / `-n` | Show what would be pushed |
| `--jobs` / `-j N` | Parallel upload jobs (default: 20) |
| `--all` | Push all stages (ignore target filtering) |

---

### `pivot pull`

Pull cached outputs from remote storage.

```bash
pivot pull [TARGETS...] [OPTIONS]
```

**Arguments:**

- `TARGETS` - Stage names or file paths to pull (optional, pulls all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--remote` / `-r NAME` | Remote name (uses default if not specified) |
| `--dry-run` / `-n` | Show what would be pulled |
| `--jobs` / `-j N` | Parallel download jobs |
| `--force` / `-f` | Overwrite existing workspace files |
| `--only-missing` | Only restore files that don't exist in workspace (the default without `--force`). A directory is replaced when every changed or extra file is pivot's own earlier output (an extra empty directory holds nothing to lose), and refused (naming the files) when one holds a local change the cache lacks |
| `--checkout-mode MODE` | `symlink`, `hardlink`, or `copy` |
| `--all` | Pull all stages (ignore target filtering) |

---

### `pivot fetch`

Fetch cached outputs from remote storage to local cache without restoring to workspace. Use `pivot pull` to also restore files, or `pivot checkout` to restore from cache.

```bash
pivot fetch [TARGETS...] [OPTIONS]
```

**Arguments:**

- `TARGETS` - Stage names or file paths to fetch (optional, fetches all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--remote` / `-r NAME` | Remote name (uses default if not specified) |
| `--dry-run` / `-n` | Show what would be fetched |
| `--jobs` / `-j N` | Parallel download jobs |
| `--all` | Fetch all stages (ignore target filtering) |

---

## Imports

### `pivot import`

Import a stage output from another Pivot repository.

```bash
pivot import REPO_URL ARTIFACT --out PATH [OPTIONS]
```

**Arguments:**

- `REPO_URL` - URL of the source Pivot repository (HTTPS, SSH, or `file://`)
- `ARTIFACT` - Identity of the output in the source repo: `pipeline/stage` for a single-output stage, `pipeline/stage:key` for one field of a multi-output stage, as printed by `pivot list --deps` there. A directory output is imported one file at a time: `pipeline/stage:path`, or `pipeline/stage:key/path` for a `TypedDict` field

**Options:**

| Option | Description |
|--------|-------------|
| `--rev REV` | Git ref to import from (branch, tag, commit; default: main) |
| `--out PATH` | Local path to write the file to (without it, the artifact's name with `/` for `:`, so `prep/split:test` writes `prep/split/test`) |
| `--force` | Overwrite existing files |
| `--no-download` | Create .pvt metadata without downloading |

**Examples:**

```bash
# Import a single-output stage's result
pivot import https://github.com/org/ml-models training/train --out data/raw/model.pkl

# Import one field of a multi-output stage, from a tag
pivot import https://github.com/org/data prep/split:test --rev v1.0 --out data/raw/test.csv

# Create metadata only (download later with pivot update)
pivot import https://github.com/org/data prep/split:test --out data/raw/test.csv --no-download
```

The imported file is tracked by a `.pvt` sidecar next to it; read it in a pipeline with `pipeline.input("model.pkl")`.

An import loads every pipeline in the project and refuses a **new** destination that overlaps a
declared stage output, whether or not that pipeline is currently selected. Use an input location
such as `data/raw/` instead. An import made there before this check can be updated with `--force`;
Pivot warns but leaves its data and `.pvt` sidecar in place until you move them outside the output.

---

### `pivot update`

Update imported artifacts from their source repositories.

```bash
pivot update [TARGETS...] [OPTIONS]
```

If no TARGETS specified, updates all imports found in the project.

**Arguments:**

- `TARGETS` - Import targets to update (optional, updates all imports if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--rev REV` | Override git ref for update |
| `--dry-run` | Show what would change without modifying |

**Examples:**

```bash
# Update all imports
pivot update

# Check for updates without applying
pivot update --dry-run

# Update specific import
pivot update data.csv

# Update to a specific revision
pivot update data.csv --rev v2.0
```

---

## Configuration

### `pivot config list`

List all configuration values.

```bash
pivot config list [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--global` | Show only global config |
| `--local` | Show only local config |
| `--json` | Output as JSON |

Without `--global` or `--local`, `config list` shows the merged configuration after validation.
It fails in one line when the local `core.state_dir` is not a string. It marks any other
`core.state_dir` that is not a path, such as an empty value, with
`invalid: core.state_dir must be a path`.

`--local` and `--global` show the values stored in that one file, as written, and exit 0 for any
stored `core.state_dir`: `core.state_dir: 5` shows as `core.state_dir = 5 (local)`. Inside a
project, the text forms label every row of a global `core.state_dir` `(global, ignored)`, whatever
its type: a mapping shows one row for each nested key. Only the project's own
`.pivot/config.yaml` sets the state directory. `--json` prints the values as JSON, with no source
labels for any key.

---

### `pivot config get`

Get a configuration value.

```bash
pivot config get KEY [OPTIONS]
```

**Arguments:**

- `KEY` - Config key (e.g., `cache.dir`, `remotes.origin`, `default_remote`)

**Options:**

| Option | Description |
|--------|-------------|
| `--global` | Get from global config |
| `--local` | Get from local config |
| `--json` | Output as JSON |

---

### `pivot config set`

Set a configuration value.

```bash
pivot config set KEY VALUE [OPTIONS]
```

**Arguments:**

- `KEY` - Config key (e.g., `cache.dir`, `remotes.origin`, `default_remote`)
- `VALUE` - Value to set

**Options:**

| Option | Description |
|--------|-------------|
| `--global` | Set in global config (~/.config/pivot/config.yaml) |

**Examples:**

```bash
# Add a remote
pivot config set remotes.origin s3://my-bucket/pivot-cache

# Set default remote
pivot config set default_remote origin

# Set global cache directory
pivot config set cache.dir /shared/cache --global

# Set max parallel workers
pivot config set core.max_workers 8
```

---

### `pivot config unset`

Remove a configuration value.

```bash
pivot config unset KEY [OPTIONS]
```

**Arguments:**

- `KEY` - Config key to remove

**Options:**

| Option | Description |
|--------|-------------|
| `--global` | Remove from global config |

**Examples:**

```bash
# Remove a remote
pivot config unset remotes.backup

# Clear default remote
pivot config unset default_remote
```

---

### Configuration Keys

| Key | Description | Default |
|-----|-------------|---------|
| `cache.dir` | Cache directory path | `<state dir>/cache` |
| `cache.checkout_mode` | Checkout mode order; a hard link refused because the cache object is at the filesystem's link limit makes a copy instead | `hardlink,symlink,copy` |
| `core.max_workers` | Most parallel workers (-1 = all CPUs); a run starts fewer when fewer of its stages can run at once | `-2` |
| `core.run_history_retention` | Keep last N runs | `100` |
| `core.state_dir` | Where Pivot keeps lock files, the state database, the default cache, the RPC socket, secret keys and plot galleries. Set it in the project's `.pivot/config.yaml`; a global value is ignored with a warning, and `config set --global core.state_dir` is refused. Two projects must never point at the same directory | `.pivot` |
| `resources` | Scheduler resource capacities | `{}` |
| `remote.jobs` | Parallel transfer jobs | `20` |
| `remote.retries` | Transfer retry count | `10` |
| `remote.connect_timeout` | Connection timeout (seconds) | `30` |
| `watch.debounce` | Watch debounce (milliseconds) | `300` |
| `display.precision` | Float display precision | `5` |
| `diff.max_rows` | Max rows for diff | `10000` |
| `default_remote` | Default remote name | (none) |

---

## Project Setup

### `pivot init`

Initialize a new Pivot project.

```bash
pivot init [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--force` / `-f` | Overwrite `.pivot/.gitignore` only for the default state directory. A configured state directory keeps any existing entry and creates a missing one atomically. |

Writes `.pivot/.gitignore`, a `.pivotignore` if there is none, and the line
`*.listing -merge` into `.gitattributes` (created if absent), so git never
merges a listing line by line (see
[Listing files](../concepts/outputs.md#listing-files)).

---

## Utilities

### `pivot doctor`

Check environment and configuration for issues.

```bash
pivot doctor [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSONL |
| `--remote` | Also check remote connectivity |

Configuration errors identify the invalid key and value and state the required format.

---

### `pivot completion`

Generate shell completion script.

```bash
pivot completion {bash|zsh|fish}
```

**Arguments:**

- `SHELL` - Shell type: `bash`, `zsh`, or `fish`

**Examples:**

```bash
# Bash (~/.bashrc)
eval "$(pivot completion bash)"

# Zsh (~/.zshrc)
eval "$(pivot completion zsh)"

# Fish (~/.config/fish/config.fish)
pivot completion fish | source
```

---

### `pivot fingerprint`

Manage function fingerprinting cache.

#### `pivot fingerprint reset`

Reset cached function fingerprints and cached stage manifests. Use after encountering stale cache issues or when troubleshooting unexpected stage re-runs.

```bash
pivot fingerprint reset
```

**Example:**

```bash
# Clear all cached fingerprints
pivot fingerprint reset
# Cleared 42 cached fingerprint entries.
```

---

### `pivot check-ignore`

Check if paths are ignored by .pivotignore.

```bash
pivot check-ignore [TARGETS...] [OPTIONS]
```

**Arguments:**

- `TARGETS` - Paths to check

**Options:**

| Option | Description |
|--------|-------------|
| `--details` / `-d` | Show matching pattern and source |
| `--json` | Output as JSON |
| `--show-defaults` | Show default patterns for starter .pivotignore |

Exit code 0 if any target is ignored, 1 if none are ignored.

It reads only the project's `.pivotignore` and answers as `git check-ignore` (git 2.52) answers
for the same patterns in a root `.gitignore`: a path under an ignored directory is ignored,
whatever later `!` patterns say, and a symlink is tested as a file even when it points to a
directory. (A git older than 2.52 answers otherwise for a `**` right after a pattern's literal
prefix: it took `foo**/bar` to ignore `foobar` too.) It differs from `git check-ignore` in four
ways: a relative path is taken from the project root, not the current directory; `pipeline.py`
and `.pivot/` at the project root are never ignored, whatever the patterns say; the project
root itself is never ignored, where `git check-ignore .` reports it ignored by `*`; and a line
with more than 256 runs of `*` tried at more than one place (not one at the end of the line,
nor a lone `*` right before a `/`; consecutive `**/` count as one) is matched differently, with
a warning: a negation is taken to match every path, and any other such line is not used.
Hashing applies the same rules to a directory input's members, except for an input directory
that an earlier rule ignores: an over-limit negation does not decide that directory is unignored,
so it is hashed whole. It is also whole when `.pivotignore` itself ignores it; `check-ignore`
reports its files ignored, but they count for its hash. See
[Directory inputs and `.pivotignore`](../concepts/dependencies.md#directory-inputs-and-pivotignore).

**Examples:**

```bash
# Check single file
pivot check-ignore app.log

# Show matching pattern details
pivot check-ignore --details *.pyc

# JSON output for scripting
pivot check-ignore --json build/ temp.log
```
