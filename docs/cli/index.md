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

**Path arguments.** `pivot track`, `push`, `fetch` and `pull` read a relative path from the
current directory. `diff`, `checkout`, `repro`, `dag`, `get` and `check-ignore` read it from the
project root, so from `sub/` they take the file `data/x.csv` as `data/x.csv`, not
`../data/x.csv`.

An absolute path that reaches the project through a symlink outside it counts as the place it
points to in `diff`, `checkout`, `repro`, `dag`, `push`, `fetch`, `pull` and `track`. The
symlink can point to the project root, to one of its ancestors, or to a directory inside the
project: from a `link -> project` symlink, `link/data/x.csv` names `data/x.csv`. A symlink
inside the project is not followed: from `project/dlink -> data`, `dlink/x.csv` matches no
output in `diff`, `checkout`, `repro`, `dag`, `push`, `fetch` and `pull`, so give the path it
points to, `data/x.csv`. `track` tracks a file through such a symlink, with a warning that the
file is inside a symlinked directory.

`pivot get` refuses every absolute path ("Path traversal not allowed in target").
`pivot check-ignore` keeps a path under a symlink as typed, so a pattern anchored to the project
root, such as `data/*.tmp`, does not match it. Give those two the path relative to the project
root.

A path that leaves the project, including `link/../x.csv`, is refused by `checkout`, `push`,
`fetch`, `pull` and `track` (`resolves outside project root`) and matches no output in
`diff`, `repro` and `dag`.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PIVOT_CACHE_DIR` | Override cache directory location |

**`PIVOT_CACHE_DIR`** takes precedence over the `cache.dir` config setting. Relative paths are resolved against the project root. Empty or whitespace-only values are treated as unset, falling back to the config file (or `.pivot/cache` if no config is set).

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
| `--no-commit` | Run stages without writing locks, cache, or StateDB |
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
| `--no-commit` | Run stages without writing locks, cache, or StateDB |
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
| `--force` / `-f` | Overwrite existing .pvt files |

Creates `.pvt` manifest files for tracking files outside of stage outputs.

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

---

### `pivot get`

Retrieve files or stage outputs from a specific git revision.

Each output comes from the local cache, then git, then the remote: `get` downloads the cache objects the output is missing and restores it from the cache. An output that fails is reported with its reason; `get` still restores and reports the other targets.

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
| `--force` / `-f` | Overwrite existing files |

A downloaded cache object has no executable bit, so an executable file in a directory output cannot be hard-linked or symlinked to it. With a mode order that has no `copy` (`--checkout-mode hardlink` or `symlink`, or a `cache.checkout_mode` without `copy`), that output fails with an error naming the file. The default order, hardlink, symlink, copy, ends with `copy`, which restores the bit.

**Examples:**

```bash
# Get a stage's outputs from a specific commit
pivot get demo/train --rev abc123

# Get stage output from branch
pivot get demo/train --rev feature-branch

# Get a stage's single output to a custom path
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
| `--output` / `-o PATH` | Output HTML path (default: `.pivot/plots/index.html`, ignored by git). Image paths are relative to the HTML file. |
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
| `cache.dir` | Cache directory | `.pivot/cache` |
| `cache.checkout_mode` | Checkout mode order; a hard link refused because the cache object is at the filesystem's link limit makes a copy instead | `hardlink,symlink,copy` |
| `core.max_workers` | Most parallel workers (-1 = all CPUs); a run starts fewer when fewer of its stages can run at once | `-2` |
| `core.run_history_retention` | Keep last N runs | `100` |
| `core.state_dir` | State directory | `.pivot` |
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
| `--force` / `-f` | Overwrite existing .pivot/.gitignore |

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

**Examples:**

```bash
# Check single file
pivot check-ignore app.log

# Show matching pattern details
pivot check-ignore --details *.pyc

# JSON output for scripting
pivot check-ignore --json build/ temp.log
```
