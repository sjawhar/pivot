# CLI Reference

Complete reference for all Pivot command-line commands.

## Quick Reference

| Task | Command |
|------|---------|
| Run pipeline | `pivot repro` |
| Run specific stages + deps | `pivot repro stage1 stage2` |
| Run single stage (no deps) | `pivot run stage` |
| See what would run | `pivot repro -n` |
| Understand why stage runs | `pivot status --explain stage` |
| List all stages | `pivot list` |
| Show stage status | `pivot status` |
| Push outputs to remote | `pivot push` |
| Pull outputs from remote | `pivot pull` |
| Watch for changes | `pivot repro --watch` |

---

## Global Options

All commands support:

| Option | Description |
|--------|-------------|
| `--verbose` / `-v` | Show detailed output |
| `--quiet` / `-q` | Suppress non-essential output |
| `--help` | Show help message |

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

- `STAGES` - Stage names to run (optional, runs all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--dry-run` / `-n` | Show what would run without executing |
| `--explain` / `-e` | Show detailed breakdown of why stages run |
| `--force` / `-f` | Force re-run of stages, ignoring cache (in --watch mode, first run only) |
| `--watch` / `-w` | Watch for file changes and re-run affected stages |
| `--debounce MS` | Debounce delay in milliseconds (default: 300, requires --watch) |
| `--tui` | Use interactive TUI display (default: plain text) |
| `--json` | Output results as JSON |
| `--tui-log PATH` | Write TUI messages to JSONL file for monitoring |
| `--no-commit` | Defer lock files to pending dir for faster iteration |
| `--no-cache` | Skip caching outputs entirely for maximum iteration speed |
| `--keep-going` / `-k` | Continue running stages after failures (default: fail-fast) |
| `--serve` | Start RPC server for agent control (requires --watch) |
| `--allow-uncached-incremental` | Allow running stages with IncrementalOut files not in cache |
| `--checkout-missing` | Restore tracked files from cache before running |
| `--allow-missing` | Allow missing dep files if tracked (only with --dry-run or --explain) |

**Examples:**

```bash
# Run entire pipeline
pivot repro

# Run specific stages and their dependencies
pivot repro train evaluate

# See what would run
pivot repro --dry-run

# Watch mode - re-run on file changes
pivot repro --watch

# Continue after failures
pivot repro --keep-going
```

---

### `pivot run`

Execute specified stages directly, without resolving dependencies. Use this when you want to run specific stages in a specific order.

```bash
pivot run STAGES... [OPTIONS]
```

**Arguments:**

- `STAGES` - Stage names to run (required, at least one)

**Options:**

| Option | Description |
|--------|-------------|
| `--force` / `-f` | Force re-run of stages, ignoring cache |
| `--tui` | Use interactive TUI display (default: plain text) |
| `--json` | Output results as JSON |
| `--tui-log PATH` | Write TUI messages to JSONL file for monitoring |
| `--no-commit` | Defer lock files to pending dir for faster iteration |
| `--no-cache` | Skip caching outputs entirely for maximum iteration speed |
| `--fail-fast` | Stop on first failure (default: keep-going) |
| `--allow-uncached-incremental` | Allow running stages with IncrementalOut files not in cache |
| `--checkout-missing` | Restore tracked files from cache before running |

**Examples:**

```bash
# Run a single stage (no dependencies)
pivot run train

# Run multiple stages in order
pivot run preprocess train

# Stop immediately on failure
pivot run preprocess train --fail-fast
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
| `--json` | Output as JSON |
| `--stages-only` | Show only pipeline status |
| `--tracked-only` | Show only tracked files |
| `--remote-only` | Show only remote status |
| `--remote` / `-r` | Include remote sync status |

---

### `pivot commit`

Commit pending locks from `--no-commit` runs.

```bash
pivot commit [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--list` | List pending stages without committing |
| `--discard` | Discard pending changes without committing |

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

### `pivot export`

Export pipeline to DVC YAML format.

```bash
pivot export [STAGES...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--output` / `-o PATH` | Output path (default: `dvc.yaml`) |

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
| `--force` / `-f` | Overwrite existing files |
| `--only-missing` | Only restore files that don't exist on disk |

---

### `pivot data get`

Retrieve files or stage outputs from a specific git revision.

```bash
pivot data get TARGETS... --rev REVISION [OPTIONS]
```

**Arguments:**

- `TARGETS` - File paths or stage names (required)

**Options:**

| Option | Description |
|--------|-------------|
| `--rev` / `-r REV` | Git revision (SHA, branch, tag) - required |
| `--output` / `-o PATH` | Output path (single file only) |
| `--checkout-mode MODE` | `symlink`, `hardlink`, or `copy` |
| `--force` / `-f` | Overwrite existing files |

**Examples:**

```bash
# Get file from specific commit
pivot data get model.pkl --rev abc123

# Get stage output from branch
pivot data get train --rev feature-branch

# Get with custom output path
pivot data get model.pkl --rev v1.0 --output old_model.pkl
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
| `--output` / `-o PATH` | Output HTML path (default: `pivot_plots/index.html`) |
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

### `pivot data diff`

Compare data files against git HEAD.

```bash
pivot data diff TARGETS... [OPTIONS]
```

**Arguments:**

- `TARGETS` - Data files to compare (required)

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
pivot data diff output.csv

# Key-based row matching
pivot data diff output.csv --key id,timestamp

# JSON output for scripting
pivot data diff output.csv --json
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
| `--jobs` / `-j N` | Parallel download jobs (default: 20) |

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
| `cache.checkout_mode` | Checkout mode order | `hardlink,symlink,copy` |
| `core.max_workers` | Parallel workers (-1 = all CPUs) | `-2` |
| `core.state_dir` | State directory | `.pivot` |
| `remote.jobs` | Parallel transfer jobs | `20` |
| `remote.retries` | Transfer retry count | `10` |
| `remote.connect_timeout` | Connection timeout (seconds) | `30` |
| `watch.debounce` | Watch debounce (milliseconds) | `300` |
| `display.precision` | Float display precision | `5` |
| `diff.max_rows` | Max rows for data diff | `10000` |
| `default_remote` | Default remote name | (none) |
| `remotes.<name>` | Remote URL (S3) | (none) |

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

### `pivot import-dvc`

Import DVC pipeline and convert to Pivot format.

```bash
pivot import-dvc [OPTIONS]
```

Reads `dvc.yaml` (and optionally `dvc.lock`, `params.yaml`) and generates `pivot.yaml` with migration notes for manual review.

**Options:**

| Option | Description |
|--------|-------------|
| `--input` / `-i PATH` | Path to dvc.yaml (default: auto-detect) |
| `--output` / `-o PATH` | Output path for pivot.yaml (default: pivot.yaml) |
| `--params` / `-p PATH` | Path to params.yaml (default: auto-detect) |
| `--notes` / `-n PATH` | Path for migration notes (default: .pivot/migration-notes.md) |
| `--force` / `-f` | Overwrite existing files |
| `--dry-run` | Show what would be generated without writing files |

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

### `pivot schema`

Output JSON Schema for pivot.yaml configuration.

```bash
pivot schema [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--indent N` | JSON indentation (0 for compact) |

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
