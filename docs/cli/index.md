# CLI Reference

Complete reference for all Pivot command-line commands.

## Global Options

All commands support:

| Option | Description |
|--------|-------------|
| `--verbose` / `-v` | Show detailed output |
| `--help` | Show help message |

---

## Pipeline Execution

### `pivot run`

Execute pipeline stages.

```bash
pivot run [STAGES...] [OPTIONS]
```

**Arguments:**

- `STAGES` - Stage names to run (optional, runs all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--single-stage` / `-s` | Run only specified stages without dependencies |
| `--cache-dir PATH` | Custom cache directory |
| `--dry-run` / `-n` | Show what would run without executing |
| `--explain` / `-e` | Show detailed breakdown of why stages run |
| `--watch` / `-w [PATTERNS]` | Watch for changes and re-run (optional glob patterns) |
| `--debounce MS` | Debounce delay in milliseconds (default: 300) |

**Examples:**

```bash
# Run all stages
pivot run

# Run specific stages
pivot run preprocess train

# Run single stage without dependencies
pivot run train --single-stage

# Dry run
pivot run --dry-run

# Watch mode
pivot run --watch
pivot run --watch "*.py,*.csv"
```

---

### `pivot dry-run`

Show what would run without executing (terse output).

```bash
pivot dry-run [STAGES...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--single-stage` / `-s` | Check only specified stages |
| `--cache-dir PATH` | Custom cache directory |

---

### `pivot explain`

Show detailed breakdown of why stages would run.

```bash
pivot explain [STAGES...] [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--single-stage` / `-s` | Explain only specified stages |
| `--cache-dir PATH` | Custom cache directory |

**Example Output:**

```
Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Code changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4

  Param changes:
    learning_rate: 0.01 -> 0.001
```

---

## Pipeline Introspection

### `pivot list`

List all registered stages.

```bash
pivot list
```

With `--verbose`, shows dependencies and outputs for each stage.

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

---

### `pivot get`

Retrieve files from a specific git revision.

```bash
pivot get TARGETS... --rev REVISION [OPTIONS]
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
pivot get model.pkl --rev abc123

# Get stage output from branch
pivot get train --rev feature-branch

# Get with custom output path
pivot get model.pkl --rev v1.0 --output old_model.pkl
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

### `pivot remote add`

Add a remote storage location.

```bash
pivot remote add NAME URL [OPTIONS]
```

**Arguments:**

- `NAME` - Remote name (e.g., `origin`)
- `URL` - S3 URL (e.g., `s3://bucket/prefix`)

**Options:**

| Option | Description |
|--------|-------------|
| `--default` / `-d` | Set as default remote |

---

### `pivot remote remove`

Remove a remote storage location.

```bash
pivot remote remove NAME
```

---

### `pivot remote list`

List configured remote storage locations.

```bash
pivot remote list
```

---

### `pivot remote default`

Set the default remote.

```bash
pivot remote default NAME
```

---

### `pivot push`

Push cached outputs to remote storage.

```bash
pivot push [STAGES...] [OPTIONS]
```

**Arguments:**

- `STAGES` - Stages to push (optional, pushes all if not specified)

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
pivot pull [STAGES...] [OPTIONS]
```

**Arguments:**

- `STAGES` - Stages to pull (optional, pulls all if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--remote` / `-r NAME` | Remote name (uses default if not specified) |
| `--dry-run` / `-n` | Show what would be pulled |
| `--jobs` / `-j N` | Parallel download jobs (default: 20) |
