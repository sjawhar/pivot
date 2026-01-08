# Pivot: High-Performance Python Pipeline Tool

**Status:** 🚧 In Development (MVP Nearly Complete)
**Python:** 3.13+ required
**License:** TBD

---

## What is Pivot?

Pivot is a Python-native pipeline tool designed to eliminate DVC's performance bottlenecks while maintaining compatibility. It provides:

- **32x faster lock file operations** through per-stage lock files
- **Automatic code change detection** using Python introspection (no manual declarations!)
- **Warm worker pools** with preloaded imports (numpy/pandas loaded once)
- **DVC compatibility** via YAML export for code reviews

### Performance Comparison (monitoring-horizons pipeline, 176 stages)

| Component         | DVC          | Pivot       | Improvement     |
| ----------------- | ------------ | ----------- | --------------- |
| Lock file writes  | 289s (23.8%) | ~9s (0.7%)  | **32x faster**  |
| Total overhead    | 301s (24.8%) | ~20s (1.6%) | **15x faster**  |
| **Total runtime** | **1214s**    | **~950s**   | **1.3x faster** |

---

## Quick Start (Coming Soon)

```bash
# Install
pip install pivot

# Define pipeline
# pipeline.py
from pivot import stage

@stage(deps=['data.csv'], outs=['processed.parquet'])
def preprocess(input_file: str = 'data.csv'):
    import pandas as pd
    df = pd.read_csv(input_file)
    df = df.dropna()
    df.to_parquet('processed.parquet')

@stage(deps=['processed.parquet'], outs=['model.pkl'])
def train(data_file: str = 'processed.parquet', lr: float = 0.01):
    import pandas as pd
    import pickle
    df = pd.read_parquet(data_file)
    model = {'lr': lr, 'data': df.shape}
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Run pipeline
pivot run

# Export to DVC YAML for code review
pivot export --output dvc.yaml
```

---

## Key Features

### 1. Automatic Code Change Detection

**No more manual dependency declarations!** Pivot automatically detects when your Python functions change:

```python
def helper(x):
    return x * 2  # Change this...

@stage(deps=['data.csv'])
def process(file: str):
    data = load(file)
    return helper(data)  # ...and Pivot knows to re-run!
```

**How it works:**

- Uses `inspect.getclosurevars()` to find referenced functions/constants
- AST extraction for `module.attr` patterns (Google-style imports)
- Recursive fingerprinting for transitive dependencies

**Validated:** See [docs/fingerprinting.md](docs/fingerprinting.md) and [tests/fingerprint/](tests/fingerprint/)

### 2. Per-Stage Lock Files

**DVC's bottleneck:** Every stage writes the entire `dvc.lock` file (O(n²) behavior)

**Pivot's solution:** Each stage gets its own lock file

- `.pivot/stages/train.lock` (~500 bytes)
- Parallel writes without contention
- **32x faster** on large pipelines

### 3. Warm Worker Pool

**Problem:** Importing numpy/pandas can take seconds per stage

**Solution:** Preload imports once in worker processes

```bash
pivot run --executor=warm  # Default
```

**Experimental:** Python 3.14's `InterpreterPoolExecutor`

```bash
pivot run --executor=interpreter  # Lower memory, faster startup
```

### 4. DVC Compatibility

Export Pivot pipelines to DVC YAML for code review:

```bash
pivot export --validate  # Creates dvc.yaml and validates against DVC
```

### 5. Explain Mode

**Killer feature:** See WHY a stage would run

```bash
pivot run --explain

Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4
      File: src/utils.py:15
```

### 6. Pydantic Stage Parameters

**Type-safe parameters with automatic injection:**

```python
from pydantic import BaseModel
from pivot import stage

class TrainParams(BaseModel):
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32

@stage(deps=['data.csv'], outs=['model.pkl'], params=TrainParams)
def train(params: TrainParams):
    print(f"Training with lr={params.learning_rate}")
```

**Override defaults via params.yaml:**

```yaml
train:
  learning_rate: 0.001
  epochs: 200
```

**How it works:**

- Define parameters using Pydantic BaseModel with defaults
- Pivot automatically injects the params instance into your function
- YAML overrides take precedence over model defaults
- Parameter changes trigger re-execution (tracked in lock files)

### 7. Incremental Outputs

**New:** Outputs that preserve state between runs for append-only workloads:

```python
from pivot import stage, IncrementalOut

@stage(deps=['new_data.csv'], outs=[IncrementalOut('database.db')])
def append_to_database():
    # database.db is restored from cache before execution
    # Stage can append to it rather than recreating from scratch
    import sqlite3
    conn = sqlite3.connect('database.db')
    # ... append new records ...
```

**How it works:**
- Before execution, `IncrementalOut` restores the previous version from cache
- Stage modifies the file in place (append, update, etc.)
- New version is cached after execution
- Uses COPY mode (not symlinks) so stages can safely modify files

### 8. Data Diff

Compare data file changes between git HEAD and workspace:

```bash
# Interactive TUI mode (default)
pivot data diff output.csv

# Non-interactive output
pivot data diff output.csv --no-tui

# Use key columns for row matching (instead of positional)
pivot data diff output.csv --key id --key timestamp

# JSON output for scripting
pivot data diff output.csv --no-tui --json
```

**Features:**
- **Interactive TUI** - Navigate between files, view schema changes and row-level diffs
- **Key-based matching** - Match rows by key columns to detect modifications vs adds/removes
- **Positional matching** - Compare rows by position when no keys specified
- **Schema detection** - Shows added/removed/type-changed columns
- **Large file handling** - Summary-only mode for files exceeding memory threshold
- **Multiple formats** - Plain text, markdown, or JSON output

**Supported formats:** CSV, JSON, JSONL

---

## Installation (Coming Soon)

```bash
pip install pivot
```

**Requirements:**

- Python 3.13+ (3.14+ for InterpreterPoolExecutor)
- Linux/Mac (Windows support via spawn context)

---

## Documentation

- **[Design Documentation](./CLAUDE.md)** - Architecture and design decisions
- **[Source Code Docs](./src/CLAUDE.md)** - Module-by-module implementation guide
- **[Testing Strategy](./tests/CLAUDE.md)** - TDD approach and test organization

---

## Development Roadmap

### Completed ✅

- **Core pipeline execution** - DAG construction, greedy parallel scheduling, per-stage lock files
- **Automatic code change detection** - getclosurevars + AST fingerprinting, transitive dependencies
- **Content-addressable cache** - xxhash64 hashing for files and code, hardlink/copy restoration
- **Pydantic parameters** - Type-safe stage parameters with params.yaml overrides
- **Watch mode** - File system monitoring with configurable globs and debounce
- **Incremental outputs** - Restore-before-run for append-only workloads
- **DVC export** - `pivot export` command for YAML generation
- **Explain mode** - `pivot run --explain` shows detailed breakdown of WHY stages would run
- **Observability** - `pivot metrics show/diff` and `pivot plots show/diff` commands
- **Data diff** - `pivot data diff` command with interactive TUI for comparing data file changes

### In Progress

- [ ] **End-to-end benchmarks** - Comparative performance testing vs DVC

### Future

- **Version control** - `pivot get --rev` to materialize old versions, DVC lock import for migration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Pipeline Code (Python decorators)                     │
│  @stage(deps=['data.csv'], outs=['model.pkl'])              │
│  def train(lr: float = 0.01): ...                          │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage Registry → DAG Builder → Scheduler                    │
│  Automatic fingerprinting | Topological sort | Ready queue  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Warm Workers / Interpreters                                 │
│  Preloaded numpy/pandas | True parallelism                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Per-Stage Lock Files (.pivot/stages/<name>.lock)        │
│  Code manifest | Params | Deps/Outs | Fast parallel writes │
└─────────────────────────────────────────────────────────────┘
```

---

## Contributing

This is currently an internal development project. Guidelines:

### Code Quality Standards

- **TDD:** Write tests before implementation
- **Type hints:** All functions must be fully typed
- **Formatting:** ruff format with line length 100
- **Linting:** ruff check for fast linting
- **Coverage:** >90% for all modules
- **Documentation:** Update CLAUDE.md files when design changes

### Before Committing

```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest --cov=src/pivot --cov-report=term --cov-fail-under=90

# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Type check
basedpyright src/
```

---

## Comparison with DVC

| Feature                   | DVC                         | Pivot                       |
| ------------------------- | --------------------------- | --------------------------- |
| **Lock file format**      | Monolithic dvc.lock         | Per-stage .lock files       |
| **Lock file overhead**    | O(n²) - 289s for 176 stages | O(n) - ~9s for 176 stages   |
| **Code change detection** | Manual (deps: [file.py])    | Automatic (getclosurevars)  |
| **Executor**              | ThreadPoolExecutor          | Warm workers + Interpreters |
| **Explain mode**          | ❌                          | ✅ Shows WHY stages run     |
| **YAML export**           | N/A                         | ✅ For code review          |
| **Python-first**          | Config-first (YAML)         | Code-first (decorators)     |

---

## Research and Validation

### Profiling Data

Real-world DVC pipelines profiled to identify bottlenecks:

- **[eval-pipeline](../timing/TIMING_REPORT.md)** - 60 stages, 50s lock overhead
- **[monitoring-horizons](../timing/MONITORING_HORIZONS_TIMING_REPORT.md)** - 176 stages, 289s lock overhead

### Experimental Validation

- **[Fingerprinting research](./docs/fingerprinting.md)** - Why AST + getclosurevars was chosen
- **[Change detection matrix](./tests/fingerprint/README.md)** - Comprehensive behavior catalog

### DVC Architecture Study

- **[Parallel execution](../dvc/dvc/repo/reproduce.py:145-283)** - StageInfo/ready queue pattern
- **[Lock file writing](../dvc/dvc/dvcfile.py:444-473)** - Root cause of O(n²) overhead

---

## FAQ

### Q: Why not just contribute to DVC?

**A:** The per-stage lock file approach requires fundamental architectural changes that would break backward compatibility. Pivot can coexist with DVC during migration.

### Q: Can I use Pivot with existing DVC projects?

**A:** Yes! Export your Pivot pipeline to dvc.yaml and run them side-by-side. Validate outputs match before fully migrating.

### Q: What if automatic code detection doesn't work for my case?

**A:** We provide escape hatches for edge cases (dynamic dispatch, method calls). You can manually declare dependencies when needed.

### Q: Why Python 3.13+ requirement?

**A:** We use modern typing features and performance improvements. Python 3.14+ unlocks experimental InterpreterPoolExecutor.

### Q: Is this production-ready?

**A:** Not yet! Core functionality is complete and well-tested (90%+ coverage), but we're polishing the final features before the 1.0 release.

---

## License

TBD

---

## Contact

Internal development team
Questions? Check CLAUDE.md files or ask the team!

---

**Last Updated:** 2026-01-08
**Version:** 0.1.0-dev
