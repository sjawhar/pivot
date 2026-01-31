# Pivot: High-Performance Python Pipeline Tool

**Change your code. Pivot knows what to run.**

**Python:** 3.13+ | **Coverage:** 90%+ | **License:** TBD

---

## What is Pivot?

Pivot is a Python pipeline tool with automatic code change detection. Define stages with typed Python functions and annotations, and Pivot figures out what needs to re-run—no manual dependency declarations, no stale caches. It provides:

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

## Quick Start

```bash
pip install pivot
```

### Python-First Pipeline Definition

Define stages with typed Python functions using annotations:

```python
# pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs
from pivot.registry import REGISTRY


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("processed.parquet", loaders.PathOnly())]


def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> PreprocessOutputs:
    df = raw.dropna()
    out_path = pathlib.Path("processed.parquet")
    df.to_parquet(out_path)
    return PreprocessOutputs(clean=out_path)


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]


def train(
    data: Annotated[pathlib.Path, outputs.Dep("processed.parquet", loaders.PathOnly())],
) -> TrainOutputs:
    df = pandas.read_parquet(data)
    model_path = pathlib.Path("model.pkl")
    # ... train model ...
    return TrainOutputs(model=model_path)


# Register stages
REGISTRY.register(preprocess)
REGISTRY.register(train)
```

```bash
pivot run  # Runs both stages
pivot run  # Instant - nothing changed
```

Modify `preprocess`, and Pivot automatically re-runs both stages. Modify `train`, and only `train` re-runs.

**Why Python-first?**
- Full type safety and IDE autocomplete
- Dependencies and outputs declared in one place (the function)
- Loader code changes automatically trigger re-runs
- No YAML to keep in sync with your Python code

---

## Key Features

### 1. Automatic Code Change Detection

**No more manual dependency declarations!** Pivot automatically detects when your Python functions change:

```python
def helper(x):
    return x * 2  # Change this...

def process():
    data = load("data.csv")
    return helper(data)  # ...and Pivot knows to re-run!
```

**How it works:**

- Uses `inspect.getclosurevars()` to find referenced functions/constants
- AST extraction for `module.attr` patterns (Google-style imports)
- Recursive fingerprinting for transitive dependencies

**Validated:** See [Architecture: Fingerprinting](docs/architecture/fingerprinting.md) and [tests/fingerprint/](tests/fingerprint/)

### 2. Per-Stage Lock Files

**DVC's bottleneck:** Every stage writes the entire `dvc.lock` file (O(n²) behavior)

**Pivot's solution:** Each stage gets its own lock file

- `.pivot/stages/train.lock` (~500 bytes)
- Parallel writes without contention
- **32x faster** on large pipelines

### 3. Warm Worker Pool

**Problem:** Importing numpy/pandas can take seconds per stage

**Solution:** Reusable worker processes with preloaded imports. Pivot uses `loky.get_reusable_executor()` to keep workers warm between runs, avoiding repeated import overhead.

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

### 6. Stage Parameters

**Type-safe parameters via Pydantic classes:**

```python
from pivot.stage_def import StageParams

class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32

def train(
    params: TrainParams,
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> TrainOutputs:
    # params.learning_rate, params.epochs, params.batch_size available
    ...
```

**How it works:**

- Define parameters as Pydantic classes inheriting from `StageParams`
- Parameter changes trigger re-execution (tracked in lock files)

**View and compare params:**

```bash
pivot params show                    # Show current param values
pivot params show --format json      # JSON output
pivot params diff                    # Compare workspace vs last commit
pivot params diff --precision 4      # Control float precision
```

### 7. Incremental Outputs

Outputs that preserve state between runs for append-only workloads:

```python
class AppendOutputs(TypedDict):
    database: Annotated[pathlib.Path, outputs.IncrementalOut("database.db", loaders.PathOnly())]

def append_to_database(
    new_data: Annotated[pandas.DataFrame, outputs.Dep("new_data.csv", loaders.CSV())],
) -> AppendOutputs:
    # database.db is restored from cache before this runs
    # Modify in place, then return the path
    ...
```

**How it works:**
- Before execution, incremental outputs restore the previous version from cache
- Stage modifies the file in place (append, update, etc.)
- New version is cached after execution
- Uses COPY mode (not symlinks) so stages can safely modify files

### 8. S3 Remote Cache Storage

**Share cached outputs across machines and CI environments:**

```bash
# Add a remote
pivot config set remotes.origin s3://my-bucket/pivot-cache
pivot config set default_remote origin

# Push cached outputs to remote
pivot push

# Push specific stages only
pivot push train_model evaluate_model

# Pull cached outputs from remote
pivot pull

# Pull specific stages (downloads only what's needed)
pivot pull train_model
```

**How it works:**
- Uses async I/O (aioboto3) for high-throughput parallel transfers
- Local index in LMDB avoids repeated HEAD requests to S3
- Stage-level filtering enables granular push/pull operations
- AWS credentials via standard chain (env vars, ~/.aws/credentials, IAM roles)

**Remote configuration stored in `.pivot/config.yaml`:**
```yaml
remotes:
  origin: s3://my-bucket/pivot-cache
default_remote: origin
```

**Local cache structure:**
```
.pivot/
├── cache/
│   └── files/           # Content-addressable cache (pushed/pulled)
│       ├── ab/
│       │   └── cdef0123...  # File content keyed by xxhash64
│       └── ...
├── stages/              # Per-stage lock files (local only)
│   ├── preprocess.lock
│   └── train.lock
├── config.yaml          # Remote configuration (local only)
└── state.lmdb/          # Hash cache, generation tracking (local only)
```

**What gets transferred:**

| Data | Pushed | Pulled | Notes |
|------|--------|--------|-------|
| Cache files (`.pivot/cache/files/`) | ✅ | ✅ | Actual file contents, content-addressable by hash |
| Lock files (`.pivot/stages/*.lock`) | ❌ | ❌ | Reference hashes; must exist locally to pull specific stages |
| Config (`.pivot/config.yaml`) | ❌ | ❌ | Contains remote URLs; each machine has its own |
| State DB (`.pivot/state.lmdb/`) | ❌ | ❌ | Local performance cache; rebuilt automatically |

**Typical workflow:**
1. Run pipeline locally → outputs cached in `.pivot/cache/files/`
2. `pivot push` → upload cache files to S3
3. On another machine: clone repo (includes lock files in git)
4. `pivot pull train_model` → download only files needed for that stage
5. `pivot run` → stages with cached outputs skip execution

### 9. Data Diff

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

### 10. Typed Dependencies and Outputs (StageDef)

**Declare typed deps/outs that auto-load and auto-save:**

```python
import pandas as pd
from pivot import loaders
from pivot.stage_def import StageDef, Dep, Out


class TrainParams(StageDef):
    # Deps - type-safe file loading
    train_data: Dep[pd.DataFrame] = Dep("data/train.csv", loaders.CSV())
    config: Dep[dict] = Dep("config.json", loaders.JSON())

    # Outs - type-safe file saving
    model: Out[dict] = Out("models/model.pkl", loaders.Pickle())
    metrics: Out[dict] = Out("metrics.json", loaders.JSON())

    # Regular params
    learning_rate: float = 0.01
    epochs: int = 100


def train(params: TrainParams) -> None:
    # Deps are auto-loaded before function runs
    df = params.train_data  # Returns pd.DataFrame
    config = params.config  # Returns dict

    # Train model...
    model = {"weights": df.shape, "lr": params.learning_rate}
    metrics = {"accuracy": 0.95}

    # Assign outputs - auto-saved after function returns
    params.model = model
    params.metrics = metrics
```

**Benefits:**
- **Type-safe** - IDE autocomplete, type checking for loaded data
- **Auto-load/save** - No boilerplate file I/O code
- **Automatic fingerprinting** - Loader code changes trigger re-runs
- **Fully self-contained** - Deps/outs are extracted from StageDef, no external config needed

**Built-in loaders:**

| Loader | Load returns | Save accepts | Options |
|--------|--------------|--------------|---------|
| `loaders.CSV()` | DataFrame | DataFrame | `sep`, `index_col`, `dtype` |
| `loaders.JSON()` | dict/list | dict/list | `indent` |
| `loaders.YAML()` | dict/list | dict/list | - |
| `loaders.Pickle()` | Any | Any | - |
| `loaders.PathOnly()` | pathlib.Path | (validates exists) | - |

**Custom loaders:**

```python
import dataclasses
from pivot.loaders import Loader


@dataclasses.dataclass(frozen=True)
class Parquet(Loader[pd.DataFrame]):
    compression: str = "snappy"

    def load(self, path: pathlib.Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def save(self, data: pd.DataFrame, path: pathlib.Path) -> None:
        data.to_parquet(path, compression=self.compression)
```

**Note:** Custom loaders must be defined at module level (not inside functions) for pickling across processes.

---

## Installation

```bash
pip install pivot
```

**Requirements:**

- Python 3.13+ (3.14+ for InterpreterPoolExecutor)
- Unix only (Linux/macOS)

---

## Documentation

Full documentation available at the [Pivot Documentation Site](https://anthropics.github.io/pivot/).

- **[Quick Start](docs/getting-started/quickstart.md)** - Build your first pipeline in 5 minutes
- **[Core Concepts](docs/getting-started/concepts.md)** - Stages, dependencies, caching
- **[CLI Reference](docs/cli/index.md)** - All available commands
- **[Architecture](docs/architecture/overview.md)** - Design decisions and internals
- **[Comparison](docs/comparison.md)** - How Pivot compares to DVC, Prefect, Dagster

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
- **Explain mode** - `pivot run --explain` and `pivot explain` show detailed breakdown of WHY stages would run
- **Observability** - `pivot metrics show/diff`, `pivot plots show/diff`, and `pivot params show/diff` commands
- **Pipeline configuration** - Python-first with `REGISTRY.register()`, optional `pivot.yaml` for matrix expansion
- **S3 remote cache** - `pivot push/pull` with async I/O, LMDB index, per-stage filtering
- **Data diff** - `pivot data diff` command with interactive TUI for comparing data file changes
- **Version retrieval** - `pivot data get --rev` to materialize files from any git revision
- **Shell completion** - Tab completion for bash, zsh, and fish
- **Centralized configuration** - `pivot config` command for managing project settings
- **Typed deps/outs (StageDef)** - Auto-loading/saving dependencies and outputs with type-safe loaders

### Planned

- **Web UI** - DAG visualization and execution monitoring
- **Additional remotes** - GCS, Azure, SSH
- **Cloud orchestration** - Integration with cloud schedulers

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Pipeline Code (pipeline.py with REGISTRY.register())  │
│  @outputs.Dep("data.csv", loaders.CSV())                    │
│  @outputs.Out("model.pkl", loaders.PathOnly())              │
│  def train(data: ...) -> TrainOutputs: ...                  │
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
| **Python-first**          | Config-first (YAML)         | Python annotations + optional YAML |
| **Remote storage**        | S3/GCS/Azure via dvc-data   | S3 with async I/O           |

---

## Technical Details

### Fingerprinting

Pivot automatically detects code changes using Python introspection:

- **[How Fingerprinting Works](docs/architecture/fingerprinting.md)** - AST + getclosurevars approach
- **[Change Detection Matrix](tests/fingerprint/README.md)** - Comprehensive behavior catalog

### Performance

Pivot was designed to address DVC's lock file bottleneck (O(n²) writes). Benchmarks on real pipelines showed:

- 176-stage pipeline: Lock file writes reduced from 289s to ~9s (32x improvement)
- Per-stage lock files enable parallel writes without contention

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

**Last Updated:** 2026-01-10
**Version:** 0.1.0-dev

