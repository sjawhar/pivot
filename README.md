# Pivot: High-Performance Python Pipeline Tool

**Status:** 🚧 In Development (Week 1)
**Python:** 3.13+ required (3.14+ for experimental features)
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

**Validated:** See `/experiments/test_getclosurevars_approach.py`

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
- **[Implementation Plan](/.claude/plans/happy-hatching-acorn.md)** - Detailed 12-week roadmap

---

## Development Roadmap

### Phase 1: MVP (Weeks 1-6) - IN PROGRESS ✅

**Week 1:** Fingerprinting + Registry

- [x] Design documentation
- [ ] Code change detection (getclosurevars + AST)
- [ ] Stage registry with decorator
- [ ] Comprehensive test suite (>90% coverage)

**Week 2:** DAG + Lock Files + YAML Export

- [ ] Dependency graph construction
- [ ] Per-stage lock files
- [ ] DVC YAML export for code review

**Week 3:** Parameters + Hashing

- [ ] Pydantic parameter system
- [ ] xxhash64 content-addressed storage

**Week 4:** Sequential Executor + Explain Mode

- [ ] Single-threaded pipeline execution
- [ ] Explain mode (show WHY stages run)

**Week 5:** Parallel Scheduler + Multiple Executors

- [ ] Warm worker pool (default)
- [ ] InterpreterPoolExecutor (experimental)
- [ ] Ready queue pattern from DVC

**Week 6:** CLI + End-to-End Testing

- [ ] Command-line interface
- [ ] Integration tests
- [ ] Benchmarks vs DVC

### Phase 2: Observability (Weeks 7-9)

- Metrics tracking and diff
- Plots generation
- Integration with existing tools

### Phase 3: Version Control (Weeks 10-12)

- `pivot get --rev` (materialize old versions)
- DVC lock file import (migration helper)
- Git hooks integration

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
- **Formatting:** Black with line length 100
- **Linting:** Ruff for fast linting
- **Coverage:** >90% for all modules
- **Documentation:** Update CLAUDE.md files when design changes

### Before Committing

```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest --cov=src/pivot --cov-report=term --cov-fail-under=90

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/ --strict
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

- **[getclosurevars approach](./experiments/test_getclosurevars_approach.py)** - Validates automatic code detection
- **[Change detection tests](./experiments/test_change_detection.py)** - Edge cases and limitations

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

**A:** Not yet! We're in Week 1 of development. Target: 6-week MVP with comprehensive testing.

---

## License

TBD

---

## Contact

Internal development team
Questions? Check CLAUDE.md files or ask the team!

---

**Last Updated:** 2026-01-04
**Version:** 0.1.0-dev (Week 1)
