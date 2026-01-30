# Comparison with Other Tools

How Pivot compares to other pipeline and workflow tools.

## Feature Comparison

| Feature | Pivot | DVC | Prefect | Dagster |
|---------|-------|-----|---------|---------|
| **Local Iteration Speed** | | | | |
| Watch mode | Auto-rerun on file change | Manual `dvc repro` | Manual trigger | Manual materialize |
| Warm workers | Workers persist across runs | Cold start each run | Cold start | Cold start |
| **Code Change Detection** | | | | |
| Function-level tracking | AST + getclosurevars | Command string hash | Input hash only | Config hash only |
| Transitive dependencies | Automatic | Manual declaration | Manual | Manual |
| Helper function changes | Detected automatically | Not detected | Not detected | Not detected |
| **Caching** | | | | |
| Content-addressable | xxhash64 | MD5 | No | Yes |
| Per-stage locks | Parallel-safe writes | Single dvc.lock | N/A | N/A |
| Remote storage | S3 | S3, GCS, Azure, SSH | Cloud-native | Cloud-native |
| **Configuration** | | | | |
| Pure Python | YAML + typed classes | YAML + scripts | Decorators | Decorators |
| Type-safe parameters | Pydantic models | No | Pydantic | Yes |
| **Ecosystem** | | | | |
| Web UI | Planned | DVC Studio | Built-in | Built-in |
| Cloud orchestration | Local-first | Local-first | Cloud-native | Cloud-native |

## When to Use Each Tool

### Use Pivot When

- You need fast local iteration with automatic code change detection
- Your pipeline is Python-native and you want decorator-based configuration
- You're frustrated by DVC's lock file performance or manual dependency declarations
- You want watch mode (`pivot repro --watch`) for rapid development cycles

### Use DVC When

- You need mature ecosystem with extensive cloud provider support
- Your team is already using DVC and switching cost is high
- You need DVC Studio for collaboration features
- You have non-Python stages (shell commands, R scripts)

### Use Prefect/Dagster When

- You need cloud-native orchestration with scheduling
- You want a web UI for monitoring and alerting
- Your workflows span multiple services and systems
- You need enterprise features (RBAC, audit logs)

## Key Differentiators

### Automatic Code Change Detection

Most tools hash command strings or explicit inputs. Pivot parses your Python functions:

```python
def helper(x):
    return x * 2  # Change this...

def process():
    return helper(load())  # ...Pivot detects it!
```

DVC would require manually updating `dvc.yaml` to track this change.

### Per-Stage Lock Files

DVC writes a single `dvc.lock` file for all stages. With many stages, this creates contention:

```
# DVC: Every stage writes entire file
dvc.lock

# Pivot: Each stage writes its own file
.pivot/stages/
├── preprocess.lock
├── train.lock
└── evaluate.lock
```

Per-stage locks enable parallel-safe writes and avoid the monolithic lock file bottleneck.

### Warm Workers

Pivot uses `loky.get_reusable_executor()` to keep worker processes alive between runs. This avoids the startup cost of reimporting large libraries like numpy and pandas on each execution.

### Watch Mode

Rapid iteration during development:

```bash
pivot repro --watch  # Monitors files, re-runs on change
```

Edit code, save, see results immediately.

## Migration from DVC

Pivot can export to DVC format for gradual adoption:

```bash
# Generate dvc.yaml from Pivot stages
pivot export > dvc.yaml

# Team members without Pivot can still use DVC
dvc repro
```

See [Migrating from DVC](migrating-from-dvc.md) for details.
