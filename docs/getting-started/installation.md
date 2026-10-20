# Installation

## Requirements

- **Python 3.13+** (3.14+ for experimental InterpreterPoolExecutor)
- **Unix only** (Linux/macOS)

## Installing from PyPI

```bash
uv add pivot
```

## Installing with Optional Dependencies

### Remote Storage

For pushing/pulling cached outputs to S3 or Google Cloud Storage:

```bash
uv add "pivot[s3]"   # s3:// remotes
uv add "pivot[gcs]"  # gs:// remotes
```

### Matplotlib Plots

For stages that return figures tagged `plot`:

```bash
uv add "pivot[matplotlib]"
```

### All Optional Dependencies

```bash
uv add "pivot[s3,gcs,matplotlib]"
```

## Development Installation

For contributing to Pivot:

```bash
# Clone the repository
git clone https://github.com/sjawhar/pivot.git
cd pivot

# Install with uv (recommended)
uv sync --active

# Or with pip
uv pip install -e ".[dev]"
```

## Verifying Installation

```bash
# List available commands
pivot --help
```

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Build your first pipeline
- [Guides](../guides/watch-mode.md) - Watch mode, multi-pipeline projects, remote storage, CI
