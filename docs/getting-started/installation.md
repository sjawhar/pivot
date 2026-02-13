# Installation

## Requirements

- **Python 3.13+** (3.14+ for experimental InterpreterPoolExecutor)
- **Unix only** (Linux/macOS)

## Installing from PyPI

```bash
uv add pivot
```

## Installing with Optional Dependencies

### S3 Remote Storage

For pushing/pulling cached outputs to S3:

```bash
uv add "pivot[s3]"
```

### DVC Integration

For `pivot export` to generate DVC-compatible YAML:

```bash
uv add "pivot[dvc]"
```

### All Optional Dependencies

```bash
uv add "pivot[s3,dvc]"
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
