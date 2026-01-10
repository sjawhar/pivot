# Installation

## Requirements

- **Python 3.13+** (3.14+ for experimental InterpreterPoolExecutor)
- **Unix only** (Linux/macOS)

## Installing from PyPI

```bash
pip install pivot
```

## Installing with Optional Dependencies

### S3 Remote Storage

For pushing/pulling cached outputs to S3:

```bash
pip install pivot[s3]
```

### DVC Integration

For `pivot export` to generate DVC-compatible YAML:

```bash
pip install pivot[dvc]
```

### All Optional Dependencies

```bash
pip install pivot[s3,dvc]
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
pip install -e ".[dev]"
```

## Verifying Installation

```bash
# Check version
pivot --version

# List available commands
pivot --help
```

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Build your first pipeline
- [Core Concepts](concepts.md) - Understand how Pivot works
