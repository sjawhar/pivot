# Getting Started

Guide for setting up a Pivot development environment.

## Using Dev Container (Recommended)

The repository includes a dev container configuration in `.devcontainer/`:

1. Open the project in VS Code
2. Install the "Dev Containers" extension
3. Click "Reopen in Container" when prompted

## Manual Setup

```bash
# Clone the repository
git clone https://github.com/sjawhar/pivot.git
cd pivot

# Install dependencies with uv
uv sync --active
```

## Quality Commands

Run these before submitting changes:

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run basedpyright .

# Run tests (parallel)
uv run pytest tests/ -n auto
```

**All four must pass before merging.**

## Quick Verification

One-liner to run all checks:

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright . && uv run pytest tests/ -n auto
```

## Project Structure

```
src/pivot/
├── cli/                 # CLI commands
├── config/              # Configuration handling
├── executor/            # Stage execution
├── pipeline/            # Pipeline loading
├── remote/              # S3 storage
├── show/                # Display commands
├── storage/             # Cache and lock files
├── tui/                 # Terminal UI
├── watch/               # Watch mode
├── dag.py               # DAG construction
├── discovery.py         # Pipeline discovery
├── fingerprint.py       # Code fingerprinting
├── loaders.py           # Data loaders
├── outputs.py           # Output type definitions
├── registry.py          # Stage registry
├── stage_def.py         # Stage definitions
└── types.py             # Type definitions
```

## Test Structure

```
tests/
├── unit/           # Unit tests per module
├── integration/    # Full pipeline tests
└── fingerprint/    # Code change detection tests
```

## Your First Contribution

1. Pick an issue labeled `good first issue`
2. Fork the repository
3. Create a feature branch
4. Make changes with tests
5. Run all quality checks
6. Submit a pull request

## Pull Request Checklist

Before opening a PR:

- [ ] All tests pass (`pytest tests/ -n auto`)
- [ ] Code is formatted (`ruff format .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`basedpyright .`)
- [ ] New features have tests
- [ ] Docstrings added for public functions

## Getting Help

- Open an issue for questions
- Check existing issues for similar problems
- Read the [Architecture Overview](../architecture/overview.md) for context

## Next Steps

- [Code Style](style.md) - Coding conventions
- [Testing Guide](testing.md) - Writing tests
- [CLI Development](cli.md) - Adding CLI commands
