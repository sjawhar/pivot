---
tags: [python, yaml, configuration]
category: design
module: config
symptoms: ["comments lost after config update", "yaml formatting changed unexpectedly"]
---

# YAML Library Choice: ruamel.yaml vs PyYAML

## Problem

When a CLI tool reads, modifies, and writes a YAML config file, users lose their carefully formatted comments and whitespace:

```yaml
# User's original config.yaml
cache:
  # Use hardlinks for speed on local filesystem
  checkout_mode: [hardlink, symlink, copy]

  # Override default cache location for this project
  dir: /fast-storage/cache
```

After running `pivot config set cache.dir /new/path`:

```yaml
cache:
  checkout_mode:
  - hardlink
  - symlink
  - copy
  dir: /new/path
```

The comments vanish. Flow-style lists become block-style. Users reasonably expect a config tool to preserve their formatting.

PyYAML (`yaml.safe_load()`) parses YAML into plain Python dicts, discarding comments and formatting metadata. When you dump the dict back, you get PyYAML's default formatting with no memory of the original structure.

## Solution

Use **ruamel.yaml** for files users will edit (config files), and **PyYAML** for read-only parsing (pipeline definitions, lock files, data files):

```python
# EDITABLE CONFIG - preserve comments and formatting
import ruamel.yaml

def edit_config(path: Path) -> None:
    yaml = ruamel.yaml.YAML(typ="rt")  # Round-trip mode

    with path.open() as f:
        data = yaml.load(f)  # Preserves structure

    data["cache"]["dir"] = "/new/path"

    with path.open("w") as f:
        yaml.dump(data, f)  # Comments and formatting intact
```

```python
# READ-ONLY DATA - simplicity and speed preferred
import yaml
from pivot import yaml_config

def load_pipeline(path: Path) -> dict:
    with path.open() as f:
        return yaml.load(f, Loader=yaml_config.Loader)
```

The `yaml_config.Loader` prefers `CSafeLoader` (C extension) for performance, falling back to `SafeLoader` when libyaml is unavailable:

```python
# src/pivot/yaml_config.py
try:
    Loader = yaml.CSafeLoader
    Dumper = yaml.CSafeDumper
except AttributeError:
    Loader = yaml.SafeLoader
    Dumper = yaml.SafeDumper
```

### Where Each Library Is Used in Pivot

| Library | Module | Purpose |
|---------|--------|---------|
| ruamel.yaml | `config/io.py` | User config (preserves comments) |
| ruamel.yaml | `dvc_import.py` | Generated pivot.yaml (clean formatting) |
| PyYAML | `pipeline/yaml.py` | pivot.yaml parsing (read-only) |
| PyYAML | `storage/lock.py` | Lock file parsing |
| PyYAML | `loaders.py` | YAML data loader |
| PyYAML | `parameters.py` | params.yaml files |
| PyYAML | `show/metrics.py` | Metrics YAML files |
| PyYAML | CLI modules | Completion, doctor diagnostics |

### Round-Trip Mode Specifics

ruamel.yaml's `typ="rt"` (round-trip) mode preserves:

- Comments (inline and block)
- Key ordering
- Flow vs block style (`[a, b]` vs multi-line lists)
- Blank lines
- Quote style on strings

```python
yaml = ruamel.yaml.YAML(typ="rt")
yaml.default_flow_style = False  # Explicit for new keys
```

## Key Insight

The choice between YAML libraries depends on the file's lifecycle:

- **User-edited files**: Use ruamel.yaml. Preserving comments respects user investment in documenting their configuration.
- **Machine-generated/read-only files**: Use PyYAML. Simpler API, faster parsing (with C extension), no need for round-trip overhead.

The performance difference is negligible for typical config file sizes. The user experience difference is significant: losing comments feels like data loss, even though the semantic content is preserved.
