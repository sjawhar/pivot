---
tags: [python, yaml, configuration]
category: design
module: config
symptoms:
  - "comments lost after config update"
  - "yaml formatting changed unexpectedly"
  - "config value set with `config set` reads back as True, False or a number"
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

Use **ruamel.yaml** for config files, through `config/io.py`, for every read and every write. Use **PyYAML** for every other YAML file (lock files, `.pvt` files, data files, params, metrics):

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
# EVERY OTHER YAML FILE - simplicity and speed preferred
import yaml
from pivot import yaml_config

def load_lock(path: Path) -> dict:
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

### One reader for config files

Every read of a file that `config set` writes (`.pivot/config.yaml`, `~/.config/pivot/config.yaml`) goes through `config/io.py`'s ruamel loader (`load_config_file`, `read_config_file_value`, `parse_config_yaml`), never PyYAML. That includes a read-only lookup of one key, such as `core.state_dir` in `project.state_dir_for`.

ruamel writes and reads YAML 1.2; PyYAML reads YAML 1.1. Plain scalars that YAML 1.2 keeps as strings come back from PyYAML as other types, so a value written by `config set` changes meaning when PyYAML reads it:

| Written by `config set core.state_dir …` | ruamel (YAML 1.2) reads | PyYAML (YAML 1.1) reads |
|---|---|---|
| `on` | `'on'` | `True` |
| `off` | `'off'` | `False` |
| `yes` | `'yes'` | `True` |
| `no` | `'no'` | `False` |
| `1:20` | `'1:20'` | `80` |
| `y` | `'y'` | `'y'` |

(Probe: `ruamel.yaml.YAML(typ="rt").load(text)` against `yaml.safe_load(text)` for `core:\n  state_dir: <value>\n`; the round-trip tests are `packages/pivot/tests/cli/test_pr503_r6_state_dir_yaml_1_1_value.py`.)

The two libraries also fail differently: `config/io.py` reports `Invalid YAML in <file>` and `Permission denied reading <file>`, so one reader also means one error message.

Exceptions, each kept to match release 0.5.2:

- `pivot import` reads a source repository's `remotes` and `default_remote` with PyYAML (`import_artifact.read_remote_config`), as 0.5.2 does, and reads the same file's `core.state_dir` with `config_io.parse_config_yaml`. A comment at that call states that the two parses are deliberate.
- `remote/config.py:55` (`_load_raw_config`, behind `pivot remote` and remote lookup) and `cli/completion.py:46` (`_get_config_keys`, remote-name completion) still read config files with PyYAML, as on 0.5.2. A remote named `on` therefore reads back as `True` there.

### Where Each Library Is Used in Pivot

| Library | Module | Purpose |
|---------|--------|---------|
| ruamel.yaml | `config/io.py` | Every config-file read and write, including `project.state_dir_for` and import's source `core.state_dir` |
| PyYAML | `import_artifact.py` | Source repository `remotes` and `default_remote` (0.5.2 parse, kept deliberately) |
| PyYAML | `remote/config.py`, `cli/completion.py` | Remote names and URLs from config files (0.5.2 readers, not yet on `config/io.py`) |
| PyYAML | `storage/lock.py` | Lock file parsing |
| PyYAML | `storage/track.py` | `.pvt` files |
| PyYAML | `loaders.py` | YAML data loader |
| PyYAML | `parameters.py` | params.yaml files |
| PyYAML | `show/metrics.py` | Metrics YAML files |

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

The choice between YAML libraries depends on the file:

- **Config files (`config set` writes them, users edit them)**: Use ruamel.yaml, through `config/io.py`, for reads as well as writes. Preserving comments respects the user's documentation, and reading with the writer's YAML version keeps every value meaning what it meant when written.
- **Every other YAML file**: Use PyYAML. Simpler API, faster parsing (with C extension), no need for round-trip overhead.

The performance difference is negligible for typical config file sizes. The user experience difference is significant: losing comments feels like data loss, even though the semantic content is preserved.
