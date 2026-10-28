"""Reading one config file: the YAML parser, dotted keys, and the global config path.

A leaf outside the ``pivot.config`` package: it imports nothing from pivot but ``exceptions``,
so ``pivot.state_dir`` reads config files without loading ``pivot.config``, and the merged config
in ``pivot.config.io`` imports ``pivot.state_dir`` without an import cycle.
"""

import pathlib
from typing import Any, cast

import ruamel.yaml

from pivot import exceptions

NOT_FOUND: object = object()


def get_global_config_path() -> pathlib.Path:
    """Get user-level config path (~/.config/pivot/config.yaml)."""
    return pathlib.Path.home() / ".config" / "pivot" / "config.yaml"


def parse_yaml(stream: Any, source: object, *, as_dict: bool) -> Any:
    """Parse config YAML as ``config set`` writes it (YAML 1.2), with 0.5.2's error message."""
    try:
        data = ruamel.yaml.YAML(typ="rt").load(stream)
    except ruamel.yaml.YAMLError as e:
        raise exceptions.ConfigError(f"Invalid YAML in {source}: {e}") from e
    if as_dict:
        return dict(data) if data is not None else {}
    return data


def load_yaml(path: pathlib.Path, *, as_dict: bool) -> Any:
    """Load YAML config with error handling."""
    if not path.exists():
        return {} if as_dict else None

    try:
        with path.open() as f:
            return parse_yaml(f, path, as_dict=as_dict)
    except PermissionError:
        raise exceptions.ConfigError(f"Permission denied reading {path}") from None
    except OSError as e:
        raise exceptions.ConfigError(f"Error reading {path}: {e}") from e


def parse_dotted_key(key: str) -> list[str]:
    """Parse 'cache.checkout_mode' into ['cache', 'checkout_mode']."""
    return key.split(".")


def get_nested(data: dict[str, Any], keys: list[str]) -> Any:
    """Traverse nested dict by key path, returns the NOT_FOUND sentinel if not found."""
    current: Any = data
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return NOT_FOUND
        current_dict = cast("dict[str, Any]", current)
        current = current_dict[k]
    return current


def read_config_file_value(path: pathlib.Path, key: str) -> tuple[bool, Any]:
    """Read one dotted key from one config file: (present, value).

    The file is parsed as ``load_config_file`` parses it; a file that is not a mapping holds no
    key.
    """
    data = load_yaml(path, as_dict=False)
    if not isinstance(data, dict):
        return False, None
    value = get_nested(cast("dict[str, Any]", data), parse_dotted_key(key))
    if value is NOT_FOUND:
        return False, None
    return True, value
