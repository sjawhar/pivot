from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

import yaml

from pivot import exceptions, project, yaml_config
from pivot.types import RawPivotConfig

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

_S3_URL_PATTERN = re.compile(r"^s3://([^/]+)(/.*)?$")


def _get_config_path() -> pathlib.Path:
    """Get path to .pivot/config.yaml."""
    return project.get_project_root() / ".pivot" / "config.yaml"


def _load_raw_config() -> RawPivotConfig:
    """Load raw config data without caching (for write operations)."""
    path = _get_config_path()
    if not path.exists():
        return RawPivotConfig()

    try:
        with path.open() as f:
            data = yaml.load(f, Loader=yaml_config.Loader)
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in config file {path}: {e}")
        return RawPivotConfig()

    if data is None or not isinstance(data, dict):
        return RawPivotConfig()

    # Runtime validation: ensure remotes and default_remote have correct types
    result = RawPivotConfig()
    if "remotes" in data and isinstance(data["remotes"], dict):
        remotes_data = cast("dict[Any, Any]", data["remotes"])
        result["remotes"] = {str(k): str(v) for k, v in remotes_data.items()}
    if "default_remote" in data and isinstance(data["default_remote"], str):
        result["default_remote"] = data["default_remote"]
    return result


def _save_raw_config(data: RawPivotConfig) -> None:
    """Save config data to .pivot/config.yaml."""
    path = _get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(dict(data), f, default_flow_style=False, sort_keys=False)


def validate_s3_url(url: str) -> tuple[str, str]:
    """Validate and parse s3://bucket/prefix URL. Returns (bucket, prefix)."""
    match = _S3_URL_PATTERN.match(url)
    if not match:
        raise exceptions.InvalidRemoteURLError(
            f"Invalid S3 URL: {url}. Expected format: s3://bucket/prefix"
        )
    bucket = match.group(1)
    prefix = (match.group(2) or "").lstrip("/")
    return bucket, prefix


def add_remote(name: str, url: str) -> None:
    """Add or update a remote in config."""
    validate_s3_url(url)

    data = _load_raw_config()
    if "remotes" not in data:
        data["remotes"] = {}
    data["remotes"][name] = url
    _save_raw_config(data)
    logger.info(f"Added remote '{name}': {url}")


def remove_remote(name: str) -> None:
    """Remove a remote from config."""
    data = _load_raw_config()
    remotes = data["remotes"] if "remotes" in data else {}

    if name not in remotes:
        raise exceptions.RemoteNotFoundError(f"Remote '{name}' not found")

    del remotes[name]

    if "default_remote" in data and data["default_remote"] == name:
        del data["default_remote"]
        logger.info(f"Cleared default remote (was '{name}')")

    if not remotes:
        if "remotes" in data:
            del data["remotes"]
    else:
        data["remotes"] = remotes

    _save_raw_config(data)
    logger.info(f"Removed remote '{name}'")


def list_remotes() -> dict[str, str]:
    """Return all configured remotes {name: url}."""
    data = _load_raw_config()
    if "remotes" not in data:
        return {}
    return data["remotes"]


def get_remote_url(name: str | None = None) -> str:
    """Get URL for named remote or default remote."""
    data = _load_raw_config()
    remotes = data["remotes"] if "remotes" in data else {}

    if not remotes:
        raise exceptions.RemoteNotFoundError(
            "No remotes configured. Use 'pivot remote add <name> <url>' to add one."
        )

    if name is None:
        name = data["default_remote"] if "default_remote" in data else None
        if name is None:
            if len(remotes) == 1:
                name = next(iter(remotes.keys()))
            else:
                raise exceptions.RemoteNotFoundError(
                    "No default remote set and multiple remotes configured. Use -r <name> to specify a remote or 'pivot remote default <name>' to set a default."
                )

    if name not in remotes:
        raise exceptions.RemoteNotFoundError(f"Remote '{name}' not found")

    return remotes[name]


def set_default_remote(name: str) -> None:
    """Set the default remote."""
    data = _load_raw_config()
    remotes = data["remotes"] if "remotes" in data else {}

    if name not in remotes:
        raise exceptions.RemoteNotFoundError(f"Remote '{name}' not found")

    data["default_remote"] = name
    _save_raw_config(data)
    logger.info(f"Set default remote to '{name}'")


def get_default_remote() -> str | None:
    """Get the name of the default remote, or None if not set."""
    data = _load_raw_config()
    if "default_remote" not in data:
        return None
    return data["default_remote"]
