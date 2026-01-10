import logging
import pathlib
from typing import Any, TypedDict, cast

import yaml

from pivot import project, yaml_config

logger = logging.getLogger(__name__)

# Default fallback order when no config or config missing checkout_mode
DEFAULT_CHECKOUT_MODE_ORDER = ["hardlink", "symlink", "copy"]

_config_cache: "PivotConfig | None" = None


class CacheConfig(TypedDict, total=False):
    """Cache configuration options."""

    checkout_mode: list[str]


class PivotConfig(TypedDict, total=False):
    """Top-level project configuration."""

    cache: CacheConfig


def load_config(path: pathlib.Path | None = None) -> PivotConfig:
    """Load and cache .pivot/config.yaml from project root."""
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    if path is None:
        path = project.get_project_root() / ".pivot" / "config.yaml"

    if not path.exists():
        logger.debug(f"Config file not found: {path}")
        _config_cache = PivotConfig()
        return _config_cache

    try:
        with path.open() as f:
            data = yaml.load(f, Loader=yaml_config.Loader)
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in config file {path}: {e}")
        _config_cache = PivotConfig()
        return _config_cache

    if data is None:
        _config_cache = PivotConfig()
        return _config_cache

    if not isinstance(data, dict):
        logger.warning(f"Config file {path} must be a mapping, got {type(data).__name__}")
        _config_cache = PivotConfig()
        return _config_cache

    # YAML parsing produces unknown structure, cast after isinstance validation
    _config_cache = _parse_config(cast("dict[str, Any]", data))
    logger.debug(f"Loaded config from {path}: {_config_cache}")
    return _config_cache


def _parse_config(data: dict[str, Any]) -> PivotConfig:
    """Parse raw YAML data into typed config."""
    config = PivotConfig()

    cache_data = data.get("cache")
    if cache_data is not None:
        if not isinstance(cache_data, dict):
            logger.warning(f"'cache' must be a mapping, got {type(cache_data).__name__}")
        else:
            cache_config = CacheConfig()
            cache_dict = cast("dict[str, Any]", cache_data)
            checkout_mode = cache_dict.get("checkout_mode")
            if checkout_mode is not None:
                if isinstance(checkout_mode, list):
                    cache_config["checkout_mode"] = [
                        str(m) for m in cast("list[Any]", checkout_mode)
                    ]
                else:
                    logger.warning(
                        f"'cache.checkout_mode' must be a list, got {type(checkout_mode).__name__}"
                    )
            if cache_config:
                config["cache"] = cache_config

    return config


def get_checkout_mode_order(config: PivotConfig | None = None) -> list[str]:
    """Get checkout mode fallback order from config or default."""
    from pivot.config import CheckoutMode

    if config is None:
        config = load_config()

    cache_config = config.get("cache")
    if cache_config is None:
        return DEFAULT_CHECKOUT_MODE_ORDER.copy()

    checkout_modes = cache_config.get("checkout_mode")
    if checkout_modes is None:
        return DEFAULT_CHECKOUT_MODE_ORDER.copy()

    # Validate each mode against CheckoutMode enum
    valid_modes = list[str]()
    for mode in checkout_modes:
        try:
            CheckoutMode(mode)
            valid_modes.append(mode)
        except ValueError:
            logger.warning(f"Invalid checkout mode '{mode}' in config, skipping")

    if not valid_modes:
        logger.warning("No valid checkout modes in config, using defaults")
        return DEFAULT_CHECKOUT_MODE_ORDER.copy()

    return valid_modes
