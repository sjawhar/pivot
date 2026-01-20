import enum
import re
from typing import Annotated, Any, Self, cast

import pydantic

_S3_URL_PATTERN = re.compile(r"^s3://[^/]+(/.*)?$")
_REMOTES_PREFIX = "remotes."

# Valid remote name pattern (alphanumeric, hyphens, underscores) - public for completion module
VALID_REMOTE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")

# Type alias for config values after validation
ConfigValue = str | int | list[str] | dict[str, str] | None


class CheckoutMode(enum.StrEnum):
    """Strategy for checking out workspace files from cache."""

    HARDLINK = "hardlink"
    SYMLINK = "symlink"
    COPY = "copy"


class ConfigScope(enum.StrEnum):
    """Where to read/write config values."""

    LOCAL = "local"
    GLOBAL = "global"


class ConfigSource(enum.StrEnum):
    """Where a config value originated from."""

    LOCAL = "local"
    GLOBAL = "global"
    DEFAULT = "default"
    UNKNOWN = "unknown"


class CacheConfig(pydantic.BaseModel):
    """Cache configuration options."""

    dir: str = ".pivot/cache"
    checkout_mode: list[CheckoutMode] = pydantic.Field(
        default=[CheckoutMode.HARDLINK, CheckoutMode.SYMLINK, CheckoutMode.COPY]
    )

    @pydantic.field_validator("checkout_mode", mode="before")
    @classmethod
    def parse_checkout_mode(cls, v: Any) -> list[str] | Any:
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v


class CoreConfig(pydantic.BaseModel):
    """Core execution configuration."""

    max_workers: int = -2
    state_dir: str = ".pivot"
    run_history_retention: Annotated[int, pydantic.Field(gt=0)] = 100

    @pydantic.field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        """Ensure max_workers is not zero."""
        if v == 0:
            raise ValueError("max_workers cannot be 0")
        return v


class RemoteTransferConfig(pydantic.BaseModel):
    """Remote transfer configuration."""

    jobs: Annotated[int, pydantic.Field(gt=0)] = 20
    retries: Annotated[int, pydantic.Field(ge=0)] = 10
    connect_timeout: Annotated[int, pydantic.Field(gt=0)] = 30


class WatchConfig(pydantic.BaseModel):
    """Watch mode configuration."""

    debounce: Annotated[int, pydantic.Field(ge=0)] = 300


class DisplayConfig(pydantic.BaseModel):
    """Display formatting configuration."""

    precision: Annotated[int, pydantic.Field(ge=0)] = 5


class DiffConfig(pydantic.BaseModel):
    """Data diff configuration."""

    max_rows: Annotated[int, pydantic.Field(gt=0)] = 10000


def _validate_s3_url(url: str) -> str:
    """Validate that a string is a valid S3 URL."""
    if not _S3_URL_PATTERN.match(url):
        raise ValueError(f"must be a valid S3 URL (s3://bucket/prefix), got: {url}")
    return url


class PivotConfig(pydantic.BaseModel):
    """Complete Pivot configuration schema."""

    cache: CacheConfig = pydantic.Field(default_factory=CacheConfig)
    core: CoreConfig = pydantic.Field(default_factory=CoreConfig)
    remote: RemoteTransferConfig = pydantic.Field(default_factory=RemoteTransferConfig)
    watch: WatchConfig = pydantic.Field(default_factory=WatchConfig)
    display: DisplayConfig = pydantic.Field(default_factory=DisplayConfig)
    diff: DiffConfig = pydantic.Field(default_factory=DiffConfig)
    default_remote: str = ""
    remotes: dict[str, str] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("remotes")
    @classmethod
    def validate_remotes(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate remote names and URLs."""
        for name, url in v.items():
            if not VALID_REMOTE_NAME.match(name):
                raise ValueError(
                    f"Invalid remote name '{name}': must be alphanumeric, hyphens, or underscores"
                )
            try:
                _validate_s3_url(url)
            except ValueError as e:
                raise ValueError(f"Remote '{name}' {e}") from None
        return v

    @classmethod
    def get_default(cls) -> Self:
        """Get default configuration."""
        return cls()


# Config keys with descriptions (used by CLI completion)
CONFIG_KEY_DESCRIPTIONS: dict[str, str] = {
    "cache.dir": "Local cache directory",
    "cache.checkout_mode": "Checkout strategy (hardlink,symlink,copy)",
    "core.max_workers": "Parallel execution workers",
    "core.state_dir": "State directory path",
    "core.run_history_retention": "Keep last N runs",
    "remote.jobs": "Concurrent transfer jobs",
    "remote.retries": "Retry attempts",
    "remote.connect_timeout": "Connection timeout (seconds)",
    "watch.debounce": "Debounce delay (ms)",
    "display.precision": "Decimal precision for metrics",
    "diff.max_rows": "Max rows for diff operations",
    "default_remote": "Default remote name",
}

# Keys that can be set via CLI (excludes dynamic remotes.* pattern)
_KNOWN_KEYS = frozenset(CONFIG_KEY_DESCRIPTIONS.keys())


def is_valid_key(key: str) -> bool:
    """Check if a config key is valid (known key or valid remotes.* pattern)."""
    if key in _KNOWN_KEYS:
        return True
    if key.startswith(_REMOTES_PREFIX):
        remote_name = key[len(_REMOTES_PREFIX) :]
        return bool(VALID_REMOTE_NAME.match(remote_name))
    return False


def get_config_default(key: str) -> ConfigValue:
    """Get the default value for a config key."""
    defaults = PivotConfig.get_default()
    parts = key.split(".")

    if len(parts) == 1:
        if hasattr(defaults, key):
            return getattr(defaults, key)
        return None

    if len(parts) == 2:
        section, subkey = parts
        if section == "remotes":
            return None
        if hasattr(defaults, section):
            section_obj = getattr(defaults, section)
            if hasattr(section_obj, subkey):
                value = getattr(section_obj, subkey)
                if isinstance(value, list):
                    items = cast("list[CheckoutMode]", value)
                    return [str(item) for item in items]
                return value

    return None


_INT_VALIDATION_SPECS: dict[str, tuple[type[pydantic.BaseModel], str, str]] = {
    "core.run_history_retention": (
        CoreConfig,
        "run_history_retention",
        "must be a positive integer",
    ),
    "remote.jobs": (RemoteTransferConfig, "jobs", "must be a positive integer"),
    "remote.retries": (RemoteTransferConfig, "retries", "must be a non-negative integer"),
    "remote.connect_timeout": (
        RemoteTransferConfig,
        "connect_timeout",
        "must be a positive integer",
    ),
    "watch.debounce": (WatchConfig, "debounce", "must be a non-negative integer"),
    "display.precision": (DisplayConfig, "precision", "must be a non-negative integer"),
    "diff.max_rows": (DiffConfig, "max_rows", "must be a positive integer"),
}


def _validate_int_field(
    key: str, value: Any, model_cls: type[pydantic.BaseModel], field: str, error_msg: str
) -> int:
    """Validate an integer field using a Pydantic model."""
    from pivot import exceptions

    try:
        instance = model_cls(**{field: value})
        result = getattr(instance, field)
        return cast("int", result)
    except (pydantic.ValidationError, TypeError):
        raise exceptions.ConfigValidationError(f"'{key}' {error_msg}") from None


def validate_config_value(key: str, value: Any) -> ConfigValue:
    """Validate and coerce a config value based on key."""
    from pivot import exceptions

    parts = key.split(".")

    if key.startswith(_REMOTES_PREFIX) and len(parts) == 2:
        remote_name = parts[1]
        if not VALID_REMOTE_NAME.match(remote_name):
            raise exceptions.ConfigValidationError(
                f"Invalid remote name '{remote_name}': must be alphanumeric, hyphens, or underscores"
            )
        if not isinstance(value, str):
            raise exceptions.ConfigValidationError(f"'{key}' must be a string")
        try:
            return _validate_s3_url(value)
        except ValueError as e:
            raise exceptions.ConfigValidationError(f"'{key}' {e}") from None

    if not is_valid_key(key):
        raise exceptions.ConfigValidationError(f"Unknown config key: '{key}'")

    if key in _INT_VALIDATION_SPECS:
        model_cls, field, error_msg = _INT_VALIDATION_SPECS[key]
        return _validate_int_field(key, value, model_cls, field, error_msg)

    match key:
        case "cache.dir" | "core.state_dir" | "default_remote":
            return str(value) if not isinstance(value, str) else value

        case "cache.checkout_mode":
            try:
                config = CacheConfig(checkout_mode=value)
                return [str(m) for m in config.checkout_mode]
            except pydantic.ValidationError as e:
                msg = "; ".join(err["msg"] for err in e.errors())
                raise exceptions.ConfigValidationError(f"'{key}': {msg}") from None

        case "core.max_workers":
            try:
                config = CoreConfig(max_workers=value)
                return config.max_workers
            except (pydantic.ValidationError, TypeError) as e:
                if isinstance(e, pydantic.ValidationError):
                    msg = "; ".join(err["msg"] for err in e.errors())
                else:
                    msg = "must be an integer"
                raise exceptions.ConfigValidationError(f"'{key}': {msg}") from None

        case _:
            raise exceptions.ConfigValidationError(f"Unknown config key: '{key}'")
