import json
from typing import Any, cast

import click
import pydantic

from pivot import config, exceptions, project
from pivot.cli import completion


def _format_value(value: Any) -> str:
    """Format a config value for display."""
    if value is None:
        return "(not set)"
    if isinstance(value, list):
        items = cast("list[Any]", value)
        return ", ".join(str(v) for v in items)
    return str(value)


def _flatten_config(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dict to dotted keys."""
    result = dict[str, Any]()
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and key != "remotes":
            nested = cast("dict[str, Any]", value)
            result.update(_flatten_config(nested, full_key))
        else:
            result[full_key] = value
    return result


@click.group()
def config_cmd() -> None:
    """View and modify Pivot configuration."""


@config_cmd.command("list")
@click.option("--global", "use_global", is_flag=True, help="Show only global config")
@click.option("--local", "use_local", is_flag=True, help="Show only local config")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def config_list(use_global: bool, use_local: bool, output_json: bool) -> None:
    """Display all configuration values."""
    data: dict[str, Any]
    if use_global:
        data = config.load_config_file(config.get_global_config_path())
    elif use_local:
        data = config.load_config_file(config.get_local_config_path())
    else:
        merged = config.get_merged_config()
        data = merged.model_dump()

    if output_json:
        click.echo(json.dumps(data, indent=2))
        return

    flat = _flatten_config(data)

    if not flat:
        click.echo("No configuration set")
        return

    for key in sorted(flat.keys()):
        value = flat[key]
        if key == "remotes" and isinstance(value, dict):
            remotes_dict = cast("dict[str, str]", value)
            for remote_name, remote_url in remotes_dict.items():
                _, source = config.get_config_value(f"remotes.{remote_name}")
                click.echo(f"remotes.{remote_name} = {remote_url} ({source})")
        else:
            configured_value, source = config.get_config_value(key)
            if source == config.ConfigSource.GLOBAL_IGNORED:
                value = configured_value
            row = f"{key} = {_format_value(value)} ({source})"
            if key == "core.state_dir" and not project.is_valid_state_dir_value(value):
                row += " invalid: core.state_dir must be a path"
            click.echo(row)


def _get_nested_value(data: dict[str, Any], key: str) -> Any:
    """Get nested value from dict using dotted key."""
    keys = key.split(".")
    current: dict[str, Any] = data
    for i, k in enumerate(keys):
        if k not in current:
            return None
        value: Any = current[k]
        if i < len(keys) - 1:
            if not isinstance(value, dict):
                return None
            current = cast("dict[str, Any]", value)
        else:
            return value
    return None


@config_cmd.command("get")
@click.argument("key", shell_complete=completion.complete_config_keys)
@click.option("--global", "use_global", is_flag=True, help="Get from global config only")
@click.option("--local", "use_local", is_flag=True, help="Get from local config only")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def config_get(key: str, use_global: bool, use_local: bool, output_json: bool) -> None:
    """Get a configuration value by dotted key."""
    if use_global and use_local:
        raise click.ClickException("Cannot use both --global and --local")

    if not config.is_valid_key(key):
        raise click.ClickException(f"Unknown config key: '{key}'")

    if use_global:
        data = config.load_config_file(config.get_global_config_path())
        value = _get_nested_value(data, key)
        source = config.ConfigScope.GLOBAL
        # Outside a project nothing ignores the value. An unreadable local config only drops
        # the warning: labeling a global value never needs the local file.
        global_state_dir = key == "core.state_dir" and value is not None
        if global_state_dir and project.report_ignored_global_state_dir():
            source = config.ConfigSource.GLOBAL_IGNORED
    elif use_local:
        data = config.load_config_file(config.get_local_config_path())
        value = _get_nested_value(data, key)
        source = config.ConfigScope.LOCAL
    else:
        value, source = config.get_config_value(key)

    if output_json:
        click.echo(json.dumps({"key": key, "value": value, "source": str(source)}))
        return

    click.echo(f"{key} = {_format_value(value)} ({source})")


@config_cmd.command("set")
@click.argument("key", shell_complete=completion.complete_config_keys)
@click.argument("value")
@click.option("--global", "use_global", is_flag=True, help="Set in global config")
def config_set(key: str, value: str, use_global: bool) -> None:
    """Set a configuration value."""
    try:
        validated = config.validate_config_value(key, value)
    except exceptions.ConfigValidationError as e:
        raise click.ClickException(str(e)) from e
    except pydantic.ValidationError as e:
        errors = "; ".join(err["msg"] for err in e.errors())
        raise click.ClickException(f"Invalid value for '{key}': {errors}") from e

    scope = config.ConfigScope.GLOBAL if use_global else config.ConfigScope.LOCAL
    try:
        notices = config.set_config_value(key, validated, scope)
    except exceptions.ConfigValidationError as e:
        raise click.ClickException(str(e)) from e
    for notice in notices:
        click.echo(notice)
    click.echo(f"Set {key} = {_format_value(validated)} in {scope} config")


@config_cmd.command("unset")
@click.argument("key", shell_complete=completion.complete_config_keys)
@click.option("--global", "use_global", is_flag=True, help="Unset from global config")
def config_unset(key: str, use_global: bool) -> None:
    """Remove a configuration value (reverts to default or inherited value)."""
    if not config.is_valid_key(key):
        raise click.ClickException(f"Unknown config key: '{key}'")

    scope = config.ConfigScope.GLOBAL if use_global else config.ConfigScope.LOCAL
    removed, notices = config.unset_config_value(key, scope)
    for notice in notices:
        click.echo(notice)

    if removed:
        click.echo(f"Removed {key} from {scope} config")
    else:
        click.echo(f"Key {key} not set in {scope} config")
