import json
from typing import Any, cast

import click
import pydantic

from pivot import config, config_files, exceptions
from pivot import state_dir as state_dir_mod
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators

# The key path of `core.state_dir` in a config file.
_STATE_DIR_PATH = ("core", "state_dir")


def _format_value(value: Any) -> str:
    """Format a config value for display."""
    if value is None:
        return "(not set)"
    if isinstance(value, list):
        items = cast("list[Any]", value)
        return ", ".join(str(v) for v in items)
    return str(value)


def _flatten_config(data: dict[str, Any], path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """Flatten a nested config dict to the key path of each value, as the file nests them."""
    result = dict[tuple[str, ...], Any]()
    for key, value in data.items():
        full_path = (*path, key)
        if isinstance(value, dict) and key != "remotes":
            nested = cast("dict[str, Any]", value)
            result.update(_flatten_config(nested, full_path))
        else:
            result[full_path] = value
    return result


def _raise_for_non_string_state_dir(data: dict[str, Any]) -> None:
    """Keep config-list's release failure concise when its stored state value has the wrong type."""
    value = _get_nested_value(data, "core.state_dir")
    if value is not None and not isinstance(value, str):
        raise click.ClickException("core.state_dir must be a string")


@click.group()
def config_cmd() -> None:
    """View and modify Pivot configuration."""


@config_cmd.command("list")
@click.option("--global", "use_global", is_flag=True, help="Show only global config")
@click.option("--local", "use_local", is_flag=True, help="Show only local config")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@cli_decorators.with_error_handling
def config_list(use_global: bool, use_local: bool, output_json: bool) -> None:
    """Display all configuration values."""
    data: dict[str, Any]
    if use_global:
        data = config.load_config_file(config_files.get_global_config_path())
    elif use_local:
        data = config.load_config_file(config.get_local_config_path())
    else:
        _raise_for_non_string_state_dir(config.load_config_file(config.get_local_config_path()))
        merged = config.get_merged_config()
        data = merged.model_dump()

    if output_json:
        click.echo(json.dumps(data, indent=2))
        return

    flat = _flatten_config(data)

    if not flat:
        click.echo("No configuration set")
        return

    # `--global` and `--local` print the stored text as 0.5.2 does; the merged listing marks a
    # value the state directory cannot use.
    merged_listing = not (use_global or use_local)
    # Inside a project, every `--global` row that comes from the file's `core.state_dir` entry,
    # whatever that value's type, is ignored: a mapping there flattens to several rows.
    global_state_dir_ignored = (
        use_global
        and any(path[:2] == _STATE_DIR_PATH for path in flat)
        and state_dir_mod.report_ignored_global_state_dir()
    )
    for path in sorted(flat, key=".".join):
        key = ".".join(path)
        value = flat[path]
        if key == "remotes" and isinstance(value, dict):
            remotes_dict = cast("dict[str, str]", value)
            for remote_name, remote_url in remotes_dict.items():
                if use_global:
                    source = config.ConfigSource.GLOBAL
                else:
                    _, source = config.get_config_value(f"remotes.{remote_name}")
                click.echo(f"remotes.{remote_name} = {remote_url} ({source})")
        else:
            if use_global:
                # Every row comes from the global file, so every label does too.
                source = config.ConfigSource.GLOBAL
                if global_state_dir_ignored and path[:2] == _STATE_DIR_PATH:
                    source = config.ConfigSource.GLOBAL_IGNORED
            else:
                configured_value, source = config.get_config_value(key)
                if source == config.ConfigSource.GLOBAL_IGNORED:
                    value = configured_value
            row = f"{key} = {_format_value(value)} ({source})"
            if (
                merged_listing
                and key == "core.state_dir"
                and not state_dir_mod.is_valid_state_dir_value(value)
            ):
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
@cli_decorators.with_error_handling
def config_get(key: str, use_global: bool, use_local: bool, output_json: bool) -> None:
    """Get a configuration value by dotted key."""
    if use_global and use_local:
        raise click.ClickException("Cannot use both --global and --local")

    if not config.is_valid_key(key):
        raise click.ClickException(f"Unknown config key: '{key}'")

    if use_global:
        data = config.load_config_file(config_files.get_global_config_path())
        value = _get_nested_value(data, key)
        source = config.ConfigScope.GLOBAL
        # Outside a project nothing ignores the value. An unreadable local config only drops
        # the warning: labeling a global value never needs the local file.
        global_state_dir = key == "core.state_dir" and value is not None
        if global_state_dir and state_dir_mod.report_ignored_global_state_dir():
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
@cli_decorators.with_error_handling
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
@cli_decorators.with_error_handling
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
