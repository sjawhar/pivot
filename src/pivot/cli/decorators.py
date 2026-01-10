from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import click

from pivot import discovery, exceptions

if TYPE_CHECKING:
    from collections.abc import Callable


def _handle_pivot_error(e: exceptions.PivotError) -> click.ClickException:
    """Convert PivotError to user-friendly ClickException."""
    message = e.format_user_message()
    if suggestion := e.get_suggestion():
        message = f"{message}\n\nTip: {suggestion}"
    return click.ClickException(message)


def with_error_handling[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Wrap function with Pivot error handling.

    Use this decorator with @group.command() for group subcommands:

        @remote.command("add")
        @with_error_handling
        def remote_add(...):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except click.ClickException:
            raise
        except exceptions.PivotError as e:
            raise _handle_pivot_error(e) from e
        except Exception as e:
            raise click.ClickException(repr(e)) from e

    return wrapper


def pivot_command(
    name: str | None = None, *, auto_discover: bool = True, **attrs: Any
) -> Callable[[Callable[..., Any]], click.Command]:
    """Create a Click command with Pivot error handling and optional auto-discovery.

    Combines @click.command() with automatic error handling that converts
    PivotError to user-friendly messages with suggestions.

    Args:
        name: Optional command name (defaults to function name)
        auto_discover: If True (default), automatically discover and register
            stages before running the command. Set to False for commands that
            don't need the registry (e.g., init, schema).
        **attrs: Additional arguments passed to click.command()

    Returns:
        Decorator that creates a click.Command with error handling
    """

    def decorator(func: Callable[..., Any]) -> click.Command:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if auto_discover and not discovery.has_registered_stages():
                try:
                    discovery.discover_and_register()
                except discovery.DiscoveryError as e:
                    raise click.ClickException(str(e)) from e
            return func(*args, **kwargs)

        wrapped = with_error_handling(wrapper)
        return click.command(name=name, **attrs)(wrapped)

    return decorator
