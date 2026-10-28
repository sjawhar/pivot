from __future__ import annotations

import click

from pivot import exceptions


def format_pivot_error(e: exceptions.PivotError) -> str:
    """Format one PivotError and its actionable suggestion."""
    message = e.format_user_message()
    if suggestion := e.get_suggestion():
        message = f"{message}\n\nTip: {suggestion}"
    return message


def handle_pivot_error(e: exceptions.PivotError) -> click.ClickException:
    """Convert PivotError to user-friendly ClickException."""
    return click.ClickException(format_pivot_error(e))


def format_unhandled_error(error: BaseException) -> str:
    """The message for an exception a command did not handle itself.

    A PivotError gets its suggestion, a click exception keeps its own message, an exception group
    (an async TaskGroup's) is each exception it holds in turn, and anything else is worded by
    `exceptions.describe_error`: an OSError as its message and paths, any other exception by its
    type. Never a Python repr.
    """
    if isinstance(error, BaseExceptionGroup):
        return "\n".join(format_unhandled_error(inner) for inner in error.exceptions)
    if isinstance(error, exceptions.PivotError):
        return format_pivot_error(error)
    if isinstance(error, click.ClickException):
        return error.format_message()
    return exceptions.describe_error(error)
