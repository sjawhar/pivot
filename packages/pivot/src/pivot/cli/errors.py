from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from pivot import exceptions


def handle_pivot_error(e: exceptions.PivotError) -> click.ClickException:
    """Convert PivotError to user-friendly ClickException."""
    message = e.format_user_message()
    if suggestion := e.get_suggestion():
        message = f"{message}\n\nTip: {suggestion}"
    return click.ClickException(message)
