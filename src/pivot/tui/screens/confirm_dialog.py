from __future__ import annotations

from typing import ClassVar, override

import rich.text
import textual.app
import textual.binding
import textual.containers
import textual.screen
import textual.widgets


class ConfirmDialog(textual.screen.ModalScreen[bool]):
    """Reusable Yes/No confirmation dialog."""

    CSS_PATH: ClassVar[str] = "../styles/modal.tcss"  # pyright: ignore[reportIncompatibleVariableOverride]

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("y", "confirm(True)", "Yes", priority=True),
        textual.binding.Binding("n", "confirm(False)", "No", priority=True),
        textual.binding.Binding("escape", "confirm(False)", "Cancel", priority=True),
    ]

    _message: str

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    @override
    def compose(self) -> textual.app.ComposeResult:
        with textual.containers.Container(id="dialog"):
            yield textual.widgets.Static(self._message, id="message")
            yield textual.widgets.Static(rich.text.Text.from_markup(r"\[y]es  \[n]o"), id="hints")

    def action_confirm(self, result: bool) -> None:
        self.dismiss(result)


class ConfirmCommitScreen(ConfirmDialog):
    """Confirmation dialog for committing changes on exit."""

    def __init__(self) -> None:
        super().__init__("You have uncommitted changes. Commit before exit?")


class ConfirmKillWorkersScreen(ConfirmDialog):
    """Confirmation dialog for quitting with running stages."""

    def __init__(self) -> None:
        super().__init__("Stages are still running. Kill them and quit?")
