from __future__ import annotations

from typing import ClassVar, override

import textual.app
import textual.binding
import textual.containers
import textual.screen
import textual.widgets


class ConfirmCommitScreen(textual.screen.ModalScreen[bool]):
    """Modal screen for confirming commit on exit."""

    CSS_PATH: ClassVar[str] = "../styles/modal.tcss"  # pyright: ignore[reportIncompatibleVariableOverride]

    BINDINGS: list[textual.binding.BindingType] = [
        textual.binding.Binding("y", "confirm(True)", "Yes"),
        textual.binding.Binding("n", "confirm(False)", "No"),
        textual.binding.Binding("escape", "confirm(False)", "Cancel"),
    ]

    @override
    def compose(self) -> textual.app.ComposeResult:
        with textual.containers.Container(id="dialog"):
            yield textual.widgets.Static(
                "You have uncommitted changes. Commit before exit?", id="message"
            )
            yield textual.widgets.Static("[y] Yes  [n] No  [Esc] Cancel")

    def action_confirm(self, result: bool) -> None:
        self.dismiss(result)
