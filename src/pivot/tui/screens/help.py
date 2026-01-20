from __future__ import annotations

from typing import ClassVar, override

import textual.app
import textual.binding
import textual.containers
import textual.screen
import textual.widgets

_HELP_TEXT = """\
[bold cyan]Navigation[/]
  j/k or Up/Down    Navigate items
  h/l or Left/Right Navigate tabs / switch panels
  Tab               Switch panel focus
  1-9               Quick-select stage

[bold cyan]Detail Panel[/]
  Enter             Expand to full width
  Escape            Collapse / go back
  n/N               Next/prev changed item
  L/I/O             Jump to Logs/Input/Output tab

[bold cyan]History[/]
  [ / ]             View older/newer execution
  G                 Return to live view
  H                 Show history list

[bold cyan]Actions[/]
  c                 Commit changes
  a                 Toggle all-logs view
  g                 Toggle keep-going mode (watch)
  ~                 Toggle debug panel
  q                 Quit"""


class HelpScreen(textual.screen.ModalScreen[None]):
    """Modal screen showing all keybindings."""

    CSS_PATH: ClassVar[str] = "../styles/modal.tcss"  # pyright: ignore[reportIncompatibleVariableOverride]

    BINDINGS: list[textual.binding.BindingType] = [
        textual.binding.Binding("escape", "dismiss", "Close"),
        textual.binding.Binding("?", "dismiss", "Close", show=False),
        textual.binding.Binding("q", "dismiss", "Close", show=False),
    ]

    @override
    def compose(self) -> textual.app.ComposeResult:
        with textual.containers.Container(id="help-dialog"):
            yield textual.widgets.Static("[bold]Keyboard Shortcuts[/]", id="help-title")
            yield textual.widgets.Static(_HELP_TEXT, id="help-content")
            yield textual.widgets.Static("[dim]Press Esc, ? or q to close[/]", id="help-footer")
