"""Context-aware footer showing relevant keyboard shortcuts."""

from __future__ import annotations

from enum import Enum, auto
from typing import ClassVar

import textual.widgets


class FooterContext(Enum):
    """Current UI context for footer shortcuts."""

    STAGE_LIST = auto()
    LOGS = auto()
    DIFF = auto()


_SHORTCUTS: dict[FooterContext, list[tuple[str, str]]] = {
    FooterContext.STAGE_LIST: [
        ("j/k", "Up/Down"),
        ("/", "Filter"),
        ("Enter", "Toggle"),
        ("q", "Quit"),
        ("?", "Help"),
    ],
    FooterContext.LOGS: [
        ("Ctrl+j/k", "Scroll"),
        ("L", "Logs"),
        ("I", "Input"),
        ("O", "Output"),
        ("q", "Quit"),
        ("?", "Help"),
    ],
    FooterContext.DIFF: [
        ("Ctrl+j/k", "Scroll"),
        ("n/N", "Next/Prev"),
        ("Enter", "Expand"),
        ("L", "Logs"),
        ("q", "Quit"),
        ("?", "Help"),
    ],
}


class PivotFooter(textual.widgets.Static):
    """Context-aware footer showing relevant keyboard shortcuts."""

    DEFAULT_CSS: ClassVar[str] = """
    PivotFooter {
        color: $text-muted;
    }
    """

    _footer_context: FooterContext

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._footer_context = FooterContext.STAGE_LIST

    def set_context(self, context: FooterContext) -> None:
        """Update the footer context and refresh display."""
        self._footer_context = context
        self.update(self.get_shortcuts_text())

    def get_shortcuts_text(self) -> str:
        """Get formatted shortcuts string for current context."""
        shortcuts = _SHORTCUTS[self._footer_context]
        parts = [f"[bold]{key}[/] {desc}" for key, desc in shortcuts]
        return "  ".join(parts)

    def on_mount(self) -> None:  # pragma: no cover
        """Initialize footer content on mount."""
        self.update(self.get_shortcuts_text())
