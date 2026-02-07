from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot.tui.widgets import FooterContext, PivotFooter

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# =============================================================================
# Context State Tests
# =============================================================================


def test_footer_default_context_is_stage_list() -> None:
    """Footer defaults to stage list context."""
    footer = PivotFooter()
    assert footer._footer_context == FooterContext.STAGE_LIST


@pytest.mark.parametrize(
    "context",
    [FooterContext.STAGE_LIST, FooterContext.LOGS, FooterContext.DIFF],
)
def test_footer_set_context_updates_internal_state(context: FooterContext) -> None:
    """set_context updates the internal context."""
    footer = PivotFooter()
    footer.set_context(context)
    assert footer._footer_context == context


# =============================================================================
# Context Shortcut Content Tests (Consolidated)
# =============================================================================


@pytest.mark.parametrize(
    ("context", "expected_substrings"),
    [
        pytest.param(
            FooterContext.STAGE_LIST,
            ["j/k", "/", "Filter", "Enter", "Toggle", "q", "Quit", "?", "Help"],
            id="stage_list",
        ),
        pytest.param(
            FooterContext.LOGS,
            ["Ctrl+j/k", "Scroll", "Ctrl+F", "Search", "n/N", "Match", "L/I/O", "Tabs", "q", "?"],
            id="logs",
        ),
        pytest.param(
            FooterContext.DIFF,
            ["Ctrl+j/k", "Scroll", "n/N", "Next/Prev", "Enter", "Expand", "L", "Logs", "q", "?"],
            id="diff",
        ),
    ],
)
def test_footer_context_shows_expected_shortcuts(
    context: FooterContext, expected_substrings: list[str]
) -> None:
    """Each context displays its expected shortcuts."""
    footer = PivotFooter()
    footer.set_context(context)
    content = footer.get_shortcuts_text()

    for substring in expected_substrings:
        assert substring in content, f"Expected '{substring}' in {context.name} context"


# =============================================================================
# Formatting Tests
# =============================================================================


def test_footer_shortcuts_text_formatting() -> None:
    """Shortcuts use bold markup and are separated by spaces."""
    footer = PivotFooter()
    content = footer.get_shortcuts_text()

    assert "[bold]" in content, "Keys should use bold markup"
    assert "[/]" in content, "Bold markup should be closed"
    assert "  " in content, "Shortcuts should be separated by double spaces"


# =============================================================================
# Content Structure Tests
# =============================================================================


def test_footer_shortcuts_text_structure() -> None:
    """Shortcuts are formatted as key-description pairs with bold markup."""
    footer = PivotFooter()
    content = footer.get_shortcuts_text()

    # Should have multiple shortcuts separated by double spaces
    parts = content.split("  ")
    assert len(parts) >= 3, "Should have at least 3 shortcuts"

    # Each part should have bold markup for key
    for part in parts:
        if part.strip():
            assert "[bold]" in part, f"Shortcut part '{part}' should have bold markup"


def test_footer_contexts_have_distinct_content() -> None:
    """Each context displays different shortcuts."""
    footer = PivotFooter()

    contents: dict[FooterContext, str] = {}
    for context in FooterContext:
        footer.set_context(context)
        contents[context] = footer.get_shortcuts_text()

    # All contexts should have unique content
    content_values = list(contents.values())
    assert len(content_values) == len(set(content_values)), (
        "Each context should have unique content"
    )


def test_footer_all_contexts_have_quit_and_help() -> None:
    """All contexts show quit and help shortcuts."""
    footer = PivotFooter()

    for context in FooterContext:
        footer.set_context(context)
        content = footer.get_shortcuts_text()
        assert "q" in content, f"Context {context} should show quit shortcut"
        assert "?" in content, f"Context {context} should show help shortcut"


def test_footer_set_same_context_updates_anyway(mocker: MockerFixture) -> None:
    """Setting same context currently updates display (documents current behavior)."""

    footer = PivotFooter()
    footer.set_context(FooterContext.LOGS)

    # Mock the update method to count calls
    update_mock = mocker.patch.object(footer, "update", wraps=footer.update)

    # Set same context again
    footer.set_context(FooterContext.LOGS)

    # Current implementation calls update even for same context
    assert update_mock.call_count == 1, "Current implementation updates even for same context"


def test_footer_get_shortcuts_without_mount() -> None:
    """get_shortcuts_text should work before widget is mounted."""
    footer = PivotFooter()

    # Should work even before mount
    text = footer.get_shortcuts_text()

    assert "q" in text, "Should include quit shortcut"
    assert "?" in text, "Should include help shortcut"
    assert "[bold]" in text, "Should include markup"
