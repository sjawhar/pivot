"""Tests for console output module."""

from io import StringIO

from pivot.cli.console import Console
from pivot.types import ArtifactIdentity, ChangeType, StageExplanation


def test_explain_stage_dep_changes() -> None:
    """Console.explain_stage should render dep changes without crashing."""
    output = StringIO()
    con = Console(stream=output, color=False)
    explanation: StageExplanation = {
        "stage_name": "train",
        "will_run": True,
        "is_forced": False,
        "reason": "Dependencies changed",
        "code_changes": [],
        "param_changes": [],
        "dep_changes": [
            {
                "identity": ArtifactIdentity("input_data", None),
                "old_hash": "aaa",
                "new_hash": "bbb",
                "change_type": ChangeType.MODIFIED,
            }
        ],
        "upstream_stale": [],
    }
    con.explain_stage(explanation)
    captured = output.getvalue()
    assert "input_data" in captured
