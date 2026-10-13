# pyright: reportMissingImports=false, reportImplicitRelativeImport=false
from __future__ import annotations

import inspect
import json
import typing
from typing import TYPE_CHECKING

import yaml

from pivot import loaders
from pivot.registry import PipelineLike, RegistryStageInfo
from pivot.show import plots
from pivot.types import (
    ArtifactIdentity,
    ArtifactRef,
    ArtifactTag,
    ChangeType,
    OutputFormat,
)

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import GitRepo
    from pivot.compose import Pipeline


# =============================================================================
# Helpers: register stages whose outputs have real identities
# =============================================================================


def _ref(
    stage: str, key: str | None, tag: ArtifactTag, fmt: loaders.Writer[typing.Any]
) -> ArtifactRef:
    return ArtifactRef(
        identity=ArtifactIdentity(stage, key), format=fmt, python_type=object, tag=tag
    )


def _register_stage(test_pipeline: PipelineLike, name: str, outs: list[ArtifactRef]) -> None:
    def _stage_func() -> None:
        pass

    pipeline = typing.cast("Pipeline", test_pipeline)
    pipeline._registry._stages[name] = RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        outs=outs,
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint={"_code": "fake_hash"},
        params_arg_name=None,
        state_dir=None,
        collection_params={},
        no_fingerprint=False,
    )


def _register_plot_stage(test_pipeline: PipelineLike, name: str) -> None:
    """Register a stage with one plot output, written by pivot to plots/<name>.png."""
    _register_stage(
        test_pipeline, name, [_ref(name, None, ArtifactTag.PLOT, loaders.MatplotlibFigure())]
    )


def _write_lock(repo_path: Path, stage_name: str, outs: list[dict[str, object]]) -> None:
    stages_dir = repo_path / ".pivot" / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    lock_content = {"code_manifest": {}, "params": {}, "deps": [], "outs": outs}
    (stages_dir / f"{stage_name}.lock").write_text(yaml.dump(lock_content))


# =============================================================================
# collect_plots_from_stages Tests
# =============================================================================


def test_collect_plots_from_stages_empty(mock_discovery: PipelineLike) -> None:
    """Empty registry returns empty list."""
    result = plots.collect_plots_from_stages()

    assert result == []


def test_collect_plots_from_stages_finds_plots(mock_discovery: PipelineLike) -> None:
    """A stage's plot output is listed at its workspace path."""
    _register_plot_stage(mock_discovery, "my_stage")

    result = plots.collect_plots_from_stages()

    assert result == [plots.PlotInfo(path="plots/my_stage.png", stage_name="my_stage")]


def test_collect_plots_from_stages_ignores_non_plots(mock_discovery: PipelineLike) -> None:
    """Ignores data and metric outputs, only returns plot outputs."""
    _register_stage(
        mock_discovery,
        "mixed_stage",
        [
            _ref("mixed_stage", "out", ArtifactTag.DATA, loaders.CSV()),
            _ref("mixed_stage", "score", ArtifactTag.METRIC, loaders.YAML()),
            _ref("mixed_stage", "chart", ArtifactTag.PLOT, loaders.MatplotlibFigure()),
        ],
    )

    result = plots.collect_plots_from_stages()

    assert result == [plots.PlotInfo(path="plots/mixed_stage/chart.png", stage_name="mixed_stage")]


# =============================================================================
# get_plot_hashes_from_head Tests
# =============================================================================


def test_get_plot_hashes_from_head_no_git_repo(mock_discovery: PipelineLike) -> None:
    """Without a git HEAD every plot maps to None."""
    _register_plot_stage(mock_discovery, "test_stage")

    result = plots.get_plot_hashes_from_head()

    assert result == {"plots/test_stage.png": None}


def test_get_plot_hashes_from_head_returns_committed_hash(
    git_repo: GitRepo,
    mock_discovery: PipelineLike,
) -> None:
    """Returns hash from lock file committed to HEAD."""
    repo_path, commit = git_repo
    _register_plot_stage(mock_discovery, "test_stage")
    _write_lock(
        repo_path, "test_stage", [{"key": None, "hash": "committed_hash_123", "tag": "plot"}]
    )
    commit("initial")

    result = plots.get_plot_hashes_from_head()

    assert result == {"plots/test_stage.png": "committed_hash_123"}


def test_get_plot_hashes_from_head_matches_keyed_output(
    git_repo: GitRepo,
    mock_discovery: PipelineLike,
) -> None:
    """A keyed plot output takes the hash recorded under its own key."""
    repo_path, commit = git_repo
    _register_stage(
        mock_discovery,
        "test_stage",
        [_ref("test_stage", "chart", ArtifactTag.PLOT, loaders.MatplotlibFigure())],
    )
    _write_lock(
        repo_path,
        "test_stage",
        [
            {"key": "chart", "hash": "chart_hash", "tag": "plot"},
            {"key": "table", "hash": "table_hash", "tag": "data"},
        ],
    )
    commit("initial")

    result = plots.get_plot_hashes_from_head()

    assert result == {"plots/test_stage/chart.png": "chart_hash"}


def test_get_plot_hashes_from_head_ignores_uncommitted_changes(
    git_repo: GitRepo,
    mock_discovery: PipelineLike,
) -> None:
    """Returns committed hash, not uncommitted changes."""
    repo_path, commit = git_repo
    _register_plot_stage(mock_discovery, "test_stage")
    _write_lock(repo_path, "test_stage", [{"key": None, "hash": "original_hash", "tag": "plot"}])
    commit("initial")
    _write_lock(repo_path, "test_stage", [{"key": None, "hash": "modified_hash", "tag": "plot"}])

    result = plots.get_plot_hashes_from_head()

    assert result == {"plots/test_stage.png": "original_hash"}, (
        "Should return committed hash, not modified"
    )


def test_get_plot_hashes_from_head_no_lock_in_head(
    git_repo: GitRepo,
    mock_discovery: PipelineLike,
) -> None:
    """Returns None for plots with no lock file in HEAD."""
    repo_path, commit = git_repo
    (repo_path / "readme.txt").write_text("readme")
    commit("initial")
    _register_plot_stage(mock_discovery, "test_stage")

    result = plots.get_plot_hashes_from_head()

    assert result == {"plots/test_stage.png": None}


# =============================================================================
# get_plot_hashes_from_workspace Tests
# =============================================================================


def test_get_plot_hashes_from_workspace_existing_file(tmp_path: Path) -> None:
    """Computes hash for existing files."""
    plot_file = tmp_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    result = plots.get_plot_hashes_from_workspace([str(plot_file)])

    assert str(plot_file) in result
    assert len(result[str(plot_file)]) == 16  # xxhash64 hex length


def test_get_plot_hashes_from_workspace_missing_file(tmp_path: Path) -> None:
    """Missing files are not included in result."""
    missing_path = str(tmp_path / "nonexistent.png")

    result = plots.get_plot_hashes_from_workspace([missing_path])

    assert missing_path not in result


def test_get_plot_hashes_from_workspace_multiple_files(tmp_path: Path) -> None:
    """Handles multiple files."""
    file1 = tmp_path / "plot1.png"
    file2 = tmp_path / "plot2.png"
    file1.write_bytes(b"data1")
    file2.write_bytes(b"data2")

    result = plots.get_plot_hashes_from_workspace([str(file1), str(file2)])

    assert len(result) == 2
    assert result[str(file1)] != result[str(file2)]


# =============================================================================
# diff_plots Tests
# =============================================================================


def test_diff_plots_no_changes() -> None:
    """No changes when hashes match."""
    old = {"plot.png": "abc123"}
    new = {"plot.png": "abc123"}

    result = plots.diff_plots(old, new)

    assert result == []


def test_diff_plots_modified() -> None:
    """Detects modified files (different hashes)."""
    old = {"plot.png": "abc123"}
    new = {"plot.png": "def456"}

    result = plots.diff_plots(old, new)

    assert len(result) == 1
    assert result[0]["path"] == "plot.png"
    assert result[0]["old_hash"] == "abc123"
    assert result[0]["new_hash"] == "def456"
    assert result[0]["change_type"] == "modified"


def test_diff_plots_added() -> None:
    """Detects added files (in new but not old)."""
    old: dict[str, str | None] = {}
    new = {"plot.png": "abc123"}

    result = plots.diff_plots(old, new)

    assert len(result) == 1
    assert result[0]["path"] == "plot.png"
    assert result[0]["old_hash"] is None
    assert result[0]["new_hash"] == "abc123"
    assert result[0]["change_type"] == "added"


def test_diff_plots_added_from_none_hash() -> None:
    """Detects added when old hash is None (uncached output)."""
    old: dict[str, str | None] = {"plot.png": None}
    new = {"plot.png": "abc123"}

    result = plots.diff_plots(old, new)

    assert len(result) == 1
    assert result[0]["path"] == "plot.png"
    assert result[0]["old_hash"] is None
    assert result[0]["new_hash"] == "abc123"
    assert result[0]["change_type"] == "added", "Should report 'added' when old hash is None"


def test_diff_plots_removed() -> None:
    """Detects removed files (in old but not new)."""
    old = {"plot.png": "abc123"}
    new: dict[str, str] = {}

    result = plots.diff_plots(old, new)

    assert len(result) == 1
    assert result[0]["path"] == "plot.png"
    assert result[0]["old_hash"] == "abc123"
    assert result[0]["new_hash"] is None
    assert result[0]["change_type"] == "removed"


def test_diff_plots_multiple_changes() -> None:
    """Handles multiple changes (modified, added, removed)."""
    old = {"a.png": "hash_a", "b.png": "hash_b"}
    new = {"a.png": "hash_a_modified", "c.png": "hash_c"}

    result = plots.diff_plots(old, new)

    assert len(result) == 3
    paths = {r["path"]: r for r in result}

    assert paths["a.png"]["change_type"] == "modified"
    assert paths["b.png"]["change_type"] == "removed"
    assert paths["c.png"]["change_type"] == "added"


def test_diff_plots_sorted_by_path() -> None:
    """Results are sorted by path."""
    old = {"z.png": "1", "a.png": "2", "m.png": "3"}
    new = {"z.png": "x", "a.png": "y", "m.png": "z"}

    result = plots.diff_plots(old, new)

    assert [r["path"] for r in result] == ["a.png", "m.png", "z.png"]


# =============================================================================
# format_diff_table Tests
# =============================================================================


def test_format_diff_table_empty() -> None:
    """Empty diffs returns 'No plot changes.'"""
    result = plots.format_diff_table([], None)

    assert result == "No plot changes."


def test_format_diff_table_plain() -> None:
    """Plain text format uses tabulate."""
    diffs = [
        plots.PlotDiffEntry(
            path="plot.png", old_hash="abc123", new_hash="def456", change_type=ChangeType.MODIFIED
        )
    ]

    result = plots.format_diff_table(diffs, None)

    assert "plot.png" in result
    assert "abc123"[:8] in result  # Truncated hash
    assert "modified" in result


def test_format_diff_table_json() -> None:
    """JSON format returns valid JSON."""
    diffs = [
        plots.PlotDiffEntry(
            path="plot.png", old_hash="abc123", new_hash="def456", change_type=ChangeType.MODIFIED
        )
    ]

    result = plots.format_diff_table(diffs, OutputFormat.JSON)

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["path"] == "plot.png"
    assert parsed[0]["change_type"] == "modified"


def test_format_diff_table_markdown() -> None:
    """Markdown format uses github tablefmt."""
    diffs = [
        plots.PlotDiffEntry(
            path="plot.png", old_hash="abc123", new_hash="def456", change_type=ChangeType.MODIFIED
        )
    ]

    result = plots.format_diff_table(diffs, OutputFormat.MD)

    assert "|" in result
    assert "---" in result


def test_format_diff_table_no_path() -> None:
    """show_path=False hides path column."""
    diffs = [
        plots.PlotDiffEntry(
            path="plot.png", old_hash="abc123", new_hash="def456", change_type=ChangeType.MODIFIED
        )
    ]

    result = plots.format_diff_table(diffs, None, show_path=False)

    assert "Path" not in result


# =============================================================================
# render_plots_html Tests
# =============================================================================


def test_render_plots_html_creates_file(tmp_path: Path) -> None:
    """Creates HTML output file."""
    plot_file = tmp_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    plot_info = plots.PlotInfo(path=str(plot_file), stage_name="test_stage")
    output_path = tmp_path / "output" / "index.html"

    result = plots.render_plots_html([plot_info], output_path)

    assert result.exists()
    content = result.read_text()
    assert "<title>Pivot Plots</title>" in content
    assert "test_stage" in content


def test_render_plots_html_empty_list(tmp_path: Path) -> None:
    """Handles empty plot list."""
    output_path = tmp_path / "index.html"

    result = plots.render_plots_html([], output_path)

    assert result.exists()
    content = result.read_text()
    assert "No plots found." in content


def test_render_plots_html_skips_missing_files(tmp_path: Path) -> None:
    """Skips plots whose files don't exist."""
    plot_info = plots.PlotInfo(path=str(tmp_path / "nonexistent.png"), stage_name="test_stage")
    output_path = tmp_path / "index.html"

    result = plots.render_plots_html([plot_info], output_path)

    content = result.read_text()
    assert "0 plot(s)" in content


def test_render_plots_html_creates_parent_dirs(tmp_path: Path) -> None:
    """Creates parent directories if they don't exist."""
    plot_file = tmp_path / "plot.png"
    plot_file.write_bytes(b"data")

    plot_info = plots.PlotInfo(path=str(plot_file), stage_name="test")
    output_path = tmp_path / "deep" / "nested" / "index.html"

    result = plots.render_plots_html([plot_info], output_path)

    assert result.exists()
    assert result.parent.exists()


def test_render_plots_html_escapes_xss(tmp_path: Path) -> None:
    """Escapes HTML special characters to prevent XSS."""
    plot_file = tmp_path / "plot.png"
    plot_file.write_bytes(b"data")

    plot_info = plots.PlotInfo(path=str(plot_file), stage_name="<script>alert('xss')</script>")
    output_path = tmp_path / "index.html"

    result = plots.render_plots_html([plot_info], output_path)

    content = result.read_text()
    assert "<script>" not in content, "Script tags should be escaped"
    assert "&lt;script&gt;" in content, "HTML entities should be used"
