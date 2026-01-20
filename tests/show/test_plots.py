from __future__ import annotations

import inspect
import json
import subprocess
from typing import TYPE_CHECKING

from pivot import loaders, outputs
from pivot.registry import REGISTRY, RegistryStageInfo
from pivot.show import plots
from pivot.storage import lock
from pivot.types import ChangeType, LockData, OutputFormat

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Helper to register Plot outputs
# =============================================================================


def _register_plot_stage(
    name: str,
    plot_path: str,
    x: str | None = None,
    y: str | None = None,
    template: str | None = None,
) -> None:
    """Register a test stage with a Plot output directly in the registry.

    This bypasses the annotation-based registration since Plot outputs
    can't be expressed through annotations (they require outputs.Plot).
    """

    def _stage_func() -> None:
        pass

    REGISTRY._stages[name] = RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        deps_paths=[],
        outs=[outputs.Plot(path=plot_path, loader=loaders.PathOnly(), x=x, y=y, template=template)],
        outs_paths=[plot_path],
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint={"_code": "fake_hash"},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
    )


def _register_mixed_output_stage(
    name: str,
    out_path: str,
    metric_path: str,
    plot_path: str,
) -> None:
    """Register a test stage with Out, Metric, and Plot outputs."""

    def _stage_func() -> None:
        pass

    REGISTRY._stages[name] = RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        deps_paths=[],
        outs=[
            outputs.Out(path=out_path, loader=loaders.PathOnly()),
            outputs.Metric(path=metric_path),
            outputs.Plot(path=plot_path, loader=loaders.PathOnly()),
        ],
        outs_paths=[out_path, metric_path, plot_path],
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint={"_code": "fake_hash"},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
    )


# =============================================================================
# collect_plots_from_stages Tests
# =============================================================================


def test_collect_plots_from_stages_empty(clean_registry: None) -> None:
    """Empty registry returns empty list."""
    result = plots.collect_plots_from_stages()

    assert result == []


def test_collect_plots_from_stages_finds_plots(set_project_root: Path) -> None:
    """Finds Plot outputs from registered stages."""
    plot_path = set_project_root / "plot.png"
    plot_path.write_bytes(b"data")

    _register_plot_stage("my_stage", str(plot_path), x="epoch", y="loss")

    result = plots.collect_plots_from_stages()

    assert len(result) == 1
    assert result[0]["stage_name"] == "my_stage"
    assert result[0]["x"] == "epoch"
    assert result[0]["y"] == "loss"


def test_collect_plots_from_stages_ignores_non_plots(set_project_root: Path) -> None:
    """Ignores Out and Metric outputs, only returns Plot outputs."""
    out_file = set_project_root / "output.txt"
    metric_file = set_project_root / "metrics.json"
    plot_file = set_project_root / "chart.png"
    out_file.write_text("data")
    metric_file.write_text("{}")
    plot_file.write_bytes(b"png")

    _register_mixed_output_stage(
        "mixed_stage",
        str(out_file),
        str(metric_file),
        str(plot_file),
    )

    result = plots.collect_plots_from_stages()

    assert len(result) == 1
    assert result[0]["path"] == str(plot_file)


# =============================================================================
# get_plot_hashes_from_lock Tests
# =============================================================================


def test_get_plot_hashes_from_lock_no_lock_file(set_project_root: Path) -> None:
    """Returns None for plots without lock files."""
    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"data")

    _register_plot_stage("test_stage", str(plot_file))

    result = plots.get_plot_hashes_from_lock()

    # Result keys are relative to project root
    assert "plot.png" in result
    assert result["plot.png"] is None


def test_get_plot_hashes_from_lock_with_hash(set_project_root: Path) -> None:
    """Returns hash from lock file."""
    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"data")

    _register_plot_stage("test_stage", str(plot_file))

    # Create lock file with hash
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)
    stage_lock = lock.StageLock("test_stage", stages_dir)
    stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={str(plot_file): {"hash": "abc123def456"}},
            dep_generations={},
        )
    )

    result = plots.get_plot_hashes_from_lock()

    # Result keys are relative to project root
    assert result["plot.png"] == "abc123def456"


def test_get_plot_hashes_from_lock_with_none_hash(set_project_root: Path) -> None:
    """Returns None for plots with null hash in lock file."""
    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"data")

    _register_plot_stage("test_stage", str(plot_file))

    # Create lock file with None hash (uncached output)
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)
    stage_lock = lock.StageLock("test_stage", stages_dir)
    stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={str(plot_file): None},
            dep_generations={},
        )
    )

    result = plots.get_plot_hashes_from_lock()

    # Result keys are relative to project root
    assert result["plot.png"] is None


# =============================================================================
# get_plot_hashes_from_head Tests
# =============================================================================


def _setup_git_repo(tmp_path: Path) -> None:
    """Initialize a git repo with user config."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def test_get_plot_hashes_from_head_no_git_repo(set_project_root: Path) -> None:
    """Returns empty dict when not in a git repo."""
    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"data")

    _register_plot_stage("test_stage", str(plot_file))

    result = plots.get_plot_hashes_from_head()

    # No git repo, so all plots have None hash
    assert "plot.png" in result
    assert result["plot.png"] is None


def test_get_plot_hashes_from_head_returns_committed_hash(
    set_project_root: Path,
) -> None:
    """Returns hash from lock file committed to HEAD."""
    _setup_git_repo(set_project_root)

    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"data")

    _register_plot_stage("test_stage", str(plot_file))

    # Create and commit lock file with hash
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)
    stage_lock = lock.StageLock("test_stage", stages_dir)
    stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={str(plot_file): {"hash": "committed_hash_123"}},
            dep_generations={},
        )
    )

    subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=set_project_root,
        check=True,
        capture_output=True,
    )

    result = plots.get_plot_hashes_from_head()

    assert result["plot.png"] == "committed_hash_123"


def test_get_plot_hashes_from_head_ignores_uncommitted_changes(
    set_project_root: Path,
) -> None:
    """Returns committed hash, not uncommitted changes."""
    _setup_git_repo(set_project_root)

    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"data")

    _register_plot_stage("test_stage", str(plot_file))

    # Create and commit lock file with original hash
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)
    stage_lock = lock.StageLock("test_stage", stages_dir)
    stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={str(plot_file): {"hash": "original_hash"}},
            dep_generations={},
        )
    )

    subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=set_project_root,
        check=True,
        capture_output=True,
    )

    # Update lock file but don't commit
    stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={str(plot_file): {"hash": "modified_hash"}},
            dep_generations={},
        )
    )

    result = plots.get_plot_hashes_from_head()

    assert result["plot.png"] == "original_hash", "Should return committed hash, not modified"


def test_get_plot_hashes_from_head_no_lock_in_head(set_project_root: Path) -> None:
    """Returns None for plots with no lock file in HEAD."""
    _setup_git_repo(set_project_root)

    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"data")

    # Create initial commit with something else
    (set_project_root / "readme.txt").write_text("readme")
    subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=set_project_root,
        check=True,
        capture_output=True,
    )

    _register_plot_stage("test_stage", str(plot_file))

    result = plots.get_plot_hashes_from_head()

    assert "plot.png" in result
    assert result["plot.png"] is None


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

    plot_info = plots.PlotInfo(
        path=str(plot_file),
        stage_name="test_stage",
        x=None,
        y=None,
        template=None,
    )
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
    plot_info = plots.PlotInfo(
        path=str(tmp_path / "nonexistent.png"),
        stage_name="test_stage",
        x=None,
        y=None,
        template=None,
    )
    output_path = tmp_path / "index.html"

    result = plots.render_plots_html([plot_info], output_path)

    content = result.read_text()
    assert "0 plot(s)" in content


def test_render_plots_html_creates_parent_dirs(tmp_path: Path) -> None:
    """Creates parent directories if they don't exist."""
    plot_file = tmp_path / "plot.png"
    plot_file.write_bytes(b"data")

    plot_info = plots.PlotInfo(
        path=str(plot_file),
        stage_name="test",
        x=None,
        y=None,
        template=None,
    )
    output_path = tmp_path / "deep" / "nested" / "index.html"

    result = plots.render_plots_html([plot_info], output_path)

    assert result.exists()
    assert result.parent.exists()


def test_render_plots_html_escapes_xss(tmp_path: Path) -> None:
    """Escapes HTML special characters to prevent XSS."""
    plot_file = tmp_path / "plot.png"
    plot_file.write_bytes(b"data")

    plot_info = plots.PlotInfo(
        path=str(plot_file),
        stage_name="<script>alert('xss')</script>",
        x=None,
        y=None,
        template=None,
    )
    output_path = tmp_path / "index.html"

    result = plots.render_plots_html([plot_info], output_path)

    content = result.read_text()
    assert "<script>" not in content, "Script tags should be escaped"
    assert "&lt;script&gt;" in content, "HTML entities should be used"


# =============================================================================
# get_output_hashes_from_revision Tests
# =============================================================================


def test_get_output_hashes_from_revision_no_git_repo(set_project_root: Path) -> None:
    """Returns empty dict when not in a git repo."""
    result = plots.get_output_hashes_from_revision("HEAD")

    assert result == {}


def test_get_output_hashes_from_revision_returns_hashes(set_project_root: Path) -> None:
    """Returns output hashes from lock files at revision."""
    _setup_git_repo(set_project_root)

    # Create and commit lock file with hash
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)
    stage_lock = lock.StageLock("test_stage", stages_dir)
    stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={"output.csv": {"hash": "abc123"}},
            dep_generations={},
        )
    )

    subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=set_project_root,
        check=True,
        capture_output=True,
    )

    result = plots.get_output_hashes_from_revision("HEAD")

    assert result == {"output.csv": "abc123"}


def test_get_output_hashes_from_revision_multiple_stages(set_project_root: Path) -> None:
    """Returns hashes from multiple stages."""
    _setup_git_repo(set_project_root)

    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)

    # Create lock files for two stages
    lock.StageLock("stage1", stages_dir).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={"plots/chart1.png": {"hash": "hash1"}},
            dep_generations={},
        )
    )
    lock.StageLock("stage2", stages_dir).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={"plots/chart2.png": {"hash": "hash2"}},
            dep_generations={},
        )
    )

    subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=set_project_root,
        check=True,
        capture_output=True,
    )

    result = plots.get_output_hashes_from_revision("HEAD")

    assert result == {"plots/chart1.png": "hash1", "plots/chart2.png": "hash2"}


def test_get_output_hashes_from_revision_normalizes_paths(set_project_root: Path) -> None:
    """Normalizes paths (e.g., ./foo.csv -> foo.csv)."""
    _setup_git_repo(set_project_root)

    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)
    stage_lock = lock.StageLock("test_stage", stages_dir)
    stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={"./output.csv": {"hash": "abc123"}},
            dep_generations={},
        )
    )

    subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=set_project_root,
        check=True,
        capture_output=True,
    )

    result = plots.get_output_hashes_from_revision("HEAD")

    assert "output.csv" in result, "Path should be normalized"
    assert "./output.csv" not in result


def test_get_output_hashes_from_revision_invalid_revision(set_project_root: Path) -> None:
    """Returns empty dict for invalid revision."""
    _setup_git_repo(set_project_root)

    (set_project_root / "readme.txt").write_text("content")
    subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=set_project_root,
        check=True,
        capture_output=True,
    )

    result = plots.get_output_hashes_from_revision("nonexistent-branch")

    assert result == {}


def test_get_output_hashes_from_revision_no_lock_files(set_project_root: Path) -> None:
    """Returns empty dict when no lock files exist at revision."""
    _setup_git_repo(set_project_root)

    (set_project_root / "readme.txt").write_text("content")
    subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=set_project_root,
        check=True,
        capture_output=True,
    )

    result = plots.get_output_hashes_from_revision("HEAD")

    assert result == {}
