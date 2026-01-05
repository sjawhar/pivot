"""Integration tests for DVC compat with real pipelines.

Tests import functionality against real DVC pipelines in pipelines/eval-pipeline/.
These tests require the eval-pipeline directory to be present and not git-ignored.
"""

import pathlib
import sys

import pytest
import yaml

# Add src to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from pivot import dvc_compat, outputs

# Base path to eval-pipeline
EVAL_PIPELINE_DIR = pathlib.Path(__file__).parent.parent / "pipelines" / "eval-pipeline"


def is_pipeline_available() -> bool:
    """Check if eval-pipeline is available and DVC can see it (not git-ignored)."""
    difficulty_yaml = EVAL_PIPELINE_DIR / "eval_pipeline" / "difficulty" / "dvc.yaml"
    if not difficulty_yaml.exists():
        return False

    # DVC needs git to track files - check if repo returns stages
    try:
        import dvc.repo

        repo = dvc.repo.Repo(str(difficulty_yaml.parent))
        return len(repo.index.stages) > 0
    except Exception:
        return False


# Skip all tests if pipeline not available
pytestmark = pytest.mark.skipif(
    not is_pipeline_available(),
    reason="eval-pipeline not available (may be git-ignored)",
)


class TestDifficultyPipeline:
    """Tests for difficulty pipeline (simple, no foreach/matrix)."""

    @pytest.fixture
    def dvc_yaml(self) -> pathlib.Path:
        return EVAL_PIPELINE_DIR / "eval_pipeline" / "difficulty" / "dvc.yaml"

    def test_import_stage_count(self, dvc_yaml: pathlib.Path) -> None:
        """Should import exactly 5 stages from difficulty pipeline."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)
        assert len(specs) == 5

    def test_import_stage_names(self, dvc_yaml: pathlib.Path) -> None:
        """Should import all expected stage names."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)
        expected = {
            "fetch_baselines",
            "patch_baselines",
            "compile_manifests",
            "compute_task_difficulty",
            "compile_human_run_data",
        }
        assert set(specs.keys()) == expected

    def test_import_preserves_cmd(self, dvc_yaml: pathlib.Path) -> None:
        """Should preserve command structure (variables resolved)."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # fetch_baselines has single command
        fb = specs["fetch_baselines"]
        assert isinstance(fb.cmd, str)
        assert "python -m eval_pipeline.difficulty.src.fetch_baselines" in fb.cmd

        # compile_manifests has list of commands
        cm = specs["compile_manifests"]
        assert isinstance(cm.cmd, list)
        assert len(cm.cmd) == 2

    def test_import_preserves_deps(self, dvc_yaml: pathlib.Path) -> None:
        """Should preserve dependencies (DVC adds params.yaml)."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # fetch_baselines: 2 deps in yaml + params.yaml = 3
        fb = specs["fetch_baselines"]
        assert len(fb.deps) >= 2
        dep_names = [pathlib.Path(d).name for d in fb.deps]
        assert "fetch_baselines.py" in dep_names
        assert "fetch_baselines.sql.jinja" in dep_names

    def test_import_preserves_output_types(self, dvc_yaml: pathlib.Path) -> None:
        """Should correctly identify metrics vs regular outs."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # patch_baselines: 2 outs + 1 metric
        pb = specs["patch_baselines"]
        metrics = [o for o in pb.outs if isinstance(o, outputs.Metric)]
        regular = [o for o in pb.outs if isinstance(o, outputs.Out)]
        assert len(metrics) == 1
        assert len(regular) == 2


class TestBasePipeline:
    """Tests for base pipeline (has foreach expansion)."""

    @pytest.fixture
    def dvc_yaml(self) -> pathlib.Path:
        return EVAL_PIPELINE_DIR / "eval_pipeline" / "base" / "dvc.yaml"

    def test_import_expands_foreach(self, dvc_yaml: pathlib.Path) -> None:
        """Should expand foreach stages."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # base has 10 stage definitions, 5 with foreach (current/legacy)
        # Expected: 5 non-foreach + 5*2 = 15 total
        assert len(specs) == 15

    def test_foreach_creates_variants(self, dvc_yaml: pathlib.Path) -> None:
        """Should create @current and @legacy variants for foreach stages."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        foreach_bases = [
            "merge_data",
            "normalize_and_bin_scores",
            "zero_out_cheating_runs",
            "fake_gpt2_data",
            "create_virtual_rebench_tasks",
        ]

        for base in foreach_bases:
            variants = [n for n in specs if n.startswith(f"{base}@")]
            assert len(variants) == 2, f"{base} should have 2 variants"
            assert f"{base}@current" in specs
            assert f"{base}@legacy" in specs

    def test_foreach_resolves_variables(self, dvc_yaml: pathlib.Path) -> None:
        """Should resolve ${item.X} variables in foreach stages."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # merge_data@current should have resolved paths
        current = specs["merge_data@current"]
        assert "merged.jsonl" in str(current.cmd)  # no ${item.suffix}

        # merge_data@legacy should have _legacy_human_runs suffix
        legacy = specs["merge_data@legacy"]
        assert "_legacy_human_runs" in str(legacy.cmd)


class TestHorizonPipeline:
    """Tests for horizon pipeline (has foreach, matrix, YAML anchors)."""

    @pytest.fixture
    def dvc_yaml(self) -> pathlib.Path:
        return EVAL_PIPELINE_DIR / "eval_pipeline" / "horizon" / "dvc.yaml"

    def test_import_expands_all(self, dvc_yaml: pathlib.Path) -> None:
        """Should expand both foreach and matrix stages."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)
        # 31 definitions → 60 expanded
        assert len(specs) == 60

    def test_matrix_expansion(self, dvc_yaml: pathlib.Path) -> None:
        """Should expand matrix stages correctly."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # plot_logistic_individual uses matrix: show_p50: [true, false]
        variants = [n for n in specs if "plot_logistic_individual" in n]
        assert len(variants) == 2
        assert "plot_logistic_individual@true" in specs
        assert "plot_logistic_individual@false" in specs

    def test_yaml_anchor_expansion(self, dvc_yaml: pathlib.Path) -> None:
        """Should resolve YAML anchors and expand foreach."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # wrangle_bootstrap_logistic uses &foreach_logistic anchor
        variants = [n for n in specs if "wrangle_bootstrap_logistic@" in n]
        assert len(variants) == 4
        expected_keys = {"headline", "ga_rebench", "swe_bench", "partial_scoring"}
        actual_keys = {n.split("@")[1] for n in variants}
        assert actual_keys == expected_keys

    def test_preserves_plots(self, dvc_yaml: pathlib.Path) -> None:
        """Should identify plot outputs correctly."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # plot_cost has many plot outputs
        if "plot_cost" in specs:
            pc = specs["plot_cost"]
            plots = [o for o in pc.outs if isinstance(o, outputs.Plot)]
            assert len(plots) >= 1


class TestGaPaperPipeline:
    """Tests for ga_paper pipeline (has matrix)."""

    @pytest.fixture
    def dvc_yaml(self) -> pathlib.Path:
        return EVAL_PIPELINE_DIR / "eval_pipeline" / "ga_paper" / "dvc.yaml"

    def test_import_with_matrix(self, dvc_yaml: pathlib.Path) -> None:
        """Should expand matrix stages."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)
        # 12 definitions, 1 with matrix (2 variants) = 13
        assert len(specs) == 13

    def test_preserves_all_output_types(self, dvc_yaml: pathlib.Path) -> None:
        """Should correctly categorize all output types."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # generate_latex_tables has only metrics
        glt = specs["generate_latex_tables"]
        metrics = [o for o in glt.outs if isinstance(o, outputs.Metric)]
        regular = [o for o in glt.outs if isinstance(o, outputs.Out)]
        assert len(metrics) == 6
        assert len(regular) == 0

        # plot_average_agent_performance has only plots
        pap = specs["plot_average_agent_performance"]
        plots = [o for o in pap.outs if isinstance(o, outputs.Plot)]
        assert len(plots) == 6

    def test_preserves_persist_option(self, dvc_yaml: pathlib.Path) -> None:
        """Should preserve persist: true option."""
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # generate_task_summaries has persist: true
        gts = specs["generate_task_summaries"]
        persisted = [o for o in gts.outs if o.persist]
        assert len(persisted) >= 1


class TestRoundTripReconstruction:
    """Tests for round-trip data preservation."""

    def test_difficulty_round_trip_structure(self) -> None:
        """Should preserve all structural data through import."""
        dvc_yaml = EVAL_PIPELINE_DIR / "eval_pipeline" / "difficulty" / "dvc.yaml"
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        with open(dvc_yaml) as f:
            original = yaml.safe_load(f)

        # Verify all stages present
        assert set(specs.keys()) == set(original["stages"].keys())

        # Verify each stage has required data
        for name, spec in specs.items():
            assert spec.cmd, f"{name} missing cmd"
            assert isinstance(spec.deps, list), f"{name} deps not list"
            assert isinstance(spec.outs, list), f"{name} outs not list"

    def test_base_round_trip_foreach_expansion(self) -> None:
        """Should expand foreach and preserve all data."""
        dvc_yaml = EVAL_PIPELINE_DIR / "eval_pipeline" / "base" / "dvc.yaml"
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        with open(dvc_yaml) as f:
            original = yaml.safe_load(f)

        # Count expected expansions
        foreach_count = 0
        non_foreach_count = 0
        for stage in original["stages"].values():
            if "foreach" in stage:
                # Count items in foreach
                foreach_items = len(stage["foreach"])
                foreach_count += foreach_items
            else:
                non_foreach_count += 1

        assert len(specs) == foreach_count + non_foreach_count

    def test_import_preserves_frozen_state(self) -> None:
        """Should preserve frozen flag from original YAML."""
        dvc_yaml = EVAL_PIPELINE_DIR / "eval_pipeline" / "base" / "dvc.yaml"
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # Check stages that have frozen: false explicitly
        if "fetch_agent_runs_vivaria" in specs:
            assert specs["fetch_agent_runs_vivaria"].frozen is False

    def test_import_preserves_descriptions(self) -> None:
        """Should preserve stage descriptions."""
        dvc_yaml = EVAL_PIPELINE_DIR / "eval_pipeline" / "base" / "dvc.yaml"
        specs = dvc_compat.import_dvc_yaml(dvc_yaml, register=False)

        # Check stage with desc
        if "fetch_agent_runs_vivaria" in specs:
            assert specs["fetch_agent_runs_vivaria"].desc is not None
            assert "Vivaria" in specs["fetch_agent_runs_vivaria"].desc
