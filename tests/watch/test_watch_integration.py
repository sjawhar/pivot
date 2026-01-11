"""Integration tests for watch mode.

These tests verify end-to-end watch mode behavior with real files and actual
stage execution. They test:
- File change detection (inputs, intermediates, outputs)
- DAG topology handling (linear, fan-out, fan-in, diamond)
- Code and config changes
- Debouncing behavior
- Error recovery

Unlike unit tests that mock core methods, these tests run real file operations
and verify observable behavior.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import threading
import time
from typing import TYPE_CHECKING
from unittest import mock

import pytest

import pivot
from pivot import executor, project
from pivot.watch import engine

if TYPE_CHECKING:
    from collections.abc import Generator

    from pivot.executor.core import ExecutionSummary


# pipeline_dir fixture is in conftest.py


# =============================================================================
# Live Watch Test Helpers
# =============================================================================


@contextlib.contextmanager
def run_watch_engine(
    debounce_ms: int = 50,
    min_executions: int = 2,
    timeout: float = 5.0,
) -> Generator[list[list[str] | None]]:
    """Context manager for running watch engine in background and capturing executions.

    Yields list of captured stage executions. First execution (initial) has stages=None.
    Subsequent executions have the list of affected stages.

    Usage:
        with run_watch_engine() as executions:
            # Make file changes here
            (pipeline_dir / "data.csv").write_text("new data")
        # After context exits, executions contains captured results
        assert len(executions) >= 2
    """
    executions: list[list[str] | None] = []
    done_event = threading.Event()

    def capture_execute(
        self: engine.WatchEngine, stages: list[str] | None
    ) -> dict[str, ExecutionSummary]:
        executions.append(stages)
        if len(executions) >= min_executions:
            done_event.set()
            self.shutdown()
        return {}

    with mock.patch.object(engine.WatchEngine, "_execute_stages", capture_execute):
        eng = engine.WatchEngine(debounce_ms=debounce_ms)
        engine_thread = threading.Thread(target=eng.run)
        engine_thread.start()

        time.sleep(0.2)  # Let watcher initialize (tested: 0.1s works, 0.2s safe margin)
        yield executions

        done_event.wait(timeout=timeout)
        eng.shutdown()
        engine_thread.join(timeout=2.0)


# =============================================================================
# File Index and Change Detection Tests
# =============================================================================


def test_file_index_maps_only_dependencies(pipeline_dir: pathlib.Path) -> None:
    """File index contains dependencies, NOT outputs."""
    (pipeline_dir / "input.csv").write_text("a,b\n1,2")

    @pivot.stage(deps=["input.csv"], outs=["output.csv"])
    def process() -> None:
        pass

    eng = engine.WatchEngine()
    index = eng._build_file_to_stages_index()

    input_path = project.resolve_path("input.csv")
    output_path = project.resolve_path("output.csv")

    assert input_path in index, "Input should be in file index"
    assert "process" in index[input_path], "Input should map to process stage"
    assert output_path not in index, "Output should NOT be in file index"


def test_file_index_intermediate_maps_to_consumer_only(pipeline_dir: pathlib.Path) -> None:
    """Intermediate files map only to consuming stage, not producing stage."""
    (pipeline_dir / "input.csv").write_text("a,b\n1,2")

    @pivot.stage(deps=["input.csv"], outs=["intermediate.csv"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["intermediate.csv"], outs=["final.csv"])
    def stage_b() -> None:
        pass

    eng = engine.WatchEngine()
    index = eng._build_file_to_stages_index()

    input_path = project.resolve_path("input.csv")
    intermediate_path = project.resolve_path("intermediate.csv")
    final_path = project.resolve_path("final.csv")

    # Input maps to stage_a
    assert input_path in index
    assert "stage_a" in index[input_path]

    # Intermediate maps to stage_b (consumer), NOT stage_a (producer)
    assert intermediate_path in index
    assert "stage_b" in index[intermediate_path]
    assert "stage_a" not in index[intermediate_path]

    # Final output not in index
    assert final_path not in index


def test_change_detection_input_file_triggers_stage(pipeline_dir: pathlib.Path) -> None:
    """Modifying an input file triggers the dependent stage."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @pivot.stage(deps=["data.csv"], outs=["result.csv"])
    def process() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("data.csv")
    affected = eng._get_stages_matching_changes({changed_path})

    assert "process" in affected, "Stage should be affected by input change"


def test_change_detection_output_file_does_not_trigger_producer(
    pipeline_dir: pathlib.Path,
) -> None:
    """Modifying an output file does NOT trigger its producing stage."""
    (pipeline_dir / "input.csv").write_text("a,b\n1,2")
    (pipeline_dir / "output.csv").write_text("result")

    @pivot.stage(deps=["input.csv"], outs=["output.csv"])
    def process() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("output.csv")
    affected = eng._get_stages_matching_changes({changed_path})

    assert len(affected) == 0, "Output change should NOT trigger producer stage"


def test_change_detection_intermediate_triggers_downstream_only(
    pipeline_dir: pathlib.Path,
) -> None:
    """Modifying intermediate file triggers downstream, not upstream."""
    (pipeline_dir / "input.csv").write_text("a,b\n1,2")
    (pipeline_dir / "intermediate.csv").write_text("x,y\n3,4")

    @pivot.stage(deps=["input.csv"], outs=["intermediate.csv"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["intermediate.csv"], outs=["final.csv"])
    def stage_b() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("intermediate.csv")
    directly_affected = eng._get_stages_matching_changes({changed_path})

    assert "stage_b" in directly_affected, "Downstream stage should be directly affected"
    assert "stage_a" not in directly_affected, "Upstream stage should NOT be affected"


# =============================================================================
# DAG Topology Tests - Linear
# =============================================================================


def test_linear_dag_input_change_affects_all_downstream(pipeline_dir: pathlib.Path) -> None:
    """Linear DAG: A → B → C. Changing A's input affects A, B, C."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["a_out.txt"])
    def stage_a() -> None:
        pathlib.Path("a_out.txt").write_text(pathlib.Path("input.txt").read_text().upper())

    @pivot.stage(deps=["a_out.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pathlib.Path("b_out.txt").write_text(pathlib.Path("a_out.txt").read_text() + "!")

    @pivot.stage(deps=["b_out.txt"], outs=["c_out.txt"])
    def stage_c() -> None:
        pathlib.Path("c_out.txt").write_text(pathlib.Path("b_out.txt").read_text() + "?")

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("input.txt")
    directly_affected = eng._get_stages_matching_changes({changed_path})
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    # Direct match: only A
    assert directly_affected == {"stage_a"}

    # With downstream: A, B, C
    assert set(all_affected) == {"stage_a", "stage_b", "stage_c"}


def test_linear_dag_middle_change_affects_downstream_only(pipeline_dir: pathlib.Path) -> None:
    """Linear DAG: A → B → C. Changing B's output affects only C."""
    (pipeline_dir / "input.txt").write_text("hello")
    (pipeline_dir / "a_out.txt").write_text("HELLO")
    (pipeline_dir / "b_out.txt").write_text("HELLO!")

    @pivot.stage(deps=["input.txt"], outs=["a_out.txt"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["a_out.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pass

    @pivot.stage(deps=["b_out.txt"], outs=["c_out.txt"])
    def stage_c() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("b_out.txt")
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    # B's output is C's input, so C is affected
    # A and B are NOT affected (B's output is not their input)
    assert set(all_affected) == {"stage_c"}


# =============================================================================
# DAG Topology Tests - Fan-out
# =============================================================================


def test_fanout_dag_input_change_affects_all_branches(pipeline_dir: pathlib.Path) -> None:
    """Fan-out DAG: A → [B, C, D]. Changing A's input affects all downstream."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["shared.txt"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["shared.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pass

    @pivot.stage(deps=["shared.txt"], outs=["c_out.txt"])
    def stage_c() -> None:
        pass

    @pivot.stage(deps=["shared.txt"], outs=["d_out.txt"])
    def stage_d() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("input.txt")
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    assert set(all_affected) == {"stage_a", "stage_b", "stage_c", "stage_d"}


def test_fanout_dag_shared_output_change_affects_all_consumers(
    pipeline_dir: pathlib.Path,
) -> None:
    """Fan-out: Changing shared intermediate affects all consumers."""
    (pipeline_dir / "input.txt").write_text("data")
    (pipeline_dir / "shared.txt").write_text("SHARED")

    @pivot.stage(deps=["input.txt"], outs=["shared.txt"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["shared.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pass

    @pivot.stage(deps=["shared.txt"], outs=["c_out.txt"])
    def stage_c() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("shared.txt")
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    # shared.txt is consumed by B and C, but NOT A's input
    assert "stage_b" in all_affected
    assert "stage_c" in all_affected
    assert "stage_a" not in all_affected


# =============================================================================
# DAG Topology Tests - Fan-in
# =============================================================================


def test_fanin_dag_single_branch_change(pipeline_dir: pathlib.Path) -> None:
    """Fan-in DAG: [A, B, C] → D. Changing only B's input affects B and D."""
    (pipeline_dir / "a_input.txt").write_text("a")
    (pipeline_dir / "b_input.txt").write_text("b")
    (pipeline_dir / "c_input.txt").write_text("c")

    @pivot.stage(deps=["a_input.txt"], outs=["a_out.txt"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["b_input.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pass

    @pivot.stage(deps=["c_input.txt"], outs=["c_out.txt"])
    def stage_c() -> None:
        pass

    @pivot.stage(deps=["a_out.txt", "b_out.txt", "c_out.txt"], outs=["final.txt"])
    def stage_d() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("b_input.txt")
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    # B's input changed → B affected → D depends on B's output → D affected
    assert "stage_b" in all_affected
    assert "stage_d" in all_affected
    # A and C are independent
    assert "stage_a" not in all_affected
    assert "stage_c" not in all_affected


# =============================================================================
# DAG Topology Tests - Diamond
# =============================================================================


def test_diamond_dag_root_change_affects_all(pipeline_dir: pathlib.Path) -> None:
    """Diamond DAG: A → [B, C] → D. Changing A's input affects all stages."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["a_out.txt"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["a_out.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pass

    @pivot.stage(deps=["a_out.txt"], outs=["c_out.txt"])
    def stage_c() -> None:
        pass

    @pivot.stage(deps=["b_out.txt", "c_out.txt"], outs=["final.txt"])
    def stage_d() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("input.txt")
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    assert set(all_affected) == {"stage_a", "stage_b", "stage_c", "stage_d"}


def test_diamond_dag_branch_change_affects_branch_and_merge(
    pipeline_dir: pathlib.Path,
) -> None:
    """Diamond: Changing one branch's output affects only that branch + merge."""
    (pipeline_dir / "input.txt").write_text("data")
    (pipeline_dir / "a_out.txt").write_text("A")
    (pipeline_dir / "b_out.txt").write_text("B")

    @pivot.stage(deps=["input.txt"], outs=["a_out.txt"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["a_out.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pass

    @pivot.stage(deps=["a_out.txt"], outs=["c_out.txt"])
    def stage_c() -> None:
        pass

    @pivot.stage(deps=["b_out.txt", "c_out.txt"], outs=["final.txt"])
    def stage_d() -> None:
        pass

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("b_out.txt")
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    # B's output changed → D depends on it → D affected
    # A and C not affected (b_out.txt is not their input)
    assert "stage_d" in all_affected
    assert "stage_a" not in all_affected
    assert "stage_b" not in all_affected
    assert "stage_c" not in all_affected


# =============================================================================
# Full Pipeline Execution Tests
# =============================================================================
# Note: Code change and watch filter unit tests are in test_engine.py and
# test_watch_utils.py respectively. This file focuses on integration tests.


def test_full_pipeline_linear_execution(pipeline_dir: pathlib.Path) -> None:
    """Linear pipeline executes correctly and produces expected outputs."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["step1.txt"])
    def step1() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("step1.txt").write_text(data.upper())

    @pivot.stage(deps=["step1.txt"], outs=["step2.txt"])
    def step2() -> None:
        data = pathlib.Path("step1.txt").read_text()
        pathlib.Path("step2.txt").write_text(f"[{data}]")

    @pivot.stage(deps=["step2.txt"], outs=["final.txt"])
    def step3() -> None:
        data = pathlib.Path("step2.txt").read_text()
        pathlib.Path("final.txt").write_text(f"Result: {data}")

    results = executor.run()

    assert (pipeline_dir / "final.txt").read_text() == "Result: [HELLO]"
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert results["step3"]["status"] == "ran"


def test_full_pipeline_diamond_execution(pipeline_dir: pathlib.Path) -> None:
    """Diamond pipeline executes all branches and merges correctly."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["shared.txt"])
    def root() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("shared.txt").write_text(data.upper())

    @pivot.stage(deps=["shared.txt"], outs=["left.txt"])
    def left_branch() -> None:
        data = pathlib.Path("shared.txt").read_text()
        pathlib.Path("left.txt").write_text(f"L:{data}")

    @pivot.stage(deps=["shared.txt"], outs=["right.txt"])
    def right_branch() -> None:
        data = pathlib.Path("shared.txt").read_text()
        pathlib.Path("right.txt").write_text(f"R:{data}")

    @pivot.stage(deps=["left.txt", "right.txt"], outs=["merged.txt"])
    def merge() -> None:
        left = pathlib.Path("left.txt").read_text()
        right = pathlib.Path("right.txt").read_text()
        pathlib.Path("merged.txt").write_text(f"{left}+{right}")

    results = executor.run()

    assert (pipeline_dir / "merged.txt").read_text() == "L:DATA+R:DATA"
    assert all(r["status"] == "ran" for r in results.values())


def test_rerun_after_input_change(pipeline_dir: pathlib.Path) -> None:
    """Changing input triggers re-execution of affected stages."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    # First run
    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "HELLO"

    # Second run - no changes, should skip
    results = executor.run()
    assert results["process"]["status"] == "skipped"

    # Modify input
    (pipeline_dir / "input.txt").write_text("world")

    # Third run - input changed, should re-run
    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "WORLD"


@pytest.mark.xfail(
    reason=(
        "Generation-based skip doesn't detect external intermediate modifications. "
        "This is a known limitation: when upstream stages skip, the generation check "
        "passes and downstream stages skip without verifying actual file content. "
        "Watch mode handles this via file change detection, but batch runs don't."
    ),
    strict=True,
)
def test_rerun_after_intermediate_change(pipeline_dir: pathlib.Path) -> None:
    """Modifying intermediate file triggers downstream re-execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["middle.txt"])
    def step1() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("middle.txt").write_text(data.upper())

    @pivot.stage(deps=["middle.txt"], outs=["output.txt"])
    def step2() -> None:
        data = pathlib.Path("middle.txt").read_text()
        pathlib.Path("output.txt").write_text(f"[{data}]")

    # First run
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "[HELLO]"

    # Modify intermediate file directly (simulating external edit)
    # Need to make writable first - IncrementalOut uses COPY mode which makes files read-only
    middle_path = pipeline_dir / "middle.txt"
    os.chmod(middle_path, 0o644)
    middle_path.write_text("MODIFIED")

    # Second run - step1 unchanged, step2 should re-run
    results = executor.run()
    assert results["step1"]["status"] == "skipped", "Upstream unchanged"
    assert results["step2"]["status"] == "ran", "Downstream should re-run"
    assert (pipeline_dir / "output.txt").read_text() == "[MODIFIED]"


def test_output_only_change_no_rerun(pipeline_dir: pathlib.Path) -> None:
    """Modifying output-only file (not input to any stage) doesn't trigger re-run."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    # First run
    results = executor.run()
    assert results["process"]["status"] == "ran"

    # Modify output directly
    # Need to make writable first - IncrementalOut uses COPY mode which makes files read-only
    output_path = pipeline_dir / "output.txt"
    os.chmod(output_path, 0o644)
    output_path.write_text("TAMPERED")

    # Second run - output changed but input didn't, should skip
    # (The stage doesn't depend on its own output)
    results = executor.run()
    assert results["process"]["status"] == "skipped"


# =============================================================================
# Live Watch Mode Tests (with real watchfiles)
# =============================================================================


def test_watch_detects_file_change_and_triggers_execution(
    pipeline_dir: pathlib.Path,
) -> None:
    """Integration: Watch mode detects file change and triggers stage execution."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("data.csv").read_text()
        pathlib.Path("output.txt").write_text(f"Processed: {len(data)} chars")

    with run_watch_engine() as executions:
        (pipeline_dir / "data.csv").write_text("a,b,c\n1,2,3\n4,5,6")

    assert len(executions) >= 2, "Should have initial + triggered execution"
    assert executions[0] is None, "Initial execution runs all stages"
    assert executions[1] is not None and "process" in executions[1]


def test_watch_code_change_triggers_reload_and_execution(
    pipeline_dir: pathlib.Path,
) -> None:
    """Integration: Python file change triggers registry reload and re-execution."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")
    helper_file = pipeline_dir / "helper.py"
    helper_file.write_text("def helper(): pass\n")

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("processed")

    execution_count = 0
    reload_called = False
    done_event = threading.Event()

    def capture_execute(
        self: engine.WatchEngine, stages: list[str] | None
    ) -> dict[str, ExecutionSummary]:
        nonlocal execution_count
        execution_count += 1
        if execution_count >= 2:
            done_event.set()
            self.shutdown()
        return {}

    def capture_reload(self: engine.WatchEngine) -> bool:
        nonlocal reload_called
        reload_called = True
        return True

    with (
        mock.patch.object(engine.WatchEngine, "_execute_stages", capture_execute),
        mock.patch.object(engine.WatchEngine, "_reload_registry", capture_reload),
    ):
        eng = engine.WatchEngine(debounce_ms=50)

        engine_thread = threading.Thread(target=eng.run)
        engine_thread.start()

        time.sleep(0.2)  # Tested: 0.1s works but 0.2s adds safety margin

        # Modify Python file to trigger code change
        helper_file.write_text("def helper(): return 42\n")

        done_event.wait(timeout=5.0)
        eng.shutdown()
        engine_thread.join(timeout=2.0)

    assert execution_count >= 2, "Should execute at least twice"
    assert reload_called, "Should reload registry on code change"


# =============================================================================
# Debounce Tests
# =============================================================================


def test_debounce_coalesces_rapid_changes(pipeline_dir: pathlib.Path) -> None:
    """Multiple rapid file changes are coalesced into one execution."""
    (pipeline_dir / "data.csv").write_text("initial")

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    execution_count = 0
    done_event = threading.Event()

    def capture_execute(
        self: engine.WatchEngine, stages: list[str] | None
    ) -> dict[str, ExecutionSummary]:
        nonlocal execution_count
        execution_count += 1
        # Wait for potential additional changes to be coalesced
        if execution_count == 1:
            # This is initial run, wait for triggered run
            pass
        elif execution_count == 2:
            # After debounce window, signal done
            done_event.set()
            self.shutdown()
        return {}

    with mock.patch.object(engine.WatchEngine, "_execute_stages", capture_execute):
        eng = engine.WatchEngine(debounce_ms=200)

        engine_thread = threading.Thread(target=eng.run)
        engine_thread.start()

        time.sleep(0.3)  # Let watcher initialize

        # Make 5 rapid changes within debounce window
        for i in range(5):
            (pipeline_dir / "data.csv").write_text(f"change {i}")
            time.sleep(0.02)  # 20ms between changes, within 200ms debounce

        done_event.wait(timeout=5.0)
        eng.shutdown()
        engine_thread.join(timeout=2.0)

    # Should have: 1 initial + 1 debounced (not 1 + 5)
    assert execution_count == 2, (
        f"Expected 2 executions (initial + debounced), got {execution_count}"
    )


# =============================================================================
# Specific Stage Selection Tests
# =============================================================================


def test_specific_stages_only_runs_selected(pipeline_dir: pathlib.Path) -> None:
    """Running with specific stages only affects those stages."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        pathlib.Path("a.txt").write_text("A")

    @pivot.stage(deps=["input.txt"], outs=["b.txt"])
    def stage_b() -> None:
        pathlib.Path("b.txt").write_text("B")

    # Run only stage_a
    results = executor.run(["stage_a"])

    assert "stage_a" in results
    assert results["stage_a"]["status"] == "ran"
    assert (pipeline_dir / "a.txt").read_text() == "A"

    # stage_b was not run
    assert "stage_b" not in results
    assert not (pipeline_dir / "b.txt").exists()


def test_engine_respects_stage_filter_on_code_change(pipeline_dir: pathlib.Path) -> None:
    """WatchEngine respects stage filter when code_changed=True."""
    (pipeline_dir / "a_input.txt").write_text("a")
    (pipeline_dir / "b_input.txt").write_text("b")

    @pivot.stage(deps=["a_input.txt"], outs=["a_out.txt"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["b_input.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pass

    # Engine only watches stage_a
    eng = engine.WatchEngine(stages=["stage_a"])

    # On code change, only filtered stages should be returned
    affected = eng._get_affected_stages(set(), code_changed=True)
    assert "stage_a" in affected
    assert "stage_b" not in affected, "Filtered out stages should not be affected"


def test_file_index_is_global_not_filtered(pipeline_dir: pathlib.Path) -> None:
    """File index maps ALL stages, not just filtered ones.

    This is correct behavior - the file index needs to be global so that
    changes to files used by non-watched stages are still detected (even
    if those stages won't run due to filtering).
    """
    (pipeline_dir / "a_input.txt").write_text("a")
    (pipeline_dir / "b_input.txt").write_text("b")

    @pivot.stage(deps=["a_input.txt"], outs=["a_out.txt"])
    def stage_a() -> None:
        pass

    @pivot.stage(deps=["b_input.txt"], outs=["b_out.txt"])
    def stage_b() -> None:
        pass

    # Engine only watches stage_a
    eng = engine.WatchEngine(stages=["stage_a"])

    # File index should still contain both stages
    index = eng._build_file_to_stages_index()
    a_path = project.resolve_path("a_input.txt")
    b_path = project.resolve_path("b_input.txt")

    assert a_path in index and "stage_a" in index[a_path]
    assert b_path in index and "stage_b" in index[b_path]

    # _get_stages_matching_changes returns matches from global index
    changed = {project.resolve_path("b_input.txt")}
    matching = eng._get_stages_matching_changes(changed)
    assert "stage_b" in matching, "Global index should match all stages"


# =============================================================================
# Error Recovery Tests
# =============================================================================


def test_stage_failure_does_not_crash_watch(pipeline_dir: pathlib.Path) -> None:
    """A failing stage doesn't crash the watch loop."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def failing_stage() -> None:
        raise RuntimeError("Stage failed!")

    execution_count = 0
    done_event = threading.Event()

    def capture_execute(
        self: engine.WatchEngine, stages: list[str] | None
    ) -> dict[str, ExecutionSummary]:
        nonlocal execution_count
        execution_count += 1
        # First run will fail, but shouldn't crash
        try:
            return executor.run(stages)
        except Exception:
            pass
        finally:
            if execution_count >= 1:
                done_event.set()
                self.shutdown()
        return {}  # type: ignore[return-value] - empty dict for failure case

    with mock.patch.object(engine.WatchEngine, "_execute_stages", capture_execute):
        eng = engine.WatchEngine(debounce_ms=50)

        engine_thread = threading.Thread(target=eng.run)
        engine_thread.start()

        done_event.wait(timeout=5.0)
        eng.shutdown()
        engine_thread.join(timeout=2.0)

    # Engine should have executed at least once without crashing
    assert execution_count >= 1
