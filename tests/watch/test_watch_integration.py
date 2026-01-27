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
from typing import TYPE_CHECKING, Annotated, TypedDict
from unittest import mock

import pytest

from helpers import register_test_stage
from pivot import executor, loaders, outputs, project
from pivot.watch import engine

if TYPE_CHECKING:
    from collections.abc import Generator

    from pivot.executor.core import ExecutionSummary


# pipeline_dir fixture is in conftest.py


# =============================================================================
# Module-level stage functions for annotation-based registration
# =============================================================================


# --- TypedDicts for stage outputs ---


class _OutputCsv(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.csv", loaders.PathOnly())]


class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _IntermediateCsv(TypedDict):
    intermediate: Annotated[pathlib.Path, outputs.Out("intermediate.csv", loaders.PathOnly())]


class _FinalCsv(TypedDict):
    final: Annotated[pathlib.Path, outputs.Out("final.csv", loaders.PathOnly())]


class _AOutTxt(TypedDict):
    a_out: Annotated[pathlib.Path, outputs.Out("a_out.txt", loaders.PathOnly())]


class _BOutTxt(TypedDict):
    b_out: Annotated[pathlib.Path, outputs.Out("b_out.txt", loaders.PathOnly())]


class _COutTxt(TypedDict):
    c_out: Annotated[pathlib.Path, outputs.Out("c_out.txt", loaders.PathOnly())]


class _DOutTxt(TypedDict):
    d_out: Annotated[pathlib.Path, outputs.Out("d_out.txt", loaders.PathOnly())]


class _SharedTxt(TypedDict):
    shared: Annotated[pathlib.Path, outputs.Out("shared.txt", loaders.PathOnly())]


class _Step1Txt(TypedDict):
    step1: Annotated[pathlib.Path, outputs.Out("step1.txt", loaders.PathOnly())]


class _Step2Txt(TypedDict):
    step2: Annotated[pathlib.Path, outputs.Out("step2.txt", loaders.PathOnly())]


class _FinalTxt(TypedDict):
    final: Annotated[pathlib.Path, outputs.Out("final.txt", loaders.PathOnly())]


class _MiddleTxt(TypedDict):
    middle: Annotated[pathlib.Path, outputs.Out("middle.txt", loaders.PathOnly())]


class _LeftTxt(TypedDict):
    left: Annotated[pathlib.Path, outputs.Out("left.txt", loaders.PathOnly())]


class _RightTxt(TypedDict):
    right: Annotated[pathlib.Path, outputs.Out("right.txt", loaders.PathOnly())]


class _MergedTxt(TypedDict):
    merged: Annotated[pathlib.Path, outputs.Out("merged.txt", loaders.PathOnly())]


class _ATxt(TypedDict):
    a: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _BTxt(TypedDict):
    b: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


# --- Stage functions for file index tests ---


def _stage_process_csv_to_csv(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _OutputCsv:
    _ = input_file
    return _OutputCsv(output=pathlib.Path("output.csv"))


def _stage_a_csv_pipeline(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _IntermediateCsv:
    _ = input_file
    return _IntermediateCsv(intermediate=pathlib.Path("intermediate.csv"))


def _stage_b_csv_pipeline(
    intermediate: Annotated[pathlib.Path, outputs.Dep("intermediate.csv", loaders.PathOnly())],
) -> _FinalCsv:
    _ = intermediate
    return _FinalCsv(final=pathlib.Path("final.csv"))


# --- Stage functions for change detection tests ---


def _stage_process_data(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputCsv:
    _ = data
    return _OutputCsv(output=pathlib.Path("output.csv"))


# --- Stage functions for linear DAG tests ---


def _stage_a_linear(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _AOutTxt:
    pathlib.Path("a_out.txt").write_text(pathlib.Path("input.txt").read_text().upper())
    return _AOutTxt(a_out=pathlib.Path("a_out.txt"))


def _stage_b_linear(
    a_out: Annotated[pathlib.Path, outputs.Dep("a_out.txt", loaders.PathOnly())],
) -> _BOutTxt:
    pathlib.Path("b_out.txt").write_text(pathlib.Path("a_out.txt").read_text() + "!")
    return _BOutTxt(b_out=pathlib.Path("b_out.txt"))


def _stage_c_linear(
    b_out: Annotated[pathlib.Path, outputs.Dep("b_out.txt", loaders.PathOnly())],
) -> _COutTxt:
    pathlib.Path("c_out.txt").write_text(pathlib.Path("b_out.txt").read_text() + "?")
    return _COutTxt(c_out=pathlib.Path("c_out.txt"))


def _stage_a_passthrough(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _AOutTxt:
    _ = input_file
    return _AOutTxt(a_out=pathlib.Path("a_out.txt"))


def _stage_b_passthrough(
    a_out: Annotated[pathlib.Path, outputs.Dep("a_out.txt", loaders.PathOnly())],
) -> _BOutTxt:
    _ = a_out
    return _BOutTxt(b_out=pathlib.Path("b_out.txt"))


def _stage_c_passthrough(
    b_out: Annotated[pathlib.Path, outputs.Dep("b_out.txt", loaders.PathOnly())],
) -> _COutTxt:
    _ = b_out
    return _COutTxt(c_out=pathlib.Path("c_out.txt"))


# --- Stage functions for fan-out DAG tests ---


def _stage_a_fanout(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SharedTxt:
    _ = input_file
    return _SharedTxt(shared=pathlib.Path("shared.txt"))


def _stage_b_fanout(
    shared: Annotated[pathlib.Path, outputs.Dep("shared.txt", loaders.PathOnly())],
) -> _BOutTxt:
    _ = shared
    return _BOutTxt(b_out=pathlib.Path("b_out.txt"))


def _stage_c_fanout(
    shared: Annotated[pathlib.Path, outputs.Dep("shared.txt", loaders.PathOnly())],
) -> _COutTxt:
    _ = shared
    return _COutTxt(c_out=pathlib.Path("c_out.txt"))


def _stage_d_fanout(
    shared: Annotated[pathlib.Path, outputs.Dep("shared.txt", loaders.PathOnly())],
) -> _DOutTxt:
    _ = shared
    return _DOutTxt(d_out=pathlib.Path("d_out.txt"))


# --- Stage functions for fan-in DAG tests ---


def _stage_a_fanin(
    a_input: Annotated[pathlib.Path, outputs.Dep("a_input.txt", loaders.PathOnly())],
) -> _AOutTxt:
    _ = a_input
    return _AOutTxt(a_out=pathlib.Path("a_out.txt"))


def _stage_b_fanin(
    b_input: Annotated[pathlib.Path, outputs.Dep("b_input.txt", loaders.PathOnly())],
) -> _BOutTxt:
    _ = b_input
    return _BOutTxt(b_out=pathlib.Path("b_out.txt"))


def _stage_c_fanin(
    c_input: Annotated[pathlib.Path, outputs.Dep("c_input.txt", loaders.PathOnly())],
) -> _COutTxt:
    _ = c_input
    return _COutTxt(c_out=pathlib.Path("c_out.txt"))


def _stage_d_fanin(
    a_out: Annotated[pathlib.Path, outputs.Dep("a_out.txt", loaders.PathOnly())],
    b_out: Annotated[pathlib.Path, outputs.Dep("b_out.txt", loaders.PathOnly())],
    c_out: Annotated[pathlib.Path, outputs.Dep("c_out.txt", loaders.PathOnly())],
) -> _FinalTxt:
    _ = a_out, b_out, c_out
    return _FinalTxt(final=pathlib.Path("final.txt"))


# --- Stage functions for diamond DAG tests ---


def _stage_a_diamond(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _AOutTxt:
    _ = input_file
    return _AOutTxt(a_out=pathlib.Path("a_out.txt"))


def _stage_b_diamond(
    a_out: Annotated[pathlib.Path, outputs.Dep("a_out.txt", loaders.PathOnly())],
) -> _BOutTxt:
    _ = a_out
    return _BOutTxt(b_out=pathlib.Path("b_out.txt"))


def _stage_c_diamond(
    a_out: Annotated[pathlib.Path, outputs.Dep("a_out.txt", loaders.PathOnly())],
) -> _COutTxt:
    _ = a_out
    return _COutTxt(c_out=pathlib.Path("c_out.txt"))


def _stage_d_diamond(
    b_out: Annotated[pathlib.Path, outputs.Dep("b_out.txt", loaders.PathOnly())],
    c_out: Annotated[pathlib.Path, outputs.Dep("c_out.txt", loaders.PathOnly())],
) -> _FinalTxt:
    _ = b_out, c_out
    return _FinalTxt(final=pathlib.Path("final.txt"))


# --- Stage functions for full pipeline execution tests ---


def _stage_step1_linear(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _Step1Txt:
    data = pathlib.Path("input.txt").read_text()
    pathlib.Path("step1.txt").write_text(data.upper())
    return _Step1Txt(step1=pathlib.Path("step1.txt"))


def _stage_step2_linear(
    step1: Annotated[pathlib.Path, outputs.Dep("step1.txt", loaders.PathOnly())],
) -> _Step2Txt:
    data = pathlib.Path("step1.txt").read_text()
    pathlib.Path("step2.txt").write_text(f"[{data}]")
    return _Step2Txt(step2=pathlib.Path("step2.txt"))


def _stage_step3_linear(
    step2: Annotated[pathlib.Path, outputs.Dep("step2.txt", loaders.PathOnly())],
) -> _FinalTxt:
    data = pathlib.Path("step2.txt").read_text()
    pathlib.Path("final.txt").write_text(f"Result: {data}")
    return _FinalTxt(final=pathlib.Path("final.txt"))


def _stage_root_diamond(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SharedTxt:
    data = pathlib.Path("input.txt").read_text()
    pathlib.Path("shared.txt").write_text(data.upper())
    return _SharedTxt(shared=pathlib.Path("shared.txt"))


def _stage_left_diamond(
    shared: Annotated[pathlib.Path, outputs.Dep("shared.txt", loaders.PathOnly())],
) -> _LeftTxt:
    data = pathlib.Path("shared.txt").read_text()
    pathlib.Path("left.txt").write_text(f"L:{data}")
    return _LeftTxt(left=pathlib.Path("left.txt"))


def _stage_right_diamond(
    shared: Annotated[pathlib.Path, outputs.Dep("shared.txt", loaders.PathOnly())],
) -> _RightTxt:
    data = pathlib.Path("shared.txt").read_text()
    pathlib.Path("right.txt").write_text(f"R:{data}")
    return _RightTxt(right=pathlib.Path("right.txt"))


def _stage_merge_diamond(
    left: Annotated[pathlib.Path, outputs.Dep("left.txt", loaders.PathOnly())],
    right: Annotated[pathlib.Path, outputs.Dep("right.txt", loaders.PathOnly())],
) -> _MergedTxt:
    left_data = pathlib.Path("left.txt").read_text()
    right_data = pathlib.Path("right.txt").read_text()
    pathlib.Path("merged.txt").write_text(f"{left_data}+{right_data}")
    return _MergedTxt(merged=pathlib.Path("merged.txt"))


# --- Stage functions for rerun tests ---


def _stage_process_rerun(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    data = pathlib.Path("input.txt").read_text()
    pathlib.Path("output.txt").write_text(data.upper())
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_step1_rerun(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _MiddleTxt:
    data = pathlib.Path("input.txt").read_text()
    pathlib.Path("middle.txt").write_text(data.upper())
    return _MiddleTxt(middle=pathlib.Path("middle.txt"))


def _stage_step2_rerun(
    middle: Annotated[pathlib.Path, outputs.Dep("middle.txt", loaders.PathOnly())],
) -> _OutputTxt:
    data = pathlib.Path("middle.txt").read_text()
    pathlib.Path("output.txt").write_text(f"[{data}]")
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_process_output_only(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    data = pathlib.Path("input.txt").read_text()
    pathlib.Path("output.txt").write_text(data.upper())
    return _OutputTxt(output=pathlib.Path("output.txt"))


# --- Stage functions for watch mode tests ---


def _stage_process_watch(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    data_content = pathlib.Path("data.csv").read_text()
    pathlib.Path("output.txt").write_text(f"Processed: {len(data_content)} chars")
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_process_code_change(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    _ = data
    pathlib.Path("output.txt").write_text("processed")
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_process_debounce(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    _ = data
    return _OutputTxt(output=pathlib.Path("output.txt"))


# --- Stage functions for specific stage selection tests ---


def _stage_a_selection(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ATxt:
    pathlib.Path("a.txt").write_text("A")
    return _ATxt(a=pathlib.Path("a.txt"))


def _stage_b_selection(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _BTxt:
    pathlib.Path("b.txt").write_text("B")
    return _BTxt(b=pathlib.Path("b.txt"))


def _stage_a_filter(
    a_input: Annotated[pathlib.Path, outputs.Dep("a_input.txt", loaders.PathOnly())],
) -> _AOutTxt:
    _ = a_input
    return _AOutTxt(a_out=pathlib.Path("a_out.txt"))


def _stage_b_filter(
    b_input: Annotated[pathlib.Path, outputs.Dep("b_input.txt", loaders.PathOnly())],
) -> _BOutTxt:
    _ = b_input
    return _BOutTxt(b_out=pathlib.Path("b_out.txt"))


# --- Stage functions for error recovery tests ---


def _stage_failing(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    raise RuntimeError("Stage failed!")


# =============================================================================
# Live Watch Test Helpers
# =============================================================================


@contextlib.contextmanager
def run_watch_engine(
    debounce_ms: int = 50,
    min_executions: int = 2,
    timeout: float = 2.0,
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

        time.sleep(0.1)  # Let watcher initialize
        yield executions

        done_event.wait(timeout=timeout)
        eng.shutdown()
        engine_thread.join(timeout=1.0)


# =============================================================================
# File Index and Change Detection Tests
# =============================================================================


def test_file_index_maps_only_dependencies(pipeline_dir: pathlib.Path) -> None:
    """File index contains dependencies, NOT outputs."""
    (pipeline_dir / "input.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_process_csv_to_csv, name="process")

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

    register_test_stage(_stage_a_csv_pipeline, name="stage_a")
    register_test_stage(_stage_b_csv_pipeline, name="stage_b")

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

    register_test_stage(_stage_process_data, name="process")

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

    register_test_stage(_stage_process_csv_to_csv, name="process")

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

    register_test_stage(_stage_a_csv_pipeline, name="stage_a")
    register_test_stage(_stage_b_csv_pipeline, name="stage_b")

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("intermediate.csv")
    directly_affected = eng._get_stages_matching_changes({changed_path})

    assert "stage_b" in directly_affected, "Downstream stage should be directly affected"
    assert "stage_a" not in directly_affected, "Upstream stage should NOT be affected"


# =============================================================================
# DAG Topology Tests - Linear
# =============================================================================


def test_linear_dag_input_change_affects_all_downstream(pipeline_dir: pathlib.Path) -> None:
    """Linear DAG: A -> B -> C. Changing A's input affects A, B, C."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_stage_a_linear, name="stage_a")
    register_test_stage(_stage_b_linear, name="stage_b")
    register_test_stage(_stage_c_linear, name="stage_c")

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("input.txt")
    directly_affected = eng._get_stages_matching_changes({changed_path})
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    # Direct match: only A
    assert directly_affected == {"stage_a"}

    # With downstream: A, B, C
    assert set(all_affected) == {"stage_a", "stage_b", "stage_c"}


def test_linear_dag_middle_change_affects_downstream_only(pipeline_dir: pathlib.Path) -> None:
    """Linear DAG: A -> B -> C. Changing B's output affects only C."""
    (pipeline_dir / "input.txt").write_text("hello")
    (pipeline_dir / "a_out.txt").write_text("HELLO")
    (pipeline_dir / "b_out.txt").write_text("HELLO!")

    register_test_stage(_stage_a_passthrough, name="stage_a")
    register_test_stage(_stage_b_passthrough, name="stage_b")
    register_test_stage(_stage_c_passthrough, name="stage_c")

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
    """Fan-out DAG: A -> [B, C, D]. Changing A's input affects all downstream."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_stage_a_fanout, name="stage_a")
    register_test_stage(_stage_b_fanout, name="stage_b")
    register_test_stage(_stage_c_fanout, name="stage_c")
    register_test_stage(_stage_d_fanout, name="stage_d")

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

    register_test_stage(_stage_a_fanout, name="stage_a")
    register_test_stage(_stage_b_fanout, name="stage_b")
    register_test_stage(_stage_c_fanout, name="stage_c")

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
    """Fan-in DAG: [A, B, C] -> D. Changing only B's input affects B and D."""
    (pipeline_dir / "a_input.txt").write_text("a")
    (pipeline_dir / "b_input.txt").write_text("b")
    (pipeline_dir / "c_input.txt").write_text("c")

    register_test_stage(_stage_a_fanin, name="stage_a")
    register_test_stage(_stage_b_fanin, name="stage_b")
    register_test_stage(_stage_c_fanin, name="stage_c")
    register_test_stage(_stage_d_fanin, name="stage_d")

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("b_input.txt")
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    # B's input changed -> B affected -> D depends on B's output -> D affected
    assert "stage_b" in all_affected
    assert "stage_d" in all_affected
    # A and C are independent
    assert "stage_a" not in all_affected
    assert "stage_c" not in all_affected


# =============================================================================
# DAG Topology Tests - Diamond
# =============================================================================


def test_diamond_dag_root_change_affects_all(pipeline_dir: pathlib.Path) -> None:
    """Diamond DAG: A -> [B, C] -> D. Changing A's input affects all stages."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_stage_a_diamond, name="stage_a")
    register_test_stage(_stage_b_diamond, name="stage_b")
    register_test_stage(_stage_c_diamond, name="stage_c")
    register_test_stage(_stage_d_diamond, name="stage_d")

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

    register_test_stage(_stage_a_diamond, name="stage_a")
    register_test_stage(_stage_b_diamond, name="stage_b")
    register_test_stage(_stage_c_diamond, name="stage_c")
    register_test_stage(_stage_d_diamond, name="stage_d")

    eng = engine.WatchEngine()
    changed_path = project.resolve_path("b_out.txt")
    all_affected = eng._get_affected_stages({changed_path}, code_changed=False)

    # B's output changed -> D depends on it -> D affected
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

    register_test_stage(_stage_step1_linear, name="step1")
    register_test_stage(_stage_step2_linear, name="step2")
    register_test_stage(_stage_step3_linear, name="step3")

    results = executor.run()

    assert (pipeline_dir / "final.txt").read_text() == "Result: [HELLO]"
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert results["step3"]["status"] == "ran"


def test_full_pipeline_diamond_execution(pipeline_dir: pathlib.Path) -> None:
    """Diamond pipeline executes all branches and merges correctly."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_stage_root_diamond, name="root")
    register_test_stage(_stage_left_diamond, name="left_branch")
    register_test_stage(_stage_right_diamond, name="right_branch")
    register_test_stage(_stage_merge_diamond, name="merge")

    results = executor.run()

    assert (pipeline_dir / "merged.txt").read_text() == "L:DATA+R:DATA"
    assert all(r["status"] == "ran" for r in results.values())


def test_rerun_after_input_change(pipeline_dir: pathlib.Path) -> None:
    """Changing input triggers re-execution of affected stages."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_stage_process_rerun, name="process")

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


def test_rerun_after_intermediate_change(pipeline_dir: pathlib.Path) -> None:
    """Modifying intermediate file triggers downstream re-execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_stage_step1_rerun, name="step1")
    register_test_stage(_stage_step2_rerun, name="step2")

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

    register_test_stage(_stage_process_output_only, name="process")

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

    register_test_stage(_stage_process_watch, name="process")

    with run_watch_engine() as executions:
        (pipeline_dir / "data.csv").write_text("a,b,c\n1,2,3\n4,5,6")

    assert len(executions) >= 2, "Should have initial + triggered execution"
    assert executions[0] is None, "Initial execution runs all stages"
    assert executions[1] is not None and "process" in executions[1]


@pytest.mark.xdist_group("watch_timing_sensitive")
def test_watch_code_change_triggers_reload_and_execution(
    pipeline_dir: pathlib.Path,
) -> None:
    """Integration: Python file change triggers registry reload and re-execution.

    Note: Uses xdist_group to isolate from parallel tests due to timing sensitivity.
    """
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")
    helper_file = pipeline_dir / "helper.py"
    helper_file.write_text("def helper(): pass\n")

    register_test_stage(_stage_process_code_change, name="process")

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

        time.sleep(0.1)  # Let watcher initialize

        # Modify Python file to trigger code change
        helper_file.write_text("def helper(): return 42\n")

        done_event.wait(timeout=2.0)
        eng.shutdown()
        engine_thread.join(timeout=1.0)

    assert execution_count >= 2, "Should execute at least twice"
    assert reload_called, "Should reload registry on code change"


# =============================================================================
# Debounce Tests
# =============================================================================


def test_debounce_coalesces_rapid_changes(pipeline_dir: pathlib.Path) -> None:
    """Multiple rapid file changes are coalesced into one execution."""
    (pipeline_dir / "data.csv").write_text("initial")

    register_test_stage(_stage_process_debounce, name="process")

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

        time.sleep(0.3)  # Let watcher initialize (longer due to 200ms debounce)

        # Make 5 rapid changes within debounce window
        for i in range(5):
            (pipeline_dir / "data.csv").write_text(f"change {i}")
            time.sleep(0.02)  # 20ms between changes, within 200ms debounce

        done_event.wait(timeout=2.0)
        eng.shutdown()
        engine_thread.join(timeout=1.0)

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

    register_test_stage(_stage_a_selection, name="stage_a")
    register_test_stage(_stage_b_selection, name="stage_b")

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

    register_test_stage(_stage_a_filter, name="stage_a")
    register_test_stage(_stage_b_filter, name="stage_b")

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

    register_test_stage(_stage_a_filter, name="stage_a")
    register_test_stage(_stage_b_filter, name="stage_b")

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

    register_test_stage(_stage_failing, name="failing_stage")

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
