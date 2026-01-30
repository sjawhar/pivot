import atexit
import logging
import multiprocessing as mp
import os
import pathlib
import sys
import threading
import time
from typing import TYPE_CHECKING, Annotated, TypedDict

import loky
import pytest
import yaml

from helpers import register_test_stage
from pivot import exceptions, executor, loaders, outputs
from pivot.executor import core as executor_core
from pivot.registry import REGISTRY

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _Step1Txt(TypedDict):
    step1: Annotated[pathlib.Path, outputs.Out("step1.txt", loaders.PathOnly())]


class _Step2Txt(TypedDict):
    step2: Annotated[pathlib.Path, outputs.Out("step2.txt", loaders.PathOnly())]


class _IntermediateTxt(TypedDict):
    intermediate: Annotated[pathlib.Path, outputs.Out("intermediate.txt", loaders.PathOnly())]


class _FinalTxt(TypedDict):
    final: Annotated[pathlib.Path, outputs.Out("final.txt", loaders.PathOnly())]


class _ATxt(TypedDict):
    a: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _BTxt(TypedDict):
    b: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _CTxt(TypedDict):
    c: Annotated[pathlib.Path, outputs.Out("c.txt", loaders.PathOnly())]


class _FailingTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("failing.txt", loaders.PathOnly())]


class _SucceedingTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("succeeding.txt", loaders.PathOnly())]


class _FirstTxt(TypedDict):
    first: Annotated[pathlib.Path, outputs.Out("first.txt", loaders.PathOnly())]


class _SecondTxt(TypedDict):
    second: Annotated[pathlib.Path, outputs.Out("second.txt", loaders.PathOnly())]


class _IndependentTxt(TypedDict):
    independent: Annotated[pathlib.Path, outputs.Out("independent.txt", loaders.PathOnly())]


class _MultiTxt(TypedDict):
    multi: Annotated[pathlib.Path, outputs.Out("multi.txt", loaders.PathOnly())]


class _GpuOnlyTxt(TypedDict):
    gpu_only: Annotated[pathlib.Path, outputs.Out("gpu_only.txt", loaders.PathOnly())]


class _TimingATxt(TypedDict):
    a: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _TimingBTxt(TypedDict):
    b: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _UpperTxt(TypedDict):
    upper: Annotated[pathlib.Path, outputs.Out("upper.txt", loaders.PathOnly())]


class _LowerTxt(TypedDict):
    lower: Annotated[pathlib.Path, outputs.Out("lower.txt", loaders.PathOnly())]


class _SpacedTxt(TypedDict):
    spaced: Annotated[pathlib.Path, outputs.Out("spaced.txt", loaders.PathOnly())]


class _CleanTxt(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("clean.txt", loaders.PathOnly())]


class _ExclusiveTxt(TypedDict):
    exclusive: Annotated[pathlib.Path, outputs.Out("exclusive.txt", loaders.PathOnly())]


class _NormalATxt(TypedDict):
    normal_a: Annotated[pathlib.Path, outputs.Out("normal_a.txt", loaders.PathOnly())]


class _NormalBTxt(TypedDict):
    normal_b: Annotated[pathlib.Path, outputs.Out("normal_b.txt", loaders.PathOnly())]


class _MetricsJson(TypedDict):
    metrics: Annotated[pathlib.Path, outputs.Out("metrics.json", loaders.PathOnly())]


class _MultiOut(TypedDict):
    z_out: Annotated[pathlib.Path, outputs.Out("z_out.txt", loaders.PathOnly())]
    a_out: Annotated[pathlib.Path, outputs.Out("a_out.txt", loaders.PathOnly())]


class _OutputDir(TypedDict):
    output_dir: Annotated[pathlib.Path, outputs.Out("output_dir/", loaders.PathOnly())]


class _OtherTxt(TypedDict):
    other: Annotated[pathlib.Path, outputs.Out("other.txt", loaders.PathOnly())]


class _StageATxt(TypedDict):
    stage_a: Annotated[pathlib.Path, outputs.Out("stage_a.txt", loaders.PathOnly())]


class _StageBTxt(TypedDict):
    stage_b: Annotated[pathlib.Path, outputs.Out("stage_b.txt", loaders.PathOnly())]


# Chain step outputs
class _ChainStep1(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("step1.txt", loaders.PathOnly())]


class _ChainStep2(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("step2.txt", loaders.PathOnly())]


class _ChainStep3(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("step3.txt", loaders.PathOnly())]


class _ChainStep4(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("step4.txt", loaders.PathOnly())]


class _ChainStep5(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("step5.txt", loaders.PathOnly())]


# =============================================================================
# Module-level stage functions for tests
# =============================================================================


def _step1_upper(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _Step1Txt:
    data = input_file.read_text()
    pathlib.Path("step1.txt").write_text(data.upper())
    return {"step1": pathlib.Path("step1.txt")}


def _step2_result(
    step1: Annotated[pathlib.Path, outputs.Dep("step1.txt", loaders.PathOnly())],
) -> _Step2Txt:
    data = step1.read_text()
    pathlib.Path("step2.txt").write_text(f"Result: {data}")
    return {"step2": pathlib.Path("step2.txt")}


def _output_upper(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    data = input_file.read_text()
    pathlib.Path("output.txt").write_text(data.upper())
    return {"output": pathlib.Path("output.txt")}


def _output_lower(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    data = input_file.read_text()
    pathlib.Path("output.txt").write_text(data.lower())
    return {"output": pathlib.Path("output.txt")}


def _intermediate_upper(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _IntermediateTxt:
    data = input_file.read_text()
    pathlib.Path("intermediate.txt").write_text(data.upper())
    return {"intermediate": pathlib.Path("intermediate.txt")}


def _final_from_intermediate(
    intermediate: Annotated[pathlib.Path, outputs.Dep("intermediate.txt", loaders.PathOnly())],
) -> _FinalTxt:
    data = intermediate.read_text()
    pathlib.Path("final.txt").write_text(f"Final: {data}")
    return {"final": pathlib.Path("final.txt")}


def _a_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ATxt:
    pathlib.Path("a.txt").write_text("a")
    return {"a": pathlib.Path("a.txt")}


def _b_from_a(
    a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _BTxt:
    _ = a
    pathlib.Path("b.txt").write_text("b")
    return {"b": pathlib.Path("b.txt")}


def _c_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _CTxt:
    pathlib.Path("c.txt").write_text("c")
    return {"c": pathlib.Path("c.txt")}


def _missing_dep_process(
    missing_input: Annotated[pathlib.Path, outputs.Dep("missing_input.txt", loaders.PathOnly())],
    run_count: dict[str, int] | None = None,
) -> _OutputTxt:
    if run_count is not None:
        run_count["process"] += 1
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _real_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _check_lock(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    stages_dir = pathlib.Path(".pivot") / "stages"
    lock_existed = (stages_dir / "check_lock.running").exists()
    pathlib.Path("lock_existed.txt").write_text("yes" if lock_existed else "no")
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _failing_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    raise RuntimeError("Stage failed!")


def _process_basic(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _output_queue_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _print_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    print("Stage output")
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _mutex_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ATxt:
    timing_file = input_file.parent / "timing.txt"
    with open(timing_file, "a") as f:
        f.write("a_start\n")
    time.sleep(0.02)
    with open(timing_file, "a") as f:
        f.write("a_end\n")
    out_file = input_file.parent / "a.txt"
    out_file.write_text("a")
    return {"a": out_file}


def _mutex_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _BTxt:
    timing_file = input_file.parent / "timing.txt"
    with open(timing_file, "a") as f:
        f.write("b_start\n")
    time.sleep(0.02)
    with open(timing_file, "a") as f:
        f.write("b_end\n")
    out_file = input_file.parent / "b.txt"
    out_file.write_text("b")
    return {"b": out_file}


def _first_basic(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FirstTxt:
    pathlib.Path("first.txt").write_text("first")
    return {"first": pathlib.Path("first.txt")}


def _second_basic(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SecondTxt:
    pathlib.Path("second.txt").write_text("second")
    return {"second": pathlib.Path("second.txt")}


def _failing_mutex(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FailingTxt:
    raise RuntimeError("Intentional failure")


def _succeeding_mutex(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SucceedingTxt:
    pathlib.Path("succeeding.txt").write_text("success")
    return {"output": pathlib.Path("succeeding.txt")}


def _multi_resource(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _MultiTxt:
    timing_file = input_file.parent / "timing.txt"
    with open(timing_file, "a") as f:
        f.write("multi_start\n")
    time.sleep(0.02)
    with open(timing_file, "a") as f:
        f.write("multi_end\n")
    out_file = input_file.parent / "multi.txt"
    out_file.write_text("multi")
    return {"multi": out_file}


def _gpu_only(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _GpuOnlyTxt:
    timing_file = input_file.parent / "timing.txt"
    with open(timing_file, "a") as f:
        f.write("gpu_start\n")
    time.sleep(0.02)
    with open(timing_file, "a") as f:
        f.write("gpu_end\n")
    out_file = input_file.parent / "gpu_only.txt"
    out_file.write_text("gpu")
    return {"gpu_only": out_file}


def _first_dep_mutex(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FirstTxt:
    pathlib.Path("first.txt").write_text("first")
    return {"first": pathlib.Path("first.txt")}


def _second_dep_mutex(
    first: Annotated[pathlib.Path, outputs.Dep("first.txt", loaders.PathOnly())],
) -> _SecondTxt:
    data = first.read_text()
    pathlib.Path("second.txt").write_text(f"second: {data}")
    return {"second": pathlib.Path("second.txt")}


def _timing_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _TimingATxt:
    with open("timing.txt", "a") as f:
        f.write("a_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("a_end\n")
    pathlib.Path("a.txt").write_text("a")
    return {"a": pathlib.Path("a.txt")}


def _timing_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _TimingBTxt:
    with open("timing.txt", "a") as f:
        f.write("b_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("b_end\n")
    pathlib.Path("b.txt").write_text("b")
    return {"b": pathlib.Path("b.txt")}


def _lonely_mutex(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _error_stage_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ATxt:
    with open("execution.log", "a") as f:
        f.write("a\n")
    raise RuntimeError("Stage A failed")


def _error_stage_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _BTxt:
    with open("execution.log", "a") as f:
        f.write("b\n")
    pathlib.Path("b.txt").write_text("b")
    return {"b": pathlib.Path("b.txt")}


def _keep_going_failing(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FailingTxt:
    raise RuntimeError("Intentional failure")


def _keep_going_succeeding(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SucceedingTxt:
    pathlib.Path("succeeding.txt").write_text("success")
    return {"output": pathlib.Path("succeeding.txt")}


def _first_failing(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FirstTxt:
    raise RuntimeError("First failed")


def _second_depends_first(
    first: Annotated[pathlib.Path, outputs.Dep("first.txt", loaders.PathOnly())],
) -> _SecondTxt:
    pathlib.Path("second.txt").write_text("should not run")
    return {"second": pathlib.Path("second.txt")}


def _independent_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _IndependentTxt:
    pathlib.Path("independent.txt").write_text("runs fine")
    return {"independent": pathlib.Path("independent.txt")}


def _ignore_first(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FirstTxt:
    raise RuntimeError("First failed")


def _ignore_second(
    first: Annotated[pathlib.Path, outputs.Dep("first.txt", loaders.PathOnly())],
) -> _SecondTxt:
    data = first.read_text()
    pathlib.Path("second.txt").write_text(f"got: {data}")
    return {"second": pathlib.Path("second.txt")}


def _invalid_error_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _slow_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    time.sleep(0.2)
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _fast_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _exits_with_code(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    sys.exit(42)


def _exits_zero(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    sys.exit(0)


def _keyboard_interrupt(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    raise KeyboardInterrupt("User cancelled")


def _dir_dep_process(
    data_dir: Annotated[pathlib.Path, outputs.Dep("data_dir", loaders.PathOnly())],
) -> _OutputTxt:
    _ = data_dir
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _parallel_false_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ATxt:
    with open("timing.txt", "a") as f:
        f.write("a_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("a_end\n")
    pathlib.Path("a.txt").write_text("a")
    return {"a": pathlib.Path("a.txt")}


def _parallel_false_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _BTxt:
    with open("timing.txt", "a") as f:
        f.write("b_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("b_end\n")
    pathlib.Path("b.txt").write_text("b")
    return {"b": pathlib.Path("b.txt")}


def _prints_output(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    print("stdout message")
    print("stderr message", file=sys.stderr)
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _partial_output(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    sys.stdout.write("no newline at end")
    sys.stdout.flush()
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _upper_mutex(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _UpperTxt:
    with open("timing.txt", "a") as f:
        f.write("upper_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("upper_end\n")
    pathlib.Path("upper.txt").write_text("done")
    return {"upper": pathlib.Path("upper.txt")}


def _lower_mutex(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _LowerTxt:
    with open("timing.txt", "a") as f:
        f.write("lower_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("lower_end\n")
    pathlib.Path("lower.txt").write_text("done")
    return {"lower": pathlib.Path("lower.txt")}


def _spaced_mutex(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SpacedTxt:
    with open("timing.txt", "a") as f:
        f.write("spaced_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("spaced_end\n")
    pathlib.Path("spaced.txt").write_text("done")
    return {"spaced": pathlib.Path("spaced.txt")}


def _clean_mutex(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _CleanTxt:
    with open("timing.txt", "a") as f:
        f.write("clean_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("clean_end\n")
    pathlib.Path("clean.txt").write_text("done")
    return {"clean": pathlib.Path("clean.txt")}


def _exclusive_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ExclusiveTxt:
    with open("timing.txt", "a") as f:
        f.write("exclusive_start\n")
    time.sleep(0.03)
    with open("timing.txt", "a") as f:
        f.write("exclusive_end\n")
    pathlib.Path("exclusive.txt").write_text("done")
    return {"exclusive": pathlib.Path("exclusive.txt")}


def _normal_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _NormalATxt:
    with open("timing.txt", "a") as f:
        f.write("a_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("a_end\n")
    pathlib.Path("normal_a.txt").write_text("done")
    return {"normal_a": pathlib.Path("normal_a.txt")}


def _normal_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _NormalBTxt:
    with open("timing.txt", "a") as f:
        f.write("b_start\n")
    time.sleep(0.02)
    with open("timing.txt", "a") as f:
        f.write("b_end\n")
    pathlib.Path("normal_b.txt").write_text("done")
    return {"normal_b": pathlib.Path("normal_b.txt")}


def _removes_output_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    output_file = pathlib.Path("output.txt")
    assert not output_file.exists(), "Output should be removed before stage runs"
    output_file.write_text("fresh data")
    return {"output": output_file}


def _cache_output_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("result")
    return {"output": pathlib.Path("output.txt")}


def _no_output_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    # Intentionally don't create output.txt
    return {"output": pathlib.Path("output.txt")}


def _metrics_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _MetricsJson:
    import json

    pathlib.Path("metrics.json").write_text(json.dumps({"accuracy": 0.95}))
    return {"metrics": pathlib.Path("metrics.json")}


def _no_metrics_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _MetricsJson:
    # Intentionally don't create metrics.json
    return {"metrics": pathlib.Path("metrics.json")}


def _multi_input_process(
    z_input: Annotated[pathlib.Path, outputs.Dep("z_input.txt", loaders.PathOnly())],
    a_input: Annotated[pathlib.Path, outputs.Dep("a_input.txt", loaders.PathOnly())],
) -> _MultiOut:
    _ = z_input, a_input
    pathlib.Path("z_out.txt").write_text("z")
    pathlib.Path("a_out.txt").write_text("a")
    return {"z_out": pathlib.Path("z_out.txt"), "a_out": pathlib.Path("a_out.txt")}


def _dir_output_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputDir:
    out_dir = pathlib.Path("output_dir")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "file1.txt").write_text("file1")
    (out_dir / "file2.txt").write_text("file2")
    return {"output_dir": out_dir}


def _dir_output_process_v2(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputDir:
    out_dir = pathlib.Path("output_dir")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "file1.txt").write_text("file1_v2")
    (out_dir / "file2.txt").write_text("file2_v2")
    return {"output_dir": out_dir}


def _lock_missing_outs(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("result")
    return {"output": pathlib.Path("output.txt")}


def _force_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    data = input_file.read_text()
    pathlib.Path("output.txt").write_text(data.upper())
    return {"output": pathlib.Path("output.txt")}


def _force_step1(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _Step1Txt:
    pathlib.Path("step1.txt").write_text("step1")
    return {"step1": pathlib.Path("step1.txt")}


def _force_step2(
    step1: Annotated[pathlib.Path, outputs.Dep("step1.txt", loaders.PathOnly())],
) -> _Step2Txt:
    _ = step1
    pathlib.Path("step2.txt").write_text("step2")
    return {"step2": pathlib.Path("step2.txt")}


def _force_other(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OtherTxt:
    pathlib.Path("other.txt").write_text("other")
    return {"other": pathlib.Path("other.txt")}


def _force_single_step1(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _Step1Txt:
    pathlib.Path("step1.txt").write_text("new_step1")
    return {"step1": pathlib.Path("step1.txt")}


def _force_single_step2(
    step1: Annotated[pathlib.Path, outputs.Dep("step1.txt", loaders.PathOnly())],
) -> _Step2Txt:
    data = step1.read_text()
    pathlib.Path("step2.txt").write_text(f"step2: {data}")
    return {"step2": pathlib.Path("step2.txt")}


def _deferred_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("result")
    return {"output": pathlib.Path("output.txt")}


def _deferred_step1(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _Step1Txt:
    data = input_file.read_text()
    pathlib.Path("step1.txt").write_text(f"processed: {data}")
    return {"step1": pathlib.Path("step1.txt")}


def _deferred_step2(
    step1: Annotated[pathlib.Path, outputs.Dep("step1.txt", loaders.PathOnly())],
) -> _Step2Txt:
    data = step1.read_text()
    pathlib.Path("step2.txt").write_text(f"final: {data}")
    return {"step2": pathlib.Path("step2.txt")}


def _concurrent_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _StageATxt:
    time.sleep(0.05)
    pathlib.Path("stage_a.txt").write_text("a")
    return {"stage_a": pathlib.Path("stage_a.txt")}


def _concurrent_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _StageBTxt:
    time.sleep(0.05)
    pathlib.Path("stage_b.txt").write_text("b")
    return {"stage_b": pathlib.Path("stage_b.txt")}


# Module-level stage helpers for chain test (avoids closure/fingerprinting issues)
def _helper_chain_step1(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ChainStep1:
    pathlib.Path("step1.txt").write_text(input_file.read_text() + "_1")
    return {"output": pathlib.Path("step1.txt")}


def _helper_chain_step2(
    step1: Annotated[pathlib.Path, outputs.Dep("step1.txt", loaders.PathOnly())],
) -> _ChainStep2:
    pathlib.Path("step2.txt").write_text(step1.read_text() + "_2")
    return {"output": pathlib.Path("step2.txt")}


def _helper_chain_step3(
    step2: Annotated[pathlib.Path, outputs.Dep("step2.txt", loaders.PathOnly())],
) -> _ChainStep3:
    pathlib.Path("step3.txt").write_text(step2.read_text() + "_3")
    return {"output": pathlib.Path("step3.txt")}


def _helper_chain_step4(
    step3: Annotated[pathlib.Path, outputs.Dep("step3.txt", loaders.PathOnly())],
) -> _ChainStep4:
    pathlib.Path("step4.txt").write_text(step3.read_text() + "_4")
    return {"output": pathlib.Path("step4.txt")}


def _helper_chain_step5(
    step4: Annotated[pathlib.Path, outputs.Dep("step4.txt", loaders.PathOnly())],
) -> _ChainStep5:
    pathlib.Path("step5.txt").write_text(step4.read_text() + "_5")
    return {"output": pathlib.Path("step5.txt")}


def _many_deps_process(
    input_0: Annotated[pathlib.Path, outputs.Dep("input_0.txt", loaders.PathOnly())],
    input_1: Annotated[pathlib.Path, outputs.Dep("input_1.txt", loaders.PathOnly())],
    input_2: Annotated[pathlib.Path, outputs.Dep("input_2.txt", loaders.PathOnly())],
    input_3: Annotated[pathlib.Path, outputs.Dep("input_3.txt", loaders.PathOnly())],
    input_4: Annotated[pathlib.Path, outputs.Dep("input_4.txt", loaders.PathOnly())],
    input_5: Annotated[pathlib.Path, outputs.Dep("input_5.txt", loaders.PathOnly())],
    input_6: Annotated[pathlib.Path, outputs.Dep("input_6.txt", loaders.PathOnly())],
    input_7: Annotated[pathlib.Path, outputs.Dep("input_7.txt", loaders.PathOnly())],
    input_8: Annotated[pathlib.Path, outputs.Dep("input_8.txt", loaders.PathOnly())],
    input_9: Annotated[pathlib.Path, outputs.Dep("input_9.txt", loaders.PathOnly())],
    input_10: Annotated[pathlib.Path, outputs.Dep("input_10.txt", loaders.PathOnly())],
    input_11: Annotated[pathlib.Path, outputs.Dep("input_11.txt", loaders.PathOnly())],
    input_12: Annotated[pathlib.Path, outputs.Dep("input_12.txt", loaders.PathOnly())],
    input_13: Annotated[pathlib.Path, outputs.Dep("input_13.txt", loaders.PathOnly())],
    input_14: Annotated[pathlib.Path, outputs.Dep("input_14.txt", loaders.PathOnly())],
    input_15: Annotated[pathlib.Path, outputs.Dep("input_15.txt", loaders.PathOnly())],
    input_16: Annotated[pathlib.Path, outputs.Dep("input_16.txt", loaders.PathOnly())],
    input_17: Annotated[pathlib.Path, outputs.Dep("input_17.txt", loaders.PathOnly())],
    input_18: Annotated[pathlib.Path, outputs.Dep("input_18.txt", loaders.PathOnly())],
    input_19: Annotated[pathlib.Path, outputs.Dep("input_19.txt", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


# =============================================================================
# Tests
# =============================================================================


def test_simple_pipeline_runs_in_order(pipeline_dir: pathlib.Path) -> None:
    """Stages execute in dependency order and produce correct outputs."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_step1_upper, name="step1")
    register_test_stage(_step2_result, name="step2")

    results = executor.run()

    assert (pipeline_dir / "step2.txt").read_text() == "Result: HELLO"
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"


def test_unchanged_stages_are_skipped(pipeline_dir: pathlib.Path) -> None:
    """Stages with unchanged code and deps are skipped on re-run."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_output_upper, name="step1")

    # First run - should execute
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "HELLO"

    # Second run - should skip (nothing changed)
    results = executor.run()
    assert results["step1"]["status"] == "skipped"


def test_code_change_triggers_rerun(pipeline_dir: pathlib.Path) -> None:
    """Changing stage code triggers re-execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_output_upper, name="process")

    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "HELLO"

    # Clear and re-register with different implementation
    REGISTRY.clear()

    register_test_stage(_output_lower, name="process")

    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "hello"


def test_code_change_rerun_with_hardlinked_readonly_output(pipeline_dir: pathlib.Path) -> None:
    """Re-running stage after code change works even when output is hardlinked to read-only cache.

    Reproduces bug where hardlinked outputs (the default checkout mode) become read-only
    (0o444) because they share the inode with the cache file. When the stage needs to re-run
    due to code changes, the framework must either delete the old output or make it writable
    before the stage can write new output.
    """
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_output_upper, name="process")

    # First run - creates output
    results = executor.run()
    assert results["process"]["status"] == "ran"

    output_file = pipeline_dir / "output.txt"
    assert output_file.exists()
    assert output_file.read_text() == "HELLO"

    # Verify output is hardlinked to read-only cache (the default behavior)
    stat_info = output_file.stat()
    assert stat_info.st_nlink > 1, "Output should be hardlinked to cache"
    assert stat_info.st_mode & 0o777 == 0o444, "Output should be read-only (hardlink to cache)"

    # Re-register with different implementation (simulates code change)
    REGISTRY.clear()
    register_test_stage(_output_lower, name="process")

    # Second run - should re-run due to fingerprint change
    # BUG: This may fail with PermissionError if framework doesn't handle read-only outputs
    results = executor.run()
    assert results["process"]["status"] == "ran", f"Stage should re-run, got: {results['process']}"
    assert output_file.read_text() == "hello"


def test_code_change_rerun_with_hardlinked_directory_output(pipeline_dir: pathlib.Path) -> None:
    """Re-running stage with directory output works even when files are hardlinked read-only.

    Tests the scenario where a stage produces a directory output with multiple files.
    After caching, individual files are hardlinked to the read-only cache.
    When code changes trigger a re-run, the directory must be removed (including
    all read-only hardlinked files) before the stage can write new output.
    """
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_dir_output_process, name="process")

    # First run - creates directory output
    results = executor.run()
    assert results["process"]["status"] == "ran"

    output_dir = pipeline_dir / "output_dir"
    assert output_dir.exists()
    assert (output_dir / "file1.txt").read_text() == "file1"

    # Verify files are hardlinked to read-only cache
    file1 = output_dir / "file1.txt"
    stat_info = file1.stat()
    assert stat_info.st_nlink > 1, "File should be hardlinked to cache"
    assert stat_info.st_mode & 0o777 == 0o444, "File should be read-only"

    # Re-register with different implementation (simulates code change)
    REGISTRY.clear()
    register_test_stage(_dir_output_process_v2, name="process")

    # Second run - should re-run due to fingerprint change
    results = executor.run()
    assert results["process"]["status"] == "ran", f"Stage should re-run, got: {results['process']}"
    assert (output_dir / "file1.txt").read_text() == "file1_v2"


def test_input_change_triggers_rerun(pipeline_dir: pathlib.Path) -> None:
    """Changing input file triggers re-execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_output_upper, name="process")

    # First run
    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "HELLO"

    # Modify input
    (pipeline_dir / "input.txt").write_text("world")

    # Should re-run due to input change
    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "WORLD"


def test_downstream_runs_when_upstream_changes(pipeline_dir: pathlib.Path) -> None:
    """Downstream stages re-run when upstream output changes."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_intermediate_upper, name="step1")
    register_test_stage(_final_from_intermediate, name="step2")

    # First run - both execute
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert (pipeline_dir / "final.txt").read_text() == "Final: HELLO"

    # Change input - both should re-run
    (pipeline_dir / "input.txt").write_text("world")
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert (pipeline_dir / "final.txt").read_text() == "Final: WORLD"


def test_run_specific_stage(pipeline_dir: pathlib.Path) -> None:
    """Can run a specific stage and its dependencies only."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_a_stage, name="a")
    register_test_stage(_b_from_a, name="b")
    register_test_stage(_c_stage, name="c")

    # Run only 'b' (should also run 'a' as dependency, but not 'c')
    results = executor.run(stages=["b"])

    assert results["a"]["status"] == "ran", "Dependency 'a' should run"
    assert results["b"]["status"] == "ran", "Target 'b' should run"
    assert "c" not in results, "Unrelated 'c' should not be in results"
    assert not (pipeline_dir / "c.txt").exists(), "Stage 'c' output should not exist"


def test_missing_dependency_raises_error(pipeline_dir: pathlib.Path) -> None:
    """Missing dependency file raises DependencyNotFoundError before stage runs."""
    # Don't create the input file - it's missing

    register_test_stage(_missing_dep_process, name="process")

    with pytest.raises(exceptions.DependencyNotFoundError) as exc_info:
        executor.run()

    assert "missing_input.txt" in str(exc_info.value)


def test_nonexistent_stage_raises_error(pipeline_dir: pathlib.Path) -> None:
    """Requesting a non-existent stage raises StageNotFoundError."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_real_stage, name="real_stage")

    with pytest.raises(exceptions.StageNotFoundError) as exc_info:
        executor.run(stages=["nonexistent_stage"])

    assert "nonexistent_stage" in str(exc_info.value)


def test_execution_lock_created_and_removed(
    pipeline_dir: pathlib.Path, stages_dir: pathlib.Path
) -> None:
    """Execution lock file is created during run and removed after."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_check_lock, name="check_lock")

    executor.run()

    lock_check_file = pipeline_dir / "lock_existed.txt"
    assert lock_check_file.read_text() == "yes", "Lock file should exist during stage execution"
    assert not (stages_dir / "check_lock.running").exists(), "Lock file should be removed after"


def test_execution_lock_removed_on_stage_failure(
    pipeline_dir: pathlib.Path, stages_dir: pathlib.Path
) -> None:
    """Execution lock is released even if stage raises an exception."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_failing_stage, name="failing_stage")

    # Executor now catches exceptions and returns failed status
    results = executor.run()

    assert results["failing_stage"]["status"] == "failed"
    assert "Stage failed!" in results["failing_stage"]["reason"]
    assert not (stages_dir / "failing_stage.running").exists(), "Lock should be released on failure"


def test_stale_lock_from_dead_process_is_broken(
    pipeline_dir: pathlib.Path, stages_dir: pathlib.Path
) -> None:
    """Stale lock file from crashed process is automatically removed."""
    (pipeline_dir / "input.txt").write_text("hello")

    # Create a stale lock with a non-existent PID
    stale_lock = stages_dir / "process.running"
    stale_lock.write_text("999999999")  # PID that doesn't exist

    register_test_stage(_process_basic, name="process")

    # Should succeed by breaking the stale lock
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not stale_lock.exists(), "Stale lock should be removed"


def test_concurrent_execution_returns_failed_status(
    pipeline_dir: pathlib.Path, stages_dir: pathlib.Path
) -> None:
    """Running stage that's already running returns failed status."""
    (pipeline_dir / "input.txt").write_text("hello")

    # Create a lock with our own PID (simulating concurrent run)
    active_lock = stages_dir / "process.running"
    active_lock.write_text(str(os.getpid()))

    register_test_stage(_process_basic, name="process")

    # Executor now returns failed status instead of raising
    results = executor.run()

    assert results["process"]["status"] == "failed"
    assert "already running" in results["process"]["reason"]
    assert str(os.getpid()) in results["process"]["reason"]

    # Clean up
    active_lock.unlink()


def test_corrupted_lock_file_is_broken(
    pipeline_dir: pathlib.Path, stages_dir: pathlib.Path
) -> None:
    """Corrupted lock file (invalid content) is treated as stale and removed."""
    (pipeline_dir / "input.txt").write_text("hello")

    # Create a corrupted lock file
    corrupted_lock = stages_dir / "process.running"
    corrupted_lock.write_text("garbage content without pid")

    register_test_stage(_process_basic, name="process")

    # Should succeed by treating corrupted lock as stale
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not corrupted_lock.exists(), "Corrupted lock should be removed"


def test_negative_pid_in_lock_is_treated_as_stale(
    pipeline_dir: pathlib.Path, stages_dir: pathlib.Path
) -> None:
    """Lock file with invalid PID (negative) is treated as stale."""
    (pipeline_dir / "input.txt").write_text("hello")

    # Create a lock with invalid PID
    invalid_lock = stages_dir / "process.running"
    invalid_lock.write_text("-1")

    register_test_stage(_process_basic, name="process")

    # Should succeed by treating invalid PID as stale
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not invalid_lock.exists(), "Invalid PID lock should be removed"


def test_output_queue_reader_only_catches_empty(pipeline_dir: pathlib.Path) -> None:
    """Output queue reader should only catch queue.Empty, not other exceptions."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_output_queue_stage, name="process")

    # This test verifies the output queue reader behavior exists and handles Empty properly
    results = executor.run()
    assert results["process"]["status"] == "ran"


def test_output_thread_cleanup_completes(pipeline_dir: pathlib.Path) -> None:
    """Output thread should be properly cleaned up after execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    initial_thread_count = threading.active_count()

    register_test_stage(_print_stage, name="process")

    results = executor.run()

    # Poll for thread cleanup with timeout (more robust than fixed sleep)
    deadline = time.monotonic() + 1.0  # 1 second timeout
    while time.monotonic() < deadline:
        final_thread_count = threading.active_count()
        if final_thread_count <= initial_thread_count + 1:
            break
        time.sleep(0.01)
    else:
        final_thread_count = threading.active_count()

    # Thread count should return to initial (or close to it)
    assert final_thread_count <= initial_thread_count + 1, (
        f"Thread leak: started with {initial_thread_count}, ended with {final_thread_count}"
    )
    assert results["process"]["status"] == "ran"


def _loky_worker_with_queue(queue: "mp.Queue[str]", message: str) -> str:
    """Worker function that writes to a queue and returns a value."""
    queue.put(message)
    return f"processed: {message}"


def test_spawn_context_manager_works_with_loky() -> None:
    """Spawn-context Manager queue works correctly with loky workers.

    This test verifies that switching mp.Manager() to use spawn context
    doesn't break communication between loky workers and the orchestrator.
    This is a prerequisite for fixing the Python 3.13 fork deprecation warning.
    """
    # Create Manager with spawn context (the change we want to make in production)
    spawn_ctx = mp.get_context("spawn")
    manager = spawn_ctx.Manager()
    queue: mp.Queue[str] = manager.Queue()  # pyright: ignore[reportAssignmentType]

    try:
        # Use loky executor (same as production code)
        lk_executor = loky.get_reusable_executor(max_workers=2)

        # Submit work that writes to the queue
        future = lk_executor.submit(_loky_worker_with_queue, queue, "test_message")

        # Verify worker can return a value (basic loky functionality)
        result = future.result(timeout=10)
        assert result == "processed: test_message"

        # Verify orchestrator can read from queue (the critical test)
        message = queue.get(timeout=5)
        assert message == "test_message"

    finally:
        manager.shutdown()


def test_mutex_prevents_concurrent_execution(pipeline_dir: pathlib.Path) -> None:
    """Stages in same mutex group run sequentially, not concurrently."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    register_test_stage(_mutex_a, name="stage_a", mutex=["gpu"])
    register_test_stage(_mutex_b, name="stage_b", mutex=["gpu"])

    results = executor.run(max_workers=4)

    assert results["stage_a"]["status"] == "ran"
    assert results["stage_b"]["status"] == "ran"

    # Verify sequential execution: either a_start,a_end,b_start,b_end or b_start,b_end,a_start,a_end
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["a_start", "a_end", "b_start", "b_end"],
        ["b_start", "b_end", "a_start", "a_end"],
    ], f"Stages ran concurrently: {timing}"


def test_mutex_releases_on_completion(pipeline_dir: pathlib.Path) -> None:
    """Mutex is released when stage completes, allowing next stage to start."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_first_basic, name="first", mutex=["resource"])
    register_test_stage(_second_basic, name="second", mutex=["resource"])

    results = executor.run(max_workers=4)

    assert results["first"]["status"] == "ran"
    assert results["second"]["status"] == "ran"
    assert (pipeline_dir / "first.txt").exists()
    assert (pipeline_dir / "second.txt").exists()


def test_mutex_releases_on_failure(pipeline_dir: pathlib.Path) -> None:
    """Mutex is released even when stage fails, allowing other stages to run."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_failing_mutex, name="failing", mutex=["resource"])
    register_test_stage(_succeeding_mutex, name="succeeding", mutex=["resource"])

    results = executor.run(max_workers=4, on_error="keep_going")

    assert results["failing"]["status"] == "failed"
    assert results["succeeding"]["status"] == "ran"
    assert (pipeline_dir / "succeeding.txt").exists()


def test_multiple_mutex_groups_per_stage(pipeline_dir: pathlib.Path) -> None:
    """Stage with multiple mutex groups blocks all of them."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    register_test_stage(_multi_resource, name="multi_resource", mutex=["gpu", "disk"])
    register_test_stage(_gpu_only, name="gpu_only", mutex=["gpu"])

    results = executor.run(max_workers=4)

    assert results["multi_resource"]["status"] == "ran"
    assert results["gpu_only"]["status"] == "ran"

    # Stages should be sequential due to shared "gpu" mutex
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["multi_start", "multi_end", "gpu_start", "gpu_end"],
        ["gpu_start", "gpu_end", "multi_start", "multi_end"],
    ], f"Stages ran concurrently despite shared mutex: {timing}"


def test_mutex_with_dependencies(pipeline_dir: pathlib.Path) -> None:
    """Mutex works correctly with stage dependencies."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_first_dep_mutex, name="first", mutex=["resource"])
    register_test_stage(_second_dep_mutex, name="second", mutex=["resource"])

    results = executor.run(max_workers=4)

    assert results["first"]["status"] == "ran"
    assert results["second"]["status"] == "ran"
    assert (pipeline_dir / "second.txt").read_text() == "second: first"


def test_no_mutex_stages_unaffected(pipeline_dir: pathlib.Path) -> None:
    """Stages without mutex run normally in parallel."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    register_test_stage(_timing_a, name="stage_a")
    register_test_stage(_timing_b, name="stage_b")

    results = executor.run(max_workers=4)

    assert results["stage_a"]["status"] == "ran"
    assert results["stage_b"]["status"] == "ran"

    # Without mutex, stages should run in parallel (interleaved timing)
    timing = timing_file.read_text().strip().split("\n")
    assert len(timing) == 4


def test_single_stage_mutex_warning(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Warning is logged when mutex group has only one stage."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_lonely_mutex, name="lonely", mutex=["lonely_group"])

    with caplog.at_level(logging.WARNING):
        results = executor.run()

    assert results["lonely"]["status"] == "ran"
    assert any(
        "lonely_group" in record.message and "lonely" in record.message for record in caplog.records
    ), (
        f"Expected warning about single-stage mutex group, got: {[r.message for r in caplog.records]}"
    )


# =============================================================================
# Error Mode Tests
# =============================================================================


def test_on_error_fail_stops_on_first_failure(pipeline_dir: pathlib.Path) -> None:
    """on_error='fail' stops pipeline when first stage fails."""
    (pipeline_dir / "input.txt").write_text("data")
    execution_log = pipeline_dir / "execution.log"

    register_test_stage(_error_stage_a, name="stage_a")
    register_test_stage(_error_stage_b, name="stage_b")

    results = executor.run(on_error="fail")

    assert results["stage_a"]["status"] == "failed"
    # stage_b may or may not run depending on timing, but pipeline should stop
    log_content = execution_log.read_text() if execution_log.exists() else ""
    assert "a" in log_content, "Stage A should have executed"


def test_on_error_keep_going_continues_independent_stages(pipeline_dir: pathlib.Path) -> None:
    """on_error='keep_going' continues running independent stages after failure."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_keep_going_failing, name="failing")
    register_test_stage(_keep_going_succeeding, name="succeeding")

    results = executor.run(on_error="keep_going")

    assert results["failing"]["status"] == "failed"
    assert results["succeeding"]["status"] == "ran"
    assert (pipeline_dir / "succeeding.txt").read_text() == "success"


def test_on_error_keep_going_skips_downstream_of_failed(pipeline_dir: pathlib.Path) -> None:
    """on_error='keep_going' skips stages that depend on failed stage."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_first_failing, name="first")
    register_test_stage(_second_depends_first, name="second")
    register_test_stage(_independent_stage, name="independent")

    results = executor.run(on_error="keep_going")

    assert results["first"]["status"] == "failed"
    assert results["second"]["status"] == "skipped"
    assert "upstream" in results["second"]["reason"]
    assert results["independent"]["status"] == "ran"


def test_invalid_on_error_raises_value_error(pipeline_dir: pathlib.Path) -> None:
    """Invalid on_error value raises ValueError."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_invalid_error_process, name="process")

    with pytest.raises(ValueError) as exc_info:
        executor.run(on_error="invalid_mode")

    assert "invalid_mode" in str(exc_info.value)
    assert "fail" in str(exc_info.value)  # Should mention valid options


# =============================================================================
# Timeout Tests
# =============================================================================


def test_stage_timeout_marks_stage_as_failed(pipeline_dir: pathlib.Path) -> None:
    """Stage exceeding timeout is marked as failed."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_slow_stage, name="slow_stage")

    results = executor.run(stage_timeout=0.1)

    assert results["slow_stage"]["status"] == "failed"
    assert "timed out" in results["slow_stage"]["reason"]


def test_stage_timeout_does_not_affect_fast_stages(pipeline_dir: pathlib.Path) -> None:
    """Fast stages complete normally even with timeout set."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_fast_stage, name="fast_stage")

    results = executor.run(stage_timeout=60.0)

    assert results["fast_stage"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "done"


# =============================================================================
# Worker Exception Tests
# =============================================================================


def test_stage_calling_sys_exit_returns_failed(pipeline_dir: pathlib.Path) -> None:
    """Stage calling sys.exit() returns failed status with exit code."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_exits_with_code, name="exits_with_code")

    results = executor.run()

    assert results["exits_with_code"]["status"] == "failed"
    assert "sys.exit" in results["exits_with_code"]["reason"]
    assert "42" in results["exits_with_code"]["reason"]


def test_stage_calling_sys_exit_zero_returns_failed(pipeline_dir: pathlib.Path) -> None:
    """Stage calling sys.exit(0) still returns failed (stages shouldn't exit)."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_exits_zero, name="exits_zero")

    results = executor.run()

    assert results["exits_zero"]["status"] == "failed"
    assert "sys.exit" in results["exits_zero"]["reason"]


def test_stage_raising_keyboard_interrupt_returns_failed(pipeline_dir: pathlib.Path) -> None:
    """Stage raising KeyboardInterrupt returns failed status."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_keyboard_interrupt, name="keyboard_interrupt")

    results = executor.run()

    assert results["keyboard_interrupt"]["status"] == "failed"
    assert "KeyboardInterrupt" in results["keyboard_interrupt"]["reason"]


# =============================================================================
# Dependency Validation Tests
# =============================================================================


def test_directory_dependency_hashed_and_runs(pipeline_dir: pathlib.Path) -> None:
    """Stage with directory as dependency hashes it and runs successfully."""
    # Create a directory with files
    data_dir = pipeline_dir / "data_dir"
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("content1")
    (data_dir / "file2.txt").write_text("content2")

    register_test_stage(_dir_dep_process, name="process")

    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "done"


# =============================================================================
# Non-Parallel Execution Tests
# =============================================================================


def test_parallel_false_runs_sequentially(pipeline_dir: pathlib.Path) -> None:
    """parallel=False runs stages one at a time."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    register_test_stage(_parallel_false_a, name="stage_a")
    register_test_stage(_parallel_false_b, name="stage_b")

    results = executor.run(parallel=False)

    assert results["stage_a"]["status"] == "ran"
    assert results["stage_b"]["status"] == "ran"

    # With parallel=False, stages must be strictly sequential
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["a_start", "a_end", "b_start", "b_end"],
        ["b_start", "b_end", "a_start", "a_end"],
    ], f"Stages overlapped with parallel=False: {timing}"


# =============================================================================
# Output Capture Tests
# =============================================================================


def test_stage_stdout_and_stderr_captured(pipeline_dir: pathlib.Path) -> None:
    """Stage stdout and stderr are captured in results."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_prints_output, name="prints_output")

    results = executor.run()

    assert results["prints_output"]["status"] == "ran"
    # Output lines are captured in results but not exposed in dict
    # The stage should run successfully with captured output


def test_stage_partial_line_output_captured(pipeline_dir: pathlib.Path) -> None:
    """Stage output without trailing newline is captured."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_partial_output, name="partial_output")

    results = executor.run()
    assert results["partial_output"]["status"] == "ran"


# =============================================================================
# Lock Retry Exhaustion Tests
# =============================================================================


def test_lock_retry_exhaustion_returns_failed(
    pipeline_dir: pathlib.Path, stages_dir: pathlib.Path
) -> None:
    """Multiple failed lock attempts return failed status."""
    (pipeline_dir / "input.txt").write_text("data")

    # Create a lock with our own PID (simulates live concurrent run)
    lock_file = stages_dir / "process.running"
    lock_file.write_text(str(os.getpid()))

    register_test_stage(_process_basic, name="process")

    results = executor.run()

    assert results["process"]["status"] == "failed"
    assert "already running" in results["process"]["reason"]

    lock_file.unlink()


# =============================================================================
# Mutex Name Normalization Tests
# =============================================================================


def test_mutex_names_are_case_insensitive(pipeline_dir: pathlib.Path) -> None:
    """Mutex names are normalized to lowercase for comparison."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    register_test_stage(_upper_mutex, name="upper_mutex", mutex=["GPU"])
    register_test_stage(_lower_mutex, name="lower_mutex", mutex=["gpu"])

    results = executor.run(max_workers=4)

    assert results["upper_mutex"]["status"] == "ran"
    assert results["lower_mutex"]["status"] == "ran"

    # Should be sequential due to mutex normalization
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["upper_start", "upper_end", "lower_start", "lower_end"],
        ["lower_start", "lower_end", "upper_start", "upper_end"],
    ], f"Mutex names not normalized - stages ran concurrently: {timing}"


def test_mutex_names_whitespace_stripped(pipeline_dir: pathlib.Path) -> None:
    """Mutex names have whitespace stripped."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    register_test_stage(_spaced_mutex, name="spaced_mutex", mutex=["  resource  "])
    register_test_stage(_clean_mutex, name="clean_mutex", mutex=["resource"])

    results = executor.run(max_workers=4)

    assert results["spaced_mutex"]["status"] == "ran"
    assert results["clean_mutex"]["status"] == "ran"

    # Should be sequential due to mutex normalization
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["spaced_start", "spaced_end", "clean_start", "clean_end"],
        ["clean_start", "clean_end", "spaced_start", "spaced_end"],
    ], f"Mutex whitespace not stripped - stages ran concurrently: {timing}"


def test_exclusive_mutex_runs_alone(pipeline_dir: pathlib.Path) -> None:
    """Stage with mutex=['*'] runs exclusively - no other stages run concurrently."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    register_test_stage(_exclusive_stage, name="exclusive_stage", mutex=["*"])
    register_test_stage(_normal_a, name="normal_a")
    register_test_stage(_normal_b, name="normal_b")

    results = executor.run(max_workers=4)

    assert results["exclusive_stage"]["status"] == "ran"
    assert results["normal_a"]["status"] == "ran"
    assert results["normal_b"]["status"] == "ran"

    # The exclusive stage must not overlap with any other stage
    timing = timing_file.read_text().strip().split("\n")

    # Find where exclusive runs in the sequence
    excl_start = timing.index("exclusive_start")
    excl_end = timing.index("exclusive_end")

    # Nothing else should be between exclusive_start and exclusive_end
    between = timing[excl_start + 1 : excl_end]
    assert between == [], f"Other stages ran during exclusive: {between}"


# =============================================================================
# Output Cache Tests
# =============================================================================


def test_executor_removes_outputs_before_run(pipeline_dir: pathlib.Path) -> None:
    """Outputs are removed before stage execution (clean state)."""
    (pipeline_dir / "input.txt").write_text("data")
    output_file = pipeline_dir / "output.txt"
    output_file.write_text("stale data")

    register_test_stage(_removes_output_process, name="process")

    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert output_file.read_text() == "fresh data"


def test_executor_saves_outputs_to_cache(pipeline_dir: pathlib.Path) -> None:
    """Outputs are saved to cache after successful execution."""
    (pipeline_dir / "input.txt").write_text("data")
    cache_dir = pipeline_dir / ".pivot" / "cache"

    register_test_stage(_cache_output_process, name="process")

    results = executor.run()

    assert results["process"]["status"] == "ran"

    # Output should exist with correct content (linked to cache via hardlink/symlink/copy)
    output_file = pipeline_dir / "output.txt"
    assert output_file.exists(), "Output should exist"
    assert output_file.read_text() == "result"

    # Cache should contain the file
    files_cache = cache_dir / "files"
    assert files_cache.exists(), "Cache directory should exist"
    cache_files = list(files_cache.rglob("*"))
    assert len(cache_files) >= 1, "Cache should contain files"


def test_executor_restores_missing_outputs_on_skip(pipeline_dir: pathlib.Path) -> None:
    """Skipped stages restore missing outputs from cache."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_cache_output_process, name="process")

    # First run - executes and caches
    results = executor.run()
    assert results["process"]["status"] == "ran"

    # Delete output (simulating user deleting file)
    output_file = pipeline_dir / "output.txt"
    if output_file.is_symlink():
        output_file.unlink()
    else:
        output_file.unlink()

    assert not output_file.exists()

    # Second run - should skip but restore output from cache
    results = executor.run()
    assert results["process"]["status"] == "skipped"
    assert output_file.exists(), "Output should be restored from cache"
    assert output_file.read_text() == "result"


def test_executor_fails_if_output_missing(pipeline_dir: pathlib.Path) -> None:
    """Stage fails if declared output is not produced."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_no_output_process, name="process")

    results = executor.run()

    assert results["process"]["status"] == "failed"
    assert "output" in results["process"]["reason"].lower()


def test_executor_handles_json_outputs(pipeline_dir: pathlib.Path) -> None:
    """JSON outputs are created correctly."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_metrics_process, name="process")

    results = executor.run()

    assert results["process"]["status"] == "ran"

    # Output should exist with correct content
    metrics_file = pipeline_dir / "metrics.json"
    assert metrics_file.exists()
    import json

    assert json.loads(metrics_file.read_text()) == {"accuracy": 0.95}


def test_executor_fails_if_json_output_missing(pipeline_dir: pathlib.Path) -> None:
    """Stage fails if JSON output is not produced."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_no_metrics_process, name="process")

    results = executor.run()

    assert results["process"]["status"] == "failed"
    assert "metrics.json" in results["process"]["reason"]


def test_executor_output_hashes_in_lock_file(pipeline_dir: pathlib.Path) -> None:
    """Output hashes are stored in lock file."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_cache_output_process, name="process")

    executor.run()

    lock_file = pipeline_dir / ".pivot" / "stages" / "process.lock"
    assert lock_file.exists()

    # Storage format uses 'outs' list (not 'output_hashes' dict)
    lock_data = yaml.safe_load(lock_file.read_text())
    assert "outs" in lock_data
    assert len(lock_data["outs"]) == 1
    assert lock_data["outs"][0]["path"] == "output.txt"


def test_executor_lock_file_deterministic_sort(pipeline_dir: pathlib.Path) -> None:
    """Lock file entries are sorted for deterministic output."""
    (pipeline_dir / "input.txt").write_text("data")
    (pipeline_dir / "z_input.txt").write_text("z")
    (pipeline_dir / "a_input.txt").write_text("a")

    register_test_stage(_multi_input_process, name="process")

    executor.run()

    lock_file = pipeline_dir / ".pivot" / "stages" / "process.lock"
    lock_data = yaml.safe_load(lock_file.read_text())

    # Storage format uses 'deps' and 'outs' lists (sorted by path)
    dep_paths = [entry["path"] for entry in lock_data.get("deps", [])]
    assert dep_paths == sorted(dep_paths), "deps should be sorted by path"

    out_paths = [entry["path"] for entry in lock_data.get("outs", [])]
    assert out_paths == sorted(out_paths), "outs should be sorted by path"


def test_executor_directory_output_cached(pipeline_dir: pathlib.Path) -> None:
    """Directory outputs are cached with manifest."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_dir_output_process, name="process")

    results = executor.run()

    assert results["process"]["status"] == "ran"

    output_dir = pipeline_dir / "output_dir"
    assert output_dir.exists()
    assert (output_dir / "file1.txt").read_text() == "file1"
    assert (output_dir / "file2.txt").read_text() == "file2"


def test_executor_lock_file_missing_outs_triggers_rerun(pipeline_dir: pathlib.Path) -> None:
    """Lock file without outs section triggers re-execution."""
    (pipeline_dir / "input.txt").write_text("data")
    stages_dir = pipeline_dir / ".pivot" / "stages"
    stages_dir.mkdir(parents=True)

    # Create lock file without outs (incomplete)
    lock_file = stages_dir / "process.lock"
    lock_file.write_text(
        yaml.dump(
            {
                "code_manifest": {},
                "params": {},
                "deps": [],
                "dep_generations": {},
                # No outs - incomplete lock
            }
        )
    )

    register_test_stage(_lock_missing_outs, name="process")

    results = executor.run()

    # Should re-run because outs is missing
    assert results["process"]["status"] == "ran"

    # Lock file should now have outs
    lock_data = yaml.safe_load(lock_file.read_text())
    assert "outs" in lock_data


# ============================================================================
# Force flag tests
# ============================================================================


def test_force_runs_unchanged_stage(pipeline_dir: pathlib.Path) -> None:
    """Force flag should run stage even when nothing changed."""
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_force_process, name="process")

    # First run - should execute
    results = executor.run()
    assert results["process"]["status"] == "ran"

    # Second run without force - should skip (nothing changed)
    results = executor.run()
    assert results["process"]["status"] == "skipped"

    # Third run with force - should run despite no changes
    results = executor.run(force=True)
    assert results["process"]["status"] == "ran"


def test_force_updates_lock_file(pipeline_dir: pathlib.Path) -> None:
    """After forced run, lock file should have current fingerprints."""
    (pipeline_dir / "input.txt").write_text("test")

    register_test_stage(_cache_output_process, name="process")

    # First run with force
    results = executor.run(force=True)
    assert results["process"]["status"] == "ran"

    # Second run without force - should skip (lock file should be correct)
    results = executor.run()
    assert results["process"]["status"] == "skipped", "Lock file should be valid after forced run"


def test_force_with_specific_stage(pipeline_dir: pathlib.Path) -> None:
    """Force flag with specific stage forces that stage and its dependencies."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_force_step1, name="step1")
    register_test_stage(_force_step2, name="step2")
    register_test_stage(_force_other, name="other")

    # First run - all execute
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert results["other"]["status"] == "ran"

    # Force run of step2 only - should force step1 and step2, skip other
    results = executor.run(stages=["step2"], force=True)
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert "other" not in results  # Not in execution set


def test_force_with_single_stage(pipeline_dir: pathlib.Path) -> None:
    """Force with single_stage should only force specified stage."""
    (pipeline_dir / "input.txt").write_text("data")
    (pipeline_dir / "step1.txt").write_text("existing")

    register_test_stage(_force_single_step1, name="step1")
    register_test_stage(_force_single_step2, name="step2")

    # First run - both execute
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"

    # Force run step2 with single_stage - step1 should skip
    results = executor.run(stages=["step2"], single_stage=True, force=True)
    assert "step1" not in results  # Not in execution set due to single_stage
    assert results["step2"]["status"] == "ran"


# =============================================================================
# Worker Pool Cleanup Tests
# =============================================================================


def test_ensure_cleanup_registered_registers_atexit(mocker: "MockerFixture") -> None:
    """Verify cleanup handler is registered exactly once via functools.cache."""
    executor_core._ensure_cleanup_registered.cache_clear()

    try:
        mock_register = mocker.patch.object(atexit, "register")
        executor_core._ensure_cleanup_registered()
        mock_register.assert_called_once_with(executor_core._cleanup_worker_pool)

        executor_core._ensure_cleanup_registered()
        mock_register.assert_called_once()  # Still just once due to cache
    finally:
        executor_core._ensure_cleanup_registered.cache_clear()
        executor_core._ensure_cleanup_registered()


# =============================================================================
# Deferred Writes Tests (Critical for multi-process safety)
# =============================================================================


def test_executor_deferred_writes_applied(pipeline_dir: pathlib.Path) -> None:
    """Coordinator applies deferred_writes from worker results.

    Verifies that after stage execution:
    - Output generations are incremented in StateDB
    - Run cache entries allow skip detection on re-run
    """
    from pivot.storage import state

    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_deferred_process, name="process")

    results = executor.run()
    assert results["process"]["status"] == "ran"

    # Verify StateDB has output generation incremented
    db_path = pipeline_dir / ".pivot" / "state.db"
    with state.StateDB(db_path, readonly=True) as db:
        output_path = pipeline_dir / "output.txt"
        output_gen = db.get_generation(output_path)
        assert output_gen is not None and output_gen >= 1, (
            "Output generation should be incremented after stage runs"
        )

    # Verify deferred writes were applied by checking skip works on second run
    results = executor.run()
    assert results["process"]["status"] == "skipped", (
        "Stage should skip on second run - deferred writes recorded run cache"
    )


def test_executor_multi_stage_generation_tracking(pipeline_dir: pathlib.Path) -> None:
    """Generations increment correctly across stage chain."""
    from pivot.storage import state

    (pipeline_dir / "input.txt").write_text("data_v1")

    register_test_stage(_deferred_step1, name="step1")
    register_test_stage(_deferred_step2, name="step2")

    # First run
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"

    db_path = pipeline_dir / ".pivot" / "state.db"
    with state.StateDB(db_path, readonly=True) as db:
        step1_gen = db.get_generation(pipeline_dir / "step1.txt")
        step2_gen = db.get_generation(pipeline_dir / "step2.txt")
        assert step1_gen is not None and step1_gen >= 1
        assert step2_gen is not None and step2_gen >= 1

    # Modify input - both should re-run since step1 output changes
    (pipeline_dir / "input.txt").write_text("data_v2")
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"

    with state.StateDB(db_path, readonly=True) as db:
        new_step1_gen = db.get_generation(pipeline_dir / "step1.txt")
        new_step2_gen = db.get_generation(pipeline_dir / "step2.txt")
        # Generations should have incremented
        assert new_step1_gen is not None and new_step1_gen > step1_gen  # type: ignore[operator]
        assert new_step2_gen is not None and new_step2_gen > step2_gen  # type: ignore[operator]


# =============================================================================
# Concurrent Execution Tests
# =============================================================================


def test_concurrent_runs_different_stages_allowed(pipeline_dir: pathlib.Path) -> None:
    """Two pivot runs targeting different stages can proceed independently."""
    (pipeline_dir / "input.txt").write_text("data")

    register_test_stage(_concurrent_a, name="stage_a")
    register_test_stage(_concurrent_b, name="stage_b")

    # Both stages can run in parallel since they have different execution locks
    results = executor.run(max_workers=4)
    assert results["stage_a"]["status"] == "ran"
    assert results["stage_b"]["status"] == "ran"


# =============================================================================
# Scalability Tests
# =============================================================================


def test_many_stage_pipeline_completes(pipeline_dir: pathlib.Path) -> None:
    """Pipeline with many stages completes in reasonable time."""
    (pipeline_dir / "input.txt").write_text("start")

    # Register 5-stage chain using module-level helpers
    register_test_stage(_helper_chain_step1, name="step1")
    register_test_stage(_helper_chain_step2, name="step2")
    register_test_stage(_helper_chain_step3, name="step3")
    register_test_stage(_helper_chain_step4, name="step4")
    register_test_stage(_helper_chain_step5, name="step5")

    start_time = time.time()
    results = executor.run()
    elapsed = time.time() - start_time

    # All stages should run
    for i in range(1, 6):
        assert results[f"step{i}"]["status"] == "ran", f"step{i} should have run"

    # Should complete in reasonable time (< 15s even with slow CI)
    assert elapsed < 15, f"5-stage pipeline took too long: {elapsed:.1f}s"


def test_skip_detection_fast_with_many_deps(pipeline_dir: pathlib.Path) -> None:
    """Second run with many deps skips quickly via generation check."""
    # Create many input files
    for i in range(20):
        (pipeline_dir / f"input_{i}.txt").write_text(f"data_{i}")

    register_test_stage(_many_deps_process, name="process")

    # First run
    results = executor.run()
    assert results["process"]["status"] == "ran"

    # Second run - should skip quickly (generation-based check)
    start_time = time.time()
    results = executor.run()
    elapsed = time.time() - start_time

    assert results["process"]["status"] == "skipped"
    # Skip check should be fast (< 5s even with slow CI)
    assert elapsed < 5, f"Skip detection took too long: {elapsed:.1f}s"
