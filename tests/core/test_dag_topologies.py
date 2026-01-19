from __future__ import annotations

import pathlib
from typing import Annotated, TypedDict

from helpers import register_test_stage
from pivot import executor, loaders, outputs

# =============================================================================
# TypedDict output types for module-level stage functions
# =============================================================================


class _OutA(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _OutB(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _OutC(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("c.txt", loaders.PathOnly())]


class _OutD(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("d.txt", loaders.PathOnly())]


class _OutE(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("e.txt", loaders.PathOnly())]


class _OutF(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("f.txt", loaders.PathOnly())]


class _OutG(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("g.txt", loaders.PathOnly())]


class _OutH(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("h.txt", loaders.PathOnly())]


class _OutX(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("x.txt", loaders.PathOnly())]


class _OutY(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("y.txt", loaders.PathOnly())]


class _OutOutput(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _OutSum(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("sum.txt", loaders.PathOnly())]


class _OutStage1(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("stage1.txt", loaders.PathOnly())]


class _OutStage2(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("stage2.txt", loaders.PathOnly())]


class _OutStage3(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("stage3.txt", loaders.PathOnly())]


class _OutStage4(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("stage4.txt", loaders.PathOnly())]


class _OutStage5(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("stage5.txt", loaders.PathOnly())]


# =============================================================================
# Module-level stage functions for test_linear_dag_three_stages
# =============================================================================


def _linear3_stage_a(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    data = pathlib.Path("input.txt").read_text()
    pathlib.Path("a.txt").write_text(f"{data}->A")
    return _OutA(output=pathlib.Path("a.txt"))


def _linear3_stage_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    data = pathlib.Path("a.txt").read_text()
    assert data == "START->A", f"Expected 'START->A', got '{data}'"
    pathlib.Path("b.txt").write_text(f"{data}->B")
    return _OutB(output=pathlib.Path("b.txt"))


def _linear3_stage_c(
    _b: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    data = pathlib.Path("b.txt").read_text()
    assert data == "START->A->B", f"Expected 'START->A->B', got '{data}'"
    pathlib.Path("c.txt").write_text(f"{data}->C")
    return _OutC(output=pathlib.Path("c.txt"))


# =============================================================================
# Module-level stage functions for test_linear_dag_five_stages
# =============================================================================


def _linear5_stage1(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutStage1:
    with open("execution_log.txt", "a") as f:
        f.write("1\n")
    n = int(pathlib.Path("input.txt").read_text())
    pathlib.Path("stage1.txt").write_text(str(n + 1))
    return _OutStage1(output=pathlib.Path("stage1.txt"))


def _linear5_stage2(
    _s1: Annotated[pathlib.Path, outputs.Dep("stage1.txt", loaders.PathOnly())],
) -> _OutStage2:
    with open("execution_log.txt", "a") as f:
        f.write("2\n")
    n = int(pathlib.Path("stage1.txt").read_text())
    assert n == 1, f"stage1 must run first, got {n}"
    pathlib.Path("stage2.txt").write_text(str(n + 1))
    return _OutStage2(output=pathlib.Path("stage2.txt"))


def _linear5_stage3(
    _s2: Annotated[pathlib.Path, outputs.Dep("stage2.txt", loaders.PathOnly())],
) -> _OutStage3:
    with open("execution_log.txt", "a") as f:
        f.write("3\n")
    n = int(pathlib.Path("stage2.txt").read_text())
    assert n == 2, f"stage2 must run first, got {n}"
    pathlib.Path("stage3.txt").write_text(str(n + 1))
    return _OutStage3(output=pathlib.Path("stage3.txt"))


def _linear5_stage4(
    _s3: Annotated[pathlib.Path, outputs.Dep("stage3.txt", loaders.PathOnly())],
) -> _OutStage4:
    with open("execution_log.txt", "a") as f:
        f.write("4\n")
    n = int(pathlib.Path("stage3.txt").read_text())
    assert n == 3, f"stage3 must run first, got {n}"
    pathlib.Path("stage4.txt").write_text(str(n + 1))
    return _OutStage4(output=pathlib.Path("stage4.txt"))


def _linear5_stage5(
    _s4: Annotated[pathlib.Path, outputs.Dep("stage4.txt", loaders.PathOnly())],
) -> _OutStage5:
    with open("execution_log.txt", "a") as f:
        f.write("5\n")
    n = int(pathlib.Path("stage4.txt").read_text())
    assert n == 4, f"stage4 must run first, got {n}"
    pathlib.Path("stage5.txt").write_text(str(n + 1))
    return _OutStage5(output=pathlib.Path("stage5.txt"))


# =============================================================================
# Module-level stage functions for test_tree_dag_one_root_two_children
# =============================================================================


def _tree1_stage_a(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    data = pathlib.Path("input.txt").read_text()
    pathlib.Path("a.txt").write_text(f"{data}->A")
    return _OutA(output=pathlib.Path("a.txt"))


def _tree1_stage_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    data = pathlib.Path("a.txt").read_text()
    assert "->A" in data, "stage_a must run before stage_b"
    pathlib.Path("b.txt").write_text(f"{data}->B")
    return _OutB(output=pathlib.Path("b.txt"))


def _tree1_stage_c(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    data = pathlib.Path("a.txt").read_text()
    assert "->A" in data, "stage_a must run before stage_c"
    pathlib.Path("c.txt").write_text(f"{data}->C")
    return _OutC(output=pathlib.Path("c.txt"))


# =============================================================================
# Module-level stage functions for test_tree_dag_deeper
# =============================================================================


def _tree2_stage_a(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    pathlib.Path("a.txt").write_text("A")
    return _OutA(output=pathlib.Path("a.txt"))


def _tree2_stage_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    assert pathlib.Path("a.txt").read_text() == "A"
    pathlib.Path("b.txt").write_text("B")
    return _OutB(output=pathlib.Path("b.txt"))


def _tree2_stage_c(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    assert pathlib.Path("a.txt").read_text() == "A"
    pathlib.Path("c.txt").write_text("C")
    return _OutC(output=pathlib.Path("c.txt"))


def _tree2_stage_d(
    _b: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
) -> _OutD:
    with open("execution_log.txt", "a") as f:
        f.write("d\n")
    assert pathlib.Path("b.txt").read_text() == "B"
    pathlib.Path("d.txt").write_text("D")
    return _OutD(output=pathlib.Path("d.txt"))


def _tree2_stage_e(
    _c: Annotated[pathlib.Path, outputs.Dep("c.txt", loaders.PathOnly())],
) -> _OutE:
    with open("execution_log.txt", "a") as f:
        f.write("e\n")
    assert pathlib.Path("c.txt").read_text() == "C"
    pathlib.Path("e.txt").write_text("E")
    return _OutE(output=pathlib.Path("e.txt"))


# =============================================================================
# Module-level stage functions for test_diamond_dag
# =============================================================================


def _diamond1_stage_a(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    pathlib.Path("a.txt").write_text("A_OUTPUT")
    return _OutA(output=pathlib.Path("a.txt"))


def _diamond1_stage_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    data = pathlib.Path("a.txt").read_text()
    assert data == "A_OUTPUT", "stage_a must run before stage_b"
    pathlib.Path("b.txt").write_text("B_OUTPUT")
    return _OutB(output=pathlib.Path("b.txt"))


def _diamond1_stage_c(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    data = pathlib.Path("a.txt").read_text()
    assert data == "A_OUTPUT", "stage_a must run before stage_c"
    pathlib.Path("c.txt").write_text("C_OUTPUT")
    return _OutC(output=pathlib.Path("c.txt"))


def _diamond1_stage_d(
    _b: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
    _c: Annotated[pathlib.Path, outputs.Dep("c.txt", loaders.PathOnly())],
) -> _OutD:
    with open("execution_log.txt", "a") as f:
        f.write("d\n")
    b_data = pathlib.Path("b.txt").read_text()
    c_data = pathlib.Path("c.txt").read_text()
    assert b_data == "B_OUTPUT", "stage_b must run before stage_d"
    assert c_data == "C_OUTPUT", "stage_c must run before stage_d"
    pathlib.Path("d.txt").write_text(f"D({b_data}+{c_data})")
    return _OutD(output=pathlib.Path("d.txt"))


# =============================================================================
# Module-level stage functions for test_diamond_dag_with_shared_data
# =============================================================================


def _diamond2_compute_a(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    n = int(pathlib.Path("input.txt").read_text())
    pathlib.Path("a.txt").write_text(str(n))
    return _OutA(output=pathlib.Path("a.txt"))


def _diamond2_double_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    n = int(pathlib.Path("a.txt").read_text())
    pathlib.Path("b.txt").write_text(str(n * 2))  # 20
    return _OutB(output=pathlib.Path("b.txt"))


def _diamond2_triple_c(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutC:
    n = int(pathlib.Path("a.txt").read_text())
    pathlib.Path("c.txt").write_text(str(n * 3))  # 30
    return _OutC(output=pathlib.Path("c.txt"))


def _diamond2_sum_d(
    _b: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
    _c: Annotated[pathlib.Path, outputs.Dep("c.txt", loaders.PathOnly())],
) -> _OutD:
    b = int(pathlib.Path("b.txt").read_text())
    c = int(pathlib.Path("c.txt").read_text())
    pathlib.Path("d.txt").write_text(str(b + c))  # 50
    return _OutD(output=pathlib.Path("d.txt"))


# =============================================================================
# Module-level stage functions for test_fanout_dag
# =============================================================================


def _fanout1_stage_a(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    pathlib.Path("a.txt").write_text("A_DATA")
    return _OutA(output=pathlib.Path("a.txt"))


def _fanout1_stage_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    data = pathlib.Path("a.txt").read_text()
    assert data == "A_DATA", "stage_a must run before stage_b"
    pathlib.Path("b.txt").write_text("B")
    return _OutB(output=pathlib.Path("b.txt"))


def _fanout1_stage_c(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    data = pathlib.Path("a.txt").read_text()
    assert data == "A_DATA", "stage_a must run before stage_c"
    pathlib.Path("c.txt").write_text("C")
    return _OutC(output=pathlib.Path("c.txt"))


def _fanout1_stage_d(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutD:
    with open("execution_log.txt", "a") as f:
        f.write("d\n")
    data = pathlib.Path("a.txt").read_text()
    assert data == "A_DATA", "stage_a must run before stage_d"
    pathlib.Path("d.txt").write_text("D")
    return _OutD(output=pathlib.Path("d.txt"))


# =============================================================================
# Module-level stage functions for test_fanout_dag_wide
# =============================================================================


def _fanout2_root_stage(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    pathlib.Path("a.txt").write_text("ROOT")
    return _OutA(output=pathlib.Path("a.txt"))


def _fanout2_consumer_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    assert pathlib.Path("a.txt").read_text() == "ROOT"
    pathlib.Path("b.txt").write_text("B")
    return _OutB(output=pathlib.Path("b.txt"))


def _fanout2_consumer_c(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    assert pathlib.Path("a.txt").read_text() == "ROOT"
    pathlib.Path("c.txt").write_text("C")
    return _OutC(output=pathlib.Path("c.txt"))


def _fanout2_consumer_d(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutD:
    with open("execution_log.txt", "a") as f:
        f.write("d\n")
    assert pathlib.Path("a.txt").read_text() == "ROOT"
    pathlib.Path("d.txt").write_text("D")
    return _OutD(output=pathlib.Path("d.txt"))


def _fanout2_consumer_e(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutE:
    with open("execution_log.txt", "a") as f:
        f.write("e\n")
    assert pathlib.Path("a.txt").read_text() == "ROOT"
    pathlib.Path("e.txt").write_text("E")
    return _OutE(output=pathlib.Path("e.txt"))


def _fanout2_consumer_f(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutF:
    with open("execution_log.txt", "a") as f:
        f.write("f\n")
    assert pathlib.Path("a.txt").read_text() == "ROOT"
    pathlib.Path("f.txt").write_text("F")
    return _OutF(output=pathlib.Path("f.txt"))


# =============================================================================
# Module-level stage functions for test_fanin_dag
# =============================================================================


def _fanin1_stage_a(
    _input_a: Annotated[pathlib.Path, outputs.Dep("input_a.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    pathlib.Path("a.txt").write_text("A_OUT")
    return _OutA(output=pathlib.Path("a.txt"))


def _fanin1_stage_b(
    _input_b: Annotated[pathlib.Path, outputs.Dep("input_b.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    pathlib.Path("b.txt").write_text("B_OUT")
    return _OutB(output=pathlib.Path("b.txt"))


def _fanin1_stage_c(
    _input_c: Annotated[pathlib.Path, outputs.Dep("input_c.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    pathlib.Path("c.txt").write_text("C_OUT")
    return _OutC(output=pathlib.Path("c.txt"))


def _fanin1_stage_d(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
    _b: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
    _c: Annotated[pathlib.Path, outputs.Dep("c.txt", loaders.PathOnly())],
) -> _OutD:
    with open("execution_log.txt", "a") as f:
        f.write("d\n")
    a = pathlib.Path("a.txt").read_text()
    b = pathlib.Path("b.txt").read_text()
    c = pathlib.Path("c.txt").read_text()
    assert a == "A_OUT", "stage_a must run before stage_d"
    assert b == "B_OUT", "stage_b must run before stage_d"
    assert c == "C_OUT", "stage_c must run before stage_d"
    pathlib.Path("d.txt").write_text(f"{a}+{b}+{c}")
    return _OutD(output=pathlib.Path("d.txt"))


# =============================================================================
# Module-level stage functions for test_fanin_dag_with_computation
# =============================================================================


def _fanin2_compute_a(
    _input_a: Annotated[pathlib.Path, outputs.Dep("input_a.txt", loaders.PathOnly())],
) -> _OutA:
    n = int(pathlib.Path("input_a.txt").read_text())
    pathlib.Path("a.txt").write_text(str(n))
    return _OutA(output=pathlib.Path("a.txt"))


def _fanin2_compute_b(
    _input_b: Annotated[pathlib.Path, outputs.Dep("input_b.txt", loaders.PathOnly())],
) -> _OutB:
    n = int(pathlib.Path("input_b.txt").read_text())
    pathlib.Path("b.txt").write_text(str(n))
    return _OutB(output=pathlib.Path("b.txt"))


def _fanin2_compute_c(
    _input_c: Annotated[pathlib.Path, outputs.Dep("input_c.txt", loaders.PathOnly())],
) -> _OutC:
    n = int(pathlib.Path("input_c.txt").read_text())
    pathlib.Path("c.txt").write_text(str(n))
    return _OutC(output=pathlib.Path("c.txt"))


def _fanin2_compute_sum(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
    _b: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
    _c: Annotated[pathlib.Path, outputs.Dep("c.txt", loaders.PathOnly())],
) -> _OutSum:
    a = int(pathlib.Path("a.txt").read_text())
    b = int(pathlib.Path("b.txt").read_text())
    c = int(pathlib.Path("c.txt").read_text())
    pathlib.Path("sum.txt").write_text(str(a + b + c))
    return _OutSum(output=pathlib.Path("sum.txt"))


# =============================================================================
# Module-level stage functions for test_complex_dag_tree_then_diamond
# =============================================================================


def _complex1_stage_a(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    pathlib.Path("a.txt").write_text("A")
    return _OutA(output=pathlib.Path("a.txt"))


def _complex1_stage_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    assert pathlib.Path("a.txt").read_text() == "A"
    pathlib.Path("b.txt").write_text("B")
    return _OutB(output=pathlib.Path("b.txt"))


def _complex1_stage_c(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    assert pathlib.Path("a.txt").read_text() == "A"
    pathlib.Path("c.txt").write_text("C")
    return _OutC(output=pathlib.Path("c.txt"))


def _complex1_stage_d(
    _b: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
) -> _OutD:
    with open("execution_log.txt", "a") as f:
        f.write("d\n")
    assert pathlib.Path("b.txt").read_text() == "B"
    pathlib.Path("d.txt").write_text("D")
    return _OutD(output=pathlib.Path("d.txt"))


def _complex1_stage_e(
    _c: Annotated[pathlib.Path, outputs.Dep("c.txt", loaders.PathOnly())],
) -> _OutE:
    with open("execution_log.txt", "a") as f:
        f.write("e\n")
    assert pathlib.Path("c.txt").read_text() == "C"
    pathlib.Path("e.txt").write_text("E")
    return _OutE(output=pathlib.Path("e.txt"))


def _complex1_stage_f(
    _d: Annotated[pathlib.Path, outputs.Dep("d.txt", loaders.PathOnly())],
    _e: Annotated[pathlib.Path, outputs.Dep("e.txt", loaders.PathOnly())],
) -> _OutF:
    with open("execution_log.txt", "a") as f:
        f.write("f\n")
    d = pathlib.Path("d.txt").read_text()
    e = pathlib.Path("e.txt").read_text()
    assert d == "D", "stage_d must run before stage_f"
    assert e == "E", "stage_e must run before stage_f"
    pathlib.Path("f.txt").write_text(f"F({d},{e})")
    return _OutF(output=pathlib.Path("f.txt"))


# =============================================================================
# Module-level stage functions for test_complex_dag_multiple_diamonds
# =============================================================================


def _complex2_root_a(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    pathlib.Path("a.txt").write_text("1")
    return _OutA(output=pathlib.Path("a.txt"))


def _complex2_mid_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    pathlib.Path("b.txt").write_text("B")
    return _OutB(output=pathlib.Path("b.txt"))


def _complex2_mid_c(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutC:
    with open("execution_log.txt", "a") as f:
        f.write("c\n")
    pathlib.Path("c.txt").write_text("C")
    return _OutC(output=pathlib.Path("c.txt"))


def _complex2_mid_d(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutD:
    with open("execution_log.txt", "a") as f:
        f.write("d\n")
    pathlib.Path("d.txt").write_text("D")
    return _OutD(output=pathlib.Path("d.txt"))


def _complex2_lower_e(
    _b: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
    _c: Annotated[pathlib.Path, outputs.Dep("c.txt", loaders.PathOnly())],
) -> _OutE:
    with open("execution_log.txt", "a") as f:
        f.write("e\n")
    b = pathlib.Path("b.txt").read_text()
    c = pathlib.Path("c.txt").read_text()
    pathlib.Path("e.txt").write_text(f"E({b}{c})")
    return _OutE(output=pathlib.Path("e.txt"))


def _complex2_lower_f(
    _c: Annotated[pathlib.Path, outputs.Dep("c.txt", loaders.PathOnly())],
    _d: Annotated[pathlib.Path, outputs.Dep("d.txt", loaders.PathOnly())],
) -> _OutF:
    with open("execution_log.txt", "a") as f:
        f.write("f\n")
    c = pathlib.Path("c.txt").read_text()
    d = pathlib.Path("d.txt").read_text()
    pathlib.Path("f.txt").write_text(f"F({c}{d})")
    return _OutF(output=pathlib.Path("f.txt"))


def _complex2_lower_g(
    _d: Annotated[pathlib.Path, outputs.Dep("d.txt", loaders.PathOnly())],
) -> _OutG:
    with open("execution_log.txt", "a") as f:
        f.write("g\n")
    d = pathlib.Path("d.txt").read_text()
    pathlib.Path("g.txt").write_text(f"G({d})")
    return _OutG(output=pathlib.Path("g.txt"))


def _complex2_final_h(
    _e: Annotated[pathlib.Path, outputs.Dep("e.txt", loaders.PathOnly())],
    _f: Annotated[pathlib.Path, outputs.Dep("f.txt", loaders.PathOnly())],
    _g: Annotated[pathlib.Path, outputs.Dep("g.txt", loaders.PathOnly())],
) -> _OutH:
    with open("execution_log.txt", "a") as f:
        f.write("h\n")
    e = pathlib.Path("e.txt").read_text()
    f_val = pathlib.Path("f.txt").read_text()
    g = pathlib.Path("g.txt").read_text()
    pathlib.Path("h.txt").write_text(f"H[{e},{f_val},{g}]")
    return _OutH(output=pathlib.Path("h.txt"))


# =============================================================================
# Module-level stage functions for test_single_stage_dag
# =============================================================================


def _single_only_stage(
    _input: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutOutput:
    data = pathlib.Path("input.txt").read_text()
    pathlib.Path("output.txt").write_text(f"PROCESSED:{data}")
    return _OutOutput(output=pathlib.Path("output.txt"))


# =============================================================================
# Module-level stage functions for test_disconnected_dags
# =============================================================================


def _disconn_stage_a(
    _input_a: Annotated[pathlib.Path, outputs.Dep("input_a.txt", loaders.PathOnly())],
) -> _OutA:
    with open("execution_log.txt", "a") as f:
        f.write("a\n")
    pathlib.Path("a.txt").write_text("A_OUT")
    return _OutA(output=pathlib.Path("a.txt"))


def _disconn_stage_b(
    _a: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _OutB:
    with open("execution_log.txt", "a") as f:
        f.write("b\n")
    assert pathlib.Path("a.txt").read_text() == "A_OUT"
    pathlib.Path("b.txt").write_text("B_OUT")
    return _OutB(output=pathlib.Path("b.txt"))


def _disconn_stage_x(
    _input_x: Annotated[pathlib.Path, outputs.Dep("input_x.txt", loaders.PathOnly())],
) -> _OutX:
    with open("execution_log.txt", "a") as f:
        f.write("x\n")
    pathlib.Path("x.txt").write_text("X_OUT")
    return _OutX(output=pathlib.Path("x.txt"))


def _disconn_stage_y(
    _x: Annotated[pathlib.Path, outputs.Dep("x.txt", loaders.PathOnly())],
) -> _OutY:
    with open("execution_log.txt", "a") as f:
        f.write("y\n")
    assert pathlib.Path("x.txt").read_text() == "X_OUT"
    pathlib.Path("y.txt").write_text("Y_OUT")
    return _OutY(output=pathlib.Path("y.txt"))


# =============================================================================
# Linear DAG: A -> B -> C
# =============================================================================


def test_linear_dag_three_stages(pipeline_dir: pathlib.Path) -> None:
    """Linear DAG A -> B -> C executes in correct order.

    Each stage appends to the data, creating a chain that proves order.
    If any stage runs out of order, assertions will fail.
    """
    (pipeline_dir / "input.txt").write_text("START")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_linear3_stage_a, name="stage_a")
    register_test_stage(_linear3_stage_b, name="stage_b")
    register_test_stage(_linear3_stage_c, name="stage_c")

    results = executor.run()

    # Verify all stages ran
    assert all(r["status"] == "ran" for r in results.values())

    # Verify execution order via file-based log
    execution_log = log_file.read_text().strip().split("\n")
    assert execution_log == ["a", "b", "c"], f"Expected ['a', 'b', 'c'], got {execution_log}"

    # Verify final output proves correct chaining
    final_output = (pipeline_dir / "c.txt").read_text()
    assert final_output == "START->A->B->C"


def test_linear_dag_five_stages(pipeline_dir: pathlib.Path) -> None:
    """Longer linear DAG: A -> B -> C -> D -> E."""
    (pipeline_dir / "input.txt").write_text("0")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_linear5_stage1, name="stage1")
    register_test_stage(_linear5_stage2, name="stage2")
    register_test_stage(_linear5_stage3, name="stage3")
    register_test_stage(_linear5_stage4, name="stage4")
    register_test_stage(_linear5_stage5, name="stage5")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")
    assert execution_log == ["1", "2", "3", "4", "5"]
    assert (pipeline_dir / "stage5.txt").read_text() == "5"


# =============================================================================
# Tree DAG: A branches to B and C (no convergence)
#
#       A
#      / \
#     B   C
# =============================================================================


def test_tree_dag_one_root_two_children(pipeline_dir: pathlib.Path) -> None:
    """Tree DAG: A -> B, A -> C (B and C both depend on A, but not each other)."""
    (pipeline_dir / "input.txt").write_text("ROOT")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_tree1_stage_a, name="stage_a")
    register_test_stage(_tree1_stage_b, name="stage_b")
    register_test_stage(_tree1_stage_c, name="stage_c")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # A must run first
    assert execution_log[0] == "a", "stage_a must run first"

    # B and C can run in either order, but both must run
    assert set(execution_log[1:]) == {"b", "c"}

    # Verify outputs
    assert (pipeline_dir / "b.txt").read_text() == "ROOT->A->B"
    assert (pipeline_dir / "c.txt").read_text() == "ROOT->A->C"


def test_tree_dag_deeper(pipeline_dir: pathlib.Path) -> None:
    """Deeper tree: A -> B -> D, A -> C -> E.

         A
        / \
       B   C
       |   |
       D   E
    """
    (pipeline_dir / "input.txt").write_text("0")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_tree2_stage_a, name="stage_a")
    register_test_stage(_tree2_stage_b, name="stage_b")
    register_test_stage(_tree2_stage_c, name="stage_c")
    register_test_stage(_tree2_stage_d, name="stage_d")
    register_test_stage(_tree2_stage_e, name="stage_e")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # Verify order constraints
    assert execution_log.index("a") < execution_log.index("b")
    assert execution_log.index("a") < execution_log.index("c")
    assert execution_log.index("b") < execution_log.index("d")
    assert execution_log.index("c") < execution_log.index("e")


# =============================================================================
# Diamond DAG: Classic diamond pattern
#
#       A
#      / \
#     B   C
#      \ /
#       D
# =============================================================================


def test_diamond_dag(pipeline_dir: pathlib.Path) -> None:
    """Diamond DAG: A -> B -> D, A -> C -> D.

    D depends on both B and C, which both depend on A.
    """
    (pipeline_dir / "input.txt").write_text("INPUT")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_diamond1_stage_a, name="stage_a")
    register_test_stage(_diamond1_stage_b, name="stage_b")
    register_test_stage(_diamond1_stage_c, name="stage_c")
    register_test_stage(_diamond1_stage_d, name="stage_d")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # Verify order constraints
    assert execution_log.index("a") < execution_log.index("b")
    assert execution_log.index("a") < execution_log.index("c")
    assert execution_log.index("b") < execution_log.index("d")
    assert execution_log.index("c") < execution_log.index("d")

    # D must be last
    assert execution_log[-1] == "d"

    # Verify final output
    assert (pipeline_dir / "d.txt").read_text() == "D(B_OUTPUT+C_OUTPUT)"


def test_diamond_dag_with_shared_data(pipeline_dir: pathlib.Path) -> None:
    """Diamond DAG where D combines data from both paths.

    A produces a number, B doubles it, C triples it, D sums both.
    Final result proves all stages ran in correct order.
    """
    (pipeline_dir / "input.txt").write_text("10")

    register_test_stage(_diamond2_compute_a, name="compute_a")
    register_test_stage(_diamond2_double_b, name="double_b")
    register_test_stage(_diamond2_triple_c, name="triple_c")
    register_test_stage(_diamond2_sum_d, name="sum_d")

    executor.run()

    # If execution order was wrong, this would be incorrect
    assert (pipeline_dir / "d.txt").read_text() == "50"


# =============================================================================
# Fan-out DAG: One stage feeds many
#
#       A
#     / | \
#    B  C  D
# =============================================================================


def test_fanout_dag(pipeline_dir: pathlib.Path) -> None:
    """Fan-out DAG: A -> B, A -> C, A -> D (one source, three consumers)."""
    (pipeline_dir / "input.txt").write_text("SOURCE")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_fanout1_stage_a, name="stage_a")
    register_test_stage(_fanout1_stage_b, name="stage_b")
    register_test_stage(_fanout1_stage_c, name="stage_c")
    register_test_stage(_fanout1_stage_d, name="stage_d")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # A must run first
    assert execution_log[0] == "a"

    # B, C, D can run in any order
    assert set(execution_log[1:]) == {"b", "c", "d"}


def test_fanout_dag_wide(pipeline_dir: pathlib.Path) -> None:
    """Wide fan-out: A -> B, C, D, E, F (five consumers)."""
    consumer_names = ["b", "c", "d", "e", "f"]

    (pipeline_dir / "input.txt").write_text("1")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_fanout2_root_stage, name="root_stage")
    register_test_stage(_fanout2_consumer_b, name="consumer_b")
    register_test_stage(_fanout2_consumer_c, name="consumer_c")
    register_test_stage(_fanout2_consumer_d, name="consumer_d")
    register_test_stage(_fanout2_consumer_e, name="consumer_e")
    register_test_stage(_fanout2_consumer_f, name="consumer_f")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")
    assert execution_log[0] == "a"
    assert set(execution_log[1:]) == set(consumer_names)


# =============================================================================
# Fan-in DAG: Many stages feed one
#
#    A  B  C
#     \ | /
#       D
# =============================================================================


def test_fanin_dag(pipeline_dir: pathlib.Path) -> None:
    """Fan-in DAG: A -> D, B -> D, C -> D (three sources, one consumer)."""
    # Create separate input files for each source
    (pipeline_dir / "input_a.txt").write_text("A_INPUT")
    (pipeline_dir / "input_b.txt").write_text("B_INPUT")
    (pipeline_dir / "input_c.txt").write_text("C_INPUT")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_fanin1_stage_a, name="stage_a")
    register_test_stage(_fanin1_stage_b, name="stage_b")
    register_test_stage(_fanin1_stage_c, name="stage_c")
    register_test_stage(_fanin1_stage_d, name="stage_d")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # A, B, C can run in any order, but D must be last
    assert execution_log[-1] == "d"
    assert set(execution_log[:-1]) == {"a", "b", "c"}

    # Verify final output
    assert (pipeline_dir / "d.txt").read_text() == "A_OUT+B_OUT+C_OUT"


def test_fanin_dag_with_computation(pipeline_dir: pathlib.Path) -> None:
    """Fan-in where D computes sum of all inputs.

    A=10, B=20, C=30 -> D=60
    """
    (pipeline_dir / "input_a.txt").write_text("10")
    (pipeline_dir / "input_b.txt").write_text("20")
    (pipeline_dir / "input_c.txt").write_text("30")

    register_test_stage(_fanin2_compute_a, name="compute_a")
    register_test_stage(_fanin2_compute_b, name="compute_b")
    register_test_stage(_fanin2_compute_c, name="compute_c")
    register_test_stage(_fanin2_compute_sum, name="compute_sum")

    executor.run()

    assert (pipeline_dir / "sum.txt").read_text() == "60"


# =============================================================================
# Complex DAG: Combination of patterns
#
#       A
#      / \
#     B   C
#     |   |
#     D   E
#      \ /
#       F
# =============================================================================


def test_complex_dag_tree_then_diamond(pipeline_dir: pathlib.Path) -> None:
    r"""Complex DAG combining tree and diamond patterns.

         A
        / \
       B   C
       |   |
       D   E
        \ /
         F
    """
    (pipeline_dir / "input.txt").write_text("X")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_complex1_stage_a, name="stage_a")
    register_test_stage(_complex1_stage_b, name="stage_b")
    register_test_stage(_complex1_stage_c, name="stage_c")
    register_test_stage(_complex1_stage_d, name="stage_d")
    register_test_stage(_complex1_stage_e, name="stage_e")
    register_test_stage(_complex1_stage_f, name="stage_f")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # Verify order constraints
    ai = execution_log.index("a")
    bi = execution_log.index("b")
    ci = execution_log.index("c")
    di = execution_log.index("d")
    ei = execution_log.index("e")
    fi = execution_log.index("f")

    assert ai < bi and ai < ci, "A must run before B and C"
    assert bi < di, "B must run before D"
    assert ci < ei, "C must run before E"
    assert di < fi and ei < fi, "D and E must run before F"

    assert (pipeline_dir / "f.txt").read_text() == "F(D,E)"


def test_complex_dag_multiple_diamonds(pipeline_dir: pathlib.Path) -> None:
    r"""Two diamonds sharing a common root.

           A
          /|\
         B C D
         |X| |
         E F G
          \|/
           H
    """
    (pipeline_dir / "input.txt").write_text("0")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_complex2_root_a, name="root_a")
    register_test_stage(_complex2_mid_b, name="mid_b")
    register_test_stage(_complex2_mid_c, name="mid_c")
    register_test_stage(_complex2_mid_d, name="mid_d")
    register_test_stage(_complex2_lower_e, name="lower_e")
    register_test_stage(_complex2_lower_f, name="lower_f")
    register_test_stage(_complex2_lower_g, name="lower_g")
    register_test_stage(_complex2_final_h, name="final_h")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # A must be first, H must be last
    assert execution_log[0] == "a"
    assert execution_log[-1] == "h"

    # Verify complex ordering constraints
    assert execution_log.index("a") < execution_log.index("b")
    assert execution_log.index("a") < execution_log.index("c")
    assert execution_log.index("a") < execution_log.index("d")
    assert execution_log.index("b") < execution_log.index("e")
    assert execution_log.index("c") < execution_log.index("e")
    assert execution_log.index("c") < execution_log.index("f")
    assert execution_log.index("d") < execution_log.index("f")
    assert execution_log.index("d") < execution_log.index("g")
    assert execution_log.index("e") < execution_log.index("h")
    assert execution_log.index("f") < execution_log.index("h")
    assert execution_log.index("g") < execution_log.index("h")

    # Verify final output proves all paths executed correctly
    result = (pipeline_dir / "h.txt").read_text()
    assert result == "H[E(BC),F(CD),G(D)]"


# =============================================================================
# Edge cases
# =============================================================================


def test_single_stage_dag(pipeline_dir: pathlib.Path) -> None:
    """Single stage with no dependencies on other stages."""
    (pipeline_dir / "input.txt").write_text("DATA")

    register_test_stage(_single_only_stage, name="only_stage")

    results = executor.run()

    assert results["only_stage"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "PROCESSED:DATA"


def test_disconnected_dags(pipeline_dir: pathlib.Path) -> None:
    """Two independent pipelines in same registry.

    Pipeline 1: A -> B
    Pipeline 2: X -> Y (completely independent)
    """
    (pipeline_dir / "input_a.txt").write_text("A")
    (pipeline_dir / "input_x.txt").write_text("X")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    register_test_stage(_disconn_stage_a, name="stage_a")
    register_test_stage(_disconn_stage_b, name="stage_b")
    register_test_stage(_disconn_stage_x, name="stage_x")
    register_test_stage(_disconn_stage_y, name="stage_y")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # All stages ran
    assert set(execution_log) == {"a", "b", "x", "y"}

    # Each pipeline maintains internal order
    assert execution_log.index("a") < execution_log.index("b")
    assert execution_log.index("x") < execution_log.index("y")

    # Both outputs correct
    assert (pipeline_dir / "b.txt").read_text() == "B_OUT"
    assert (pipeline_dir / "y.txt").read_text() == "Y_OUT"
