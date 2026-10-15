"""Execution order across DAG shapes, run through the compositional API.

Each stage appends its letter to ``execution_log.txt`` (in the project root, the
worker's cwd) and chains data through its handles, so a stage that runs out of
order either breaks an assertion inside the stage or produces the wrong output.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import executor
from tests import helpers

if TYPE_CHECKING:
    from pivot.registry import PipelineLike


def _helper_log(letter: str) -> None:
    with pathlib.Path("execution_log.txt").open("a") as f:
        f.write(f"{letter}\n")


def _helper_read_log(pipeline_dir: pathlib.Path) -> list[str]:
    return (pipeline_dir / "execution_log.txt").read_text().strip().split("\n")


def _helper_output(stage: str) -> str:
    return helpers.get_output_path(stage).read_text()


# =============================================================================
# Linear DAG: A -> B -> C
# =============================================================================


def _linear3_stage_a(data: str) -> str:
    _helper_log("a")
    return f"{data}->A"


def _linear3_stage_b(a: str) -> str:
    _helper_log("b")
    assert a == "START->A", f"Expected 'START->A', got '{a}'"
    return f"{a}->B"


def _linear3_stage_c(b: str) -> str:
    _helper_log("c")
    assert b == "START->A->B", f"Expected 'START->A->B', got '{b}'"
    return f"{b}->C"


def test_linear_dag_three_stages(pipeline_dir: pathlib.Path, test_pipeline: PipelineLike) -> None:
    """Linear DAG A -> B -> C executes in correct order."""
    (pipeline_dir / "input.txt").write_text("START")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_linear3_stage_a, name="stage_a", deps={"data": "input.txt"})
    b = helpers.register_test_stage(_linear3_stage_b, name="stage_b", deps={"a": a})
    helpers.register_test_stage(_linear3_stage_c, name="stage_c", deps={"b": b})

    results = executor.run(pipeline=test_pipeline)

    assert all(r["status"] == "ran" for r in results.values())
    assert _helper_read_log(pipeline_dir) == ["a", "b", "c"]
    assert _helper_output("stage_c") == "START->A->B->C"


# =============================================================================
# Tree DAG: A branches to B and C (no convergence)
# =============================================================================


def _tree1_stage_a(data: str) -> str:
    _helper_log("a")
    return f"{data}->A"


def _tree1_stage_b(a: str) -> str:
    _helper_log("b")
    assert "->A" in a, "stage_a must run before stage_b"
    return f"{a}->B"


def _tree1_stage_c(a: str) -> str:
    _helper_log("c")
    assert "->A" in a, "stage_a must run before stage_c"
    return f"{a}->C"


def test_tree_dag_one_root_two_children(
    pipeline_dir: pathlib.Path, test_pipeline: PipelineLike
) -> None:
    """Tree DAG: A -> B, A -> C (B and C both depend on A, but not each other)."""
    (pipeline_dir / "input.txt").write_text("ROOT")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_tree1_stage_a, name="stage_a", deps={"data": "input.txt"})
    helpers.register_test_stage(_tree1_stage_b, name="stage_b", deps={"a": a})
    helpers.register_test_stage(_tree1_stage_c, name="stage_c", deps={"a": a})

    executor.run(pipeline=test_pipeline)

    execution_log = _helper_read_log(pipeline_dir)
    assert execution_log[0] == "a", "stage_a must run first"
    assert set(execution_log[1:]) == {"b", "c"}
    assert _helper_output("stage_b") == "ROOT->A->B"
    assert _helper_output("stage_c") == "ROOT->A->C"


def _tree2_stage_a(data: str) -> str:
    _ = data
    _helper_log("a")
    return "A"


def _tree2_stage_b(a: str) -> str:
    _helper_log("b")
    assert a == "A"
    return "B"


def _tree2_stage_c(a: str) -> str:
    _helper_log("c")
    assert a == "A"
    return "C"


def _tree2_stage_d(b: str) -> str:
    _helper_log("d")
    assert b == "B"
    return "D"


def _tree2_stage_e(c: str) -> str:
    _helper_log("e")
    assert c == "C"
    return "E"


def test_tree_dag_deeper(pipeline_dir: pathlib.Path, test_pipeline: PipelineLike) -> None:
    """Deeper tree: A -> B -> D, A -> C -> E."""
    (pipeline_dir / "input.txt").write_text("0")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_tree2_stage_a, name="stage_a", deps={"data": "input.txt"})
    b = helpers.register_test_stage(_tree2_stage_b, name="stage_b", deps={"a": a})
    c = helpers.register_test_stage(_tree2_stage_c, name="stage_c", deps={"a": a})
    helpers.register_test_stage(_tree2_stage_d, name="stage_d", deps={"b": b})
    helpers.register_test_stage(_tree2_stage_e, name="stage_e", deps={"c": c})

    results = executor.run(pipeline=test_pipeline)

    assert all(r["status"] == "ran" for r in results.values()), results
    execution_log = _helper_read_log(pipeline_dir)
    assert execution_log.index("a") < execution_log.index("b")
    assert execution_log.index("a") < execution_log.index("c")
    assert execution_log.index("b") < execution_log.index("d")
    assert execution_log.index("c") < execution_log.index("e")


# =============================================================================
# Diamond DAG: A -> B -> D, A -> C -> D
# =============================================================================


def _diamond1_stage_a(data: str) -> str:
    _ = data
    _helper_log("a")
    return "A_OUTPUT"


def _diamond1_stage_b(a: str) -> str:
    _helper_log("b")
    assert a == "A_OUTPUT", "stage_a must run before stage_b"
    return "B_OUTPUT"


def _diamond1_stage_c(a: str) -> str:
    _helper_log("c")
    assert a == "A_OUTPUT", "stage_a must run before stage_c"
    return "C_OUTPUT"


def _diamond1_stage_d(b: str, c: str) -> str:
    _helper_log("d")
    assert b == "B_OUTPUT", "stage_b must run before stage_d"
    assert c == "C_OUTPUT", "stage_c must run before stage_d"
    return f"D({b}+{c})"


def test_diamond_dag(pipeline_dir: pathlib.Path, test_pipeline: PipelineLike) -> None:
    """Diamond DAG: D depends on both B and C, which both depend on A."""
    (pipeline_dir / "input.txt").write_text("INPUT")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_diamond1_stage_a, name="stage_a", deps={"data": "input.txt"})
    b = helpers.register_test_stage(_diamond1_stage_b, name="stage_b", deps={"a": a})
    c = helpers.register_test_stage(_diamond1_stage_c, name="stage_c", deps={"a": a})
    helpers.register_test_stage(_diamond1_stage_d, name="stage_d", deps={"b": b, "c": c})

    executor.run(pipeline=test_pipeline)

    execution_log = _helper_read_log(pipeline_dir)
    assert execution_log.index("a") < execution_log.index("b")
    assert execution_log.index("a") < execution_log.index("c")
    assert execution_log.index("b") < execution_log.index("d")
    assert execution_log.index("c") < execution_log.index("d")
    assert execution_log[-1] == "d"
    assert _helper_output("stage_d") == "D(B_OUTPUT+C_OUTPUT)"


def _diamond2_compute_a(data: str) -> str:
    return str(int(data))


def _diamond2_double_b(a: str) -> str:
    return str(int(a) * 2)


def _diamond2_triple_c(a: str) -> str:
    return str(int(a) * 3)


def _diamond2_sum_d(b: str, c: str) -> str:
    return str(int(b) + int(c))


def test_diamond_dag_with_shared_data(
    pipeline_dir: pathlib.Path, test_pipeline: PipelineLike
) -> None:
    """A produces a number, B doubles it, C triples it, D sums both."""
    (pipeline_dir / "input.txt").write_text("10")

    a = helpers.register_test_stage(
        _diamond2_compute_a, name="compute_a", deps={"data": "input.txt"}
    )
    b = helpers.register_test_stage(_diamond2_double_b, name="double_b", deps={"a": a})
    c = helpers.register_test_stage(_diamond2_triple_c, name="triple_c", deps={"a": a})
    helpers.register_test_stage(_diamond2_sum_d, name="sum_d", deps={"b": b, "c": c})

    executor.run(pipeline=test_pipeline)

    assert _helper_output("sum_d") == "50"


# =============================================================================
# Fan-out DAG: One stage feeds many
# =============================================================================


def _fanout1_stage_a(data: str) -> str:
    _ = data
    _helper_log("a")
    return "A_DATA"


def _fanout1_stage_b(a: str) -> str:
    _helper_log("b")
    assert a == "A_DATA", "stage_a must run before stage_b"
    return "B"


def _fanout1_stage_c(a: str) -> str:
    _helper_log("c")
    assert a == "A_DATA", "stage_a must run before stage_c"
    return "C"


def _fanout1_stage_d(a: str) -> str:
    _helper_log("d")
    assert a == "A_DATA", "stage_a must run before stage_d"
    return "D"


def test_fanout_dag(pipeline_dir: pathlib.Path, test_pipeline: PipelineLike) -> None:
    """Fan-out DAG: A -> B, A -> C, A -> D (one source, three consumers)."""
    (pipeline_dir / "input.txt").write_text("SOURCE")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_fanout1_stage_a, name="stage_a", deps={"data": "input.txt"})
    helpers.register_test_stage(_fanout1_stage_b, name="stage_b", deps={"a": a})
    helpers.register_test_stage(_fanout1_stage_c, name="stage_c", deps={"a": a})
    helpers.register_test_stage(_fanout1_stage_d, name="stage_d", deps={"a": a})

    executor.run(pipeline=test_pipeline)

    execution_log = _helper_read_log(pipeline_dir)
    assert execution_log[0] == "a"
    assert set(execution_log[1:]) == {"b", "c", "d"}


def _fanout2_root_stage(data: str) -> str:
    _ = data
    _helper_log("a")
    return "ROOT"


def _fanout2_consumer_b(a: str) -> str:
    _helper_log("b")
    assert a == "ROOT"
    return "B"


def _fanout2_consumer_c(a: str) -> str:
    _helper_log("c")
    assert a == "ROOT"
    return "C"


def _fanout2_consumer_d(a: str) -> str:
    _helper_log("d")
    assert a == "ROOT"
    return "D"


def _fanout2_consumer_e(a: str) -> str:
    _helper_log("e")
    assert a == "ROOT"
    return "E"


def _fanout2_consumer_f(a: str) -> str:
    _helper_log("f")
    assert a == "ROOT"
    return "F"


def test_fanout_dag_wide(pipeline_dir: pathlib.Path, test_pipeline: PipelineLike) -> None:
    """Wide fan-out: A -> B, C, D, E, F (five consumers)."""
    (pipeline_dir / "input.txt").write_text("1")
    (pipeline_dir / "execution_log.txt").write_text("")

    root = helpers.register_test_stage(
        _fanout2_root_stage, name="root_stage", deps={"data": "input.txt"}
    )
    helpers.register_test_stage(_fanout2_consumer_b, name="consumer_b", deps={"a": root})
    helpers.register_test_stage(_fanout2_consumer_c, name="consumer_c", deps={"a": root})
    helpers.register_test_stage(_fanout2_consumer_d, name="consumer_d", deps={"a": root})
    helpers.register_test_stage(_fanout2_consumer_e, name="consumer_e", deps={"a": root})
    helpers.register_test_stage(_fanout2_consumer_f, name="consumer_f", deps={"a": root})

    executor.run(pipeline=test_pipeline)

    execution_log = _helper_read_log(pipeline_dir)
    assert execution_log[0] == "a"
    assert set(execution_log[1:]) == {"b", "c", "d", "e", "f"}


# =============================================================================
# Fan-in DAG: Many stages feed one
# =============================================================================


def _fanin1_stage_a(data: str) -> str:
    _ = data
    _helper_log("a")
    return "A_OUT"


def _fanin1_stage_b(data: str) -> str:
    _ = data
    _helper_log("b")
    return "B_OUT"


def _fanin1_stage_c(data: str) -> str:
    _ = data
    _helper_log("c")
    return "C_OUT"


def _fanin1_stage_d(a: str, b: str, c: str) -> str:
    _helper_log("d")
    assert a == "A_OUT", "stage_a must run before stage_d"
    assert b == "B_OUT", "stage_b must run before stage_d"
    assert c == "C_OUT", "stage_c must run before stage_d"
    return f"{a}+{b}+{c}"


def test_fanin_dag(pipeline_dir: pathlib.Path, test_pipeline: PipelineLike) -> None:
    """Fan-in DAG: A -> D, B -> D, C -> D (three sources, one consumer)."""
    (pipeline_dir / "input_a.txt").write_text("A_INPUT")
    (pipeline_dir / "input_b.txt").write_text("B_INPUT")
    (pipeline_dir / "input_c.txt").write_text("C_INPUT")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_fanin1_stage_a, name="stage_a", deps={"data": "input_a.txt"})
    b = helpers.register_test_stage(_fanin1_stage_b, name="stage_b", deps={"data": "input_b.txt"})
    c = helpers.register_test_stage(_fanin1_stage_c, name="stage_c", deps={"data": "input_c.txt"})
    helpers.register_test_stage(_fanin1_stage_d, name="stage_d", deps={"a": a, "b": b, "c": c})

    executor.run(pipeline=test_pipeline)

    execution_log = _helper_read_log(pipeline_dir)
    assert execution_log[-1] == "d"
    assert set(execution_log[:-1]) == {"a", "b", "c"}
    assert _helper_output("stage_d") == "A_OUT+B_OUT+C_OUT"


def _fanin2_compute(data: str) -> str:
    return str(int(data))


def _fanin2_compute_sum(a: str, b: str, c: str) -> str:
    return str(int(a) + int(b) + int(c))


def test_fanin_dag_with_computation(
    pipeline_dir: pathlib.Path, test_pipeline: PipelineLike
) -> None:
    """Fan-in where D computes sum of all inputs: A=10, B=20, C=30 -> D=60."""
    (pipeline_dir / "input_a.txt").write_text("10")
    (pipeline_dir / "input_b.txt").write_text("20")
    (pipeline_dir / "input_c.txt").write_text("30")

    a = helpers.register_test_stage(_fanin2_compute, name="compute_a", deps={"data": "input_a.txt"})
    b = helpers.register_test_stage(_fanin2_compute, name="compute_b", deps={"data": "input_b.txt"})
    c = helpers.register_test_stage(_fanin2_compute, name="compute_c", deps={"data": "input_c.txt"})
    helpers.register_test_stage(
        _fanin2_compute_sum, name="compute_sum", deps={"a": a, "b": b, "c": c}
    )

    executor.run(pipeline=test_pipeline)

    assert _helper_output("compute_sum") == "60"


# =============================================================================
# Complex DAGs
# =============================================================================


def _complex1_stage_a(data: str) -> str:
    _ = data
    _helper_log("a")
    return "A"


def _complex1_stage_b(a: str) -> str:
    _helper_log("b")
    assert a == "A"
    return "B"


def _complex1_stage_c(a: str) -> str:
    _helper_log("c")
    assert a == "A"
    return "C"


def _complex1_stage_d(b: str) -> str:
    _helper_log("d")
    assert b == "B"
    return "D"


def _complex1_stage_e(c: str) -> str:
    _helper_log("e")
    assert c == "C"
    return "E"


def _complex1_stage_f(d: str, e: str) -> str:
    _helper_log("f")
    assert d == "D", "stage_d must run before stage_f"
    assert e == "E", "stage_e must run before stage_f"
    return f"F({d},{e})"


def test_complex_dag_tree_then_diamond(
    pipeline_dir: pathlib.Path, test_pipeline: PipelineLike
) -> None:
    """A -> B -> D -> F and A -> C -> E -> F."""
    (pipeline_dir / "input.txt").write_text("X")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_complex1_stage_a, name="stage_a", deps={"data": "input.txt"})
    b = helpers.register_test_stage(_complex1_stage_b, name="stage_b", deps={"a": a})
    c = helpers.register_test_stage(_complex1_stage_c, name="stage_c", deps={"a": a})
    d = helpers.register_test_stage(_complex1_stage_d, name="stage_d", deps={"b": b})
    e = helpers.register_test_stage(_complex1_stage_e, name="stage_e", deps={"c": c})
    helpers.register_test_stage(_complex1_stage_f, name="stage_f", deps={"d": d, "e": e})

    executor.run(pipeline=test_pipeline)

    execution_log = _helper_read_log(pipeline_dir)
    ai, bi, ci, di, ei, fi = (execution_log.index(x) for x in "abcdef")
    assert ai < bi and ai < ci, "A must run before B and C"
    assert bi < di, "B must run before D"
    assert ci < ei, "C must run before E"
    assert di < fi and ei < fi, "D and E must run before F"
    assert _helper_output("stage_f") == "F(D,E)"


def _complex2_root_a(data: str) -> str:
    _ = data
    _helper_log("a")
    return "1"


def _complex2_mid_b(a: str) -> str:
    _ = a
    _helper_log("b")
    return "B"


def _complex2_mid_c(a: str) -> str:
    _ = a
    _helper_log("c")
    return "C"


def _complex2_mid_d(a: str) -> str:
    _ = a
    _helper_log("d")
    return "D"


def _complex2_lower_e(b: str, c: str) -> str:
    _helper_log("e")
    return f"E({b}{c})"


def _complex2_lower_f(c: str, d: str) -> str:
    _helper_log("f")
    return f"F({c}{d})"


def _complex2_lower_g(d: str) -> str:
    _helper_log("g")
    return f"G({d})"


def _complex2_final_h(e: str, f: str, g: str) -> str:
    _helper_log("h")
    return f"H[{e},{f},{g}]"


def test_complex_dag_multiple_diamonds(
    pipeline_dir: pathlib.Path, test_pipeline: PipelineLike
) -> None:
    """Two diamonds sharing a common root: A -> {B,C,D} -> {E,F,G} -> H."""
    (pipeline_dir / "input.txt").write_text("0")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_complex2_root_a, name="root_a", deps={"data": "input.txt"})
    b = helpers.register_test_stage(_complex2_mid_b, name="mid_b", deps={"a": a})
    c = helpers.register_test_stage(_complex2_mid_c, name="mid_c", deps={"a": a})
    d = helpers.register_test_stage(_complex2_mid_d, name="mid_d", deps={"a": a})
    e = helpers.register_test_stage(_complex2_lower_e, name="lower_e", deps={"b": b, "c": c})
    f = helpers.register_test_stage(_complex2_lower_f, name="lower_f", deps={"c": c, "d": d})
    g = helpers.register_test_stage(_complex2_lower_g, name="lower_g", deps={"d": d})
    helpers.register_test_stage(_complex2_final_h, name="final_h", deps={"e": e, "f": f, "g": g})

    executor.run(pipeline=test_pipeline)

    execution_log = _helper_read_log(pipeline_dir)
    assert execution_log[0] == "a"
    assert execution_log[-1] == "h"
    for before, after in [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "e"),
        ("c", "e"),
        ("c", "f"),
        ("d", "f"),
        ("d", "g"),
        ("e", "h"),
        ("f", "h"),
        ("g", "h"),
    ]:
        assert execution_log.index(before) < execution_log.index(after), f"{before} < {after}"
    assert _helper_output("final_h") == "H[E(BC),F(CD),G(D)]"


# =============================================================================
# Edge cases
# =============================================================================


def _single_only_stage(data: str) -> str:
    return f"PROCESSED:{data}"


def test_single_stage_dag(pipeline_dir: pathlib.Path, test_pipeline: PipelineLike) -> None:
    """Single stage with no dependencies on other stages."""
    (pipeline_dir / "input.txt").write_text("DATA")

    helpers.register_test_stage(_single_only_stage, name="only_stage", deps={"data": "input.txt"})

    results = executor.run(pipeline=test_pipeline)

    assert results["test/only_stage"]["status"] == "ran"
    assert _helper_output("only_stage") == "PROCESSED:DATA"


def _disconn_stage_a(data: str) -> str:
    _ = data
    _helper_log("a")
    return "A_OUT"


def _disconn_stage_b(a: str) -> str:
    _helper_log("b")
    assert a == "A_OUT"
    return "B_OUT"


def _disconn_stage_x(data: str) -> str:
    _ = data
    _helper_log("x")
    return "X_OUT"


def _disconn_stage_y(x: str) -> str:
    _helper_log("y")
    assert x == "X_OUT"
    return "Y_OUT"


def test_disconnected_dags(pipeline_dir: pathlib.Path, test_pipeline: PipelineLike) -> None:
    """Two independent chains in one pipeline: A -> B and X -> Y."""
    (pipeline_dir / "input_a.txt").write_text("A")
    (pipeline_dir / "input_x.txt").write_text("X")
    (pipeline_dir / "execution_log.txt").write_text("")

    a = helpers.register_test_stage(_disconn_stage_a, name="stage_a", deps={"data": "input_a.txt"})
    helpers.register_test_stage(_disconn_stage_b, name="stage_b", deps={"a": a})
    x = helpers.register_test_stage(_disconn_stage_x, name="stage_x", deps={"data": "input_x.txt"})
    helpers.register_test_stage(_disconn_stage_y, name="stage_y", deps={"x": x})

    executor.run(pipeline=test_pipeline)

    execution_log = _helper_read_log(pipeline_dir)
    assert set(execution_log) == {"a", "b", "x", "y"}
    assert execution_log.index("a") < execution_log.index("b")
    assert execution_log.index("x") < execution_log.index("y")
    assert _helper_output("stage_b") == "B_OUT"
    assert _helper_output("stage_y") == "Y_OUT"
