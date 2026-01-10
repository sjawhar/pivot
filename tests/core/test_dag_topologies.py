import pathlib

import pytest

from pivot import executor, project
from pivot.registry import stage


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)
    return tmp_path


# =============================================================================
# Linear DAG: A → B → C
# =============================================================================


def test_linear_dag_three_stages(pipeline_dir: pathlib.Path) -> None:
    """Linear DAG A → B → C executes in correct order.

    Each stage appends to the data, creating a chain that proves order.
    If any stage runs out of order, assertions will fail.
    """
    (pipeline_dir / "input.txt").write_text("START")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("a.txt").write_text(f"{data}->A")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        data = pathlib.Path("a.txt").read_text()
        assert data == "START->A", f"Expected 'START->A', got '{data}'"
        pathlib.Path("b.txt").write_text(f"{data}->B")

    @stage(deps=["b.txt"], outs=["c.txt"])
    def stage_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        data = pathlib.Path("b.txt").read_text()
        assert data == "START->A->B", f"Expected 'START->A->B', got '{data}'"
        pathlib.Path("c.txt").write_text(f"{data}->C")

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
    """Longer linear DAG: A → B → C → D → E."""
    (pipeline_dir / "input.txt").write_text("0")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input.txt"], outs=["stage1.txt"])
    def stage1() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("1\n")
        n = int(pathlib.Path("input.txt").read_text())
        pathlib.Path("stage1.txt").write_text(str(n + 1))

    @stage(deps=["stage1.txt"], outs=["stage2.txt"])
    def stage2() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("2\n")
        n = int(pathlib.Path("stage1.txt").read_text())
        assert n == 1, f"stage1 must run first, got {n}"
        pathlib.Path("stage2.txt").write_text(str(n + 1))

    @stage(deps=["stage2.txt"], outs=["stage3.txt"])
    def stage3() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("3\n")
        n = int(pathlib.Path("stage2.txt").read_text())
        assert n == 2, f"stage2 must run first, got {n}"
        pathlib.Path("stage3.txt").write_text(str(n + 1))

    @stage(deps=["stage3.txt"], outs=["stage4.txt"])
    def stage4() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("4\n")
        n = int(pathlib.Path("stage3.txt").read_text())
        assert n == 3, f"stage3 must run first, got {n}"
        pathlib.Path("stage4.txt").write_text(str(n + 1))

    @stage(deps=["stage4.txt"], outs=["stage5.txt"])
    def stage5() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("5\n")
        n = int(pathlib.Path("stage4.txt").read_text())
        assert n == 4, f"stage4 must run first, got {n}"
        pathlib.Path("stage5.txt").write_text(str(n + 1))

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
    """Tree DAG: A → B, A → C (B and C both depend on A, but not each other)."""
    (pipeline_dir / "input.txt").write_text("ROOT")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("a.txt").write_text(f"{data}->A")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        data = pathlib.Path("a.txt").read_text()
        assert "->A" in data, "stage_a must run before stage_b"
        pathlib.Path("b.txt").write_text(f"{data}->B")

    @stage(deps=["a.txt"], outs=["c.txt"])
    def stage_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        data = pathlib.Path("a.txt").read_text()
        assert "->A" in data, "stage_a must run before stage_c"
        pathlib.Path("c.txt").write_text(f"{data}->C")

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
    """Deeper tree: A → B → D, A → C → E.

         A
        / \
       B   C
       |   |
       D   E
    """
    (pipeline_dir / "input.txt").write_text("0")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        pathlib.Path("a.txt").write_text("A")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        assert pathlib.Path("a.txt").read_text() == "A"
        pathlib.Path("b.txt").write_text("B")

    @stage(deps=["a.txt"], outs=["c.txt"])
    def stage_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        assert pathlib.Path("a.txt").read_text() == "A"
        pathlib.Path("c.txt").write_text("C")

    @stage(deps=["b.txt"], outs=["d.txt"])
    def stage_d() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("d\n")
        assert pathlib.Path("b.txt").read_text() == "B"
        pathlib.Path("d.txt").write_text("D")

    @stage(deps=["c.txt"], outs=["e.txt"])
    def stage_e() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("e\n")
        assert pathlib.Path("c.txt").read_text() == "C"
        pathlib.Path("e.txt").write_text("E")

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
    """Diamond DAG: A → B → D, A → C → D.

    D depends on both B and C, which both depend on A.
    """
    (pipeline_dir / "input.txt").write_text("INPUT")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        pathlib.Path("a.txt").write_text("A_OUTPUT")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        data = pathlib.Path("a.txt").read_text()
        assert data == "A_OUTPUT", "stage_a must run before stage_b"
        pathlib.Path("b.txt").write_text("B_OUTPUT")

    @stage(deps=["a.txt"], outs=["c.txt"])
    def stage_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        data = pathlib.Path("a.txt").read_text()
        assert data == "A_OUTPUT", "stage_a must run before stage_c"
        pathlib.Path("c.txt").write_text("C_OUTPUT")

    @stage(deps=["b.txt", "c.txt"], outs=["d.txt"])
    def stage_d() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("d\n")
        b_data = pathlib.Path("b.txt").read_text()
        c_data = pathlib.Path("c.txt").read_text()
        assert b_data == "B_OUTPUT", "stage_b must run before stage_d"
        assert c_data == "C_OUTPUT", "stage_c must run before stage_d"
        pathlib.Path("d.txt").write_text(f"D({b_data}+{c_data})")

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

    @stage(deps=["input.txt"], outs=["a.txt"])
    def compute_a() -> None:
        n = int(pathlib.Path("input.txt").read_text())
        pathlib.Path("a.txt").write_text(str(n))

    @stage(deps=["a.txt"], outs=["b.txt"])
    def double_b() -> None:
        n = int(pathlib.Path("a.txt").read_text())
        pathlib.Path("b.txt").write_text(str(n * 2))  # 20

    @stage(deps=["a.txt"], outs=["c.txt"])
    def triple_c() -> None:
        n = int(pathlib.Path("a.txt").read_text())
        pathlib.Path("c.txt").write_text(str(n * 3))  # 30

    @stage(deps=["b.txt", "c.txt"], outs=["d.txt"])
    def sum_d() -> None:
        b = int(pathlib.Path("b.txt").read_text())
        c = int(pathlib.Path("c.txt").read_text())
        pathlib.Path("d.txt").write_text(str(b + c))  # 50

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
    """Fan-out DAG: A → B, A → C, A → D (one source, three consumers)."""
    (pipeline_dir / "input.txt").write_text("SOURCE")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        pathlib.Path("a.txt").write_text("A_DATA")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        data = pathlib.Path("a.txt").read_text()
        assert data == "A_DATA", "stage_a must run before stage_b"
        pathlib.Path("b.txt").write_text("B")

    @stage(deps=["a.txt"], outs=["c.txt"])
    def stage_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        data = pathlib.Path("a.txt").read_text()
        assert data == "A_DATA", "stage_a must run before stage_c"
        pathlib.Path("c.txt").write_text("C")

    @stage(deps=["a.txt"], outs=["d.txt"])
    def stage_d() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("d\n")
        data = pathlib.Path("a.txt").read_text()
        assert data == "A_DATA", "stage_a must run before stage_d"
        pathlib.Path("d.txt").write_text("D")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # A must run first
    assert execution_log[0] == "a"

    # B, C, D can run in any order
    assert set(execution_log[1:]) == {"b", "c", "d"}


def test_fanout_dag_wide(pipeline_dir: pathlib.Path) -> None:
    """Wide fan-out: A → B, C, D, E, F (five consumers)."""
    consumer_names = ["b", "c", "d", "e", "f"]

    (pipeline_dir / "input.txt").write_text("1")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input.txt"], outs=["a.txt"])
    def root_stage() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        pathlib.Path("a.txt").write_text("ROOT")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def consumer_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        assert pathlib.Path("a.txt").read_text() == "ROOT"
        pathlib.Path("b.txt").write_text("B")

    @stage(deps=["a.txt"], outs=["c.txt"])
    def consumer_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        assert pathlib.Path("a.txt").read_text() == "ROOT"
        pathlib.Path("c.txt").write_text("C")

    @stage(deps=["a.txt"], outs=["d.txt"])
    def consumer_d() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("d\n")
        assert pathlib.Path("a.txt").read_text() == "ROOT"
        pathlib.Path("d.txt").write_text("D")

    @stage(deps=["a.txt"], outs=["e.txt"])
    def consumer_e() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("e\n")
        assert pathlib.Path("a.txt").read_text() == "ROOT"
        pathlib.Path("e.txt").write_text("E")

    @stage(deps=["a.txt"], outs=["f.txt"])
    def consumer_f() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("f\n")
        assert pathlib.Path("a.txt").read_text() == "ROOT"
        pathlib.Path("f.txt").write_text("F")

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
    """Fan-in DAG: A → D, B → D, C → D (three sources, one consumer)."""
    # Create separate input files for each source
    (pipeline_dir / "input_a.txt").write_text("A_INPUT")
    (pipeline_dir / "input_b.txt").write_text("B_INPUT")
    (pipeline_dir / "input_c.txt").write_text("C_INPUT")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input_a.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        pathlib.Path("a.txt").write_text("A_OUT")

    @stage(deps=["input_b.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        pathlib.Path("b.txt").write_text("B_OUT")

    @stage(deps=["input_c.txt"], outs=["c.txt"])
    def stage_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        pathlib.Path("c.txt").write_text("C_OUT")

    @stage(deps=["a.txt", "b.txt", "c.txt"], outs=["d.txt"])
    def stage_d() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("d\n")
        a = pathlib.Path("a.txt").read_text()
        b = pathlib.Path("b.txt").read_text()
        c = pathlib.Path("c.txt").read_text()
        assert a == "A_OUT", "stage_a must run before stage_d"
        assert b == "B_OUT", "stage_b must run before stage_d"
        assert c == "C_OUT", "stage_c must run before stage_d"
        pathlib.Path("d.txt").write_text(f"{a}+{b}+{c}")

    executor.run()

    execution_log = log_file.read_text().strip().split("\n")

    # A, B, C can run in any order, but D must be last
    assert execution_log[-1] == "d"
    assert set(execution_log[:-1]) == {"a", "b", "c"}

    # Verify final output
    assert (pipeline_dir / "d.txt").read_text() == "A_OUT+B_OUT+C_OUT"


def test_fanin_dag_with_computation(pipeline_dir: pathlib.Path) -> None:
    """Fan-in where D computes sum of all inputs.

    A=10, B=20, C=30 → D=60
    """
    (pipeline_dir / "input_a.txt").write_text("10")
    (pipeline_dir / "input_b.txt").write_text("20")
    (pipeline_dir / "input_c.txt").write_text("30")

    @stage(deps=["input_a.txt"], outs=["a.txt"])
    def compute_a() -> None:
        n = int(pathlib.Path("input_a.txt").read_text())
        pathlib.Path("a.txt").write_text(str(n))

    @stage(deps=["input_b.txt"], outs=["b.txt"])
    def compute_b() -> None:
        n = int(pathlib.Path("input_b.txt").read_text())
        pathlib.Path("b.txt").write_text(str(n))

    @stage(deps=["input_c.txt"], outs=["c.txt"])
    def compute_c() -> None:
        n = int(pathlib.Path("input_c.txt").read_text())
        pathlib.Path("c.txt").write_text(str(n))

    @stage(deps=["a.txt", "b.txt", "c.txt"], outs=["sum.txt"])
    def compute_sum() -> None:
        a = int(pathlib.Path("a.txt").read_text())
        b = int(pathlib.Path("b.txt").read_text())
        c = int(pathlib.Path("c.txt").read_text())
        pathlib.Path("sum.txt").write_text(str(a + b + c))

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

    @stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        pathlib.Path("a.txt").write_text("A")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        assert pathlib.Path("a.txt").read_text() == "A"
        pathlib.Path("b.txt").write_text("B")

    @stage(deps=["a.txt"], outs=["c.txt"])
    def stage_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        assert pathlib.Path("a.txt").read_text() == "A"
        pathlib.Path("c.txt").write_text("C")

    @stage(deps=["b.txt"], outs=["d.txt"])
    def stage_d() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("d\n")
        assert pathlib.Path("b.txt").read_text() == "B"
        pathlib.Path("d.txt").write_text("D")

    @stage(deps=["c.txt"], outs=["e.txt"])
    def stage_e() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("e\n")
        assert pathlib.Path("c.txt").read_text() == "C"
        pathlib.Path("e.txt").write_text("E")

    @stage(deps=["d.txt", "e.txt"], outs=["f.txt"])
    def stage_f() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("f\n")
        d = pathlib.Path("d.txt").read_text()
        e = pathlib.Path("e.txt").read_text()
        assert d == "D", "stage_d must run before stage_f"
        assert e == "E", "stage_e must run before stage_f"
        pathlib.Path("f.txt").write_text(f"F({d},{e})")

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

    @stage(deps=["input.txt"], outs=["a.txt"])
    def root_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        pathlib.Path("a.txt").write_text("1")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def mid_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        pathlib.Path("b.txt").write_text("B")

    @stage(deps=["a.txt"], outs=["c.txt"])
    def mid_c() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("c\n")
        pathlib.Path("c.txt").write_text("C")

    @stage(deps=["a.txt"], outs=["d.txt"])
    def mid_d() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("d\n")
        pathlib.Path("d.txt").write_text("D")

    @stage(deps=["b.txt", "c.txt"], outs=["e.txt"])
    def lower_e() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("e\n")
        b = pathlib.Path("b.txt").read_text()
        c = pathlib.Path("c.txt").read_text()
        pathlib.Path("e.txt").write_text(f"E({b}{c})")

    @stage(deps=["c.txt", "d.txt"], outs=["f.txt"])
    def lower_f() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("f\n")
        c = pathlib.Path("c.txt").read_text()
        d = pathlib.Path("d.txt").read_text()
        pathlib.Path("f.txt").write_text(f"F({c}{d})")

    @stage(deps=["d.txt"], outs=["g.txt"])
    def lower_g() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("g\n")
        d = pathlib.Path("d.txt").read_text()
        pathlib.Path("g.txt").write_text(f"G({d})")

    @stage(deps=["e.txt", "f.txt", "g.txt"], outs=["h.txt"])
    def final_h() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("h\n")
        e = pathlib.Path("e.txt").read_text()
        f_val = pathlib.Path("f.txt").read_text()
        g = pathlib.Path("g.txt").read_text()
        pathlib.Path("h.txt").write_text(f"H[{e},{f_val},{g}]")

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

    @stage(deps=["input.txt"], outs=["output.txt"])
    def only_stage() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(f"PROCESSED:{data}")

    results = executor.run()

    assert results["only_stage"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "PROCESSED:DATA"


def test_disconnected_dags(pipeline_dir: pathlib.Path) -> None:
    """Two independent pipelines in same registry.

    Pipeline 1: A → B
    Pipeline 2: X → Y (completely independent)
    """
    (pipeline_dir / "input_a.txt").write_text("A")
    (pipeline_dir / "input_x.txt").write_text("X")
    log_file = pipeline_dir / "execution_log.txt"
    log_file.write_text("")

    @stage(deps=["input_a.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("a\n")
        pathlib.Path("a.txt").write_text("A_OUT")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("b\n")
        assert pathlib.Path("a.txt").read_text() == "A_OUT"
        pathlib.Path("b.txt").write_text("B_OUT")

    @stage(deps=["input_x.txt"], outs=["x.txt"])
    def stage_x() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("x\n")
        pathlib.Path("x.txt").write_text("X_OUT")

    @stage(deps=["x.txt"], outs=["y.txt"])
    def stage_y() -> None:
        with open("execution_log.txt", "a") as f:
            f.write("y\n")
        assert pathlib.Path("x.txt").read_text() == "X_OUT"
        pathlib.Path("y.txt").write_text("Y_OUT")

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
