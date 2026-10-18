"""SKILL.md's "Common errors" table quotes the messages users actually see.

Each case builds the mistake a row describes, runs the CLI, and checks three things: the row's
quoted text is in SKILL.md, that quote matches the concrete message the case expects (with
`<placeholder>` and quoted one-letter names such as `'x'` as wildcards), and the command prints
that message, failing unless the row is a warning. A final test checks that every row of the
table has a case.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli

if TYPE_CHECKING:
    import click.testing

_SKILL = pathlib.Path(__file__).parents[4] / "skills" / "writing-pivot-stages" / "SKILL.md"

_HEAD = "import pandas as pd\n\nfrom pivot import Pipeline, loaders, stage\n\n\n"


@dataclasses.dataclass(frozen=True)
class _Case:
    doc: str
    """Text quoted in the table row."""
    observed: str
    """What the command prints for this case."""
    files: dict[str, str]
    args: tuple[str, ...] = ("list",)
    cwd: str = "."
    fails: bool = True
    """False for a warning: the command prints the message and still exits 0."""


_CASES = {
    "removed-marker-import": _Case(
        doc="`cannot import name 'Out' from 'pivot'`",
        observed="cannot import name 'Out' from 'pivot'",
        files={"pipeline.py": "from pivot import Out\n"},
    ),
    "removed-marker-attribute": _Case(
        doc="`module 'pivot' has no attribute 'Out'`",
        observed="module 'pivot' has no attribute 'Out'",
        files={"pipeline.py": "import pivot\n\npivot.Out\n"},
    ),
    "no-pipeline-variable": _Case(
        doc="`does not define a 'pipeline' variable`",
        observed="does not define a 'pipeline' variable",
        files={
            "pipeline.py": _HEAD
            + "@stage\ndef s(raw: pd.DataFrame) -> pd.DataFrame:\n    return raw\n\n\n"
            + 'with Pipeline("demo") as p:\n    s(p.input("t.csv"))\n'
        },
    ),
    "sibling-import-in-subdirectory": _Case(
        doc="`No module named '<yours>'`",
        observed="No module named 'sibling_settings'",
        files={
            "report/sibling_settings.py": "X = 1\n",
            "report/pipeline.py": "from sibling_settings import X\n"
            + _HEAD
            + 'with Pipeline("report") as pipeline:\n    pass\n',
        },
        cwd="report",
    ),
    "plain-argument": _Case(
        doc="`parameter 'x' has unsupported type`",
        observed="parameter 'factor' has unsupported type 'int'",
        files={
            "pipeline.py": _HEAD
            + "@stage\ndef s(raw: pd.DataFrame, factor) -> pd.DataFrame:\n    return raw\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    s(pipeline.input("t.csv"), factor=3)\n'
        },
    ),
    "multi-output-handle": _Case(
        doc="`received handle from multi-output stage`",
        observed="received handle from multi-output stage 'demo/two' without selecting an output",
        files={
            "pipeline.py": "from typing import TypedDict\n\n"
            + _HEAD
            + "class Two(TypedDict):\n    a: pd.DataFrame\n    b: pd.DataFrame\n\n\n"
            + '@stage\ndef two(raw: pd.DataFrame) -> Two:\n    return {"a": raw, "b": raw}\n\n\n'
            + "@stage\ndef use(x: pd.DataFrame) -> pd.DataFrame:\n    return x\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    use(two(pipeline.input("t.csv")))\n'
        },
    ),
    "unknown-return-format": _Case(
        doc="`Cannot infer serialization format`",
        observed="Cannot infer serialization format for type int",
        files={
            "pipeline.py": _HEAD
            + "@stage\ndef s(raw: pd.DataFrame) -> int:\n    return 1\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    s(pipeline.input("t.csv"))\n'
        },
    ),
    "module-level-list": _Case(
        doc="`captures non-scalar variable 'X'`",
        observed="captures non-scalar variable 'COLS' (type: list)",
        files={
            "pipeline.py": _HEAD
            + 'COLS = ["a"]\n\n\n@stage\ndef s(raw: pd.DataFrame) -> pd.DataFrame:\n'
            + "    return raw[COLS]\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    s(pipeline.input("t.csv"))\n'
        },
        args=("repro",),
    ),
    "missing-input": _Case(
        doc="`depends on input '<name>' (path data/raw/<name>), which does not exist`",
        observed="depends on input 'left.csv' (path data/raw/left.csv), which does not exist",
        files={
            "pipeline.py": _HEAD
            + "@stage\ndef s(raw: pd.DataFrame) -> pd.DataFrame:\n    return raw\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    s(pipeline.input("left.csv"))\n'
        },
        args=("repro",),
    ),
    "no-return-annotation": _Case(
        doc="`must have a return type annotation`",
        observed="Stage function s must have a return type annotation",
        files={
            "pipeline.py": _HEAD
            + "@stage\ndef s(raw: pd.DataFrame):\n    return raw\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    s(pipeline.input("t.csv"))\n'
        },
    ),
    "variant-with-space": _Case(
        doc="`Invalid variant name 'big one': use only letters, digits and the characters _ . -`",
        observed="Invalid variant name 'big one': use only letters, digits and the characters _ . -",
        files={
            "pipeline.py": _HEAD
            + "@stage\ndef s(raw: pd.DataFrame) -> pd.DataFrame:\n    return raw\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    with pipeline.variant("big one"):\n'
            + '        s(pipeline.input("t.csv"))\n'
        },
    ),
    "directory-not-a-dict": _Case(
        doc="`is tagged directory but its type is <type>`",
        observed="the return value is tagged directory but its type is DataFrame",
        files={
            "pipeline.py": "from typing import Annotated\n\n"
            + "import pandas as pd\n\nfrom pivot import Pipeline, directory, stage\n\n\n"
            + "@stage\ndef s(raw: pd.DataFrame) -> Annotated[pd.DataFrame, directory]:\n"
            + "    return raw\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    s(pipeline.input("t.csv"))\n'
        },
    ),
    "write-only-consumer": _Case(
        doc="`whose loader <loader> cannot read it back`",
        observed="whose loader MatplotlibFigure cannot read it back",
        files={
            "pipeline.py": "from typing import Annotated\n\n"
            + "import matplotlib.figure\nimport pandas as pd\n\n"
            + "from pivot import Pipeline, plot, stage\n\n\n"
            + "@stage\ndef chart(raw: pd.DataFrame) -> Annotated[matplotlib.figure.Figure, plot]:\n"
            + "    return matplotlib.figure.Figure()\n\n\n"
            + "@stage\ndef use(chart: object) -> dict:\n    return {}\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    use(chart(pipeline.input("t.csv")))\n'
        },
    ),
    "two-inputs-one-file": _Case(
        doc="`Input binding conflict: inputs 'a' and 'b' are both bound to '<path>'`",
        observed="Input binding conflict: inputs 't.csv' and 'alias' are both bound to "
        + "'data/raw/t.csv'",
        files={
            "pipeline.py": _HEAD
            + "@stage\ndef s(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:\n"
            + "    return a\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n'
            + '    s(pipeline.input("t.csv"), pipeline.input("alias", path="data/raw/t.csv"))\n'
        },
    ),
    "two-outputs-one-path": _Case(
        doc="`Output path conflict: 'p/x/y' and 'p/x:y' both write data/p/x/y.jsonl`",
        observed="Output path conflict: 'p/x/y' and 'p/x:y' both write data/p/x/y.jsonl",
        files={
            "pipeline.py": "from typing import TypedDict\n\n"
            + _HEAD
            + "class XResult(TypedDict):\n    y: pd.DataFrame\n\n\n"
            + '@stage\ndef x(raw: pd.DataFrame) -> XResult:\n    return {"y": raw}\n\n\n'
            + "@stage\ndef y(raw: pd.DataFrame) -> pd.DataFrame:\n    return raw\n\n\n"
            + 'with Pipeline("p/x") as inner:\n    y(inner.input("t.csv"))\n\n'
            + 'with Pipeline("p") as pipeline:\n    x(pipeline.input("t.csv"))\n\n'
            + "pipeline.include(inner)\n"
        },
    ),
    "lossy-yaml-output": _Case(
        doc="`wrote its YAML output in a form Pivot cannot read back`",
        observed=(
            "WARNING: Stage 'demo/s' wrote its YAML output in a form Pivot cannot read back: "
            + "YAML output must contain plain data; got DataFrame"
        ),
        files={
            "pipeline.py": _HEAD
            + "@stage\ndef s(raw: pd.DataFrame) -> dict:\n"
            + '    return {"frame": raw}\n\n\n'
            + 'with Pipeline("demo") as pipeline:\n    s(pipeline.input("t.csv"))\n'
        },
        args=("repro",),
        fails=False,
    ),
}


_PLACEHOLDER = re.compile(r"<(?P<angle>[a-z]+)>|'(?P<letter>[A-Za-z])'")


def _quote_pattern(doc: str) -> re.Pattern[str]:
    """Compile a row's quote into a pattern; one placeholder name matches one value throughout."""
    text = doc.strip("`")
    parts = list[str]()
    seen = set[str]()
    position = 0
    for match in _PLACEHOLDER.finditer(text):
        parts.append(re.escape(text[position : match.start()]))
        if match["angle"] is not None:
            group, body, quote = f"angle_{match['angle']}", ".+?", ""
        else:
            group, body, quote = f"letter_{match['letter']}", "[^']+", "'"
        parts.append(quote + (f"(?P={group})" if group in seen else f"(?P<{group}>{body})") + quote)
        seen.add(group)
        position = match.end()
    parts.append(re.escape(text[position:]))
    return re.compile("".join(parts))


def _common_error_rows() -> list[str]:
    text = _SKILL.read_text()
    section = text.split("## Common errors", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if line.startswith("| `")]


@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_common_error_is_quoted_as_the_cli_prints_it(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
) -> None:
    assert any(case.doc in row for row in _common_error_rows()), case.doc
    assert _quote_pattern(case.doc).search(case.observed), (case.doc, case.observed)

    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "t.csv").write_text("a\n1\n2\n")
        for rel, text in case.files.items():
            (cwd / rel).parent.mkdir(parents=True, exist_ok=True)
            (cwd / rel).write_text(text)
        monkeypatch.chdir(cwd / case.cwd)

        result = runner.invoke(cli.cli, list(case.args))

    assert (result.exit_code != 0) == case.fails, result.output
    # The console wraps long lines; compare the words.
    assert " ".join(case.observed.split()) in " ".join(result.output.split()), result.output


@pytest.mark.parametrize(
    ("doc", "observed"),
    [
        pytest.param(
            "`closure captures mutable variable 'X'`",
            "captures non-scalar variable 'COLS' (type: list)",
            id="stale-wording",
        ),
        pytest.param(
            "`depends on input '<name>' (path data/raw/<name>), which does not exist`",
            "depends on input 'a.csv' (path data/raw/b.csv), which does not exist",
            id="one-placeholder-two-values",
        ),
        pytest.param(
            "`Invalid variant name 'big one'`",
            "Invalid stage name 'big one'",
            id="literal-mismatch",
        ),
        pytest.param(
            "`parameter 'x' has unsupported type`",
            "parameter factor has unsupported type 'int'",
            id="quotes-are-literal",
        ),
    ],
)
def test_quote_pattern_rejects_a_misquote(doc: str, observed: str) -> None:
    """A quote that a row and its case agree on still fails when the CLI says something else."""
    assert _quote_pattern(doc).search(observed) is None


def test_every_common_error_row_has_a_case() -> None:
    rows = _common_error_rows()
    assert rows, f"No 'Common errors' table rows in {_SKILL}"
    uncovered = [row for row in rows if not any(case.doc in row for case in _CASES.values())]
    assert uncovered == []
