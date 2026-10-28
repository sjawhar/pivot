"""A directory output may contain an ordinary file whose name ends in `.pvt`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from conftest import isolated_pivot_dir
from pivot import cli

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner


_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"notes.pvt": "ordinary stage output", "value.txt": "v"}


with Pipeline("b") as pipeline:
    tree()
"""


def test_directory_output_keeps_an_ordinary_pvt_named_file_across_reruns(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A `.pvt` suffix is reserved only for a valid Pivot sidecar, not an arbitrary output name."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PIPELINE)

        first = runner.invoke(cli.cli, ["repro"])
        second = runner.invoke(cli.cli, ["repro", "--force"])

        assert first.exit_code == 0, first.output
        assert second.exit_code == 0, second.output
        assert (root / "data" / "b" / "tree" / "notes.pvt").read_text() == "ordinary stage output"
