"""`pivot commit` must save outputs into the cache so `pivot checkout` can restore them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import cli, compose, executor

if TYPE_CHECKING:
    import click.testing


@compose.stage
def _helper_shout(text: str) -> str:
    return text.upper()


def test_commit_after_no_commit_lets_checkout_restore_output(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    root = mock_discovery.root
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "words.txt").write_text("hello")
    with mock_discovery:
        _helper_shout(mock_discovery.input("words.txt", t=str))

    executor.run(pipeline=mock_discovery, no_commit=True)
    output = root / "data" / "test" / "_helper_shout.txt"
    assert output.read_text() == "HELLO"

    commit_result = runner.invoke(cli.cli, ["commit"])
    assert commit_result.exit_code == 0, commit_result.output

    output.unlink()
    checkout_result = runner.invoke(cli.cli, ["checkout", "test/_helper_shout"])

    assert checkout_result.exit_code == 0, checkout_result.output
    assert output.read_text() == "HELLO"
