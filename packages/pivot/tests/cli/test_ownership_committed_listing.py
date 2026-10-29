"""A stage's own ``*.pvt`` members are read from the listing committed beside its lock, not only
from the cache.

A lock that arrives through git (``git pull`` of a teammate's run, without ``pivot pull``) has its
committed listing file beside it, but the local cache need not hold that listing's object. Each
test deletes the lock's listing object from the cache, keeps the committed listing file, and
checks that the stage's own sidecar-shaped ``fake.pvt`` is still the stage's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from conftest import isolated_pivot_dir
from pivot import cli
from pivot.storage import cache, track

if TYPE_CHECKING:
    import pathlib

    import click.testing

_SIDECAR_SHAPED_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, stage


class TreeParams(StageParams):
    n: int = 1


@stage
def tree(params: TreeParams) -> Annotated[dict[str, str], directory]:
    return {
        "value.txt": str(params.n),
        "fake.pvt": f"path: value.txt\\nhash: abcdef{params.n:010x}\\nsize: 1\\n",
    }


with Pipeline("b") as pipeline:
    tree(params=TreeParams())
"""

_OWNERSHIP_TEXT = ("outside every stage output", "Refusing", "refuses to replace")

# "control" keeps the listing object in the cache, so a "no-cached-listing" failure is caused by
# the missing object alone.
pytestmark = pytest.mark.parametrize("drop", [True, False], ids=["no-cached-listing", "control"])


def _built_without_cached_listing(
    runner: click.testing.CliRunner, root: pathlib.Path, drop: bool
) -> None:
    (root / "pipeline.py").write_text(_SIDECAR_SHAPED_MEMBER_PIPELINE)
    first = runner.invoke(cli.cli, ["repro"])
    assert first.exit_code == 0, first.output
    fake = root / "data" / "b" / "tree" / "fake.pvt"
    assert track.read_pvt_file(fake) is not None, "precondition: fake.pvt parses as a sidecar"
    document = yaml.safe_load((root / ".pivot" / "stages" / "b" / "tree.lock").read_text())
    (out,) = document["records"]["outs"]
    assert (root / ".pivot" / "stages" / "b" / "tree=.listing").is_file()
    obj = cache.get_cache_path(root / ".pivot" / "cache" / "files", out["listing"])
    assert obj.is_file()
    if drop:
        obj.unlink()


def _no_ownership_text(output: str) -> None:
    flat = " ".join(output.split())
    for text in _OWNERSHIP_TEXT:
        assert text not in flat, output


def test_status_gives_no_saved_input_warning_for_the_stages_own_pvt(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, drop: bool
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        _built_without_cached_listing(runner, root, drop)

        status = runner.invoke(cli.cli, ["status"])

        assert status.exit_code == 0, status.output
        _no_ownership_text(status.output)


def test_a_rerun_replaces_the_stages_own_pvt(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, drop: bool
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        _built_without_cached_listing(runner, root, drop)
        (root / "params.yaml").write_text("b/tree:\n  n: 2\n")

        rerun = runner.invoke(cli.cli, ["repro"])

        assert rerun.exit_code == 0, rerun.output
        _no_ownership_text(rerun.output)
        assert (root / "data" / "b" / "tree" / "value.txt").read_text() == "2"


def test_checkout_force_of_the_stage_restores_over_its_own_pvt(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, drop: bool
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        _built_without_cached_listing(runner, root, drop)
        (root / "data" / "b" / "tree" / "value.txt").unlink()

        forced = runner.invoke(cli.cli, ["checkout", "b/tree", "--force"])

        assert forced.exit_code == 0, forced.output
        _no_ownership_text(forced.output)
        assert (root / "data" / "b" / "tree" / "value.txt").read_text() == "1"
