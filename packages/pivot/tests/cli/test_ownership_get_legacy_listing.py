"""A stage's own ``*.pvt`` member inside a listed directory, restored by ``pivot get --rev ...
--force`` from a listing only the legacy ``.pivot/cache`` holds, stays the stage's: ``status``
gives no saved-input warning and the next run replaces it.

``core.state_dir`` is set after the first run, so the first run's objects, its listing included,
are only in the legacy cache. With ``committed=False`` the listing file is left out of the commit,
so ``get`` can read that listing only from the legacy cache; with ``committed=True`` it reads the
listing at the revision (control).
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import cli, project
from pivot.storage import cache, track

if TYPE_CHECKING:
    import pathlib

    import click.testing

    from conftest import GitRepo

_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, stage


class TreeParams(StageParams):
    n: int = 1


@stage
def tree(params: TreeParams) -> Annotated[dict[str, str], directory]:
    return {{"notes.pvt": {member}, "value.txt": str(params.n)}}


with Pipeline("b") as pipeline:
    tree(params=TreeParams())
"""

# An unparsable member is the stage's wherever its path is listed; a sidecar-shaped one only
# when its content is a recorded one, so it needs the record `get` makes.
_MEMBERS = {
    "ordinary": ('f"ordinary stage output {params.n}"', "ordinary stage output {n}"),
    "sidecar": (
        'f"path: value.txt\\nhash: abcdef{params.n:010x}\\nsize: 1\\n"',
        "hash: abcdef{n:010x}",
    ),
}


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.parametrize("shape", ["ordinary", "sidecar"])
@pytest.mark.parametrize("committed", [False, True], ids=["legacy-only", "committed"])
def test_get_from_a_legacy_only_listing_keeps_the_stage_pvt_member_the_stages(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
    committed: bool,
    shape: str,
) -> None:
    member, marker = _MEMBERS[shape]
    repo, _ = git_repo
    (repo / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo)
    monkeypatch.chdir(repo)
    (repo / "pipeline.py").write_text(_PIPELINE.format(member=member))
    first = runner.invoke(cli.cli, ["repro"])
    assert first.exit_code == 0, first.output
    parsed = track.read_pvt_file(repo / "data/b/tree/notes.pvt")
    assert (parsed is not None) is (shape == "sidecar"), "precondition: member shape"

    lock_doc = yaml.safe_load((repo / ".pivot/stages/b/tree.lock").read_text())
    (out,) = lock_doc["records"]["outs"]
    first_listing = out["listing"]
    listing_file = repo / ".pivot/stages/b/tree=.listing"
    assert listing_file.is_file()
    _git(repo, "add", "-A")
    if not committed:
        _git(repo, "rm", "--cached", "-q", str(listing_file.relative_to(repo)))
    _git(repo, "commit", "-qm", "v1")
    sha = _git(repo, "rev-parse", "HEAD")
    assert (
        _git(repo, "ls-tree", "-r", "--name-only", sha).count("tree=.listing") == 1
    ) is committed

    state = repo.parent / f"{repo.name}-state"
    (repo / ".pivot/config.yaml").write_text(f"core:\n  state_dir: {state}\n")
    (repo / "params.yaml").write_text("b/tree:\n  n: 2\n")
    second = runner.invoke(cli.cli, ["repro"])
    assert second.exit_code == 0, second.output

    legacy_files = repo / ".pivot/cache/files"
    configured_files = state / "cache/files"
    assert cache.get_cache_path(legacy_files, first_listing).is_file()
    assert not cache.get_cache_path(configured_files, first_listing).exists()
    tree = repo / "data/b/tree"
    assert marker.format(n=2) in (tree / "notes.pvt").read_text()

    got = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "b/tree", "--force"])
    assert got.exit_code == 0, got.output
    assert marker.format(n=1) in (tree / "notes.pvt").read_text()

    status = runner.invoke(cli.cli, ["status"])
    rerun = runner.invoke(cli.cli, ["repro"])

    assert status.exit_code == 0, status.output
    assert "outside every stage output" not in " ".join(status.output.split()), status.output
    assert rerun.exit_code == 0, rerun.output
    assert "outside every stage output" not in " ".join(rerun.output.split()), rerun.output
    assert marker.format(n=2) in (tree / "notes.pvt").read_text()
