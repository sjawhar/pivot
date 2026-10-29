"""Round-3 recheck of PR #523: `pivot push` and a directory listing it cannot read.

The one read rule the pull request states is: a committed listing that does not hash to its
record, or cannot be read, is passed over for the cache's copy with one warning, and only when
the cache lacks that object too is it refused. `pivot verify` refuses it. `pivot push` is the
reader that does neither: it transfers nothing, counts the directory as "skipped" and exits 0,
so a teammate's `pivot pull` finds nothing on the remote.

Found by the recheck (lane PR523RecheckR3) at the round-3 head `38234b96`. Drop this file into
``packages/pivot/tests/cli/`` to run it; it uses the repository's ``moto_services`` fixture.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import uuid

import boto3
import pytest
import yaml

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
import pathlib

from pivot import Pipeline, stage


@stage
def note(corpus: pathlib.Path) -> dict:
    return {"files": sorted(p.name for p in corpus.iterdir())}


with Pipeline("demo") as pipeline:
    note(pipeline.input("corpus", t=pathlib.Path))
"""


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _run(
    cwd: pathlib.Path, *argv: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env or _env(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _git(cwd: pathlib.Path, *args: str) -> str:
    return _ok(_run(cwd, "git", "-c", "user.email=t@t", "-c", "user.name=t", *args))


def _project(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _ok(_run(root, _PIVOT, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    corpus = root / "data/raw/corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("a")
    (root / "assets/nested").mkdir(parents=True)
    (root / "assets/one.txt").write_text("alpha")
    (root / "assets/nested/two.txt").write_text("beta2")
    (root / ".gitignore").write_text("__pycache__/\n/data/\n/assets/\n")
    _ok(_run(root, _PIVOT, "track", "assets"))
    _ok(_run(root, _PIVOT, "repro"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "built")
    return root


def _remote_env(moto_services: dict[str, str], root: pathlib.Path) -> tuple[dict[str, str], str]:
    bucket = f"r3-{uuid.uuid4().hex[:8]}"
    remote = f"s3://{bucket}/cache"
    env = _env(
        AWS_ENDPOINT_URL=moto_services["s3"],
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    boto3.client(
        "s3",
        endpoint_url=moto_services["s3"],
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    ).create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
    for argv in (
        ("config", "set", "remotes.origin", remote),
        ("config", "set", "default_remote", "origin"),
    ):
        _ok(_run(root, _PIVOT, *argv, env=env))
    return env, remote


def _dir_hash(root: pathlib.Path) -> str:
    recorded = yaml.safe_load((root / "assets.pvt").read_text())
    return str(recorded["hash"])


def _cache_object(root: pathlib.Path, digest: str) -> pathlib.Path:
    return root / ".pivot/cache/files" / digest[:2] / digest[2:]


@pytest.mark.parametrize("how", ["edited", "absent"])
def test_push_does_not_report_success_when_it_cannot_read_a_directory_listing(
    tmp_path: pathlib.Path, moto_services: dict[str, str], how: str
):
    """With the committed listing unreadable and no cache object for it, `pivot push` has
    nothing it can send for that directory. It must say so and exit non-zero, as `pivot verify`
    does for the same project - not report `0 transferred, 1 skipped, 0 failed` and exit 0,
    which reads as "the remote has it"."""
    root = _project(tmp_path / "src")
    env, _ = _remote_env(moto_services, root)
    digest = _dir_hash(root)
    listing = root / "assets.pvt.listing"
    if how == "edited":
        listing.write_text("deadbeefdeadbeef 5 - one.txt\n")
    else:
        listing.unlink()
    _cache_object(root, digest).unlink()

    pushed = _run(root, _PIVOT, "push", "assets", env=env)

    out = pushed.stdout + pushed.stderr
    assert "assets.pvt.listing" in out, out
    assert "0 transferred" not in out or pushed.returncode != 0, out
    assert pushed.returncode != 0, out


def test_push_of_a_tracked_directory_still_works_when_its_listing_is_readable(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
):
    """The boundary the fix must not cross: an ordinary push still sends the directory, and a
    fresh clone restores it. Passes at the round-3 head."""
    root = _project(tmp_path / "src")
    env, remote = _remote_env(moto_services, root)

    pushed = _ok(_run(root, _PIVOT, "push", "assets", env=env))

    assert "0 transferred" not in pushed, pushed
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(root), str(clone))
    for argv in (
        ("config", "set", "remotes.origin", remote),
        ("config", "set", "default_remote", "origin"),
    ):
        _ok(_run(clone, _PIVOT, *argv, env=env))
    _ok(_run(clone, _PIVOT, "pull", "assets", env=env))
    assert (clone / "assets/one.txt").read_text() == "alpha"
    assert (clone / "assets/nested/two.txt").read_text() == "beta2"
