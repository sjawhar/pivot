"""Red CLI tests from the third acceptance round of sjawhar/pivot#544 (head 21812831).

They use the suite's `moto_services` fixture. Each test drives the installed `pivot` binary, as
the acceptance run did on pivot-slices-3; see /mnt/nvme/sami/tmp/BET408/reports/accept544.md,
"Round 3". Both are low severity: one makes Pivot rerun a stage it need not rerun, the other tells
the user the wrong command; neither lets a stage skip that should run. The round's third defect
(D9) was in the docs, which no longer name `pivot status --remote` among the commands that fill
the remote index, so it has no test here.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import uuid
from typing import Protocol, cast

import boto3
import xxhash

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
import pivot
from pivot import Pipeline, stage


@stage
def plan(pin: str) -> dict:
    return {"rows": [pivot.read_by_hash(h).read_text() for h in pin.split()]}


with Pipeline("p") as pipeline:
    plan(pipeline.input("counted.pin", t=str))
"""

_SNAPSHOT = b"taken: lemma_a\n"


class _MotoS3Client(Protocol):
    def create_bucket(self, **kwargs: object) -> object: ...

    def put_object(self, **kwargs: object) -> object: ...


def _env(endpoint: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(
        AWS_ENDPOINT_URL=endpoint,
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    return env


def _pivot(cwd: pathlib.Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _git(cwd: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True)


def _status(root: pathlib.Path, env: dict[str, str], *extra: str) -> dict[str, object]:
    proc = _pivot(root, env, "status", "--json", *extra)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def _plan(document: dict[str, object]) -> dict[str, object]:
    (stage,) = [
        s for s in cast("list[dict[str, object]]", document["stages"]) if s["name"] == "p/plan"
    ]
    return stage


def _helper_project(
    tmp_path: pathlib.Path, endpoint: str
) -> tuple[pathlib.Path, dict[str, str], str, str]:
    """A project whose stage `p/plan` read, by hash, a snapshot another project pushed to the
    remote: the read fetched it, so the run fetched it from that remote. Committed, so it can be
    cloned. Returns the root, the env, the bucket and the snapshot's hash."""
    env = _env(endpoint)
    bucket = f"reads-{uuid.uuid4().hex[:8]}"
    s3 = cast(
        "_MotoS3Client",
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name="us-east-1",
            aws_access_key_id="x",
            aws_secret_access_key="x",
        ),
    )
    s3.create_bucket(Bucket=bucket)
    snapshot = xxhash.xxh64(_SNAPSHOT).hexdigest()
    s3.put_object(Bucket=bucket, Key=f"cache/files/{snapshot[:2]}/{snapshot[2:]}", Body=_SNAPSHOT)
    root = tmp_path / "p"
    root.mkdir()
    _git(root, "init", "-q")
    _ok(_pivot(root, env, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / "data/raw").mkdir(parents=True)
    (root / "data/raw/counted.pin").write_text(f"{snapshot}\n")
    _ok(_pivot(root, env, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(root, env, "repro"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "run")
    return root, env, bucket, snapshot


def _helper_clone(
    tmp_path: pathlib.Path, root: pathlib.Path, env: dict[str, str], bucket: str
) -> pathlib.Path:
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(root), str(clone)], check=True)
    _ok(_pivot(clone, env, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    return clone


def test_an_object_a_read_fetched_from_the_remote_counts_as_on_it(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """D7. The stage's read fetched the snapshot from remote `origin`. When the cache later loses
    it (a cache cleanup, gc), `pivot status` reports the stage stale with "pivot has not seen them
    on remote 'origin'", although this project downloaded it from there, and `pivot repro` reruns
    the stage (observed: round 3, S3b step 1). `pivot pull` records what it downloads in the remote
    index; a read's fetch records nothing."""
    root, env, _, snapshot = _helper_project(tmp_path, moto_services["s3"])
    (root / ".pivot/cache/files" / snapshot[:2] / snapshot[2:]).unlink()

    stage = _plan(_status(root, env))

    assert stage["status"] == "cached", stage


def test_status_suggests_pull_for_stages_stale_only_for_missing_read_objects(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """D8. In a fresh clone every reading stage is stale with "`pivot pull` fetches them", but the
    suggestions only say "Run `pivot repro` to execute N stale stages", which reruns each stage
    rather than fetching its objects (observed: round 3, R3.2). For missing outputs, status counts
    the stages `pivot pull` restores and suggests it."""
    root, env, bucket, _ = _helper_project(tmp_path, moto_services["s3"])
    clone = _helper_clone(tmp_path, root, env, bucket)

    document = _status(clone, env)

    assert _plan(document)["status"] == "stale"
    suggestions = cast("list[str]", document["suggestions"])
    assert any("pivot pull" in suggestion for suggestion in suggestions), suggestions
