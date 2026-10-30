"""Red CLI tests from the acceptance of sjawhar/pivot#544 (pivot.read_by_hash) at 52169fd1.

Each test drives the installed `pivot` binary (and released pivot 0.6.3 via `uv tool run`) in a
fresh project, the way the acceptance run did on pivot-slices-3; see
/mnt/nvme/sami/tmp/BET408/reports/accept544.md. Place this file at
packages/pivot/tests/cli/test_read_by_hash_accept544_red.py; it uses the suite's `moto_services`
fixture.
"""

from __future__ import annotations

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
    rows = list[str]()
    for content_hash in pin.split():
        rows.append(pivot.read_by_hash(content_hash).read_text().strip())
    return {"rows": rows}


with Pipeline("p") as pipeline:
    plan(pipeline.input("counted.pin", t=str))
"""

_PIN = pathlib.Path("data/raw/counted.pin")


class _MotoS3Client(Protocol):
    def create_bucket(self, **kwargs: object) -> object: ...

    def delete_object(self, **kwargs: object) -> object: ...


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _run(
    cwd: pathlib.Path, argv: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _pivot(cwd: pathlib.Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return _run(cwd, [_PIVOT, *args], env)


def _project(root: pathlib.Path, env: dict[str, str]) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    _ok(_pivot(root, env, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / "data/raw").mkdir(parents=True)
    return root


def _object(root: pathlib.Path, env: dict[str, str], text: str) -> str:
    path = root / "store" / f"{uuid.uuid4().hex}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    _ok(_pivot(root, env, "track", str(path.relative_to(root))))
    return xxhash.xxh64(text.encode()).hexdigest()


def _cache_object(root: pathlib.Path, content_hash: str) -> pathlib.Path:
    return root / ".pivot/cache/files" / content_hash[:2] / content_hash[2:]


def _pin(root: pathlib.Path, *hashes: str) -> None:
    (root / _PIN).write_text("".join(f"{content_hash}\n" for content_hash in hashes))


def _remote(root: pathlib.Path, endpoint: str) -> tuple[dict[str, str], _MotoS3Client, str]:
    env = _env(
        AWS_ENDPOINT_URL=endpoint,
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
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
    _ok(_pivot(root, env, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    return env, s3, bucket


def test_a_targeted_pull_fetches_the_objects_the_stage_read(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """Defect 5. `pivot pull <stage>` fetches the stage's outputs and its cached dependencies
    (`include_deps=True`) so the stage can run from the clone, but not the objects its lock's
    `reads` names (observed in S8, 8c: none of A2, B1, L reached the clone's cache)."""
    root = _project(tmp_path / "p", _env())
    env, _, bucket = _remote(root, moto_services["s3"])
    snapshot = _object(root, env, "taken: lemma_a\n")
    _pin(root, snapshot)
    _ok(_pivot(root, env, "repro"))
    _ok(_pivot(root, env, "push"))
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "run"], cwd=root, check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(root), str(clone)], check=True)
    _ok(_pivot(clone, env, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))

    _ok(_pivot(clone, env, "pull", "p/plan"))

    assert _cache_object(clone, snapshot).is_file(), "the object p/plan read should be pulled"
