"""Red CLI test from the second acceptance round of sjawhar/pivot#544 (head 4621e0a4).

Place this file at packages/pivot/tests/cli/test_read_by_hash_accept544_r2_red.py. It drives the
installed `pivot` binary in a fresh project, as the acceptance run did on pivot-slices-3; see
/mnt/nvme/sami/tmp/BET408/reports/accept544.md, "Round 2".
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import uuid

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

# A plain `pivot status` is a local command in every released pivot. With the defaults
# (remote.retries 10, botocore's legacy backoff), the remote check it now makes for a read object
# the cache lacks waited more than 5 minutes against a refused connection on pivot-slices-3, where
# remote.retries 0 took 1.3 s and 0.6.3's status 3.2 s.
_STATUS_BUDGET_SECONDS = 60


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _pivot(
    cwd: pathlib.Path, *args: str, env: dict[str, str], timeout: float = 300
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def test_status_with_an_unreachable_remote_answers_promptly(tmp_path: pathlib.Path) -> None:
    """D6. A stage's lock records an object the cache no longer holds, and the remote refuses
    connections (a laptop offline, a CI job without credentials for the network). Plain
    `pivot status`, which 0.6.3 answers from local state alone, asked the remote about the
    object and, with the default retry settings, did not answer for minutes. Under decision D6
    it contacts no remote for read objects, as for outputs, and answers within a minute: the
    stage is stale because pivot has not seen the object on the remote (adapted from the red
    test's "could not be reached", which assumed a remote query)."""
    env = _env(
        AWS_ENDPOINT_URL="http://127.0.0.1:9",
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    root = tmp_path / "p"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init", env=env))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / "data/raw").mkdir(parents=True)
    store = root / "store" / f"{uuid.uuid4().hex}.txt"
    store.parent.mkdir()
    store.write_text("taken: lemma_a\n")
    _ok(_pivot(root, "track", str(store.relative_to(root)), env=env))
    snapshot = xxhash.xxh64(b"taken: lemma_a\n").hexdigest()
    (root / "data/raw/counted.pin").write_text(f"{snapshot}\n")
    _ok(_pivot(root, "repro", env=env))
    (root / ".pivot/cache/files" / snapshot[:2] / snapshot[2:]).unlink()
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://unreachable/cache", env=env))

    try:
        proc = _pivot(root, "status", "--json", env=env, timeout=_STATUS_BUDGET_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"`pivot status` gave no answer within {_STATUS_BUDGET_SECONDS} s with the remote "
            + "unreachable"
        ) from error

    (stage,) = json.loads(proc.stdout)["stages"]
    assert stage["status"] == "stale", stage
    assert snapshot in stage["reason"] and "remote 'origin'" in stage["reason"], stage
