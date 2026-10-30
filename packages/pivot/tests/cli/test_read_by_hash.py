"""A stage reads an object by the content hash a declared input names (`pivot.read_by_hash`).

Each test drives the installed `pivot` binary in a fresh project, the way a user does. The
project's one stage, `p/plan`, reads its pipeline input `counted.pin`, a file of content hashes,
and reads each object the pin names by its hash. The objects reach the cache through
`pivot track`, as a snapshot another project pushed would reach it through `pivot pull`, or sit
only on a moto S3 remote, as such a snapshot does before anything fetches it. A test whose
docstring names an acceptance round of sjawhar/pivot#544 pins a defect that round found.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import uuid
from typing import Protocol, cast

import boto3
import xxhash
import yaml

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

_LOCK = pathlib.Path(".pivot/stages/p/plan.lock")
_OUTPUT = pathlib.Path("data/p/plan.yaml")
_PIN = pathlib.Path("data/raw/counted.pin")

JsonObject = dict[str, object]


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _pivot(
    cwd: pathlib.Path, *args: str, env: dict[str, str] | None = None, timeout: float = 300
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args],
        cwd=cwd,
        env=env or _env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _project(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / "data/raw").mkdir(parents=True)
    return root


def _helper_object(root: pathlib.Path, text: str) -> str:
    """Put ``text`` in the project's cache as one object (`pivot track`) and return its hash."""
    path = root / "store" / f"{uuid.uuid4().hex}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    _ok(_pivot(root, "track", str(path.relative_to(root))))
    return xxhash.xxh64(text.encode()).hexdigest()


def _helper_cache_object(root: pathlib.Path, content_hash: str) -> pathlib.Path:
    return root / ".pivot/cache/files" / content_hash[:2] / content_hash[2:]


def _helper_pin(root: pathlib.Path, *hashes: str) -> None:
    (root / _PIN).write_text("".join(f"{content_hash}\n" for content_hash in hashes))


def _helper_lock_reads(root: pathlib.Path) -> list[str] | None:
    lock = yaml.safe_load((root / _LOCK).read_text())
    return lock["reads"] if "reads" in lock else None


def _helper_rows(root: pathlib.Path) -> list[str]:
    return yaml.safe_load((root / _OUTPUT).read_text())["rows"]


def _helper_completed(proc: subprocess.CompletedProcess[str]) -> dict[str, tuple[str, str]]:
    """Each stage `pivot repro --json` completed, with its status and reason."""
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    return {
        event["stage"]: (event["status"], event["reason"])
        for event in events
        if event["type"] == "stage_complete"
    }


def _helper_status_document(root: pathlib.Path, env: dict[str, str] | None = None) -> JsonObject:
    """What `pivot status --json` reports."""
    proc = _pivot(root, "status", "--json", env=env)
    _ = _ok(proc)
    return json.loads(proc.stdout)


def _helper_plan(document: JsonObject) -> JsonObject:
    """`p/plan`'s entry in a `pivot status --json` document."""
    (stage,) = [s for s in cast("list[JsonObject]", document["stages"]) if s["name"] == "p/plan"]
    return stage


def _helper_status(root: pathlib.Path, env: dict[str, str] | None = None) -> tuple[str, str]:
    """`p/plan`'s status and reason as `pivot status --json` reports them."""
    stage = _helper_plan(_helper_status_document(root, env))
    return cast("str", stage["status"]), cast("str", stage["reason"])


def _helper_git(cwd: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True)


def _helper_remove_object(root: pathlib.Path, content_hash: str) -> None:
    path = _helper_cache_object(root, content_hash)
    assert path.is_file(), f"{content_hash} should be in the cache"
    path.unlink()


# =============================================================================
# Reading, recording and rerunning
# =============================================================================


def test_repro_hands_the_stage_the_object_and_records_its_hash_in_the_lock(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path / "p")
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)

    first = _pivot(root, "repro", "--json")

    assert _helper_completed(first)["p/plan"][0] == "ran", first.stdout + first.stderr
    assert _helper_rows(root) == ["taken: lemma_a"]
    assert _helper_lock_reads(root) == [snapshot], "the lock should name the object read"
    second = _pivot(root, "repro", "--json")
    assert _helper_completed(second)["p/plan"][0] == "cached", second.stdout + second.stderr
    assert _helper_status(root) == ("cached", "")


def test_a_changed_pin_reruns_the_stage_and_the_lock_names_the_new_object(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path / "p")
    old = _helper_object(root, "taken: lemma_a\n")
    new = _helper_object(root, "taken: lemma_b\n")
    _helper_pin(root, old)
    _ok(_pivot(root, "repro"))
    _helper_pin(root, new)

    assert _helper_status(root) == ("stale", "Input dependencies changed")
    proc = _pivot(root, "repro", "--json")

    assert _helper_completed(proc)["p/plan"] == ("ran", "Input dependencies changed"), proc.stdout
    assert _helper_rows(root) == ["taken: lemma_b"]
    assert _helper_lock_reads(root) == [new], "the lock should name only what this run read"


def test_the_lock_lists_each_object_read_once_in_hash_order_and_is_stable(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path / "p")
    first = _helper_object(root, "one\n")
    second = _helper_object(root, "two\n")
    _helper_pin(root, second, first, second)
    _ok(_pivot(root, "repro"))
    written = (root / _LOCK).read_bytes()

    _ok(_pivot(root, "repro", "--force"))

    assert _helper_lock_reads(root) == sorted({first, second})
    assert (root / _LOCK).read_bytes() == written, "the same reads should give the same lock"


def test_a_stage_that_reads_nothing_by_hash_writes_no_reads_key(tmp_path: pathlib.Path) -> None:
    root = _project(tmp_path / "p")
    _helper_pin(root)

    _ok(_pivot(root, "repro"))

    assert _helper_lock_reads(root) is None, (root / _LOCK).read_text()


# =============================================================================
# Objects missing from the cache and the remote
# =============================================================================


def test_status_reports_stale_when_a_read_object_is_missing_and_no_remote_is_configured(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path / "p")
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro"))
    _helper_remove_object(root, snapshot)

    status, reason = _helper_status(root)

    assert status == "stale"
    assert snapshot in reason and "no remote is configured" in reason, reason
    repro = _pivot(root, "repro")
    out = repro.stdout + repro.stderr
    assert repro.returncode != 0, out
    assert f"Object {snapshot} is not in the cache" in out, out
    assert _helper_lock_reads(root) == [snapshot], "a failed run should leave the lock as it was"


def test_a_stage_whose_object_is_corrupt_in_the_cache_fails_its_read(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path / "p")
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)
    path = _helper_cache_object(root, snapshot)
    path.chmod(0o644)
    path.write_text("taken: something else\n")

    repro = _pivot(root, "repro")

    out = repro.stdout + repro.stderr
    assert repro.returncode != 0, out
    assert f"Object {snapshot}" in out and "hashes to" in out, out
    assert not (root / _LOCK).exists(), "a failed read should write no lock"


def test_a_pin_that_holds_no_content_hash_fails_the_read(tmp_path: pathlib.Path) -> None:
    root = _project(tmp_path / "p")
    (root / _PIN).write_text("../../etc/passwd\n")

    repro = _pivot(root, "repro")

    out = repro.stdout + repro.stderr
    assert repro.returncode != 0, out
    assert "is not a content hash" in out, out


# =============================================================================
# The remote
# =============================================================================


class _MotoS3Client(Protocol):
    """The boto3 S3 client calls these tests make; boto3 ships no stubs."""

    def create_bucket(self, **kwargs: object) -> object: ...

    def delete_object(self, **kwargs: object) -> object: ...

    def put_object(self, **kwargs: object) -> object: ...


def _helper_remote(root: pathlib.Path, endpoint: str) -> tuple[dict[str, str], _MotoS3Client, str]:
    """Point the project at a fresh bucket on the moto server at ``endpoint`` as its one remote.
    Returns the env that reaches it, a client for the bucket, and the bucket's name."""
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
    _ok(_pivot(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache", env=env))
    return env, s3, bucket


def _helper_put_remote(s3: _MotoS3Client, bucket: str, data: bytes) -> str:
    """Put ``data`` on the remote as the object another project pushed; return its hash."""
    content_hash = xxhash.xxh64(data).hexdigest()
    s3.put_object(
        Bucket=bucket, Key=f"cache/files/{content_hash[:2]}/{content_hash[2:]}", Body=data
    )
    return content_hash


def _helper_clone(
    tmp_path: pathlib.Path, root: pathlib.Path, env: dict[str, str], bucket: str
) -> pathlib.Path:
    """A fresh git clone of the committed project at ``root``, pointed at the same remote."""
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(root), str(clone)], check=True)
    _ok(_pivot(clone, "config", "set", "remotes.origin", f"s3://{bucket}/cache", env=env))
    return clone


def _helper_fetched_project(
    tmp_path: pathlib.Path, endpoint: str
) -> tuple[pathlib.Path, dict[str, str], str, str]:
    """A project whose stage `p/plan` read, by hash, a snapshot another project pushed to the
    remote: the read fetched it, so the run fetched it from that remote. Committed, so it can be
    cloned. Returns the root, the env, the bucket and the snapshot's hash."""
    root = _project(tmp_path / "p")
    env, s3, bucket = _helper_remote(root, endpoint)
    snapshot = _helper_put_remote(s3, bucket, b"taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro", env=env))
    _helper_git(root, "add", "-A")
    _helper_git(root, "commit", "-qm", "run")
    return root, env, bucket, snapshot


def test_a_read_fetches_the_object_from_the_remote_when_the_cache_lacks_it(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    root = _project(tmp_path / "p")
    env, _, _ = _helper_remote(root, moto_services["s3"])
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro", env=env))
    _ok(_pivot(root, "push", env=env))
    _helper_remove_object(root, snapshot)

    assert _helper_status(root, env) == ("cached", ""), "the remote holds the object"
    proc = _pivot(root, "repro", "--force", "--json", env=env)

    assert _helper_completed(proc)["p/plan"][0] == "ran", proc.stdout + proc.stderr
    assert _helper_rows(root) == ["taken: lemma_a"]
    assert _helper_cache_object(root, snapshot).read_text() == "taken: lemma_a\n"


def test_status_reports_stale_when_neither_the_cache_nor_the_remote_holds_a_read_object(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    root = _project(tmp_path / "p")
    env, s3, bucket = _helper_remote(root, moto_services["s3"])
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro", env=env))
    _ok(_pivot(root, "push", env=env))
    _helper_remove_object(root, snapshot)
    s3.delete_object(Bucket=bucket, Key=f"cache/files/{snapshot[:2]}/{snapshot[2:]}")
    shutil.rmtree(root / ".pivot/state.lmdb")  # forget that the push put it on the remote

    status, reason = _helper_status(root, env)

    assert status == "stale"
    assert snapshot in reason and "remote 'origin'" in reason, reason
    repro = _pivot(root, "repro", env=env)
    out = repro.stdout + repro.stderr
    assert repro.returncode != 0, out
    assert f"Object {snapshot}" in out and "origin" in out, out


def test_an_object_a_read_fetched_from_the_remote_counts_as_on_it(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """Acceptance round 3, D7. The stage's read fetched the snapshot from remote `origin`. When
    the cache later loses it (a cache cleanup, gc), `pivot status` reported the stage stale with
    "pivot has not seen them on remote 'origin'", although this project downloaded it from there,
    and `pivot repro` reran the stage. `pivot pull` records what it downloads in the remote index;
    a read's fetch now does too."""
    root, env, _, snapshot = _helper_fetched_project(tmp_path, moto_services["s3"])
    _helper_remove_object(root, snapshot)

    stage = _helper_plan(_helper_status_document(root, env))

    assert stage["status"] == "cached", stage


def test_status_suggests_pull_for_stages_stale_only_for_missing_read_objects(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """Acceptance round 3, D8. In a fresh clone every reading stage is stale with "`pivot pull`
    fetches them", but the suggestions said only "Run `pivot repro` to execute N stale stages",
    which reruns each stage rather than fetching its objects. For missing outputs, status counts
    the stages `pivot pull` restores and suggests it; it now counts these too."""
    root, env, bucket, _ = _helper_fetched_project(tmp_path, moto_services["s3"])
    clone = _helper_clone(tmp_path, root, env, bucket)

    document = _helper_status_document(clone, env)

    assert _helper_plan(document)["status"] == "stale"
    suggestions = cast("list[str]", document["suggestions"])
    assert any("pivot pull" in suggestion for suggestion in suggestions), suggestions


def test_status_json_says_pull_fetches_what_a_stage_read(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """The stage's recovery is a field of its JSON entry, with and without `--explain`, so a
    script can act on it without reading the reason's text; the suggestion comes from it."""
    root, env, bucket, _ = _helper_fetched_project(tmp_path, moto_services["s3"])
    clone = _helper_clone(tmp_path, root, env, bucket)

    plain = _helper_status_document(clone, env)
    explain = _pivot(clone, "status", "--explain", "--json", env=env)

    assert _helper_plan(plain)["recovery"] == "pull_reads", plain
    assert cast("list[str]", plain["suggestions"])[0] == (
        "Run `pivot pull` to fetch the objects 1 stage read by hash from the remote"
    ), plain
    _ = _ok(explain)
    explained = _helper_plan(json.loads(explain.stdout))
    assert explained["recovery"] == "pull_reads", explained


def test_a_targeted_pull_fetches_the_objects_the_stage_read(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """Acceptance round 1, defect 5. `pivot pull <stage>` fetched the stage's outputs and its
    cached dependencies (`include_deps=True`) so the stage can run from the clone, but not the
    objects its lock's `reads` names (observed in S8, 8c: none of A2, B1, L reached the clone's
    cache)."""
    root = _project(tmp_path / "p")
    env, _, bucket = _helper_remote(root, moto_services["s3"])
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro", env=env))
    _ok(_pivot(root, "push", env=env))
    _helper_git(root, "add", "-A")
    _helper_git(root, "commit", "-qm", "run")
    clone = _helper_clone(tmp_path, root, env, bucket)

    _ok(_pivot(clone, "pull", "p/plan", env=env))

    assert _helper_cache_object(clone, snapshot).is_file(), (
        "the object p/plan read should be pulled"
    )


# A plain `pivot status` is a local command in every released pivot. With the defaults
# (remote.retries 10, botocore's legacy backoff), the remote check it made in round 2 for a read
# object the cache lacks waited more than 5 minutes against a refused connection on
# pivot-slices-3, where remote.retries 0 took 1.3 s and 0.6.3's status 3.2 s.
_STATUS_BUDGET_SECONDS = 60


def test_status_with_an_unreachable_remote_answers_promptly(tmp_path: pathlib.Path) -> None:
    """Acceptance round 2, D6. A stage's lock records an object the cache no longer holds, and
    the remote refuses connections (a laptop offline, a CI job without credentials for the
    network). Plain `pivot status`, which 0.6.3 answers from local state alone, asked the remote
    about the object and, with the default retry settings, did not answer for minutes. Under
    decision D6 it contacts no remote for read objects, as for outputs, and answers within a
    minute: the stage is stale because pivot has not seen the object on the remote (adapted from
    the red test's "could not be reached", which assumed a remote query)."""
    env = _env(
        AWS_ENDPOINT_URL="http://127.0.0.1:9",
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    root = _project(tmp_path / "p")
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro", env=env))
    _helper_remove_object(root, snapshot)
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


# =============================================================================
# The run cache and `pivot commit`
# =============================================================================


def test_a_run_cache_restore_records_what_the_restored_run_read(tmp_path: pathlib.Path) -> None:
    root = _project(tmp_path / "p")
    first = _helper_object(root, "taken: lemma_a\n")
    second = _helper_object(root, "taken: lemma_b\n")
    _helper_pin(root, first)
    _ok(_pivot(root, "repro"))
    _helper_pin(root, second)
    _ok(_pivot(root, "repro"))
    _helper_pin(root, first)

    proc = _pivot(root, "repro", "--json")

    assert _helper_completed(proc)["p/plan"] == ("cached", "unchanged (run cache)"), proc.stdout
    assert _helper_lock_reads(root) == [first]
    assert _helper_rows(root) == ["taken: lemma_a"]


def test_the_run_cache_restores_nothing_whose_read_object_is_gone(tmp_path: pathlib.Path) -> None:
    root = _project(tmp_path / "p")
    first = _helper_object(root, "taken: lemma_a\n")
    second = _helper_object(root, "taken: lemma_b\n")
    _helper_pin(root, first)
    _ok(_pivot(root, "repro"))
    _helper_pin(root, second)
    _ok(_pivot(root, "repro"))
    _helper_pin(root, first)
    _helper_remove_object(root, first)

    repro = _pivot(root, "repro")

    out = repro.stdout + repro.stderr
    assert repro.returncode != 0, out
    assert f"Object {first} is not in the cache" in out, out
    assert _helper_lock_reads(root) == [second], "nothing should replace the lock"


def test_commit_after_a_no_commit_run_records_what_that_run_read(tmp_path: pathlib.Path) -> None:
    root = _project(tmp_path / "p")
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro", "--no-commit"))
    assert not (root / _LOCK).exists()

    _ok(_pivot(root, "commit"))

    assert _helper_lock_reads(root) == [snapshot]
    assert _helper_status(root) == ("cached", "")


def test_an_object_a_no_commit_run_fetched_counts_as_on_the_remote(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """A `--no-commit` run's read records what it fetched in the remote index, as a committed
    run's does: the remote holds the object whether or not the outputs are committed. So once
    `pivot commit` records the run and the cache loses the object, the stage is still cached."""
    root = _project(tmp_path / "p")
    env, s3, bucket = _helper_remote(root, moto_services["s3"])
    snapshot = _helper_put_remote(s3, bucket, b"taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro", "--no-commit", env=env))
    _ok(_pivot(root, "commit", env=env))
    _helper_remove_object(root, snapshot)

    assert _helper_lock_reads(root) == [snapshot]
    assert _helper_status(root, env) == ("cached", "")


# =============================================================================
# A team that mixes this pivot with 0.6.3
# =============================================================================


def _released(version: str) -> list[str]:
    uv = os.environ.get("PIVOT_TEST_UV") or shutil.which("uv")
    assert uv is not None, "this test runs released pivots with `uv tool run`"
    return [uv, "tool", "run", "--quiet", "--from", f"pivot=={version}", "pivot"]


def _helper_run(cwd: pathlib.Path, argv: list[str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*argv, *args], cwd=cwd, env=_env(), capture_output=True, text=True, timeout=600
    )


def test_pivot_0_6_3_reads_a_lock_that_records_reads_and_reruns_nothing(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path / "p")
    snapshot = _helper_object(root, "taken: lemma_a\n")
    _helper_pin(root, snapshot)
    _ok(_pivot(root, "repro"))
    written = (root / _LOCK).read_bytes()
    old = _released("0.6.3")

    status = _helper_run(root, old, "status", "--json")
    explain = _helper_run(root, old, "status", "--explain", "--json")
    repro = _helper_run(root, old, "repro")

    assert status.returncode == 0, status.stdout + status.stderr
    (stage,) = json.loads(status.stdout)["stages"]
    assert stage["status"] == "cached", stage
    assert explain.returncode == 0, explain.stdout + explain.stderr
    (explained,) = json.loads(explain.stdout)["stages"]
    assert explained["will_run"] is False, explained
    assert repro.returncode == 0, repro.stdout + repro.stderr
    assert (root / _LOCK).read_bytes() == written, "0.6.3 should leave the lock as it was"


def test_this_pivot_reads_a_lock_pivot_0_6_3_wrote_and_reruns_nothing(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path / "p")
    _helper_pin(root, "0123456789abcdef")
    plain = _PIPELINE.replace(
        "pivot.read_by_hash(content_hash).read_text().strip()", "content_hash"
    )
    (root / "pipeline.py").write_text(plain)
    _ok(_helper_run(root, _released("0.6.3"), "repro"))
    written = (root / _LOCK).read_bytes()

    proc = _pivot(root, "repro", "--json")

    assert _helper_completed(proc)["p/plan"][0] == "cached", proc.stdout + proc.stderr
    assert _helper_status(root) == ("cached", "")
    assert (root / _LOCK).read_bytes() == written
