"""Remote CLI commands resolve bare and qualified stage identities before path normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import boto3

from pivot import cli, compose, config, executor
from pivot.cli import helpers as cli_helpers
from pivot.cli import remote as cli_remote
from pivot.remote import storage as remote_storage
from pivot.storage import cache, lock
from pivot.types import ArtifactIdentity

if TYPE_CHECKING:
    import click.testing


class _RemoteOutput(TypedDict):
    model: str


@compose.stage
def _remote_train() -> _RemoteOutput:
    return {"model": "remote payload"}


def test_push_fetch_pull_resolve_bare_and_qualified_stage_keys(
    mock_discovery: compose.Pipeline,
    monkeypatch,
    moto_patch_session,
    runner: click.testing.CliRunner,
) -> None:
    # The remote test package's autouse fixture supplies empty outs by default;
    # checkout in pull needs the real registry data to restore the workspace file.
    monkeypatch.setattr(cli_helpers, "get_stage", mock_discovery.get_stage)

    with mock_discovery:
        _remote_train()
    executor.run(pipeline=mock_discovery)

    stage_name = "test/_remote_train"
    ref = mock_discovery.get_stage(stage_name)["outs"][0]
    output_path = cli_helpers.require_workspace_store().resolve_display_path(ref)
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(config.get_state_dir())).read()
    assert stage_lock is not None
    output_hash = stage_lock["output_hashes"][ArtifactIdentity(stage_name, "model")]["hash"]
    cache_path = cache.get_cache_path(config.get_cache_dir() / "files", output_hash)
    assert cache_path.exists()

    bucket = "pivot-identity-targets"
    boto3.Session(region_name="us-east-1").client("s3").create_bucket(Bucket=bucket)
    remote = remote_storage.S3Remote(f"s3://{bucket}/cache/")
    monkeypatch.setattr(
        cli_remote.transfer,
        "create_remote_from_name",
        lambda name: (remote, "origin"),
    )

    push = runner.invoke(cli.cli, ["push", "_remote_train:model"])
    assert push.exit_code == 0, push.output
    assert "Pushed to 'origin': 1 transferred" in push.output

    cache_path.unlink()
    fetch = runner.invoke(cli.cli, ["fetch", "test/_remote_train:model"])
    assert fetch.exit_code == 0, fetch.output
    assert "Fetched from 'origin': 1 transferred" in fetch.output
    assert cache_path.exists()

    cache_path.unlink()
    output_path.unlink()
    pull = runner.invoke(cli.cli, ["pull", "_remote_train:model"])
    assert pull.exit_code == 0, pull.output
    assert "Fetched from 'origin': 1 transferred" in pull.output
    assert output_path.read_text() == "remote payload"
