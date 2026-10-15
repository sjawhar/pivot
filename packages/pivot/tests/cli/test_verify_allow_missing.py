"""`pivot verify --allow-missing` on compositional-API pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import cli, compose, config, types
from pivot.storage import cache, lock

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture


@compose.stage
def process(input_file: str) -> str:
    return input_file.upper()


@compose.stage
def produce(input_file: str) -> str:
    return input_file.upper()


@compose.stage
def consume(produced: str) -> str:
    return f"consumed: {produced}"


def _helper_mock_remote(mocker: MockerFixture, *, exists: bool) -> None:
    """Mock the remote boundary: configured remote plus S3 existence checks."""
    mocker.patch("pivot.remote.config.list_remotes", return_value={"origin": "s3://bucket/cache"})
    mocker.patch("pivot.remote.config.get_default_remote", return_value="origin")
    mocker.patch("pivot.remote.config.get_remote_url", return_value="s3://bucket/cache")
    remote = mocker.patch("pivot.remote.storage.S3Remote").return_value

    async def _bulk_exists(hashes: list[str], concurrency: int = 20) -> dict[str, bool]:
        return dict.fromkeys(hashes, exists)

    remote.bulk_exists = _bulk_exists


def test_verify_allow_missing_finds_present_raw_input_at_its_workspace_path(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """A raw input on disk at data/raw/<name> is not missing, even if it is on no remote."""
    root = mock_discovery.root
    raw = root / "data" / "raw" / "input.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("hello")
    with mock_discovery:
        process(mock_discovery.input("input.txt", t=str))
    stage_name = mock_discovery.list_stages()[0]

    output = root / "data" / "test" / "process.txt"
    output.parent.mkdir(parents=True)
    output.write_text("HELLO")
    out_hash = cache.save_to_cache(
        output, config.get_cache_dir() / "files", checkout_mode=config.CheckoutMode.COPY
    )
    raw_hash, _ = cache.hash_file(raw)
    lock.StageLock(stage_name, lock.get_stages_dir(config.get_state_dir())).write(
        types.LockData(
            code_manifest=mock_discovery.ensure_fingerprint(stage_name),
            params={},
            dep_hashes={types.ArtifactIdentity("input.txt", None): types.FileHash(hash=raw_hash)},
            output_hashes={types.ArtifactIdentity(stage_name, None): out_hash},
        )
    )
    _helper_mock_remote(mocker, exists=False)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert "Missing files" not in result.output, result.output
    assert result.exit_code == 0, result.output


def test_verify_allow_missing_names_stale_producer_dependency(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """A lock-only consumer must fail against the producer lock and name that dependency."""
    root = mock_discovery.root
    raw = root / "data" / "raw" / "input.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("v2")
    with mock_discovery:
        intermediate = produce(mock_discovery.input("input.txt", t=str))
        consume(intermediate)
    producer_stage, consumer_stage = mock_discovery.list_stages()
    producer_hash = types.FileHash(hash="aaaaaaaaaaaaaaaa")
    consumer_hash = types.FileHash(hash="bbbbbbbbbbbbbbbb")
    state_dir = config.get_state_dir()
    raw_hash, _ = cache.hash_file(raw)
    lock.StageLock(producer_stage, lock.get_stages_dir(state_dir)).write(
        types.LockData(
            code_manifest=mock_discovery.ensure_fingerprint(producer_stage),
            params={},
            dep_hashes={types.ArtifactIdentity("input.txt", None): types.FileHash(hash=raw_hash)},
            output_hashes={types.ArtifactIdentity(producer_stage, None): producer_hash},
        )
    )
    lock.StageLock(consumer_stage, lock.get_stages_dir(state_dir)).write(
        types.LockData(
            code_manifest=mock_discovery.ensure_fingerprint(consumer_stage),
            params={},
            dep_hashes={types.ArtifactIdentity(producer_stage, None): consumer_hash},
            output_hashes={
                types.ArtifactIdentity(consumer_stage, None): types.FileHash(
                    hash="cccccccccccccccc"
                )
            },
        )
    )
    _helper_mock_remote(mocker, exists=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 1, result.output
    assert f"✗ {consumer_stage}: failed" in result.output
    assert f"Input dependencies changed: {producer_stage}" in result.output
