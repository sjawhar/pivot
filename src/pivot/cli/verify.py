from __future__ import annotations

import asyncio
import json
import pathlib
from typing import TYPE_CHECKING, Literal, TypedDict

import click

from pivot import config, exceptions, project
from pivot import status as status_mod
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.remote import config as remote_config
from pivot.remote import storage as remote_mod
from pivot.remote import sync as transfer
from pivot.storage import lock
from pivot.types import OutputHash, PipelineStatus, PipelineStatusInfo, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


VerifyStatus = Literal["passed", "failed"]


class StageVerifyInfo(TypedDict):
    """Verification info for a single stage."""

    name: str
    status: VerifyStatus
    reason: str
    missing_files: list[str]


class VerifyOutput(TypedDict):
    """JSON output structure for verify command."""

    passed: bool
    stages: list[StageVerifyInfo]


def _extract_file_hashes(hash_infos: Mapping[str, OutputHash]) -> dict[str, str]:
    """Extract individual file hashes from a hash_info dict.

    Tree hashes (directory hashes) are computed, not cached - only individual
    file hashes are stored in the cache. For directories with manifests,
    extracts each manifest entry's hash.
    """
    result = dict[str, str]()
    for path, hash_info in hash_infos.items():
        if hash_info is None:
            continue
        if is_dir_hash(hash_info):
            for entry in hash_info["manifest"]:
                entry_path = str(pathlib.Path(path) / entry["relpath"])
                result[entry_path] = entry["hash"]
        else:
            result[path] = hash_info["hash"]
    return result


def _get_stage_lock_hashes(
    stage_name: str, state_dir: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Get output and dep file hashes from a stage's lock file.

    Returns (output_hashes, dep_hashes) where each is {path: hash}.

    For both outputs and deps, includes manifest entry hashes for directories.
    """
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    if lock_data is None:
        return {}, {}

    return (
        _extract_file_hashes(lock_data["output_hashes"]),
        _extract_file_hashes(lock_data["dep_hashes"]),
    )


def _get_stage_missing_hashes(
    stage_name: str,
    state_dir: Path,
    local_hashes: set[str],
    allow_missing: bool,
    project_root: Path,
) -> dict[str, list[str]]:
    """Get hashes missing from local cache for a stage.

    Returns {hash: [paths]} for hashes not in local cache.
    When allow_missing=True, also includes deps missing locally.
    """
    output_hashes, dep_hashes = _get_stage_lock_hashes(stage_name, state_dir)
    hash_to_paths = dict[str, list[str]]()

    # Check outputs: must be in local cache or (with allow_missing) on remote
    for path, hash_val in output_hashes.items():
        if hash_val not in local_hashes:
            hash_to_paths.setdefault(hash_val, []).append(path)

    # Check deps only when allow_missing: must exist on disk OR in cache OR on remote
    if allow_missing:
        for path, hash_val in dep_hashes.items():
            # Skip if file exists on disk
            if (project_root / path).exists():
                continue
            # Skip if hash is in local cache (can be restored)
            if hash_val in local_hashes:
                continue
            # File missing from disk and cache - need to check remote
            hash_to_paths.setdefault(hash_val, []).append(path)

    return hash_to_paths


def _create_remote_if_needed(allow_missing: bool) -> remote_mod.S3Remote | None:
    """Create remote connection if allow_missing mode requires it."""
    if not allow_missing:
        return None

    remotes = remote_config.list_remotes()
    if not remotes:
        raise exceptions.RemoteNotConfiguredError(
            "No remotes configured. --allow-missing requires a remote to check for files."
        )

    try:
        remote, _ = transfer.create_remote_from_name(None)
    except exceptions.RemoteError:
        raise
    except Exception as e:
        raise exceptions.RemoteError(f"Failed to create remote connection: {e}") from e
    return remote


def _verify_stages(
    pipeline_status: list[PipelineStatusInfo],
    state_dir: Path,
    cache_dir: Path,
    allow_missing: bool,
) -> tuple[bool, list[StageVerifyInfo]]:
    """Verify all stages and return pass/fail status with details.

    Batches all S3 existence checks into a single call for performance.
    """
    remote = _create_remote_if_needed(allow_missing)
    local_hashes = transfer.get_local_cache_hashes(cache_dir)
    project_root = project.get_project_root()

    # Phase 1: Collect missing hashes from all non-stale stages
    stage_hash_to_paths = dict[str, dict[str, list[str]]]()
    all_missing_hashes = set[str]()

    for stage_info in pipeline_status:
        if stage_info["status"] == PipelineStatus.STALE:
            continue
        hash_to_paths = _get_stage_missing_hashes(
            stage_info["name"], state_dir, local_hashes, allow_missing, project_root
        )
        stage_hash_to_paths[stage_info["name"]] = hash_to_paths
        all_missing_hashes.update(hash_to_paths.keys())

    # Phase 2: Single batched S3 existence check for all hashes
    remote_exists = dict[str, bool]()
    if all_missing_hashes and remote is not None:
        try:
            remote_exists = asyncio.run(remote.bulk_exists(list(all_missing_hashes)))
        except exceptions.RemoteError:
            raise
        except Exception as e:
            raise exceptions.RemoteError(f"Failed to check remote existence: {e}") from e

    # Phase 3: Build results using cached remote existence info
    results = list[StageVerifyInfo]()

    for stage_info in pipeline_status:
        stage_name = stage_info["name"]

        # Stale stages always fail
        if stage_info["status"] == PipelineStatus.STALE:
            results.append(
                StageVerifyInfo(
                    name=stage_name,
                    status="failed",
                    reason=stage_info["reason"] or "Stage is stale",
                    missing_files=[],
                )
            )
            continue

        hash_to_paths = stage_hash_to_paths[stage_name]

        # Determine missing files based on mode
        if not hash_to_paths:
            missing_files = list[str]()
        elif allow_missing and remote is not None:
            # With allow_missing, only files absent from remote are truly missing
            missing_files = [
                path
                for hash_val, paths in hash_to_paths.items()
                if not remote_exists.get(hash_val, False)
                for path in paths
            ]
        else:
            # Without allow_missing, all locally missing hashes are failures
            missing_files = [p for paths in hash_to_paths.values() for p in paths]

        # Create result based on whether any files are missing
        if missing_files:
            results.append(
                StageVerifyInfo(
                    name=stage_name,
                    status="failed",
                    reason=f"Missing files: {', '.join(missing_files)}",
                    missing_files=missing_files,
                )
            )
        else:
            results.append(
                StageVerifyInfo(name=stage_name, status="passed", reason="", missing_files=[])
            )

    all_passed = all(r["status"] == "passed" for r in results)
    return all_passed, results


def _output_text(passed: bool, results: list[StageVerifyInfo], quiet: bool) -> None:
    """Output verification results as formatted text."""
    if quiet:
        return

    click.echo("Verification passed" if passed else "Verification failed")
    click.echo()
    for stage in results:
        status_icon = "✓" if stage["status"] == "passed" else "✗"
        click.echo(f"  {status_icon} {stage['name']}: {stage['status']}")
        if stage["reason"]:
            click.echo(f"      {stage['reason']}")


def _output_json(passed: bool, results: list[StageVerifyInfo]) -> None:
    """Output verification results as JSON."""
    output = VerifyOutput(passed=passed, stages=results)
    click.echo(json.dumps(output, indent=2))


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option("--allow-missing", is_flag=True, help="Allow missing local files if on remote")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def verify(
    ctx: click.Context,
    stages: tuple[str, ...],
    allow_missing: bool,
    output_json: bool,
) -> None:
    """Verify pipeline was reproduced and outputs are available.

    Checks that all stages are cached (code, params, deps match lock files)
    and output files exist locally or on remote.

    With --allow-missing, both stage dependencies and outputs are verified
    to exist on the remote cache, enabling CI verification without local data.

    Use in CI pre-merge gates to ensure pipeline is reproducible.

    Exit codes:
      0 - Verification passed
      1 - Verification failed (stale stages or missing files)
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    # Validate stages exist
    stages_list = cli_helpers.stages_to_list(stages)
    cli_helpers.validate_stages_exist(stages_list)

    # Check if any stages are registered
    all_stages = cli_helpers.get_all_stages()
    if not all_stages:
        raise click.ClickException("No stages registered. Nothing to verify.")

    # Resolve directories - use same state_dir as get_pipeline_status for consistency
    state_dir = config.get_state_dir()
    cache_dir = config.get_cache_dir()

    # Get pipeline status (uses default state directory internally)
    pipeline_status, _ = status_mod.get_pipeline_status(
        stages_list,
        single_stage=False,
        all_stages=all_stages,
        stage_registry=cli_helpers.get_registry(),
        allow_missing=allow_missing,
    )

    if not pipeline_status:
        raise click.ClickException("No stages to verify.")

    # Verify stages
    passed, results = _verify_stages(pipeline_status, state_dir, cache_dir, allow_missing)

    # Output results
    if output_json:
        _output_json(passed, results)
    else:
        _output_text(passed, results, quiet)

    # Set exit code
    if not passed:
        raise SystemExit(1)
