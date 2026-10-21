from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import pathlib
import shlex
import tempfile
from typing import TYPE_CHECKING, TypedDict, cast

import aiohttp
import yaml

from pivot import exceptions, project, types
from pivot.config import io as config_io
from pivot.remote import config as remote_config
from pivot.remote import git_archive, github, protocol
from pivot.storage import cache, track
from pivot.storage import lock as storage_lock

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.types import DirManifestEntry, OutEntry, StorageLockData

logger = logging.getLogger(__name__)


class ResolvedImport(TypedDict):
    stage: str
    path: str
    hash: str
    size: int
    remote_url: str
    rev_lock: str


class ImportResult(TypedDict):
    pvt_path: str
    data_path: str
    downloaded: bool


async def _download_verified(remote_url: str, file_hash: str, data_path: pathlib.Path) -> int:
    """Download ``file_hash`` to ``data_path`` only if its content matches; return its size.

    The file is downloaded to a temp file in the destination directory and moved over
    ``data_path`` after its hash is checked, so a corrupt remote copy leaves any existing
    file (the last good import) untouched.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=data_path.parent, prefix=f".{data_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        await protocol.open_remote(remote_url).download_file(file_hash, tmp_path)
        actual_hash, _ = cache.hash_file(tmp_path)
        if actual_hash != file_hash:
            raise exceptions.RemoteError(f"Hash mismatch: expected {file_hash}, got {actual_hash}")
        size = tmp_path.stat().st_size
        os.replace(tmp_path, data_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
    return size


def _maybe_token(repo_url: str, token: str | None) -> str | None:
    if token is not None:
        return token
    if github.is_github_url(repo_url):
        return github.get_token()
    return None


async def _run_blocking[T](func: Callable[..., T], *args: object) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args))


async def _read_remote_file(
    repo_url: str,
    path: str,
    rev: str,
    token: str | None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> bytes | None:
    if github.is_github_url(repo_url):
        owner, repo = github.parse_github_url(repo_url)
        return await github.read_file(
            owner, repo, path, rev, _maybe_token(repo_url, token), session=session
        )
    return await _run_blocking(git_archive.read_file_from_remote_repo, repo_url, path, rev)


async def _resolve_ref(
    repo_url: str,
    ref: str,
    token: str | None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> str | None:
    if github.is_github_url(repo_url):
        owner, repo = github.parse_github_url(repo_url)
        return await github.resolve_ref(
            owner, repo, ref, _maybe_token(repo_url, token), session=session
        )
    return await _run_blocking(git_archive.resolve_ref_from_remote_repo, repo_url, ref)


async def read_remote_config(
    repo_url: str,
    rev: str,
    token: str | None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> tuple[str, dict[str, object]]:
    raw = await _read_remote_file(repo_url, ".pivot/config.yaml", rev, token, session=session)
    if raw is None:
        raise exceptions.RemoteError("Remote repo is missing .pivot/config.yaml")

    try:
        data = cast("object", yaml.safe_load(raw))
    except yaml.YAMLError as e:
        raise exceptions.RemoteError(f"Invalid YAML in remote config: {e}") from e

    if not isinstance(data, dict):
        raise exceptions.RemoteError("Remote config is not a mapping")

    config_data = cast("dict[str, object]", data)
    remotes_raw = config_data.get("remotes")
    if not isinstance(remotes_raw, dict) or not remotes_raw:
        raise exceptions.RemoteError("Remote config has no remotes")

    remotes_data = cast("dict[object, object]", remotes_raw)
    remotes = {str(k): str(v) for k, v in remotes_data.items()}

    default_remote = config_data.get("default_remote")
    if default_remote is not None and not isinstance(default_remote, str):
        # Warn-first: the source config is a saved input outside this repo.
        logger.warning(
            "Remote config default_remote in %s must be a remote name, got %r; ignoring it",
            repo_url,
            default_remote,
        )
        default_remote = None
    if default_remote is None:
        if len(remotes) == 1:
            default_remote = next(iter(remotes.keys()))
        else:
            raise exceptions.RemoteError("Remote config has multiple remotes but no default_remote")

    if default_remote not in remotes:
        raise exceptions.RemoteError(f"Default remote '{default_remote}' not found in config")

    url = str(remotes[default_remote])
    _ = remote_config.validate_remote_url(url)
    logger.info("Using source remote: %s", url)

    # Two parses on purpose: remotes are read above as 0.5.2 reads them (YAML 1.1, as the source's
    # `pivot remote` reads them), and the returned config, which supplies core.state_dir, as every
    # other Pivot command reads its config (YAML 1.2). Each key means what it means to the source.
    try:
        source_config = config_io.parse_config_yaml(raw, "remote config")
    except exceptions.ConfigError as e:
        raise exceptions.RemoteError(str(e)) from e
    return url, source_config


def _remote_state_dir(config_data: dict[str, object]) -> str:
    """Return the source repo's normalized state path at the imported revision."""
    core = config_data.get("core")
    if not isinstance(core, dict):
        return ".pivot"
    state_dir = cast("dict[str, object]", core).get("state_dir")
    if state_dir is None:
        return ".pivot"
    if (
        not isinstance(state_dir, str)
        or not state_dir
        or pathlib.PurePosixPath(state_dir).is_absolute()
    ):
        raise exceptions.RemoteError(
            f"Source repo's core.state_dir must be a path inside the repo, got {state_dir!r}"
        )

    normalized_parts = list[str]()
    for part in pathlib.PurePosixPath(state_dir).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized_parts:
                raise exceptions.RemoteError(
                    f"Source repo's core.state_dir is outside the repo: {state_dir!r}"
                )
            normalized_parts.pop()
        else:
            normalized_parts.append(part)
    return pathlib.PurePosixPath(*normalized_parts).as_posix()


async def _list_github_lock_paths(
    owner: str,
    repo: str,
    directory: str,
    rev: str,
    token: str | None,
    session: aiohttp.ClientSession | None,
) -> list[str] | None:
    """List lock files under directory, relative to it, descending into pipeline dirs.

    The contents API lists a single level, and pipeline-prefixed stage names
    (``smoke/total``) keep their lock files in subdirectories.
    """
    entries = await github.list_directory(owner, repo, directory, rev, token, session=session)
    if entries is None:
        return None
    lock_paths = [
        entry["name"]
        for entry in entries
        if entry["type"] == "file" and entry["name"].endswith(".lock")
    ]
    subdirs = [entry["name"] for entry in entries if entry["type"] == "dir"]
    nested = await asyncio.gather(
        *[
            _list_github_lock_paths(owner, repo, f"{directory}/{d}", rev, token, session)
            for d in subdirs
        ]
    )
    for subdir, sub_paths in zip(subdirs, nested, strict=True):
        if sub_paths is not None:
            lock_paths.extend(f"{subdir}/{p}" for p in sub_paths)
    return lock_paths


async def list_remote_lock_files(
    repo_url: str,
    rev: str,
    token: str | None,
    stages_dir: str,
    *,
    session: aiohttp.ClientSession | None = None,
) -> list[str]:
    if github.is_github_url(repo_url):
        owner, repo = github.parse_github_url(repo_url)
        entries = await _list_github_lock_paths(
            owner, repo, stages_dir, rev, _maybe_token(repo_url, token), session
        )
    else:
        entries = await _run_blocking(
            git_archive.list_directory_from_remote_repo,
            repo_url,
            stages_dir,
            rev,
        )

    if entries is None:
        raise exceptions.RemoteError(f"Remote repo has no {stages_dir} directory")

    # Entries are relative to the lock directory, so nested paths are pipeline-prefixed names.
    stage_names = [entry.removesuffix(".lock") for entry in entries if entry.endswith(".lock")]

    if not stage_names:
        raise exceptions.RemoteError("Remote repo has no lock files")

    return stage_names


async def read_remote_lock_file(
    repo_url: str,
    stage_name: str,
    rev: str,
    token: str | None,
    stages_dir: str,
    *,
    session: aiohttp.ClientSession | None = None,
) -> StorageLockData | None:
    path = f"{stages_dir}/{stage_name}.lock"
    raw = await _read_remote_file(repo_url, path, rev, token, session=session)
    if raw is None:
        return None
    try:
        data = cast("object", yaml.safe_load(raw))
    except yaml.YAMLError:
        logger.warning("Failed to parse lock file for stage '%s'", stage_name)
        return None
    if not storage_lock.is_lock_data(data):
        logger.warning("Invalid lock file structure for stage '%s'", stage_name)
        return None
    return data


def _entry_identity(stage: str, entry: OutEntry) -> types.ArtifactIdentity:
    """Identity of a lock output entry: ``stage`` or ``stage:key``."""
    return types.ArtifactIdentity(stage, entry["key"])


def _entry_size(entry: OutEntry, identity: str) -> int:
    if "size" in entry:
        return int(entry["size"])
    if "manifest" in entry:
        return sum(int(m["size"]) for m in entry["manifest"])
    logger.warning("Lock entry '%s' has no size information", identity)
    return 0


def _iter_manifest_paths(
    identity: types.ArtifactIdentity, entry: OutEntry
) -> list[tuple[str, DirManifestEntry]]:
    """Each file of a directory output, named as a stage depending on it names it."""
    if "manifest" not in entry:
        return []
    return [
        (types.identity_key(types.directory_file_identity(identity, m["relpath"])), m)
        for m in entry["manifest"]
    ]


def _renamed_message(old_name: str, directory: str, relpath: str, new_name: str) -> str:
    return (
        f"'{old_name}' is the old name of file '{relpath}' of directory output "
        + f"'{directory}'; it is now '{new_name}'"
    )


def _refuse_old_directory_file_name(
    pvt_path: pathlib.Path, source: track.ImportSource, rev: str
) -> None:
    """Refuse an import recorded as ``stage/path``, the name of a file of a directory output
    before it became ``stage:path``; the error gives the command that re-imports it."""
    prefix = f"{source['stage']}/"
    if not source["path"].startswith(prefix):
        return
    relpath = source["path"].removeprefix(prefix)
    new_name = types.identity_key(
        types.directory_file_identity(types.ArtifactIdentity(source["stage"], None), relpath)
    )
    out = project.to_relative_path(track.get_data_path(pvt_path))
    command = shlex.join(
        ["pivot", "import", source["repo"], new_name, "--rev", rev, "--out", out, "--force"]
    )
    raise exceptions.PivotError(
        _renamed_message(source["path"], source["stage"], relpath, new_name)
        + f". Re-import it: {command}"
    )


async def resolve_remote_path(
    repo_url: str,
    path: str,
    rev: str,
    token: str | None,
) -> ResolvedImport:
    async with aiohttp.ClientSession() as session:
        return await _resolve_remote_path_with_session(repo_url, path, rev, token, session)


async def _resolve_remote_path_with_session(
    repo_url: str,
    path: str,
    rev: str,
    token: str | None,
    session: aiohttp.ClientSession,
) -> ResolvedImport:
    gh_session = session if github.is_github_url(repo_url) else None
    rev_lock = await _resolve_ref(repo_url, rev, token, session=gh_session)
    if rev_lock is None:
        raise exceptions.RemoteError(f"Unable to resolve ref '{rev}'")

    remote_url, config_data = await read_remote_config(
        repo_url, rev_lock, token, session=gh_session
    )
    source_state_dir = _remote_state_dir(config_data)
    stages_dir = "stages" if source_state_dir == "." else f"{source_state_dir}/stages"
    if (
        source_state_dir != "."
        and not github.is_github_url(repo_url)
        and await _run_blocking(
            git_archive.path_is_symlink_from_remote_repo,
            repo_url,
            source_state_dir,
            rev_lock,
        )
    ):
        raise exceptions.RemoteError(
            f"Source repo's core.state_dir is a symlink and cannot be imported: {source_state_dir}"
        )

    # Use resolved SHA for all subsequent reads to avoid TOCTOU race
    stage_names = await list_remote_lock_files(
        repo_url, rev_lock, token, stages_dir, session=gh_session
    )
    lock_datas = await asyncio.gather(
        *[
            read_remote_lock_file(repo_url, stage, rev_lock, token, stages_dir, session=gh_session)
            for stage in stage_names
        ]
    )

    matches = list[tuple[str, str, str, int]]()
    available = set[str]()
    renamed: str | None = None

    for stage, lock_data in zip(stage_names, lock_datas, strict=True):
        if lock_data is None:
            continue
        for entry in lock_data["outs"]:
            entry_identity = _entry_identity(stage, entry)
            identity = types.identity_key(entry_identity)
            manifest_paths = _iter_manifest_paths(entry_identity, entry)
            available.add(identity)
            available.update(full_path for full_path, _ in manifest_paths)

            if identity == path:
                if "manifest" in entry:
                    file_example = types.identity_key(
                        types.directory_file_identity(entry_identity, "{filename}")
                    )
                    message = (
                        "Cannot import directory output "
                        + f"'{path}'. Import individual files instead, "
                        + f"e.g., '{file_example}'"
                    )
                    raise exceptions.PivotError(message)
                matches.append((stage, path, entry["hash"], _entry_size(entry, identity)))
                continue

            for full_path, manifest_entry in manifest_paths:
                if full_path == path:
                    matches.append(
                        (
                            stage,
                            path,
                            manifest_entry["hash"],
                            int(manifest_entry["size"]),
                        )
                    )
                elif f"{identity}/{manifest_entry['relpath']}" == path:
                    renamed = _renamed_message(path, identity, manifest_entry["relpath"], full_path)

    if not matches and renamed is not None:
        raise exceptions.PivotError(renamed)

    if not matches:
        max_shown = 10
        sorted_available = sorted(available)
        if len(sorted_available) > max_shown:
            shown = ", ".join(sorted_available[:max_shown])
            available_list = f"{shown} (and {len(sorted_available) - max_shown} more)"
        elif sorted_available:
            available_list = ", ".join(sorted_available)
        else:
            available_list = "(none)"
        raise exceptions.PivotError(
            f"Artifact '{path}' not found in remote outputs. Available outputs: {available_list}"
        )

    if len(matches) > 1:
        stages = ", ".join(sorted({stage for stage, _, _, _ in matches}))
        raise exceptions.PivotError(
            f"'{path}' names more than one output (of stage {stages}); "
            + "the source pipeline must rename one"
        )

    stage, matched_path, hash_, size = matches[0]
    return ResolvedImport(
        stage=stage,
        path=matched_path,
        hash=hash_,
        size=size,
        remote_url=remote_url,
        rev_lock=rev_lock,
    )


async def import_artifact(
    repo_url: str,
    path: str,
    *,
    rev: str = "main",
    out: str | None = None,
    force: bool = False,
    no_download: bool = False,
    project_root: pathlib.Path | None = None,
) -> ImportResult:
    resolved = await resolve_remote_path(repo_url, path, rev, None)

    if project_root is None:
        project_root = project.get_project_root()

    dest = out if out is not None else path.replace(":", "/")
    data_path = project.normalize_path(dest, base=project_root)
    if not data_path.is_relative_to(project_root.absolute()):
        raise exceptions.PivotError(
            f"Import destination must be within project root. Got: {data_path}"
        )
    pvt_path = track.get_pvt_path(data_path)

    if data_path.exists() and not force:
        raise exceptions.PivotError(f"'{data_path}' already exists. Use --force to overwrite.")
    if pvt_path.exists() and not force:
        raise exceptions.PivotError(f"'{pvt_path}' already exists. Use --force to overwrite.")

    if force and data_path.exists() and data_path.is_dir():
        raise exceptions.PivotError("Cannot overwrite directory with import")

    pvt_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = False
    file_size = resolved["size"]
    if not no_download:
        file_size = await _download_verified(resolved["remote_url"], resolved["hash"], data_path)
        downloaded = True

    source = track.ImportSource(
        repo=repo_url,
        rev=rev,
        rev_lock=resolved["rev_lock"],
        stage=resolved["stage"],
        path=resolved["path"],
        remote=resolved["remote_url"],
    )
    pvt_data = track.PvtData(
        path=data_path.name,
        hash=resolved["hash"],
        size=file_size,
        source=source,
    )

    track.write_pvt_file(pvt_path, pvt_data)

    return ImportResult(
        pvt_path=str(pvt_path),
        data_path=str(data_path),
        downloaded=downloaded,
    )


class UpdateCheck(TypedDict):
    available: bool
    current_rev: str
    latest_rev: str


class UpdateResult(TypedDict):
    downloaded: bool
    metadata_updated: bool
    updated: bool
    old_rev: str
    new_rev: str
    path: str


async def check_for_update(pvt_path: pathlib.Path, pvt_data: track.PvtData) -> UpdateCheck:
    """Check if import has updates. Resolves current ref, compares to rev_lock."""
    if "source" not in pvt_data:
        raise exceptions.PivotError("Not an import")
    source = pvt_data["source"]
    _refuse_old_directory_file_name(pvt_path, source, source["rev"])
    rev_lock = await _resolve_ref(source["repo"], source["rev"], None)
    if rev_lock is None:
        raise exceptions.RemoteError(f"Cannot resolve ref '{source['rev']}' from {source['repo']}")
    return UpdateCheck(
        available=rev_lock != source["rev_lock"],
        current_rev=source["rev_lock"],
        latest_rev=rev_lock,
    )


async def update_import(pvt_path: pathlib.Path, *, new_rev: str | None = None) -> UpdateResult:
    """Update an imported artifact. Re-resolves ref, re-downloads if hash changed."""
    pvt_data = track.read_pvt_file(pvt_path)
    if pvt_data is None:
        raise exceptions.PivotError(f"Invalid .pvt file: {pvt_path}")
    if "source" not in pvt_data:
        raise exceptions.PivotError(f"Not an import: {pvt_path}")

    source = pvt_data["source"]
    rev = new_rev if new_rev is not None else source["rev"]
    old_rev = source["rev_lock"]
    _refuse_old_directory_file_name(pvt_path, source, rev)

    # Re-resolve at new/current rev
    resolved = await resolve_remote_path(source["repo"], source["path"], rev, None)

    downloaded = False
    data_path = track.get_data_path(pvt_path)
    need_download = resolved["hash"] != pvt_data["hash"] or not data_path.exists()

    file_size = resolved["size"]
    if need_download:
        file_size = await _download_verified(resolved["remote_url"], resolved["hash"], data_path)
        downloaded = True
    elif data_path.exists():
        file_size = data_path.stat().st_size

    metadata_changed = (
        resolved["rev_lock"] != old_rev
        or resolved["hash"] != pvt_data["hash"]
        or rev != source["rev"]
    )
    if metadata_changed:
        new_source = track.ImportSource(
            repo=source["repo"],
            rev=rev,
            rev_lock=resolved["rev_lock"],
            stage=resolved["stage"],
            path=resolved["path"],
            remote=resolved["remote_url"],
        )
        new_pvt = track.PvtData(
            path=pvt_data["path"],
            hash=resolved["hash"],
            size=file_size,
            source=new_source,
        )
        track.write_pvt_file(pvt_path, new_pvt)

    return UpdateResult(
        downloaded=downloaded,
        metadata_updated=metadata_changed,
        updated=downloaded,
        old_rev=old_rev,
        new_rev=resolved["rev_lock"],
        path=str(data_path),
    )
