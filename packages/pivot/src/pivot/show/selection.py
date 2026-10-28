"""Select the data files a ``pivot diff`` target names at git HEAD and in the workspace."""

from __future__ import annotations

import collections
import logging
import os
import pathlib
from typing import TYPE_CHECKING, TypedDict

from pivot import config, exceptions, git, loaders, project, registry, types
from pivot.show import common
from pivot.show import data as data_module
from pivot.storage import cache, listing, lock
from pivot.types import DataFileFormat

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pivot.storage import store as store_mod

logger = logging.getLogger(__name__)


# A directory's listing at git HEAD: where its committed listing file is (relative to the
# project root), or the listing an older pivot's HEAD lock holds inline.
type _HeadListing = str | listing.DirListing


class _DiffOutput(TypedDict):
    """A path ``pivot diff`` can compare: its HEAD entry and current output state."""

    path: str
    head: types.HashInfo | None
    head_listing: _HeadListing
    in_workspace: bool
    is_directory: bool
    changed_kind: tuple[str, str] | None


class DataSelection(TypedDict):
    """The hashes and workspace paths a diff's targets select."""

    head_hashes: dict[str, str | None]
    workspace_hashes: dict[str, str]
    workspace_paths: dict[str, pathlib.Path]


def _format_from_loader(
    loader: loaders.Reader[object] | loaders.Writer[object] | loaders.Loader[object, object],
) -> DataFileFormat | None:
    if isinstance(loader, loaders.CSV):
        return DataFileFormat.CSV
    if isinstance(loader, loaders.JSON):
        return DataFileFormat.JSON
    if isinstance(loader, (loaders.JSONL, loaders.DataFrameJSONL)):
        return DataFileFormat.JSONL
    return None


def _parents(relpath: str) -> list[str]:
    """Every directory above ``relpath``, nearest first, short of the root."""
    return [str(parent) for parent in pathlib.PurePosixPath(relpath).parents[:-1]]


def _member_selected(relpath: str, prefixes: frozenset[str] | None) -> bool:
    """Whether a directory output's member is selected: every member, or one at or below a
    selected prefix."""
    return (
        prefixes is None
        or relpath in prefixes
        or any(parent in prefixes for parent in _parents(relpath))
    )


def _kind(is_directory: bool) -> str:
    return "a directory" if is_directory else "a file"


def _diff_outputs(
    stage_infos: Mapping[str, registry.RegistryStageInfo],
    store: store_mod.WorkspaceStore,
    proj_root: pathlib.Path,
) -> tuple[dict[str, _DiffOutput], dict[types.ArtifactIdentity, list[str]]]:
    """Map each comparable path to what HEAD recorded there and the output that lives there.

    Each output's HEAD entry is placed at the path its HEAD lock records, and its workspace
    side at its current path, so an output that moved is compared as removed at the old path
    and added at the new one. A moved entry is dropped when its HEAD path is a current output's
    path that has its own HEAD entry, or an earlier moved entry's HEAD path. A path compares as a
    directory when the registry tags its output DIRECTORY, its HEAD entry records a directory, or
    it is a directory on disk. Any other path is comparable when its loader or extension is CSV,
    JSON or JSONL. Also returns the paths each output identity owns, at HEAD and now, for stage
    targets.
    """
    outs_by_stage = dict[str, list[types.ArtifactRef]]()
    for stage_name, stage_info in stage_infos.items():
        outs = [
            out
            for out in stage_info["outs"]
            if out.tag not in {types.ArtifactTag.METRIC, types.ArtifactTag.PLOT}
        ]
        if outs:
            outs_by_stage[stage_name] = outs

    lock_data_by_stage = common.read_lock_files_from_head(list(outs_by_stage))
    moved_heads = dict[str, tuple[types.HashInfo, list[str], _HeadListing]]()
    current = list[
        tuple[types.ArtifactRef, pathlib.Path, str, types.HashInfo | None, _HeadListing]
    ]()
    paths_by_identity = dict[types.ArtifactIdentity, list[str]]()
    for stage_name, outs in outs_by_stage.items():
        committed = lock_data_by_stage[stage_name]
        entries = (
            {} if committed is None else {entry["key"]: entry for entry in committed.data["outs"]}
        )
        stages_dir = lock.get_stages_dir(
            registry.get_stage_state_dir(stage_infos[stage_name], config.locate_state_dir())
        )
        for out in outs:
            output_path = store.resolve_display_path(out)
            rel_path = project.to_relative_path(output_path, proj_root)
            paths = paths_by_identity.setdefault(out.identity, [rel_path])
            entry = entries.get(out.identity.key)
            head = None if entry is None else lock.entry_hash_info(entry)
            listing_at_head: _HeadListing = ""
            if head is not None and types.is_dir_hash(head):
                listing_at_head = project.to_relative_path(
                    listing.output_listing_path(stages_dir, out.identity), proj_root
                )
                if committed is not None and out.identity.key in committed.listings:
                    listing_at_head = committed.listings[out.identity.key]
            head_path = rel_path if entry is None or "path" not in entry else entry["path"]
            if head is not None and head_path != rel_path:
                moved_heads.setdefault(head_path, (head, paths, listing_at_head))
                head = None
            current.append((out, output_path, rel_path, head, listing_at_head))

    result = dict[str, _DiffOutput]()
    for out, output_path, rel_path, own_head, own_listing in current:
        moved = moved_heads.pop(rel_path, None)
        head = own_head
        head_listing = own_listing
        if head is None and moved is not None:
            head, moved_paths, head_listing = moved
            moved_paths.append(rel_path)
        head_is_directory = head is not None and types.is_dir_hash(head)
        workspace_is_directory = output_path.is_dir()
        is_directory = (
            out.tag is types.ArtifactTag.DIRECTORY or head_is_directory or workspace_is_directory
        )
        changed_kind = None
        if (
            head is not None
            and output_path.exists()
            and head_is_directory != workspace_is_directory
        ):
            changed_kind = (_kind(head_is_directory), _kind(workspace_is_directory))
        if (
            is_directory
            or _format_from_loader(out.format) is not None
            or data_module.detect_format(pathlib.Path(rel_path)) is not DataFileFormat.UNKNOWN
        ):
            result[rel_path] = _DiffOutput(
                path=rel_path,
                head=head,
                head_listing=head_listing,
                in_workspace=True,
                is_directory=is_directory,
                changed_kind=changed_kind,
            )

    for head_path, (head, moved_paths, head_listing) in moved_heads.items():
        head_is_directory = types.is_dir_hash(head)
        if (
            head_is_directory
            or data_module.detect_format(pathlib.Path(head_path)) is not DataFileFormat.UNKNOWN
        ):
            moved_paths.append(head_path)
            result[head_path] = _DiffOutput(
                path=head_path,
                head=head,
                head_listing=head_listing,
                in_workspace=False,
                is_directory=head_is_directory,
                changed_kind=None,
            )
    return result, paths_by_identity


def _select_outputs(
    targets: Sequence[str],
    outputs: Mapping[str, _DiffOutput],
    paths_by_identity: Mapping[types.ArtifactIdentity, list[str]],
    stage_infos: Mapping[str, registry.RegistryStageInfo],
    proj_root: pathlib.Path,
) -> tuple[dict[str, frozenset[str] | None], list[tuple[str, str | None]]]:
    """Map each selected output to its selected member prefixes, or None for every member.

    ``.`` selects every output, a stage the paths its outputs have at HEAD and now, and a path
    the output at it, every output below it, or, inside a directory output, the members at or
    below it. Also returns, in target order, each target that selects no output (paired with
    None) and each target inside a directory output (paired with its path), which matches only
    if a selected member lies at or below that path.
    """
    from pivot.cli import targets as cli_targets

    below = collections.defaultdict[str, list[str]](list)
    for output_path in outputs:
        for parent in _parents(output_path):
            below[parent].append(output_path)

    selected = dict[str, set[str] | None]()
    unresolved = list[tuple[str, str | None]]()
    for target in targets:
        resolved = cli_targets.resolve_cli_target(target, stage_infos)
        if resolved is not None:
            hits = [
                path
                for ref in resolved["refs"]
                for path in paths_by_identity.get(ref.identity, [])
                if path in outputs
            ]
        else:
            path = project.to_relative_path(project.normalize_path(target), proj_root)
            if path == ".":
                hits = list(outputs)
            elif path in outputs:
                hits = [path]
            else:
                hits = below.get(path, [])
            if not hits:
                container = next((parent for parent in _parents(path) if parent in outputs), None)
                if container is not None and outputs[container]["is_directory"]:
                    prefixes = selected.setdefault(container, set[str]())
                    if prefixes is not None:
                        prefixes.add(path[len(container) + 1 :])
                    unresolved.append((target, path))
                    continue
        for hit in hits:
            selected[hit] = None
        if not hits:
            unresolved.append((target, None))

    selection = {
        path: None if prefixes is None else frozenset(prefixes)
        for path, prefixes in selected.items()
    }
    return selection, unresolved


def _head_listing(at_head: _HeadListing, record: types.DirHash) -> listing.DirListing | None:
    """A directory's listing at git HEAD: the inline one an older pivot's lock held, else its
    committed listing file there or the cache's object (``listing.read_found``); None when
    none holds it."""
    if not isinstance(at_head, str):
        return at_head
    content = git.read_file_from_revision(at_head, "HEAD")
    try:
        return listing.read_found(
            content, f"{at_head} at HEAD", record, config.locate_cache_dir() / "files"
        )
    except exceptions.ListingMissingError:
        return None


def _head_members(output: _DiffOutput, prefixes: frozenset[str] | None) -> dict[str, str | None]:
    """The selected output's comparable files and hashes at git HEAD."""
    if not output["is_directory"]:
        return {output["path"]: None if output["head"] is None else output["head"]["hash"]}
    if output["head"] is None or not types.is_dir_hash(output["head"]):
        return {}
    members = _head_listing(output["head_listing"], output["head"])
    if members is None:
        return {}
    return {
        f"{output['path']}/{entry['relpath']}": entry["hash"]
        for entry in members["files"]
        if _member_selected(entry["relpath"], prefixes)
        and data_module.detect_format(pathlib.Path(entry["relpath"])) is not DataFileFormat.UNKNOWN
    }


def _workspace_members(
    output: _DiffOutput,
    prefixes: frozenset[str] | None,
    proj_root: pathlib.Path,
    cache_dir: pathlib.Path,
) -> dict[str, tuple[str, pathlib.Path]]:
    """The selected output's comparable workspace files, each with its hash and path.

    A directory output's members come from the record-time walker: a symlink it leaves out,
    and everything below one, is not a member and is never read. The warning names only the
    left-out symlinks inside the selected members, under the output's own path. A path only
    HEAD recorded, the old path of an output that moved, has no workspace side and is never
    read.
    """
    location = proj_root / output["path"]
    if not output["in_workspace"]:
        return {}
    if not output["is_directory"]:
        if not location.exists():
            return {}
        file_hash, _ = cache.hash_file(location)
        return {output["path"]: (file_hash, location)}
    if not location.is_dir():
        return {}

    result = dict[str, tuple[str, pathlib.Path]]()
    files = list[tuple[str, os.DirEntry[str]]]()
    left_out = list[pathlib.Path]()
    cache.walk_artifact_members(os.fspath(location), "", cache_dir, files, [], left_out)
    for relpath, entry in files:
        file_path = pathlib.Path(entry.path)
        if not _member_selected(relpath, prefixes):
            continue
        if data_module.detect_format(file_path) is DataFileFormat.UNKNOWN:
            continue
        file_hash, _ = cache.hash_file(file_path)
        if not cache.is_artifact_member_content(file_path, file_hash, cache_dir):
            left_out.append(file_path)
            continue
        result[f"{output['path']}/{relpath}"] = (file_hash, file_path)

    selected_parents = set[str]()
    if prefixes is not None:
        selected_parents = {parent for prefix in prefixes for parent in _parents(prefix)}
    named = list[pathlib.Path]()
    for link in left_out:
        link_relpath = link.relative_to(location).as_posix()
        if _member_selected(link_relpath, prefixes) or link_relpath in selected_parents:
            named.append(link)
    if named:
        cache.log_left_out_symlinks(location, named)
    return result


def select_data_files(targets: Sequence[str]) -> DataSelection | None:
    """Select the data files ``targets`` name, at git HEAD and in the workspace.

    One selection serves both sides (see ``_select_outputs``), and one walk per selected
    directory output builds its workspace members. Both sides are keyed by path: an output
    that moved is removed at the path git HEAD recorded and added at its current one. A path
    that holds a file at git HEAD and a directory in the workspace, or the reverse, is named in
    a warning and not compared, and every target that selects no CSV, JSON or JSONL file on
    either side is named in one warning. Returns None when no registered stage has an output
    `pivot diff` can compare.
    """
    from pivot.cli import helpers as cli_helpers

    store = cli_helpers.require_workspace_store()
    proj_root = project.get_project_root()
    stage_infos = cli_helpers.get_all_stages()
    outputs, paths_by_identity = _diff_outputs(stage_infos, store, proj_root)
    if not outputs:
        return None

    head_hashes = dict[str, str | None]()
    workspace_hashes = dict[str, str]()
    workspace_paths = dict[str, pathlib.Path]()
    cache_dir = config.locate_cache_dir() / "files"
    selection, unresolved = _select_outputs(
        targets, outputs, paths_by_identity, stage_infos, proj_root
    )
    for path, prefixes in selection.items():
        output = outputs[path]
        if output["changed_kind"] is not None:
            head_kind, workspace_kind = output["changed_kind"]
            logger.warning(
                "Output %s is %s at git HEAD and %s in the workspace, so it is not compared",
                path,
                head_kind,
                workspace_kind,
            )
            continue
        head_hashes.update(_head_members(output, prefixes))
        for member, (file_hash, file_path) in _workspace_members(
            output, prefixes, proj_root, cache_dir
        ).items():
            workspace_hashes[member] = file_hash
            workspace_paths[member] = file_path

    covered = set[str]()
    if any(path is not None for _, path in unresolved):
        for member in head_hashes.keys() | workspace_hashes.keys():
            covered.add(member)
            covered.update(_parents(member))
    unmatched = [target for target, path in unresolved if path is None or path not in covered]
    if unmatched:
        logger.warning(
            "No CSV, JSON or JSONL stage output matches %s, so it is not compared",
            ", ".join(repr(target) for target in unmatched),
        )
    return DataSelection(
        head_hashes=head_hashes,
        workspace_hashes=workspace_hashes,
        workspace_paths=workspace_paths,
    )
