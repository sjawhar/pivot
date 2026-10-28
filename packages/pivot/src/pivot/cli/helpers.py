from __future__ import annotations

import enum
import importlib
import pathlib
import sys
from typing import TYPE_CHECKING, Any, cast, override

import click
from tqdm.asyncio import tqdm as async_tqdm

if TYPE_CHECKING:
    from networkx import DiGraph

    from pivot.cli import CliContext
    from pivot.registry import PipelineLike, RegistryStageInfo
    from pivot.storage.store import WorkspaceStore

from pivot import config, display, exceptions, names, project, registry, types
from pivot.cli import decorators as cli_decorators
from pivot.storage import lock, track
from pivot.types import HashInfo


class NoPipelineError(exceptions.PivotError):
    """Raised when no Pipeline is available in context."""

    @override
    def format_user_message(self) -> str:
        return (
            "No pipeline definition found.\n"
            "\n"
            "This command requires a pipeline to be defined in one of:\n"
            "  - pipeline.py"
        )


def get_pipeline() -> PipelineLike:
    """Get Pipeline from context, raising NoPipelineError if not found."""
    pipeline = cli_decorators.get_pipeline_from_context()
    if pipeline is None:
        raise NoPipelineError()
    return pipeline


def list_stages() -> list[str]:
    """List stage names from Pipeline in context."""
    return get_pipeline().list_stages()


def get_stage(name: str) -> RegistryStageInfo:
    """Get stage info from Pipeline in context."""
    return get_pipeline().get_stage(name)


def get_all_stages() -> dict[str, RegistryStageInfo]:
    """Get all stages as a dict from Pipeline in context."""
    pipeline = get_pipeline()
    return {name: pipeline.get_stage(name) for name in pipeline.list_stages()}


def build_dag() -> DiGraph[str]:
    """Build DAG from Pipeline in context."""
    return get_pipeline().build_dag()


def get_workspace_store() -> WorkspaceStore | None:
    """Get a WorkspaceStore for display/path resolution, or None if no pipeline."""
    store_mod = importlib.import_module("pivot.storage.store")

    pipeline = cli_decorators.get_pipeline_from_context()
    if pipeline is None:
        return None
    return store_mod.WorkspaceStore(
        project_root=project.get_project_root(),
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    )


def require_workspace_store() -> WorkspaceStore:
    """Get the WorkspaceStore for the Pipeline in context, raising NoPipelineError if none."""
    store = get_workspace_store()
    if store is None:
        raise NoPipelineError()
    return store


def _json_default(obj: Any) -> Any:  # noqa: ANN401 - json.dumps default requires Any
    """Handle non-standard JSON types in JSONL output."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, set):
        # Convert to sorted list for JSON serialization (deterministic output)
        # Cast needed because isinstance(obj, set) doesn't narrow the element type
        return sorted(cast("set[Any]", obj), key=str)
    # Let json.dumps raise TypeError for truly unserializable types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def emit_jsonl(event: object) -> None:
    """Emit a single JSONL event to stdout with flush for streaming.

    Handles StrEnum values automatically. Sets are converted to sorted lists. A non-finite
    float is written as a string (``display.to_json``). Other non-serializable types will
    raise TypeError.
    """
    print(display.to_json(event, default=_json_default), flush=True)


def get_cli_context(ctx: click.Context) -> CliContext:
    """Get CLI context with defaults if not set."""
    if ctx.obj:
        return ctx.obj
    # Return dict matching CliContext structure to avoid circular import
    return {"verbose": False, "quiet": False}


def stages_to_list(stages: tuple[str, ...]) -> list[str] | None:
    """Convert Click's stage tuple to list or None if empty."""
    return list(stages) if stages else None


def resolve_stage_names(stages: list[str] | None) -> list[str] | None:
    """Resolve bare or pipeline-qualified stage names to registered stage names.

    Raises StageNotFoundError naming every target that matches no registered stage.
    """
    if not stages:
        return stages
    registered = list_stages()
    known = dict.fromkeys(registered)
    resolved = [names.resolve_stage_name(stage, known) for stage in stages]
    unknown = [stage for stage, name in zip(stages, resolved, strict=True) if name not in known]
    if unknown:
        raise exceptions.StageNotFoundError(unknown, available_stages=registered)
    return resolved


class TransferProgress:
    """Context manager for transfer progress bars."""

    _action: str
    _bar: async_tqdm[Any] | None
    _show: bool

    def __init__(self, action: str, *, quiet: bool = False) -> None:
        self._action = action
        self._bar = None
        self._show = sys.stderr.isatty() and not quiet

    def __enter__(self) -> TransferProgress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._bar is not None:
            self._bar.close()

    def callback(self, completed: int, total: int, filename: str) -> None:
        if not self._show:
            return
        if self._bar is None:
            self._bar = async_tqdm(
                total=total,
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
                unit="file",
            )
        self._bar.desc = f"{self._action} {display.escape_control(filename)}"
        self._bar.update(completed - self._bar.n)


def get_stage_output_info() -> tuple[dict[str, HashInfo], dict[str, pathlib.Path]]:
    """Get output hash info from lock files for cached stage outputs only.

    Non-cached outputs (e.g. Metric with cache=False) are excluded —
    they are git-tracked and not Pivot's responsibility to restore.

    Uses per-stage state_dir from the registry for lock file lookup. Also returns each
    output's stage state_dir, whose StateDB holds the file hashes the stage's runs and
    explanations look up.
    """
    result = dict[str, HashInfo]()
    state_dirs = dict[str, pathlib.Path]()
    store = get_workspace_store()
    if store is None:
        return result, state_dirs

    for stage_name in list_stages():
        stage_info = get_stage(stage_name)
        ref_by_identity = {out.identity: out for out in stage_info["outs"]}
        cached_identities = {
            out.identity for out in stage_info["outs"] if out.tag is not types.ArtifactTag.METRIC
        }

        stage_state_dir = registry.get_stage_state_dir(stage_info, config.get_state_dir())
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_state_dir))
        lock_data = stage_lock.read()
        if lock_data:
            for identity, out_hash in lock_data["output_hashes"].items():
                if identity not in cached_identities:
                    continue
                ref = ref_by_identity.get(identity)
                if ref is None:
                    continue
                resolved = str(store.resolve_display_path(ref))
                result[resolved] = out_hash
                state_dirs[resolved] = stage_state_dir

    return result, state_dirs


def get_locally_tracked_targets() -> list[str]:
    """Return paths of all files tracked locally via .pvt files and stage lockfiles."""
    proj_root = project.get_project_root()
    targets = list(track.discover_pvt_files(proj_root))

    pipeline = cli_decorators.get_pipeline_from_context()
    if pipeline is not None:
        targets.extend(get_stage_output_info()[0])

    return targets


def print_transfer_errors(errors: list[str], max_shown: int = 5) -> None:
    """Print transfer errors with truncation for long lists."""
    if not errors:
        return
    for err in errors[:max_shown]:
        click.echo(f"  Error: {display.safe_message(err)}", err=True)
    if len(errors) > max_shown:
        click.echo(f"  ... and {len(errors) - max_shown} more errors", err=True)
