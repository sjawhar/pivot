"""Presentation layer: materializes CAS refs as workspace symlinks.

After a successful pipeline run, creates a conventional directory tree
(data/, metrics/, plots/) with symlinks pointing to CAS ref paths.
This gives users browsable output at familiar locations while the
actual data lives in content-addressed storage.
"""

from __future__ import annotations

import logging
import pathlib  # noqa: TCH003 - used at runtime for path operations
from typing import TYPE_CHECKING

from pivot import compose, types
from pivot.storage import store as store_mod

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo

logger = logging.getLogger(__name__)


def present(
    *,
    project_root: pathlib.Path,
    pipeline_name: str,
    cache_dir: pathlib.Path,
    stages: dict[str, RegistryStageInfo],
) -> None:
    """Materialize CAS ref symlinks into workspace display paths.

    For each output of each stage, creates a symlink at the conventional
    workspace location (e.g., data/pipeline/stage.csv) pointing to the
    CAS ref path (e.g., .pivot/cache/files/refs/stage/_single).

    Only creates symlinks for outputs that have CAS refs on disk.
    """
    refs_dir = cache_dir / "refs"
    if not refs_dir.exists():
        return

    ws = store_mod.WorkspaceStore(
        project_root=project_root,
        pipeline_name=pipeline_name,
        input_bindings={},
    )

    created = 0
    for _stage_name, info in stages.items():
        for out in info["outs"]:
            ref_path = _ref_path(refs_dir, out)
            if not ref_path.exists() and not ref_path.is_symlink():
                continue

            display_path = ws.resolve_display_path(out)
            _ensure_symlink(display_path, ref_path)
            created += 1

    if created:
        logger.debug("Presentation layer: created %d symlinks", created)


def _ref_path(refs_dir: pathlib.Path, ref: types.ArtifactRef) -> pathlib.Path:
    """Compute the CAS ref path for an artifact.

    Mirrors CacheStore._ref_path: uses SINGLE_OUTPUT_KEY for key=None.
    """
    key = ref.identity.key or compose.SINGLE_OUTPUT_KEY
    return refs_dir / ref.identity.producer / key


def _ensure_symlink(display_path: pathlib.Path, ref_path: pathlib.Path) -> None:
    """Create or update a symlink at display_path pointing to ref_path."""
    display_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing symlink/file if present
    if display_path.is_symlink() or display_path.exists():
        display_path.unlink()

    display_path.symlink_to(ref_path.resolve())
