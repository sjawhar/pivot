from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import Any, Literal, cast

import click
from click.shell_completion import CompletionItem

from pivot.config.models import CONFIG_KEY_DESCRIPTIONS, VALID_REMOTE_NAME

logger = logging.getLogger(__name__)

CACHE_VERSION = "v1"


def complete_config_keys(
    _ctx: click.Context,
    _param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    """Provide config key completions with descriptions."""
    try:
        keys = _get_config_keys()
        return [
            CompletionItem(k, help=desc) for k, desc in keys.items() if k.startswith(incomplete)
        ]
    except Exception:
        logger.debug("Config key completion failed", exc_info=True)
        return []


def _get_config_keys() -> dict[str, str]:
    """Get all valid config keys with descriptions, including dynamic remotes."""
    import yaml

    keys = dict(CONFIG_KEY_DESCRIPTIONS)

    # Add remotes from config files
    for path in _get_config_paths():
        try:
            if not path.exists():
                continue
            data: object = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            typed_data = cast("dict[str, Any]", data)
            remotes: object = typed_data.get("remotes")
            if isinstance(remotes, dict):
                for remote_name in cast("dict[str, Any]", remotes):
                    if isinstance(remote_name, str) and VALID_REMOTE_NAME.match(remote_name):  # pyright: ignore[reportUnnecessaryIsInstance] - YAML can parse non-string keys
                        keys[f"remotes.{remote_name}"] = f"Remote '{remote_name}'"
        except Exception:
            logger.debug("Config completion: failed to load %s", path, exc_info=True)
            continue

    return keys


def _get_config_paths() -> list[pathlib.Path]:
    """Get config file paths without heavy imports."""
    paths = [pathlib.Path.home() / ".config" / "pivot" / "config.yaml"]

    # Use existing fast project root finder
    if root := _find_project_root_fast():
        paths.append(root / ".pivot" / "config.yaml")

    return paths


def complete_stages(
    _ctx: click.Context,
    _param: click.Parameter,
    incomplete: str,
) -> list[str]:
    """Provide stage name completions for CLI arguments."""
    try:
        stages = _get_stages_fast()
        if stages is None:
            stages = _get_stages_full()
        return [s for s in stages if s.startswith(incomplete)]
    except Exception:
        logger.debug("Completion failed", exc_info=True)
        return []


# Alias for push/pull targets (currently stages only; expand if file path completion needed)
complete_targets = complete_stages


def _detect_config_file(root: pathlib.Path) -> tuple[str, float] | None:
    """Detect config file and capture its mtime.

    Returns (relative_path, mtime) or None if no config found.
    """
    from pivot import discovery

    for name in (*discovery.PIVOT_YAML_NAMES, discovery.PIPELINE_PY_NAME):
        path = root / name
        if path.exists():
            return (name, path.stat().st_mtime)
    return None


def _get_stages_from_cache(root: pathlib.Path) -> list[str] | None:
    """Read stage names from cache if fresh.

    Returns stage names if cache exists and is valid, None otherwise.
    Cache is valid when version matches and config file mtime unchanged.
    """
    cache_path = root / ".pivot" / "cache" / "stages.cache"
    try:
        lines = cache_path.read_text().splitlines()
    except OSError:
        return None

    if len(lines) < 2 or lines[0] != CACHE_VERSION:
        return None

    header = lines[1]
    sep_idx = header.rfind(":")
    if sep_idx == -1:
        return None

    config_file, mtime_str = header[:sep_idx], header[sep_idx + 1 :]

    try:
        current_mtime = (root / config_file).stat().st_mtime
    except OSError:
        return None

    if str(current_mtime) != mtime_str:
        return None

    return [line for line in lines[2:] if line]


def _write_stages_cache(
    root: pathlib.Path, config_file: str, mtime: float, stages: list[str]
) -> None:
    """Write stage names to cache (atomic).

    Uses temp file + rename for atomic write on POSIX.
    Cleans up temp file on failure before re-raising OSError.
    Caller is responsible for handling write failures.
    """
    cache_dir = root / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "stages.cache"
    content = f"{CACHE_VERSION}\n{config_file}:{mtime}\n" + "\n".join(stages) + "\n"

    # Write to temp file in same directory, then rename atomically
    fd, tmp_path_str = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    tmp_path = pathlib.Path(tmp_path_str)
    try:
        with open(fd, "w") as f:
            f.write(content)
        tmp_path.rename(cache_path)
    except OSError:
        # Clean up temp file on failure
        tmp_path.unlink(missing_ok=True)
        raise


def _get_stages_fast() -> list[str] | None:
    """Fast path: get stage names without full Pivot imports (~10ms).

    Returns None to trigger fallback when:
    - No project root found
    - No pivot.yaml exists
    - Config has 'variants' field (requires Python imports)
    - YAML parse error
    """
    root = _find_project_root_fast()
    if root is None:
        return None

    # Try cache first (works for both YAML and pipeline.py)
    if (stages := _get_stages_from_cache(root)) is not None:
        return stages

    yaml_path = next(
        (p for p in [root / "pivot.yaml", root / "pivot.yml"] if p.exists()),
        None,
    )
    if yaml_path is None:
        return None

    # Lazy imports to avoid loading pivot package at CLI startup
    import yaml

    from pivot import matrix

    try:
        raw: object = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError:
        logger.debug("YAML parse error in %s", yaml_path)
        return None

    config = cast("dict[str, Any]", raw) if isinstance(raw, dict) else None

    if matrix.needs_fallback(config):
        return None

    return matrix.extract_stage_names(config)


def _get_stages_full() -> list[str]:
    """Fallback: get stage names via full discovery (~500ms).

    Writes stage names to cache after discovery for future fast lookups.
    Cache is only written if config file mtime is unchanged during discovery
    (prevents race condition where config changes during slow discovery).
    """
    from pivot import discovery

    # Capture config file and mtime BEFORE discovery (race condition prevention)
    root = _find_project_root_fast()
    pre_discovery = _detect_config_file(root) if root else None

    pipeline = discovery.discover_pipeline()
    if pipeline is None:
        return []

    stages = pipeline.list_stages()

    # Write cache only if config file unchanged during discovery
    if root and pre_discovery:
        config_file, pre_mtime = pre_discovery
        try:
            post_mtime = (root / config_file).stat().st_mtime
            if pre_mtime == post_mtime:
                _write_stages_cache(root, config_file, pre_mtime, stages)
        except OSError:
            pass  # Cache write failure is non-fatal

    return stages


def _find_project_root_fast() -> pathlib.Path | None:
    """Find project root without importing pivot.project.

    Walks up from cwd looking for .pivot or .git markers.
    """
    current = pathlib.Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / ".pivot").exists() or (parent / ".git").exists():
            return parent
    return None


@click.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion_cmd(shell: Literal["bash", "zsh", "fish"]) -> None:
    """Generate shell completion script.

    To enable completions, add to your shell config:

    \b
    Bash (~/.bashrc):
        eval "$(pivot completion bash)"

    \b
    Zsh (~/.zshrc):
        eval "$(pivot completion zsh)"

    \b
    Fish (~/.config/fish/config.fish):
        pivot completion fish | source
    """
    import contextlib
    import os
    import sys

    from pivot.cli import cli

    prog_name = os.path.basename(sys.argv[0]) if sys.argv else "pivot"
    complete_var = f"_{prog_name.upper().replace('-', '_')}_COMPLETE"
    os.environ[complete_var] = f"{shell}_source"

    with contextlib.suppress(SystemExit):
        cli.main(standalone_mode=False)
