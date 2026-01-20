from __future__ import annotations

import logging
import pathlib
from typing import Literal

import click
from click.shell_completion import CompletionItem

from pivot.config.models import CONFIG_KEY_DESCRIPTIONS, VALID_REMOTE_NAME

logger = logging.getLogger(__name__)


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
    from typing import Any, cast

    import yaml

    keys = dict(CONFIG_KEY_DESCRIPTIONS)

    # Add remotes from config files
    for path in _get_config_paths():
        try:
            if not path.exists():
                continue
            data = cast("dict[str, Any] | None", yaml.safe_load(path.read_text(encoding="utf-8")))
            if data is None:
                continue
            remotes = data.get("remotes")
            if isinstance(remotes, dict):
                for name in remotes:  # pyright: ignore[reportUnknownVariableType] - validated below
                    if isinstance(name, str) and VALID_REMOTE_NAME.match(name):
                        keys[f"remotes.{name}"] = f"Remote '{name}'"
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
        config = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError:
        logger.debug("YAML parse error in %s", yaml_path)
        return None

    if matrix.needs_fallback(config):
        return None

    return matrix.extract_stage_names(config)


def _get_stages_full() -> list[str]:
    """Fallback: get stage names via full discovery (~500ms)."""
    from pivot import discovery, registry

    if not discovery.has_registered_stages():
        discovery.discover_and_register()

    return registry.REGISTRY.list_stages()


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
