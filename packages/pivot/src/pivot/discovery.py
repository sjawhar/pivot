# pyright: reportImplicitRelativeImport=false
from __future__ import annotations

import dataclasses
import importlib
import importlib.machinery
import importlib.util
import logging
import pathlib
import sys
import types as _stdlib_types
from typing import Final

from pivot import fingerprint, metrics, project, types
from pivot.registry import PipelineLike

logger = logging.getLogger(__name__)

PIPELINE_PY_NAME: Final = "pipeline.py"


class DiscoveryError(Exception):
    """Error during pipeline discovery."""


# Import machinery is process-global. This session record is the narrow ownership boundary:
# a new public discovery/load restores only modules and sys.path entries that the prior session
# displaced or inserted, while the active pipeline stays importable for fingerprinting/workers.
_MISSING_MODULE = object()


@dataclasses.dataclass
class _ImportSession:
    root: pathlib.Path
    path_entry: str
    caller_path_entries: list[tuple[int, str]]
    preexisting_root_modules: dict[str, _stdlib_types.ModuleType]
    owned_modules: dict[str, _stdlib_types.ModuleType] = dataclasses.field(default_factory=dict)
    displaced_modules: dict[str, object] = dataclasses.field(default_factory=dict)


_import_session: _ImportSession | None = None


def _end_import_session(session: _ImportSession) -> None:
    """Remove session-owned imports and restore modules/sys.path displaced for it."""
    for module_name, module in session.owned_modules.items():
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
    for module_name, previous in session.displaced_modules.items():
        if isinstance(previous, _stdlib_types.ModuleType):
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
    if session.path_entry in sys.path:
        sys.path.remove(session.path_entry)
    for index, entry in session.caller_path_entries:
        sys.path.insert(min(index, len(sys.path)), entry)
    importlib.invalidate_caches()


def _start_import_session(root: pathlib.Path) -> _ImportSession:
    """Start a fresh root import session, restoring the previous session first."""
    global _import_session
    if _import_session is not None:
        _end_import_session(_import_session)
    root = root.resolve()
    preexisting_root_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name != "pivot"
        and not module_name.startswith("pivot.")
        and _module_is_under_root(module, root)
    }
    root_str = str(root)
    caller_path_entries = [
        (index, entry) for index, entry in enumerate(sys.path) if entry == root_str
    ]
    for index, _entry in reversed(caller_path_entries):
        del sys.path[index]
    sys.path.insert(0, root_str)
    _import_session = _ImportSession(root, root_str, caller_path_entries, preexisting_root_modules)
    importlib.invalidate_caches()
    return _import_session


def _displace_module(module_name: str, session: _ImportSession) -> None:
    """Save one host module before session code claims its name."""
    if module_name in session.displaced_modules:
        return
    session.displaced_modules[module_name] = sys.modules.get(module_name, _MISSING_MODULE)
    sys.modules.pop(module_name, None)


def _module_is_under_root(module: _stdlib_types.ModuleType, root: pathlib.Path) -> bool:
    """Whether a regular or namespace module is loaded from this project root."""
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        try:
            return pathlib.Path(module_file).resolve().is_relative_to(root)
        except (OSError, ValueError):
            return False
    module_paths = getattr(module, "__path__", ())
    try:
        return any(pathlib.Path(path).resolve().is_relative_to(root) for path in module_paths)
    except (OSError, TypeError, ValueError):
        return False


def _track_root_modules(session: _ImportSession) -> None:
    """Track all current user imports under root so the next session can remove them."""
    for module_name, module in list(sys.modules.items()):
        if module_name == "pivot" or module_name.startswith("pivot."):
            continue
        if (
            _module_is_under_root(module, session.root)
            and session.preexisting_root_modules.get(module_name) is not module
        ):
            session.owned_modules[module_name] = module


def _module_name_for_path(path: pathlib.Path, root: pathlib.Path) -> str:
    """Return the stable dotted module name for a pipeline.py path below root."""
    try:
        relative_path = path.resolve().relative_to(root)
    except ValueError as exc:
        raise DiscoveryError(f"Pipeline {path} is outside import root {root}") from exc
    return ".".join(relative_path.with_suffix("").parts)


def _ensure_parent_modules(module_name: str, session: _ImportSession) -> None:
    """Make root-relative parents available, displacing unrelated same-named packages."""
    parts = module_name.split(".")
    for index in range(1, len(parts)):
        parent_name = ".".join(parts[:index])
        parent_dir = session.root.joinpath(*parts[:index])
        existing = sys.modules.get(parent_name)
        if isinstance(existing, _stdlib_types.ModuleType) and _module_is_under_root(
            existing, session.root
        ):
            continue
        if existing is not None:
            _displace_module(parent_name, session)
        init_file = parent_dir / "__init__.py"
        if init_file.is_file():
            parent = importlib.import_module(parent_name)
        else:
            parent = _stdlib_types.ModuleType(parent_name)
            parent.__path__ = [str(parent_dir)]  # type: ignore[attr-defined] - package marker
            parent.__package__ = parent_name
            spec = importlib.machinery.ModuleSpec(parent_name, loader=None, is_package=True)
            spec.submodule_search_locations = [str(parent_dir)]
            parent.__spec__ = spec
            sys.modules[parent_name] = parent
        session.owned_modules[parent_name] = parent


def _rollback_import_attempt(
    session: _ImportSession,
    owned_before: set[str],
    displaced_before: set[str],
) -> None:
    """Undo root modules and collisions created by one failed pipeline import."""
    _track_root_modules(session)
    for module_name in set(session.owned_modules) - owned_before:
        module = session.owned_modules.pop(module_name)
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
    for module_name in set(session.displaced_modules) - displaced_before:
        previous = session.displaced_modules.pop(module_name)
        if isinstance(previous, _stdlib_types.ModuleType):
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)


def find_config_in_dir(directory: pathlib.Path) -> pathlib.Path | None:
    """Find the pipeline config file in a directory.

    Returns the path to pipeline.py if found.
    Returns None if not found.
    """
    pipeline_path = directory / PIPELINE_PY_NAME
    if pipeline_path.is_file():
        return pipeline_path
    return None


def discover_pipeline(
    project_root: pathlib.Path | None = None,
    *,
    all_pipelines: bool = False,
) -> PipelineLike | None:
    """Discover and return Pipeline from pipeline.py.

    Looks for pipeline config in this order:
    1. Current working directory (if within project root)
    2. Project root

    In each location, checks for:
    - pipeline.py - looks for `pipeline` variable (Pipeline instance)

    When all_pipelines=True, discovers ALL pipeline config files in the project,
    loads each, and merges them into a single Pipeline via include(). The combined
    pipeline contains stages from all discovered pipelines, each retaining its
    original state_dir.

    Args:
        project_root: Override project root (default: auto-detect)
        all_pipelines: If True, discover and combine all pipelines in project.

    Returns:
        Pipeline instance, or None if nothing found

    Raises:
        DiscoveryError: If discovery fails, or if both config types exist
    """
    _t = metrics.start()
    session: _ImportSession | None = None
    loaded = False
    try:
        root = (project_root or project.get_project_root()).resolve()
        session = _start_import_session(root)

        if all_pipelines:
            result = _discover_all_pipelines(root, session)
            loaded = result is not None
            return result
        try:
            cwd = pathlib.Path.cwd().resolve()
            root_resolved = root
        except OSError as e:
            raise DiscoveryError(f"Failed to resolve paths: {e}") from e

        # Check cwd first if it's within project root but not the root itself
        config_path: pathlib.Path | None = None
        if cwd != root_resolved and cwd.is_relative_to(root_resolved):
            config_path = find_config_in_dir(cwd)

        # Fall back to project root (use resolved path for consistency)
        if config_path is None:
            config_path = find_config_in_dir(root_resolved)

        if config_path is None:
            return None

        logger.info(f"Discovered {config_path}")

        _t_module = metrics.start()
        try:
            result = _load_pipeline_from_module(config_path, session)
            loaded = result is not None
            return result
        except SystemExit as e:
            raise DiscoveryError(f"Pipeline {config_path} called sys.exit({e.code})") from e
        except DiscoveryError:
            raise
        except Exception as e:
            raise DiscoveryError(f"Failed to load {config_path}: {e}") from e
        finally:
            metrics.end("discovery.load_module", _t_module)
            fingerprint.flush_ast_hash_cache()
            fingerprint.flush_manifest_cache()
    finally:
        if session is not None and not loaded:
            _end_import_session(session)
            if _import_session is session:
                globals()["_import_session"] = None
        metrics.end("discovery.total", _t)


def _load_pipeline_from_module(
    path: pathlib.Path, session: _ImportSession | None = None
) -> PipelineLike | None:
    """Load one pipeline.py into an active session under its stable dotted module name."""
    owns_session = session is None
    if session is None:
        session = _start_import_session(path.parent)
    resolved_path = path.resolve()
    module_name = _module_name_for_path(resolved_path, session.root)
    module = sys.modules.get(module_name)
    if isinstance(module, _stdlib_types.ModuleType):
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and pathlib.Path(module_file).resolve() == resolved_path:
            module_dict = vars(module)
        else:
            _displace_module(module_name, session)
            module = None
    else:
        module = None

    owned_before = set(session.owned_modules)
    displaced_before = set(session.displaced_modules)
    try:
        if module is None:
            _ensure_parent_modules(module_name, session)
            spec = importlib.util.spec_from_file_location(module_name, resolved_path)
            if spec is None or spec.loader is None:
                raise DiscoveryError(f"Unable to create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            session.owned_modules[module_name] = module
            spec.loader.exec_module(module)
        _track_root_modules(session)
        module_dict = vars(module)

        pipeline_obj = module_dict.get("pipeline")
        if pipeline_obj is not None:
            if not isinstance(pipeline_obj, PipelineLike):
                from typing import get_protocol_members

                required = sorted(get_protocol_members(PipelineLike))
                missing = [attr for attr in required if not hasattr(pipeline_obj, attr)]
                if missing:
                    raise DiscoveryError(
                        f"{path} defines 'pipeline' but it's missing required methods: {missing}"
                    )
                raise DiscoveryError(
                    f"{path} defines 'pipeline' but it doesn't satisfy the Pipeline interface "
                    + f"(got {type(pipeline_obj).__name__})"
                )
            return pipeline_obj

        for name, value in module_dict.items():
            if isinstance(value, type):
                continue
            if isinstance(value, PipelineLike):
                raise DiscoveryError(
                    f"{path} does not define a 'pipeline' variable. Found Pipeline instance named '{name}' - rename it to 'pipeline'."
                )
        return None
    except BaseException:
        _rollback_import_attempt(session, owned_before, displaced_before)
        if owns_session:
            _end_import_session(session)
            if _import_session is session:
                globals()["_import_session"] = None
        raise


def _discover_all_pipelines(root: pathlib.Path, session: _ImportSession) -> PipelineLike | None:
    """Discover all pipelines and combine into one import session."""
    from pivot.compose import Pipeline

    config_paths = glob_all_pipelines(root)
    if not config_paths:
        return None

    pipelines = list[PipelineLike]()
    for path in config_paths:
        try:
            pipeline = _load_pipeline_from_module(path, session)
        except SystemExit as e:
            logger.warning(f"--all: pipeline {path} called sys.exit({e.code}), skipping")
            continue
        except Exception as e:
            logger.warning(f"--all: failed to load pipeline from {path}, skipping ({e})")
            continue
        if pipeline is not None:
            pipelines.append(pipeline)
            logger.info(f"  {pipeline.name}: {path} ({len(pipeline.list_stages())} stages)")
        else:
            logger.warning(f"--all: failed to load pipeline from {path}, skipping")

    if not pipelines:
        return None

    combined = Pipeline("all", root=root)
    for pipeline in pipelines:
        combined.include(pipeline)  # Auto-prefixes on name collision

    if logger.isEnabledFor(logging.DEBUG):
        local_outputs = set[str]()
        all_deps = set[str]()
        for stage_name in combined.list_stages():
            info = combined.get_stage(stage_name)
            local_outputs.update(types.identity_key(out.identity) for out in info["outs"])
            all_deps.update(types.identity_key(dep.identity) for dep in info["deps"].values())
        unresolved = all_deps - local_outputs
        if unresolved:
            sample = ", ".join(sorted(unresolved)[:5])
            suffix = f"... ({len(unresolved)} total)" if len(unresolved) > 5 else ""
            logger.debug(
                f"--all: dependency path(s) not produced by any discovered pipeline: {sample}{suffix}"
            )

    logger.debug(
        f"Discovered {len(pipelines)} pipelines with {len(combined.list_stages())} total stages"
    )
    return combined


# Directories excluded from all-pipelines scan
_SCAN_EXCLUDE_DIRS = frozenset(
    {
        ".pivot",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".ruff_cache",
        "site-packages",
        "dist-packages",
        ".eggs",
    }
)


def glob_all_pipelines(project_root: pathlib.Path) -> list[pathlib.Path]:
    """Find all pipeline config files in the project.

    Scans project_root recursively for pipeline.py files,
    skipping common non-project directories (.venv, __pycache__, etc.).

    Args:
        project_root: Project root directory to scan.

    Returns:
        List of paths to pipeline config files.
    """
    # Collect all candidate paths grouped by directory
    by_dir = dict[pathlib.Path, list[pathlib.Path]]()
    for path in project_root.rglob(PIPELINE_PY_NAME):
        # Only check path components relative to project_root to avoid
        # false exclusions when project_root itself is inside a directory
        # named "venv", "__pycache__", etc.
        try:
            rel_parts = path.relative_to(project_root).parts
        except ValueError:
            continue
        if any(part in _SCAN_EXCLUDE_DIRS for part in rel_parts):
            continue
        by_dir.setdefault(path.parent, list[pathlib.Path]()).append(path)

    # Validate and select canonical config per directory, sorted for deterministic
    # ordering (auto-prefix collision resolution depends on include order).
    results = list[pathlib.Path]()
    for directory in sorted(by_dir):
        results.append(by_dir[directory][0])

    return results


def load_pipeline_from_path(path: pathlib.Path) -> PipelineLike | None:
    """Load a Pipeline from a pipeline.py file in a fresh direct-load session."""
    if path.name != PIPELINE_PY_NAME:
        logger.debug(f"Unknown pipeline file type: {path}")
        return None

    resolved_path = path.resolve()
    try:
        project_root = project.get_project_root().resolve()
    except Exception:
        project_root = resolved_path.parent
    import_root = (
        project_root if resolved_path.is_relative_to(project_root) else resolved_path.parent
    )
    session = _start_import_session(import_root)
    loaded = False
    try:
        pipeline = _load_pipeline_from_module(resolved_path, session)
        loaded = pipeline is not None
        return pipeline
    except DiscoveryError as e:
        logger.warning(f"Pipeline discovery issue in {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Failed to load pipeline from {path}: {e}")
        return None
    finally:
        if not loaded:
            _end_import_session(session)
            if _import_session is session:
                globals()["_import_session"] = None
        fingerprint.flush_ast_hash_cache()
        fingerprint.flush_manifest_cache()
