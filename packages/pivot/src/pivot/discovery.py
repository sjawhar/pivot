from __future__ import annotations

import _imp
import contextlib
import dataclasses
import importlib
import importlib.machinery
import importlib.util
import logging
import os
import pathlib
import py_compile
import sys
import types as _stdlib_types
from typing import TYPE_CHECKING, Final, override

import cloudpickle

from pivot import exceptions, fingerprint, metrics, project, types
from pivot.registry import PipelineLike

if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.abc import PathEntryFinder

logger = logging.getLogger(__name__)

PIPELINE_PY_NAME: Final = "pipeline.py"


class DiscoveryError(Exception):
    """Error during pipeline discovery."""


# Import machinery is process-global. This session record is the narrow ownership boundary:
# a new public discovery/load restores only modules and sys.path entries that the prior session
# displaced or inserted, while the active pipeline stays importable for fingerprinting/workers.
_MISSING_MODULE = object()

# A cached bytecode file's flags word (PEP 552): bit 0 marks it hash-based, bit 1 has every
# import check that hash against the source.
_CHECKED_HASH_PYC_FLAGS: Final = 0b11


def _is_checked_hash_pyc(path: str) -> bool:
    """Whether ``path`` holds bytecode for this Python that every import checks by source hash."""
    try:
        with open(path, "rb") as cached:
            header = cached.read(8)
    except OSError:
        return False
    return (
        header[:4] == importlib.util.MAGIC_NUMBER
        and int.from_bytes(header[4:8], "little") == _CHECKED_HASH_PYC_FLAGS
    )


class _ProjectSourceLoader(importlib.machinery.SourceFileLoader):
    """Load a user-code file from bytecode checked against its source's content, not its mtime.

    CPython's default bytecode cache is checked only by the source's size and whole-second mtime,
    so an edit that keeps the size, made in the second the previous version was written and
    cached, imports the previous version's code. A checked-hash cache (PEP 552) is checked by the
    source's content on every import, by this process and by any other that imports the file.
    Other bytecode for the file is replaced with one before the import, or removed when Python
    is told not to write bytecode (``sys.dont_write_bytecode``), so no other process reads it.

    The file is compiled from source when the cache cannot be written (a read-only
    ``__pycache__``), when the source does not compile (which reports its syntax error), under
    ``sys.dont_write_bytecode``, and when Python is told never to check hash-based bytecode
    (``--check-hash-based-pycs never``). Its bytecode is replaced in that last case too: pivot's
    workers do not inherit the flag, and they read it.

    A file whose resolved path is a library's (``fingerprint.is_library_path``: pivot's own
    packages included), as a library package symlinked into the project root, loads as CPython
    loads it, and its bytecode is left alone.
    """

    @override
    def get_code(self, fullname: str) -> _stdlib_types.CodeType | None:
        source = self.get_filename(fullname)
        if fingerprint.is_library_path(pathlib.Path(source).resolve()):
            return super().get_code(fullname)
        cached = importlib.util.cache_from_source(source)
        if not _is_checked_hash_pyc(cached):
            if sys.dont_write_bytecode:
                with contextlib.suppress(OSError):
                    os.unlink(cached)
                return self.source_to_code(self.get_data(source), source)
            try:
                py_compile.compile(
                    source,
                    cfile=cached,
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
                )
            except (py_compile.PyCompileError, OSError):
                return self.source_to_code(self.get_data(source), source)
        if _imp.check_hash_based_pycs == "never":
            return self.source_to_code(self.get_data(source), source)
        return super().get_code(fullname)


_project_finder_for = importlib.machinery.FileFinder.path_hook(
    (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES),
    (_ProjectSourceLoader, importlib.machinery.SOURCE_SUFFIXES),
    (importlib.machinery.SourcelessFileLoader, importlib.machinery.BYTECODE_SUFFIXES),
)


def _holds_user_code(path: str, root: pathlib.Path) -> bool:
    """Whether import path entry ``path`` is a directory of code fingerprinting treats as the
    user's, imported through ``_ProjectSourceLoader``.

    That is every directory but a library's (``fingerprint.is_library_path``, which counts the
    stdlib even when the interpreter lives under the root), minus, under the project root, the
    library and tooling directories discovery skips. A stage's module outside the root, as a
    package installed from a source checkout, is user code.
    """
    try:
        resolved = pathlib.Path(path or ".").resolve()
    except OSError:
        return False
    if fingerprint.is_library_path(resolved):
        return False
    if not resolved.is_relative_to(root):
        return True
    return _SCAN_EXCLUDE_DIRS.isdisjoint(resolved.relative_to(root).parts)


def _project_path_hook(root: pathlib.Path) -> Callable[[str], PathEntryFinder]:
    """A ``sys.path_hooks`` entry that imports user code with ``_ProjectSourceLoader`` and
    declines every other path entry."""

    def hook(path: str) -> PathEntryFinder:
        if not _holds_user_code(path, root):
            raise ImportError(f"{path} holds no user code")
        return _project_finder_for(path)

    return hook


def _forget_user_code_finders(root: pathlib.Path) -> None:
    """Drop the import finders cached for user-code directories, so the next import from one
    asks ``sys.path_hooks`` again."""
    for entry in list(sys.path_importer_cache):
        if _holds_user_code(entry, root):
            del sys.path_importer_cache[entry]


def _install_path_hook(root: pathlib.Path) -> Callable[[str], PathEntryFinder]:
    """Import user code through ``_ProjectSourceLoader`` from now on; return the hook."""
    path_hook = _project_path_hook(root)
    sys.path_hooks.insert(0, path_hook)
    _forget_user_code_finders(root)
    importlib.invalidate_caches()
    return path_hook


@dataclasses.dataclass
class _ImportSession:
    root: pathlib.Path
    path_entry: str
    caller_path_entries: list[tuple[int, str]]
    preexisting_root_modules: dict[str, _stdlib_types.ModuleType]
    path_hook: Callable[[str], PathEntryFinder]
    # Whether each module file lies under root; every scan of sys.modules shares one resolve.
    file_under_root: dict[str, bool] = dataclasses.field(default_factory=dict)
    owned_modules: dict[str, _stdlib_types.ModuleType] = dataclasses.field(default_factory=dict)
    displaced_modules: dict[str, object] = dataclasses.field(default_factory=dict)
    preexisting_pickle_by_value_modules: set[str] = dataclasses.field(default_factory=set)
    pickle_by_value_modules: dict[str, _stdlib_types.ModuleType] = dataclasses.field(
        default_factory=dict
    )


_import_session: _ImportSession | None = None


def _end_import_session(session: _ImportSession) -> None:
    """Remove session-owned imports and restore modules/sys.path displaced for it."""
    for module_name, module in session.owned_modules.items():
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
    for module_name, module in session.pickle_by_value_modules.items():
        if module_name in session.preexisting_pickle_by_value_modules:
            continue
        with contextlib.suppress(ValueError):
            cloudpickle.unregister_pickle_by_value(module)
    for module_name, previous in session.displaced_modules.items():
        if isinstance(previous, _stdlib_types.ModuleType):
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
    if session.path_entry in sys.path:
        sys.path.remove(session.path_entry)
    for index, entry in session.caller_path_entries:
        sys.path.insert(min(index, len(sys.path)), entry)
    if session.path_hook in sys.path_hooks:
        sys.path_hooks.remove(session.path_hook)
    _forget_user_code_finders(session.root)
    importlib.invalidate_caches()


def _start_import_session(root: pathlib.Path) -> _ImportSession:
    """Start a fresh root import session, restoring the previous session first."""
    global _import_session
    if _import_session is not None:
        _end_import_session(_import_session)
    root = root.resolve()
    file_under_root = dict[str, bool]()
    preexisting_root_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if _module_is_under_root(module, root, file_under_root)
    }
    root_str = str(root)
    caller_path_entries = [
        (index, entry) for index, entry in enumerate(sys.path) if entry == root_str
    ]
    for index, _entry in reversed(caller_path_entries):
        del sys.path[index]
    sys.path.insert(0, root_str)
    _import_session = _ImportSession(
        root,
        root_str,
        caller_path_entries,
        preexisting_root_modules,
        # User code imports through _ProjectSourceLoader while the session lasts.
        _install_path_hook(root),
        file_under_root,
        preexisting_pickle_by_value_modules=cloudpickle.list_registry_pickle_by_value(),
    )
    return _import_session


def import_session_root() -> pathlib.Path | None:
    """The project root of the import session the last discovery started, if it lasts."""
    return None if _import_session is None else _import_session.root


def install_user_code_loader(root: str) -> None:
    """Import user code through ``_ProjectSourceLoader`` in this process from now on, as the
    import session for ``root`` does: the initializer of each worker (``executor.core``).

    A worker imports user code by name: a stage's module that discovery could not send by
    value (one outside the root, or under it in a ``src`` layout) and any module a stage body
    imports. The session's loader is installed only in the process that discovered.
    """
    _install_path_hook(pathlib.Path(root))


def _displace_module(module_name: str, session: _ImportSession) -> None:
    """Save one host module before session code claims its name."""
    if module_name in session.displaced_modules:
        return
    session.displaced_modules[module_name] = sys.modules.get(module_name, _MISSING_MODULE)
    sys.modules.pop(module_name, None)


def _module_is_under_root(
    module: _stdlib_types.ModuleType, root: pathlib.Path, file_under_root: dict[str, bool]
) -> bool:
    """Whether a regular or namespace project module is loaded from this project root.

    Libraries are never project modules, even from a virtualenv inside the root: a session
    removes the modules it owns, and some extensions (numpy's) cannot be imported twice.
    ``file_under_root`` memoizes the verdict per module file for this root.
    """
    if not fingerprint.is_user_code(module):
        return False
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        if (verdict := file_under_root.get(module_file)) is None:
            try:
                verdict = pathlib.Path(module_file).resolve().is_relative_to(root)
            except (OSError, ValueError):
                verdict = False
            file_under_root[module_file] = verdict
        return verdict
    module_paths = getattr(module, "__path__", ())
    try:
        return any(pathlib.Path(path).resolve().is_relative_to(root) for path in module_paths)
    except (OSError, TypeError, ValueError):
        return False


def _module_name_matches_root_path(
    module_name: str,
    module: _stdlib_types.ModuleType,
    root: pathlib.Path,
) -> bool:
    """Return whether a module's dotted name maps directly to a path below root."""
    parts = module_name.split(".")
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        resolved = pathlib.Path(module_file).resolve()
        return resolved in {
            root.joinpath(*parts).with_suffix(".py"),
            root.joinpath(*parts, "__init__.py"),
        }
    module_paths = getattr(module, "__path__", ())
    return any(pathlib.Path(path).resolve() == root.joinpath(*parts) for path in module_paths)


def _track_root_modules(session: _ImportSession) -> None:
    """Track all current user imports under root so the next session can remove them."""
    for module_name, module in list(sys.modules.items()):
        if not _module_is_under_root(module, session.root, session.file_under_root):
            continue
        if session.preexisting_root_modules.get(module_name) is not module:
            session.owned_modules[module_name] = module
        if (
            _module_name_matches_root_path(module_name, module, session.root)
            and module_name not in session.preexisting_pickle_by_value_modules
            and module_name not in session.pickle_by_value_modules
        ):
            cloudpickle.register_pickle_by_value(module)
            session.pickle_by_value_modules[module_name] = module


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
            existing, session.root, session.file_under_root
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
    pickle_by_value_before: set[str],
) -> None:
    """Undo root modules and collisions created by one failed pipeline import."""
    _track_root_modules(session)
    for module_name in set(session.pickle_by_value_modules) - pickle_by_value_before:
        module = session.pickle_by_value_modules.pop(module_name)
        if module_name not in session.preexisting_pickle_by_value_modules:
            with contextlib.suppress(ValueError):
                cloudpickle.unregister_pickle_by_value(module)
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


def _require_project_state(pipeline: PipelineLike, project_root: pathlib.Path) -> None:
    """Refuse a pipeline whose stages keep state outside the project configuration.

    Inputs and outputs resolve against the project root, but ``root=`` moves a stage's lock
    files and half its StateDB, so a ``root=`` other than the project root splits one stage's
    state across two directories. Stages from another pipeline carry that pipeline's root.
    While the project config cannot be read or its state directory cannot be used, stages keep
    0.5.2's ``.pivot`` (``project.state_dir_or_default``), and that is what is checked.
    """
    advice = (
        "but a pipeline always uses the project's .pivot and data/ (with state in its configured "
        + f"directory; {project_root}); remove root="
    )
    if pipeline.root.resolve() != project_root:
        raise exceptions.PipelineConfigError(
            f"Pipeline '{pipeline.name}' sets root={pipeline.root}, {advice}"
        )
    expected_state_dir = project.state_dir_or_default(project_root).resolve()
    for stage_name in pipeline.list_stages():
        state_dir = pipeline.get_stage(stage_name)["state_dir"]
        if state_dir is not None and state_dir.resolve() != expected_state_dir:
            raise exceptions.PipelineConfigError(
                f"Stage '{stage_name}' comes from a pipeline that sets root={state_dir.parent}, "
                + advice
            )


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
        try:
            root = (project_root or project.get_project_root()).resolve()
            cwd = pathlib.Path.cwd().resolve()
        except OSError as e:
            raise DiscoveryError(f"Failed to resolve paths: {e}") from e
        session = _start_import_session(root)

        if all_pipelines:
            result = _discover_all_pipelines(root, session)
            if result is not None:
                _require_project_state(result, root)
            loaded = result is not None
            return result
        # Check cwd first if it's within project root but not the root itself
        config_path: pathlib.Path | None = None
        if cwd != root and cwd.is_relative_to(root):
            config_path = find_config_in_dir(cwd)

        # Fall back to project root
        if config_path is None:
            config_path = find_config_in_dir(root)

        if config_path is None:
            return None

        logger.info(f"Discovered {config_path}")

        _t_module = metrics.start()
        try:
            result = _load_pipeline_from_module(config_path, session)
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
        if result is not None:
            _require_project_state(result, root)
        loaded = result is not None
        return result
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
    pickle_by_value_before = set(session.pickle_by_value_modules)
    try:
        if module is None:
            _ensure_parent_modules(module_name, session)
            spec = importlib.util.spec_from_file_location(
                module_name,
                resolved_path,
                loader=_ProjectSourceLoader(module_name, str(resolved_path)),
            )
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
        _rollback_import_attempt(
            session,
            owned_before,
            displaced_before,
            pickle_by_value_before,
        )
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
    first_by_name = dict[str, tuple[pathlib.Path, PipelineLike]]()
    for path in config_paths:
        try:
            pipeline = _load_pipeline_from_module(path, session)
        except SystemExit as e:
            logger.warning(f"--all: pipeline {path} called sys.exit({e.code}), skipping")
            continue
        except Exception as e:
            logger.warning(f"--all: failed to load pipeline from {path}, skipping ({e})")
            continue
        if pipeline is None:
            logger.warning(f"--all: failed to load pipeline from {path}, skipping")
            continue
        # A pipeline name names its configured state directory's stages/<name>/ and
        # data/<name>/, so two files declaring one name would silently share stages, locks and
        # outputs. A file that exports a pipeline object another file already gave (a re-export)
        # declares nothing.
        first_path, first_pipeline = first_by_name.setdefault(pipeline.name, (path, pipeline))
        if first_path != path:
            if first_pipeline is not pipeline:
                raise DiscoveryError(
                    f"Pipeline name '{pipeline.name}' is declared by both "
                    + f"{first_path.relative_to(root)} and {path.relative_to(root)}. "
                    + "Pipeline names must be unique within a project; rename one of them."
                )
            logger.debug(
                f"--all: {path.relative_to(root)} exports the same pipeline object "
                + f"'{pipeline.name}' as {first_path.relative_to(root)}; skipping"
            )
            continue
        pipelines.append(pipeline)
        logger.info(f"  {pipeline.name}: {path} ({len(pipeline.list_stages())} stages)")

    if not pipelines:
        return None

    combined = Pipeline("all", root=root)
    for pipeline in pipelines:
        combined.include(pipeline)

    if logger.isEnabledFor(logging.DEBUG):
        local_outputs = set[str]()
        all_deps = set[str]()
        for stage_name in combined.list_stages():
            info = combined.get_stage(stage_name)
            local_outputs.update(types.identity_key(out.identity) for out in info["outs"])
            all_deps.update(
                types.identity_key(types.source_identity(dep)) for dep in info["deps"].values()
            )
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
    resolved_project_root = project_root.resolve()
    state_dir = project.state_dir_or_default(project_root)
    # A state directory at or above the root would exclude the whole project.
    excluded_state_dir: pathlib.Path | None = None
    if not resolved_project_root.is_relative_to(state_dir.resolve()):
        excluded_state_dir = state_dir.resolve()

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
        if excluded_state_dir is not None and path.resolve().is_relative_to(excluded_state_dir):
            continue
        by_dir.setdefault(path.parent, list[pathlib.Path]()).append(path)

    # Validate and select canonical config per directory, sorted for deterministic
    # load and include order.
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
    except DiscoveryError as e:
        logger.warning(f"Pipeline discovery issue in {path}: {e}")
        pipeline = None
    except Exception as e:
        logger.debug(f"Failed to load pipeline from {path}: {e}")
        pipeline = None
    try:
        if pipeline is not None:
            _require_project_state(pipeline, project_root)
        loaded = pipeline is not None
        return pipeline
    finally:
        if not loaded:
            _end_import_session(session)
            if _import_session is session:
                globals()["_import_session"] = None
        fingerprint.flush_ast_hash_cache()
        fingerprint.flush_manifest_cache()
