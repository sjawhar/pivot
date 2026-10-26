"""`loaded_source`: the source each user module was compiled from, as fingerprinting reads it."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import linecache
import logging
import os
import pathlib
import py_compile
import subprocess
import sys
import time
import types
import uuid
import zipfile
from typing import cast

import pytest
import xxhash

from pivot import discovery, loaded_source
from pivot.executor import core as executor_core
from pivot.executor import worker


@pytest.fixture(autouse=True)
def _isolated_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts from an empty record, as a new process does."""
    monkeypatch.setattr(loaded_source, "_by_file", {})
    monkeypatch.setattr(loaded_source, "_by_module", {})
    monkeypatch.setattr(loaded_source, "_recorded_objects", {})
    monkeypatch.setattr(loaded_source, "_line_entries", {})
    monkeypatch.setattr(loaded_source, "_token", None)
    monkeypatch.setattr(loaded_source, "_coordinator", None)
    monkeypatch.setattr(loaded_source, "_sent_token", None)
    monkeypatch.setattr(loaded_source, "_unseen", set[str]())
    monkeypatch.setattr(loaded_source, "_warned", set[str]())


def _load(path: pathlib.Path, text: str, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Write `text` to `path`, compile and run it as a module, and record it as loaded."""
    data = text.encode()
    path.write_bytes(data)
    name = f"loaded_{uuid.uuid4().hex}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(data, str(path), "exec"), vars(module))
    loaded_source.record(str(path), name, data)
    return module


def _import_with(name: str, loader: importlib.abc.Loader, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_loader(name, loader, origin=path)
    assert spec is not None and spec.loader is not None
    spec.has_location = True
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _text(path: pathlib.Path, data: bytes) -> str:
    return f"{path.resolve()}\0{xxhash.xxh64(data).hexdigest()}"


def test_getsource_reads_the_loaded_text_after_the_file_is_edited(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`inspect.getsource` re-reads an edited file; `loaded_source.getsource` returns the text
    the function was compiled from."""
    path = tmp_path / "helper.py"
    module = _load(path, 'def value():\n    return "alpha"\n', monkeypatch)
    path.write_text('def value():\n    return "bravo, edited after the load"\n')

    assert "bravo" in inspect.getsource(module.value), "inspect reads the edited file"
    assert loaded_source.getsource(module.value) == 'def value():\n    return "alpha"\n'
    assert "alpha" in inspect.getsource(module.value), "the loaded text stays in linecache"


def test_getsource_follows_a_functools_cache_wrapper_to_its_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `functools.cache` helper has no code object of its own; its source is the wrapped
    function's, read from the loaded text as for any function."""
    path = tmp_path / "cached.py"
    loaded = "import functools\n\n\n@functools.cache\ndef table(n):\n    return tuple(range(n))\n"
    module = _load(path, loaded, monkeypatch)
    path.write_text(loaded.replace("range(n)", "range(n + 1000)"))

    assert loaded_source.getsource(module.table) == (
        "@functools.cache\ndef table(n):\n    return tuple(range(n))\n"
    )


def test_getsource_restores_the_loaded_text_after_linecache_is_cleared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "helper.py"
    module = _load(path, "class Settings:\n    SCALE = 2\n", monkeypatch)
    path.write_text("class Settings:\n    SCALE = 3000\n")
    assert loaded_source.getsource(module.Settings) == "class Settings:\n    SCALE = 2\n"

    linecache.clearcache()

    assert loaded_source.getsource(module.Settings) == "class Settings:\n    SCALE = 2\n"


def test_a_file_loaded_again_replaces_its_text(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "helper.py"
    first = _load(path, 'def value():\n    return "alpha"\n', monkeypatch)
    assert loaded_source.getsource(first.value) == 'def value():\n    return "alpha"\n'
    token = loaded_source.token()

    again = _load(path, 'def value():\n    return "bravo"\n', monkeypatch)

    assert loaded_source.getsource(again.value) == 'def value():\n    return "bravo"\n'
    assert loaded_source.digest(str(path)) == (
        xxhash.xxh64(b'def value():\n    return "bravo"\n').hexdigest()
    )
    assert loaded_source.token() != token, "the token follows the record"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"def f():\r\n    return 1\r\n", id="crlf"),
        pytest.param(b"def f():\n    return 1", id="no-final-newline"),
        pytest.param(b"", id="empty"),
        pytest.param(b"\xef\xbb\xbfdef f():\n    return '\xc3\xa9'\n", id="utf-8-bom"),
        pytest.param(b"# -*- coding: latin-1 -*-\nX = '\xe9'\n", id="latin-1-cookie"),
    ],
)
def test_loaded_lines_are_the_lines_linecache_reads_from_the_file(
    tmp_path: pathlib.Path, data: bytes
) -> None:
    """The text fingerprinting hashes for an unedited file is the text `inspect` reads from it,
    so recorded fingerprints do not change."""
    path = tmp_path / "source.py"
    path.write_bytes(data)

    assert loaded_source._source_lines(data) == linecache.updatecache(str(path))


def test_digest_answers_by_module_file_and_by_resolved_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    module = _load(link / "mod.py", "VALUE = 1\n", monkeypatch)

    expected = xxhash.xxh64(b"VALUE = 1\n").hexdigest()
    resolved = (real / "mod.py").resolve()
    assert loaded_source.digest(str(link / "mod.py")) == expected
    assert loaded_source.digest(str(resolved)) == expected
    assert loaded_source.digests()[module.__name__] == _text(resolved, b"VALUE = 1\n")
    assert loaded_source.digest(str(tmp_path / "never_loaded.py")) is None


def test_prune_forgets_modules_no_longer_loaded_and_their_linecache_lines(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An import session's end removes its modules; their entries and the `linecache` lines
    pinned for them go too, so `inspect` reads such a file from disk again."""
    kept = _load(tmp_path / "kept.py", "VALUE = 1\n", monkeypatch)
    gone = _load(tmp_path / "gone.py", "def value():\n    return 2\n", monkeypatch)
    loaded_source.getsource(gone.value)
    assert str(tmp_path / "gone.py") in linecache.cache, "getsource pinned the loaded lines"
    token = loaded_source.token()
    monkeypatch.delitem(sys.modules, gone.__name__)

    loaded_source.prune()

    assert loaded_source.digest(str(tmp_path / "gone.py")) is None
    assert str(tmp_path / "gone.py") not in linecache.cache
    assert loaded_source.digest(str(tmp_path / "kept.py")) is not None
    assert kept.__name__ in loaded_source.digests()
    assert loaded_source.token() != token


def test_a_module_from_another_loader_is_recorded_from_that_loader(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record is about the module, not the loader: a module CPython's own loader imported
    (as `spec_from_file_location` or an editable install's finder gives it) is recorded from the
    bytes its loader reads, and fingerprinting then reads those after an edit."""
    path = tmp_path / "plain_loader.py"
    path.write_text('def value():\n    return "v1"\n')
    name = f"plain_loader_{uuid.uuid4().hex}"
    discovery.install_user_code_loader(str(tmp_path))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    path.write_text('def value():\n    return "v2, edited"\n')

    assert loaded_source.digests()[name] == _text(path, b'def value():\n    return "v1"\n')
    assert '"v1"' in loaded_source.getsource(module.value)


def test_recording_loader_uses_checked_source_over_a_timestamp_valid_pyc(
    tmp_path: pathlib.Path,
) -> None:
    """A direct `spec_from_file_location` source loader must not execute v1 bytecode while its
    recorder stores an edited v2 file with v1's size and timestamp."""
    path = tmp_path / "same_stat.py"
    path.write_text('def value():\n    return "alpha"\n')
    py_compile.compile(str(path), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    old = path.stat()
    path.write_text('def value():\n    return "bravo"\n')
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    discovery.install_user_code_loader(str(tmp_path))
    name = f"same_stat_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.value() == "bravo"
    assert loaded_source.digests()[name] == _text(path, b'def value():\n    return "bravo"\n')


def test_a_module_in_a_zip_is_recorded_from_the_zip(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "lib.zip"
    name = f"zipped_{uuid.uuid4().hex}"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{name}.py", 'def value():\n    return "v1"\n')
    discovery.install_user_code_loader(str(tmp_path))
    monkeypatch.syspath_prepend(str(archive))
    module = importlib.import_module(name)

    assert '"v1"' in loaded_source.getsource(module.value)
    assert not loaded_source.digests()[name].endswith("\0")


def test_a_sourceless_module_is_recorded_as_unreadable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.pyc` with no source holds code no one can read back: it is recorded with no digest,
    and a stage that uses it is never trusted, even in the process that loaded it."""
    name = f"sourceless_{uuid.uuid4().hex}"
    source = tmp_path / f"{name}.py"
    source.write_text('def value():\n    return "v1"\n')
    compiled = tmp_path / f"{name}.pyc"
    py_compile.compile(str(source), cfile=str(compiled), doraise=True)
    source.unlink()
    # Import through the normal path with the recorder active, so the RecordingLoader wraps it.
    discovery.install_user_code_loader(str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)

    assert loaded_source.digests()[name].endswith("\0")
    edited, unreadable = loaded_source.check_stage({f"func:{name}.value": "x"}, set(), "", None)
    assert (edited, unreadable) == ([], [name])


def test_a_stage_uses_modules_its_manifest_hashes_or_it_imported(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the modules a stage used are compared: one whose code its manifest hashes, or one it
    imported while it ran. A by-name `ext:` entry (a helper whose body never reruns its callers)
    and a `module:` coverage entry are not uses of the module's code."""
    helper = _load(tmp_path / "helper.py", "def value():\n    return 1\n", monkeypatch)
    name = helper.__name__
    edited_record = {name: f"{(tmp_path / 'helper.py').resolve()}\0{'0' * 16}"}
    loaded_source.set_coordinator(edited_record)
    token = loaded_source._token_of(edited_record)

    def check(manifest: dict[str, str], imported: set[str]) -> list[str]:
        return loaded_source.check_stage(manifest, imported, token, None)[0]

    assert check({f"func:{name}.value": "h"}, set()) == [name]
    assert check({f"dep:data:loader:{name}.Reader:load": "h"}, set()) == [name]
    assert check({f"const:{name}.SCALE": "2"}, set()) == [name]
    assert check({}, {name}) == [name], "imported while the stage ran"
    assert check({f"ext:{name}.value": f"{name}.value", f"module:{name}": ""}, set()) == []
    assert check({"self:unrelated": "h"}, set()) == [], "another stage's module is not compared"


def test_check_stage_compares_by_module_name_and_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that loads another file for a module name than the coordinator did (a new
    package shadowing `helpers.py`) ran other code, even with the same bytes."""
    shadow = tmp_path / "pkg"
    shadow.mkdir()
    data = b"VALUE = 1\n"
    module = _load(shadow / "__init__.py", data.decode(), monkeypatch)
    name = module.__name__
    coordinator = {name: _text(tmp_path / "module.py", data)}
    loaded_source.set_coordinator(coordinator)

    edited, _ = loaded_source.check_stage({}, {name}, loaded_source._token_of(coordinator), None)

    assert edited == [name]


def test_check_stage_uses_digests_a_stage_carries(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator loaded user code after these workers started, so the stage carries the
    records it was fingerprinted against, and the worker compares with them."""
    shared = _load(tmp_path / "shared.py", "VALUE = 2\n", monkeypatch)
    started_with = {shared.__name__: _text(tmp_path / "shared.py", b"VALUE = 2\n")}
    loaded_source.set_coordinator(started_with)
    sent = {shared.__name__: _text(tmp_path / "shared.py", b"VALUE = 1\n"), "other": "/o.py\0x"}

    assert loaded_source.check_stage({}, {shared.__name__}, "later", sent)[0] == [shared.__name__]
    assert loaded_source.check_stage(
        {}, {shared.__name__}, loaded_source._token_of(started_with), None
    ) == ([], [])


def test_check_stage_counts_every_used_module_when_the_coordinator_moved_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage sent with a token other than the one the worker started with, and without the
    records, cannot be checked: every module it used that the worker loaded counts as edited, so
    it reruns rather than reusing. A module it did not use does not count."""
    shared = _load(tmp_path / "shared.py", "VALUE = 2\n", monkeypatch)
    other = _load(tmp_path / "other.py", "VALUE = 3\n", monkeypatch)
    loaded_source.set_coordinator(loaded_source.digests())

    edited, _ = loaded_source.check_stage({}, {shared.__name__}, "a-later-token", None)

    assert edited == [shared.__name__]
    assert other.__name__ not in edited


def test_a_stage_carries_digests_only_after_the_record_changed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers get the records once, in their initializer arguments; a stage carries only the
    token until the coordinator loads more user code."""
    _load(tmp_path / "one.py", "VALUE = 1\n", monkeypatch)

    kwargs = executor_core._requested_reusable_executor_kwargs(None, None)
    initargs = cast("tuple[object, ...]", kwargs["initargs"])

    assert initargs[2] == loaded_source.digests()
    assert loaded_source.digests_for_stage() is None

    _load(tmp_path / "two.py", "VALUE = 2\n", monkeypatch)

    assert loaded_source.digests_for_stage() == loaded_source.digests()


class _TransformingLoader(importlib.machinery.SourceFileLoader):
    """A source loader subclass, as a tool that rewrites source before compiling it has."""


def _import_before_recorder(
    path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """Import `path` as a plain `sys.path` import does, with CPython's source file loader and no
    recorder installed: what `sitecustomize` or a `.pth` import line does before Pivot starts."""
    name = f"unseen_{uuid.uuid4().hex}"
    spec = discovery._original_spec_from_file_location(name, path)
    assert spec is not None and type(spec.loader) is importlib.machinery.SourceFileLoader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _started_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have this process start now, so every file written before counts as unchanged since."""
    started = time.time_ns() + loaded_source._CHANGE_TIME_SLACK_NS
    monkeypatch.setattr(loaded_source, "_process_start_ns", lambda: started)


def _reason(name: str) -> str:
    return loaded_source._record_fields(loaded_source.digests()[name])[2]


def test_a_module_imported_before_the_recorder_is_recorded_from_its_unchanged_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module imported before Pivot installed its recorder, whose file and bytecode have not
    changed since the process started, is recorded from its file: those bytes are the ones it was
    compiled from, and fingerprinting keeps reading them after an edit."""
    path = tmp_path / "preloaded.py"
    data = b'def value():\n    return "v1"\n'
    path.write_bytes(data)
    module = _import_before_recorder(path, monkeypatch)
    _started_now(monkeypatch)

    loaded_source.record_unseen_modules()
    path.write_text('def value():\n    return "v2, edited"\n')

    assert loaded_source.digests()[module.__name__] == _text(path, data)
    assert '"v1"' in loaded_source.getsource(module.value)


def test_a_module_whose_file_changed_since_the_process_started_is_unreadable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The file was written after this process started, as an edit during a run's startup is, so
    Pivot cannot show the module holds the file's code: it is unreadable, a stage that uses it is
    never reused, and the reason is logged once per process, not once per stage."""
    path = tmp_path / "preloaded.py"
    path.write_text('def value():\n    return "v1"\n')
    module = _import_before_recorder(path, monkeypatch)
    name = module.__name__

    loaded_source.record_unseen_modules()

    assert loaded_source.digests()[name] == (
        f"{path.resolve()}\0\0its file changed after this process started"
    )
    assert loaded_source.digest(str(path)) is None
    with caplog.at_level(logging.WARNING, logger="pivot.loaded_source"):
        for _ in range(2):
            assert loaded_source.check_stage({f"func:{name}.value": "h"}, set(), "", None) == (
                [],
                [name],
            )
    warnings = [r.getMessage() for r in caplog.records if r.name == "pivot.loaded_source"]
    assert len(warnings) == 1, warnings
    assert f"user module '{name}' compile" in warnings[0]
    assert "because its file changed after this process started" in warnings[0]


def test_a_module_another_loader_loaded_before_the_recorder_is_unreadable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only CPython's own source file loader is known to run its file's text; a subclass may
    transform it, so its file says nothing about the module's code."""
    path = tmp_path / "transformed.py"
    path.write_text("VALUE = 1\n")
    name = f"transformed_{uuid.uuid4().hex}"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))  # removed after the test
    _import_with(name, _TransformingLoader(name, str(path)), str(path))
    _started_now(monkeypatch)

    loaded_source.record_unseen_modules()

    assert _reason(name) == "its loader (_TransformingLoader) is not CPython's source file loader"


def test_stale_bytecode_the_import_ran_makes_the_module_unreadable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPython checks mtime-checked bytecode by the source's size and whole-second mtime, so an
    edit that keeps both imports the earlier code. The file then does not hold the module's code,
    though Python's own check accepts the bytecode for it."""
    path = tmp_path / "stale.py"
    path.write_text('def value():\n    return "alpha"\n')
    py_compile.compile(str(path), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    old = path.stat()
    path.write_text('def value():\n    return "bravo"\n')
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    module = _import_before_recorder(path, monkeypatch)
    assert module.value() == "alpha", "the import ran the earlier bytecode"
    _started_now(monkeypatch)

    loaded_source.record_unseen_modules()

    assert _reason(module.__name__) == "its cached bytecode is not the code of its file"


@pytest.mark.parametrize(
    ("mode", "trusted"),
    [
        pytest.param(py_compile.PycInvalidationMode.TIMESTAMP, True, id="mtime-checked"),
        pytest.param(py_compile.PycInvalidationMode.CHECKED_HASH, False, id="checked-hash"),
    ],
)
def test_bytecode_written_after_the_process_started(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: py_compile.PycInvalidationMode,
    trusted: bool,
) -> None:
    """Bytecode written since the process started replaced the file the import saw. CPython's
    import writes mtime-checked bytecode only after rejecting the earlier file for this same
    source, so the module was then compiled from its file; another writer (Pivot's own loader
    writes checked-hash bytecode) may have replaced bytecode the import ran."""
    path = tmp_path / "rewritten.py"
    data = b'def value():\n    return "v1"\n'
    path.write_bytes(data)
    time.sleep(2 * loaded_source._CHANGE_TIME_SLACK_NS / 1e9)
    started = time.time_ns()
    monkeypatch.setattr(loaded_source, "_process_start_ns", lambda: started)
    py_compile.compile(str(path), invalidation_mode=mode)
    module = _import_before_recorder(path, monkeypatch)

    loaded_source.record_unseen_modules()

    assert loaded_source.digests()[module.__name__] == (
        _text(path, data)
        if trusted
        else f"{path.resolve()}\0\0its bytecode file changed after this process started"
    )


def test_a_module_imported_without_bytecode_is_recorded_unless_bytecode_was_removed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no bytecode file, and none removed since the process started, the import compiled
    the file. Bytecode removed since then may be what the import ran."""
    kept = tmp_path / "kept" / "nobytecode.py"
    kept.parent.mkdir()
    kept.write_text("VALUE = 1\n")
    removed = tmp_path / "removed" / "nobytecode.py"
    removed.parent.mkdir()
    removed.write_text("VALUE = 1\n")
    py_compile.compile(str(removed), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    time.sleep(2 * loaded_source._CHANGE_TIME_SLACK_NS / 1e9)
    started = time.time_ns()
    monkeypatch.setattr(loaded_source, "_process_start_ns", lambda: started)
    pathlib.Path(importlib.util.cache_from_source(str(removed))).unlink()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    kept_module = _import_before_recorder(kept, monkeypatch)
    removed_module = _import_before_recorder(removed, monkeypatch)

    loaded_source.record_unseen_modules()

    assert loaded_source.digests()[kept_module.__name__] == _text(kept, b"VALUE = 1\n")
    assert _reason(removed_module.__name__) == (
        "its bytecode directory changed after this process started"
    )


def test_prune_forgets_what_record_unseen_modules_recorded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module no recorder saw compile is judged at each discovery: the end of an import
    session forgets its record, though the module stays loaded."""
    path = tmp_path / "preloaded.py"
    path.write_text("VALUE = 1\n")
    module = _import_before_recorder(path, monkeypatch)
    _started_now(monkeypatch)
    loaded_source.record_unseen_modules()
    assert module.__name__ in loaded_source.digests()

    loaded_source.prune()

    assert module.__name__ not in loaded_source.digests()
    loaded_source.record_unseen_modules()
    assert module.__name__ in loaded_source.digests(), "the next discovery records it again"


def test_a_worker_logs_the_coordinators_reason_for_an_unseen_module(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The coordinator's record of a module it could not vouch for carries the reason, so the
    worker whose stage uses the module says why the stage is not reused."""
    name = "coordinator_only"
    records = {name: f"{tmp_path / 'm.py'}\0\0its file changed after this process started"}
    loaded_source.set_coordinator(records)

    with caplog.at_level(logging.WARNING, logger="pivot.loaded_source"):
        checked = loaded_source.check_stage(
            {f"func:{name}.value": "h"}, set(), loaded_source._token_of(records), None
        )

    assert checked == ([], [name])
    assert f"user module '{name}' compile" in caplog.text
    assert "because its file changed after this process started" in caplog.text


def test_process_start_is_never_after_the_process_started() -> None:
    """A file changed after the process started must not count as unchanged since: the start
    Pivot reads may be early, never late."""
    before = time.time_ns()
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import time\nnow = time.time_ns()\nfrom pivot import loaded_source\n"
            + "print(now, loaded_source._process_start_ns())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    now, started = (int(field) for field in done.stdout.split())

    assert before - loaded_source._CHANGE_TIME_SLACK_NS <= started <= now


def test_a_forked_process_cannot_vouch_for_modules_its_parent_loaded() -> None:
    """A process forked without exec holds the modules its parent imported before it started,
    so its own start time says nothing about when they were compiled."""
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os\nfrom pivot import loaded_source\n"
            + "assert loaded_source._process_start_ns() is not None\n"
            + "pid = os.fork()\n"
            + "if pid == 0:\n"
            + "    os._exit(0 if loaded_source._process_start_ns() is None else 1)\n"
            + "_, status = os.waitpid(pid, 0)\n"
            + "raise SystemExit(os.waitstatus_to_exitcode(status))\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr


def test_a_worker_judges_the_modules_it_imported_before_its_initializer(
    tmp_path: pathlib.Path,
) -> None:
    """A worker imports what `sitecustomize` or a `.pth` line imports before loky runs its
    initializer, so `core._init_worker` judges those modules by discovery's rule: one whose file
    changed after the worker started is unreadable there, with the reason. Run in a child
    process, which the initializer's import hooks and signal handlers may change."""
    path = tmp_path / "worker_preloaded.py"
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import os, pathlib, sys\nsys.path.insert(0, {str(tmp_path)!r})\n"
            + f"pathlib.Path({str(path)!r}).write_text('VALUE = 1\\n')\n"
            + "import worker_preloaded\nfrom pivot import loaded_source\n"
            + "from pivot.executor import core\ncore._init_worker(os.getppid(), None, {})\n"
            + "print(repr(loaded_source.digests()['worker_preloaded']))\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == repr(
        f"{path.resolve()}\0\0its file changed after this process started"
    )


def _stage_info(func: object, coordinator: dict[str, str]) -> worker.WorkerStageInfo:
    """The fields of a stage `worker._untrusted_modules` reads, sent with `coordinator`'s token."""
    fields = cast(
        "object",
        {
            "func": func,
            "no_fingerprint": False,
            "loaded_sources_token": loaded_source._token_of(coordinator),
        },
    )
    return cast("worker.WorkerStageInfo", fields)


def test_a_stage_checks_the_module_that_defines_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage defined outside the project root reaches a worker by reference, so the worker
    imports its module by name. Its manifest hashes its body under `self:`, which names no
    module, so the module that defines it is what tells the worker that it holds other code than
    the coordinator fingerprinted."""
    path = tmp_path / "stage_module.py"
    module = _load(path, "def stage():\n    return 'v2'\n", monkeypatch)
    coordinator = {module.__name__: f"{path.resolve()}\0{'0' * 16}"}
    loaded_source.set_coordinator(coordinator)

    edited, _ = worker._untrusted_modules(
        _stage_info(module.stage, coordinator),
        {"self:stage": "h"},
        worker._StageImports(covered=frozenset()),
        tmp_path / "project",
    )

    assert edited == [module.__name__]


def test_a_stage_checks_the_modules_it_imported_while_it_ran(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module the stage imported while it ran, already loaded in the worker, is used even when
    the stage's manifest hashes none of its code: the worker holding another version than the
    coordinator makes the stage untrusted."""
    stage_path = tmp_path / "stage_module.py"
    stage_module = _load(stage_path, "def stage():\n    return 1\n", monkeypatch)
    imported_path = tmp_path / "imported_module.py"
    imported = _load(imported_path, "VALUE = 2\n", monkeypatch)
    coordinator = {
        stage_module.__name__: _text(stage_path, b"def stage():\n    return 1\n"),
        imported.__name__: f"{imported_path.resolve()}\0{'0' * 16}",
    }
    loaded_source.set_coordinator(coordinator)

    edited, _ = worker._untrusted_modules(
        _stage_info(stage_module.stage, coordinator),
        {},
        worker._StageImports(covered=frozenset(), imported={imported.__name__}),
        tmp_path,
    )

    assert edited == [imported.__name__]
