from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from pivot.storage import state

if TYPE_CHECKING:
    import pathlib


def test_state_dbs_share_environment_for_one_state_dir(tmp_path: pathlib.Path) -> None:
    """Concurrent StateDB users on one path share one LMDB environment (#465)."""
    cached_file = tmp_path / "input.txt"
    cached_file.write_text("content")
    cached_stat = cached_file.stat()

    with state.StateDB(tmp_path, readonly=True) as reader:
        with state.StateDB(tmp_path) as writer:
            writer.save(cached_file, cached_stat, "cached-hash")
            with state.StateDB(tmp_path, readonly=True) as second_reader:
                assert second_reader.get(cached_file, cached_stat) == "cached-hash"
        assert reader.get(cached_file, cached_stat) == "cached-hash"


def test_new_state_db_uses_recreated_environment(tmp_path: pathlib.Path) -> None:
    """A recreated state.lmdb is not served through an old environment."""
    holder = state.StateDB(tmp_path, readonly=True)
    shutil.rmtree(tmp_path / "state.lmdb")

    with state.StateDB(tmp_path) as replacement:
        replacement.put_raw(b"generation", b"from-recreated-environment")

    holder.close()
    with state.StateDB(tmp_path, readonly=True) as reader:
        assert reader.get_raw(b"generation") == b"from-recreated-environment"


def test_fingerprint_and_explain_share_state_db_in_one_process(tmp_path: pathlib.Path) -> None:
    """Fingerprint's cache DB must not block explain's StateDB open (#465)."""
    script = """
import pathlib
import sys

from pivot import explain, fingerprint, project
from pivot.config import io
from pivot.storage import lock
from pivot.types import LockData

root = pathlib.Path(sys.argv[1])
(root / '.git').mkdir()
state_dir = root / '.pivot'
state_dir.mkdir()
project._project_root_cache = root
io.clear_config_cache()

lock.StageLock('stage', lock.get_stages_dir(state_dir)).write(
    LockData(
        code_manifest={'self:stage': 'fingerprint'},
        params={},
        dep_hashes={},
        output_hashes={},
    )
)
assert fingerprint._get_state_db() is not None
explanation = explain.get_stage_explanation(
    stage_name='stage',
    fingerprint={'self:stage': 'fingerprint'},
    deps=[],
    outs_paths=[],
    params_instance=None,
    overrides=None,
    state_dir=state_dir,
)
assert explanation['will_run'] is False
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_state_db_reopens_after_fork(tmp_path: pathlib.Path) -> None:
    """A child gets a fresh StateDB instead of the parent's environment."""
    script = """
import os
import pathlib
import sys

from pivot.storage import state

state_dir = pathlib.Path(sys.argv[1])
parent = state.StateDB(state_dir)
parent.put_raw(b'parent', b'written-before-fork')

pid = os.fork()
if pid == 0:
    try:
        try:
            parent.get_raw(b'parent')
        except RuntimeError as error:
            assert 'inherited across fork' in str(error)
        else:
            raise AssertionError('inherited StateDB was usable in child')

        with state.StateDB(state_dir) as child:
            assert child.get_raw(b'parent') == b'written-before-fork'
            child.put_raw(b'child', b'written-by-child')
    except BaseException as error:
        print(repr(error), file=sys.stderr)
        os._exit(1)
    os._exit(0)

_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
assert parent.get_raw(b'child') == b'written-by-child'
parent.close()
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_fork_waits_for_active_write_before_reopening_state_db(tmp_path: pathlib.Path) -> None:
    """Fork waits for StateDB writes, then both processes use fresh environments."""
    script = """
import os
import pathlib
import sys
import threading

from pivot.storage import state

state_dir = pathlib.Path(sys.argv[1])
parent = state.StateDB(state_dir)
ready = threading.Event()
release = threading.Event()

def write() -> None:
    with parent._write_transaction() as txn:
        txn.put(b'parent', b'written-before-fork')
        ready.set()
        release.wait()

writer = threading.Thread(target=write)
writer.start()
assert ready.wait(5)
timer = threading.Timer(0.1, release.set)
timer.start()

pid = os.fork()
if pid == 0:
    try:
        with state.StateDB(state_dir) as child:
            assert child.get_raw(b'parent') == b'written-before-fork'
            child.put_raw(b'child', b'written-by-child')
    except BaseException as error:
        print(repr(error), file=sys.stderr)
        os._exit(1)
    os._exit(0)

_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
timer.join()
writer.join()
assert parent.get_raw(b'child') == b'written-by-child'
parent.close()
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
