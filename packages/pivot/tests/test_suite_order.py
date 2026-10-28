"""PR518RecheckR2 red test, finding 2: tests pass in either order in one process.

Put at packages/pivot/tests/test_suite_order.py. One pytest-xdist worker runs many tests in one
process, in an order the scheduler picks. A test that leaves Pivot's user-code path hook, its
module recorder or a lasting import session installed changes what a later test imports and
fingerprints. CI runs 40328718857, 40341797294 and 40357442105 on this branch failed this way.
"""

import pathlib
import subprocess
import sys

import pytest

_TESTS = pathlib.Path(__file__).parent


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            "fingerprint/test_loaded_source.py::test_a_module_from_another_loader_is_recorded_from_that_loader",
            "test_discovery.py::test_session_end_gives_user_code_cpythons_loader_back",
            id="worker-loader-installed-by-a-test",
        ),
        pytest.param(
            "test_discovery.py::test_discovery_reuses_the_bytecode_of_an_unchanged_project_file",
            "fingerprint/test_fingerprint.py::test_cached_manifest_misses_when_a_recorded_file_changes_keeping_its_stat",
            id="session-recorder-lasting-after-discovery",
        ),
        pytest.param(
            "test_discovery.py::test_discovery_reuses_the_bytecode_of_an_unchanged_project_file",
            "fingerprint/test_fingerprint.py::test_manifest_cache_miss_on_file_deleted",
            id="session-recorder-and-a-deleted-file",
        ),
    ],
)
def test_two_tests_pass_in_one_process_in_this_order(
    tmp_path: pathlib.Path, first: str, second: str
) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:xdist",
            "-p",
            "no:cacheprovider",
            "-q",
            "--basetemp",
            str(tmp_path / "bt"),
            str(_TESTS / first),
            str(_TESTS / second),
        ],
        cwd=_TESTS.parents[2],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-4000:]
