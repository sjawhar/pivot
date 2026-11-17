"""The decisions behind the `dist:` key of an installed distribution, checked through the `pivot`
CLI on real installs, then in this process.

Each CLI test makes its own venv with uv over this test environment (`tests.installed_dists`),
installs setuptools fixtures into it the way the decision names (from a path, from git, a local
wheel, as from a package index, editable, by pip), and drives its own pivot project, whose stages
call the installed code. What pivot does is read from its reports and from the stages' locks:
which files the key hashes, which install forms get one, how a stage reaches a distribution and
the distributions it depends on, the refusals of a broken install and their downgrade by unsafe
fingerprinting, and the refusal of a project file that shadows a module the stage reaches.

The in-process tests, in a section of their own at the end, call `pivot.installed_distributions`:
its pure functions on the bytes uv and pip write and on wrong and boundary inputs, and
`distribution_keys` over real installs in a venv each test makes, with each reached module's
origin given as plain data; and one fingerprints, through the registry, a stage that imports its
installed library only in its body. No fixture is imported into this process.
"""

import collections.abc
import errno
import importlib.metadata
import json
import os
import pathlib
import py_compile
import subprocess
import sys

import pytest
import pytest_mock
import xxhash
from tests import installed_dists

from pivot import (
    discovery,
    exceptions,
    fingerprint,
    import_origin,
    installed_distributions,
    loaded_source,
    project,
)

_STAGE = "p/make"
_OUTPUT = pathlib.Path("data/p/make.txt")
_UNSAFE_CONFIG = "core:\n  unsafe_fingerprinting: true\n"

# Each refusal of a broken install, and its warning under unsafe fingerprinting, ends with this.
_REFUSAL_FIX = "Fix: reinstall"

_FIXDEP = 'def base():\n    return "base-1"\n'
_FIXLIB = """\
import fixdep


def value():
    return "value-1:" + fixdep.base()


def main():
    print(value())
"""
_FIXLIB_ALONE = """\
def value():
    return "value-1"


def main():
    print(value())
"""

# A project whose stage writes the concatenated `value()` of each module it imports and names
# nothing else of them.
_PIPELINE = """\
{imports}

from pivot import Pipeline, stage


@stage
def make() -> str:
    return {expression}


with Pipeline("p") as pipeline:
    make()
"""

# Item (e): `p/mid` calls an index-installed library, `p/chain` the first of three git installs.
_CLOSURE_PIPELINE = """\
import fixa
import idxmid

from pivot import Pipeline, stage


@stage
def mid() -> str:
    return idxmid.value()


@stage
def chain() -> str:
    return fixa.value()


with Pipeline("p") as pipeline:
    mid()
    chain()
"""

# Item (j): a library's pivot loader, which writes the stage's output upper-cased.
_LOADER = """\
import dataclasses
import pathlib

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Shout(loaders.Writer[str]):
    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data.upper())
"""
_NO_FINGERPRINT_PIPELINE = """\
from typing import Annotated

from fixloader import Shout
from pivot import Pipeline, no_fingerprint, stage


@stage
@no_fingerprint()
def make() -> Annotated[str, Shout()]:
    return "hello"


with Pipeline("p") as pipeline:
    make()
"""

# Item (l): the stage reaches the installed `fixbase` only through the base class of a user class.
_FIXBASE = """\
class Base:
    def greet(self):
        return "installed"
"""
_SHADOW = """\
class Base:
    def greet(self):
        return "shadow"
"""
_BASE_CLASS_PIPELINE = """\
import fixbase

from pivot import Pipeline, stage


class Greeter(fixbase.Base):
    pass


@stage
def make() -> str:
    return Greeter().greet()


with Pipeline("p") as pipeline:
    make()
"""

# The refusals of a broken install, word for word, `{site}` the venv's resolved site-packages.
_UNCLASSIFIED_REFUSAL = (
    "Distribution 'fix-lib' ({site}/fix_lib-0.1.dist-info), which this stage reaches, is "
    "installed in a form Pivot cannot classify: its direct_url.json is not a JSON object. Pivot "
    "cannot fingerprint the installed code the stage runs, so the stage could skip a run after "
    "that code changes. Fix: reinstall 'fix-lib'."
)
_RECORDLESS_DEPENDENCY_REFUSAL = (
    "Distribution 'fixdep' ({site}/fixdep-0.1.dist-info), which a distribution this stage "
    "reaches depends on, is installed from a local directory (dir_info) and has no RECORD file, "
    "so Pivot cannot read its files. Pivot cannot fingerprint the installed code the stage runs, "
    "so the stage could skip a run after that code changes. Fix: reinstall 'fixdep'."
)
# A reached module `{module}`, whose file `{file}` no RECORD lists, beside `Fix_Lib` with no RECORD.
_UNLISTED_REFUSAL = (
    "Module '{module}' ({site}/{file}) lies in {site}, where distribution 'fix-lib' "
    "({site}/fix_lib-0.1.dist-info), installed from a local directory (dir_info), has no RECORD "
    "file, and no RECORD file there lists that file. Pivot cannot tell whether the stage runs "
    "that distribution's code, so the stage could skip a run after it changes. Fix: reinstall "
    "'fix-lib'."
)
_RECORDLESS_REACHED_REFUSAL = _UNLISTED_REFUSAL.format(
    module="fixlib", file="fixlib/__init__.py", site="{site}"
)


def _helper_fixlib_and_fixdep(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Write under `root` the projects of `Fix_Lib` (module `fixlib`, requires `fixdep`) and
    `fixdep`; returns their directories."""
    fixlib = installed_dists.write_library(
        root, "Fix_Lib", "fixlib", _FIXLIB, dependencies=("fixdep",)
    )
    fixdep = installed_dists.write_library(root, "fixdep", "fixdep", _FIXDEP)
    return fixlib, fixdep


def _helper_pipeline(*modules: str) -> str:
    """The project whose stage imports each of `modules` and writes their `value()`s joined."""
    return _PIPELINE.format(
        imports="\n".join(f"import {module}" for module in modules),
        expression=" + ".join(f"{module}.value()" for module in modules),
    )


def _helper_project(
    tmp_path: pathlib.Path, venv: installed_dists.Venv, pipeline: str
) -> installed_dists.Project:
    return installed_dists.make_project(tmp_path / "project", tmp_path / "home", venv, pipeline)


def _helper_damage(path: pathlib.Path, text: str | None) -> None:
    """Write `text` over the installed file at `path`, or delete it when `text` is None."""
    assert path.is_file(), f"{path} must be installed before the test damages it"
    if text is None:
        path.unlink()
    else:
        path.write_text(text)


def _helper_flat(text: str) -> str:
    """`text` with each run of white space made one space, as a wrapped message reads."""
    return " ".join(text.split())


def _helper_ran(project: installed_dists.Project, output: str) -> dict[str, str]:
    """`pivot repro` runs the stage, which writes `output`, and refuses and warns of nothing;
    returns the lock's code manifest."""
    status, shown = installed_dists.repro_status(project, _STAGE)
    assert status == "ran", f"`pivot repro` must run {_STAGE}:\n{shown}"
    assert _REFUSAL_FIX not in shown, f"`pivot repro` must refuse and warn of nothing:\n{shown}"
    written = (project.root / _OUTPUT).read_text()
    assert written == output, f"{_STAGE} must write {output!r}, not {written!r}"
    return installed_dists.lock_manifest(project, _STAGE)


def _helper_assert_changes(project: installed_dists.Project, keys: list[str]) -> None:
    """`pivot status --explain --json` names exactly `keys` changed, and refuses nothing."""
    changed, shown = installed_dists.code_changes(project, _STAGE)
    assert changed == keys, f"`pivot status --explain` must name exactly {keys} changed:\n{shown}"
    assert _REFUSAL_FIX not in shown, f"`pivot status` must refuse and warn of nothing:\n{shown}"


def _helper_without(manifest: dict[str, str], key: str) -> dict[str, str]:
    return {name: value for name, value in manifest.items() if name != key}


def _helper_assert_refused(ran: subprocess.CompletedProcess[str], refusal: str) -> None:
    """The pivot command `ran` failed, refusing the stage with `refusal`."""
    shown = installed_dists.shown(ran)
    assert ran.returncode == 1, f"pivot must refuse {_STAGE}:\n{shown}"
    assert refusal in _helper_flat(shown), f"pivot must refuse with:\n{refusal}\n{shown}"


def _helper_assert_runs_with_warning(
    project: installed_dists.Project, warning: str, command: str
) -> None:
    """`pivot repro` runs the stage, not restoring it from the run cache, and warns `warning`."""
    status, shown = installed_dists.repro_status(project, _STAGE)
    assert status == "ran", f"{command} must run {_STAGE}:\n{shown}"
    assert warning in _helper_flat(shown), f"{command} must warn:\n{warning}\n{shown}"
    written = (project.root / _OUTPUT).read_text()
    assert written == "value-1:base-1", f"{_STAGE} must write its value, not {written!r}"


def test_installs_by_uv_and_by_pip_in_two_venvs_give_one_value(tmp_path: pathlib.Path) -> None:
    """(a) `Fix_Lib`, with a console script, installed from its path by uv into one venv and by
    pip into a second venv at another path. Their RECORDs differ: each lists its console script,
    whose first line names its own venv, and pip's lists bytecode. The lock that the first venv's
    pivot writes is current for the second venv's pivot: `dist:fix-lib` has one value."""
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", _FIXLIB_ALONE, scripts={"fix-lib": "fixlib:main"}
    )
    by_uv = installed_dists.make_venv(tmp_path / "uv-venv")
    by_pip = installed_dists.make_venv(tmp_path / "pip-venv", seed=True)
    installed_dists.install_directory(by_uv, source)
    installed_dists.pip_install_directory(by_pip, source)

    uv_dist = installed_dists.installed(by_uv, "Fix_Lib")
    pip_dist = installed_dists.installed(by_pip, "Fix_Lib")
    uv_files = [str(path) for path in uv_dist.files or []]
    pip_files = [str(path) for path in pip_dist.files or []]
    script = "../../../bin/fix-lib"
    assert script in uv_files and script in pip_files, (
        f"both RECORDs must list the console script {script}: uv {uv_files}, pip {pip_files}"
    )
    assert any("__pycache__" in path and path.endswith(".pyc") for path in pip_files), (
        f"pip's RECORD must list bytecode: {pip_files}"
    )
    uv_script = (by_uv.python.parent / "fix-lib").read_bytes()
    pip_script = (by_pip.python.parent / "fix-lib").read_bytes()
    assert uv_script != pip_script, "the two console scripts must differ, each naming its venv"
    installers = (uv_dist.read_text("INSTALLER"), pip_dist.read_text("INSTALLER"))
    assert installers == ("uv", "pip\n"), f"the installers must be uv and pip: {installers}"
    project = _helper_project(tmp_path, by_uv, _helper_pipeline("fixlib"))
    manifest = _helper_ran(project, "value-1")
    assert installed_dists.dist_keys(manifest) == ["dist:fix-lib"], f"the lock: {manifest}"

    by_pip_project = installed_dists.Project(root=project.root, home=project.home, venv=by_pip)

    status, shown = installed_dists.stage_status(by_pip_project, _STAGE)
    assert status == "cached", f"the pip venv's pivot must report {_STAGE} cached:\n{shown}"
    _helper_assert_changes(by_pip_project, [])


def test_a_byte_edited_in_place_in_an_installed_file_changes_the_value(
    tmp_path: pathlib.Path,
) -> None:
    """(b) One byte of the literal `value()` returns, edited in place in the installed
    `fixlib/__init__.py` with no reinstall, makes the stage stale on `dist:fix-lib`: the key
    hashes the files read from disk, not the hashes RECORD holds."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(tmp_path / "src", "Fix_Lib", "fixlib", _FIXLIB_ALONE)
    installed_dists.install_directory(venv, source)
    project = _helper_project(tmp_path, venv, _helper_pipeline("fixlib"))
    _helper_ran(project, "value-1")

    installed_dists.replace_once(
        venv.site_packages / "fixlib" / "__init__.py", '"value-1"', '"value-2"'
    )

    _helper_assert_changes(project, ["dist:fix-lib"])


def test_a_reinstall_changing_an_entry_point_changes_the_value_and_one_of_the_version_does_not(
    tmp_path: pathlib.Path,
) -> None:
    """(c) `Fix_Lib` reinstalled from its path with its console script's target changed, which
    changes its `entry_points.txt` and its script and no module: the stage is stale on
    `dist:fix-lib`. After a run, `Fix_Lib` reinstalled with only its version changed, which
    changes only the files of its `.dist-info` directory and that directory's name: nothing is
    marked."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", _FIXLIB_ALONE, scripts={"fix-lib": "fixlib:main"}
    )
    installed_dists.install_directory(venv, source)
    project = _helper_project(tmp_path, venv, _helper_pipeline("fixlib"))
    _helper_ran(project, "value-1")

    installed_dists.replace_once(source / "pyproject.toml", '"fixlib:main"', '"fixlib:value"')
    installed_dists.install_directory(venv, source, reinstall=True)

    entry_points = installed_dists.installed(venv, "Fix_Lib").read_text("entry_points.txt")
    assert entry_points is not None and "fixlib:value" in entry_points, (
        f"the reinstall must change the entry point: {entry_points!r}"
    )
    _helper_assert_changes(project, ["dist:fix-lib"])
    _helper_ran(project, "value-1")

    installed_dists.replace_once(source / "pyproject.toml", 'version = "0.1"', 'version = "0.2"')
    installed_dists.install_directory(venv, source, reinstall=True)

    version = installed_dists.installed(venv, "Fix_Lib").version
    assert version == "0.2", f"Fix_Lib 0.2 must now be installed, not {version}"
    _helper_assert_changes(project, [])


def test_a_local_wheel_gets_a_key_and_an_archive_from_an_https_url_gets_none(
    tmp_path: pathlib.Path,
) -> None:
    """(d) `fixwheel` installed from a wheel file: its `direct_url.json` has `archive_info` and a
    `file:` URL, and the stage's lock holds `dist:fixwheel`. The same `direct_url.json` with the
    URL of a remote archive (`https:`) gives no key."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(tmp_path / "src", "fixwheel", "fixwheel", _FIXLIB_ALONE)
    wheels = tmp_path / "wheels"
    installed_dists.build_wheel(venv, source, wheels)
    (wheel,) = wheels.glob("*.whl")
    installed_dists.install(venv, str(wheel))
    url = installed_dists.direct_url(venv, "fixwheel")
    assert url is not None and "archive_info" in url and url["url"].startswith("file:"), (
        f"a wheel file's install must record archive_info with a file: URL: {url!r}"
    )
    project = _helper_project(tmp_path, venv, _helper_pipeline("fixwheel"))
    manifest = _helper_ran(project, "value-1")
    assert installed_dists.dist_keys(manifest) == ["dist:fixwheel"], f"the lock: {manifest}"

    url["url"] = f"https://files.example.org/packages/{wheel.name}"
    (venv.site_packages / "fixwheel-0.1.dist-info" / "direct_url.json").write_text(json.dumps(url))

    manifest = _helper_ran(project, "value-1")
    assert installed_dists.dist_keys(manifest) == [], f"the lock must hold no dist: key: {manifest}"


def test_the_closure_follows_an_index_install_and_a_chain_of_git_installs_with_a_cycle(
    tmp_path: pathlib.Path,
) -> None:
    """(e) `p/mid` calls `idxmid`, installed as from a package index, whose `Requires-Dist` names
    `fixdep`, installed from its path: `p/mid` gets `dist:fixdep` alone. `p/chain` calls `fixa`;
    `fixa`, `fixb` and `fixc` are git installs, `fixa` requires `fixb`, and `fixb` requires `fixc`
    and `fixa` (a cycle): `p/chain` gets the key of each of the three, and the run ends."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", _FIXDEP)
    idxmid = installed_dists.write_library(
        src,
        "idxmid",
        "idxmid",
        'import fixdep\n\n\ndef value():\n    return "mid:" + fixdep.base()\n',
        dependencies=("fixdep",),
    )
    fixa = installed_dists.write_library(
        src,
        "fixa",
        "fixa",
        'import fixb\n\n\ndef value():\n    return "a:" + fixb.value()\n',
        dependencies=("fixb",),
    )
    fixb = installed_dists.write_library(
        src,
        "fixb",
        "fixb",
        'import fixc\n\n\ndef value():\n    return "b:" + fixc.value()\n',
        dependencies=("fixc", "fixa"),
    )
    fixc = installed_dists.write_library(src, "fixc", "fixc", 'def value():\n    return "c"\n')
    wheels = tmp_path / "wheels"
    project = _helper_project(tmp_path, venv, _CLOSURE_PIPELINE)
    installed_dists.install_directory(venv, fixdep)
    installed_dists.build_wheel(venv, idxmid, wheels)
    installed_dists.install_from_wheels(venv, wheels, "idxmid")
    installed_dists.install(
        venv,
        *(
            installed_dists.init_git_requirement(source, project.home, name)
            for source, name in ((fixa, "fixa"), (fixb, "fixb"), (fixc, "fixc"))
        ),
    )
    assert installed_dists.direct_url(venv, "idxmid") is None, "idxmid must be in the index form"

    installed_dists.pivot_ok(project, "repro")

    written = [
        (project.root / "data" / "p" / f"{stage}.txt").read_text() for stage in ("mid", "chain")
    ]
    assert written == ["mid:base-1", "a:b:c"], f"both stages must run: {written}"
    mid = installed_dists.lock_manifest(project, "p/mid")
    assert installed_dists.dist_keys(mid) == ["dist:fixdep"], f"p/mid's lock: {mid}"
    chain = installed_dists.lock_manifest(project, "p/chain")
    assert installed_dists.dist_keys(chain) == ["dist:fixa", "dist:fixb", "dist:fixc"], (
        f"p/chain's lock: {chain}"
    )


def test_a_requirement_installed_editable_gets_no_key(tmp_path: pathlib.Path) -> None:
    """(e2) The stage calls `Fix_Lib`, installed from its path, whose `Requires-Dist` names
    `fixdep`, installed editable: the lock holds `dist:fix-lib` and no key for `fixdep`. No module
    of `fixdep` is reached by name, so only the dependency closure meets it."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = _helper_fixlib_and_fixdep(tmp_path / "src")
    installed_dists.install_directory(venv, fixdep, editable=True)
    installed_dists.install_directory(venv, fixlib)
    url = installed_dists.direct_url(venv, "fixdep")
    assert url is not None and url["dir_info"] == {"editable": True}, (
        f"fixdep must be installed editable: direct_url.json is {url!r}"
    )
    project = _helper_project(tmp_path, venv, _helper_pipeline("fixlib"))

    manifest = _helper_ran(project, "value-1:base-1")

    assert installed_dists.dist_keys(manifest) == ["dist:fix-lib"], f"the lock: {manifest}"


def test_a_namespace_package_shared_with_an_index_distribution_gives_the_local_key_alone(
    tmp_path: pathlib.Path,
) -> None:
    """(f) The namespace package `nsco` holds `nsco.local_part` of `nsco-local`, installed from
    its path, and `nsco.index_part` of `nsco-index`, installed as from a package index. The stage
    reads `nsco` by its top name: the lock holds `dist:nsco-local` and no other `dist:` key."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    local = installed_dists.write_setuptools_project(
        tmp_path / "src" / "nsco-local",
        "nsco-local",
        "0.1",
        packages=["nsco.local_part"],
        files={"nsco/local_part/__init__.py": 'def value():\n    return "local"\n'},
    )
    index = installed_dists.write_setuptools_project(
        tmp_path / "src" / "nsco-index",
        "nsco-index",
        "0.1",
        packages=["nsco.index_part"],
        files={"nsco/index_part/__init__.py": 'def value():\n    return "index"\n'},
    )
    wheels = tmp_path / "wheels"
    installed_dists.install_directory(venv, local)
    installed_dists.build_wheel(venv, index, wheels)
    installed_dists.install_from_wheels(venv, wheels, "nsco-index")
    namespace = venv.site_packages / "nsco"
    assert sorted(child.name for child in namespace.iterdir()) == ["index_part", "local_part"], (
        f"{namespace} must be a namespace package of the two parts"
    )
    project = _helper_project(
        tmp_path, venv, _helper_pipeline("nsco.local_part", "nsco.index_part")
    )

    manifest = _helper_ran(project, "localindex")

    assert installed_dists.dist_keys(manifest) == ["dist:nsco-local"], f"the lock: {manifest}"


def test_a_requirement_behind_a_false_marker_still_adds_its_key(tmp_path: pathlib.Path) -> None:
    """(g) `Fix_Lib`, installed from its path, requires `fixdep` only on Python 2; `fixdep` is
    installed from its path and never imported. Markers are not evaluated: the lock holds
    `dist:fixdep` beside `dist:fix-lib`."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    requirement = 'fixdep; python_version < "3"'
    fixlib = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", _FIXLIB_ALONE, dependencies=(requirement,)
    )
    fixdep = installed_dists.write_library(tmp_path / "src", "fixdep", "fixdep", _FIXDEP)
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install_directory(venv, fixlib)
    requires = installed_dists.installed(venv, "Fix_Lib").requires
    assert requires == [requirement], f"Fix_Lib's METADATA must require {requirement!r}: {requires}"
    project = _helper_project(tmp_path, venv, _helper_pipeline("fixlib"))

    manifest = _helper_ran(project, "value-1")

    assert installed_dists.dist_keys(manifest) == ["dist:fix-lib", "dist:fixdep"], (
        f"the lock: {manifest}"
    )


@pytest.mark.parametrize(
    ("damaged", "text", "refusal"),
    [
        pytest.param(
            "fix_lib-0.1.dist-info/direct_url.json",
            "[]",
            _UNCLASSIFIED_REFUSAL,
            id="reached-unclassified",
        ),
        pytest.param(
            "fixdep-0.1.dist-info/RECORD",
            None,
            _RECORDLESS_DEPENDENCY_REFUSAL,
            id="dependency-without-record",
        ),
        pytest.param(
            "fix_lib-0.1.dist-info/RECORD",
            None,
            _RECORDLESS_REACHED_REFUSAL,
            id="reached-without-record",
        ),
    ],
)
def test_a_broken_install_the_stage_needs_is_refused_and_unsafe_fingerprinting_warns(
    tmp_path: pathlib.Path, damaged: str, text: str | None, refusal: str
) -> None:
    """(h) `Fix_Lib` and its requirement `fixdep` installed from their paths, then one installed
    file damaged (`damaged`, relative to site-packages, overwritten with `text` or deleted): a
    `direct_url.json` of the reached `Fix_Lib` that is not a JSON object; the RECORD of `fixdep`,
    which the closure meets by name; the RECORD of the reached `Fix_Lib`. `pivot repro` refuses
    the stage with `refusal`, which names the distribution. Under `core.unsafe_fingerprinting` it
    warns with the same message and runs the stage, whose result is kept out of the run cache:
    with its lock deleted, the next `pivot repro` runs it again, where a filed result would be
    restored."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = _helper_fixlib_and_fixdep(tmp_path / "src")
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install_directory(venv, fixlib)
    project = _helper_project(tmp_path, venv, _helper_pipeline("fixlib"))
    _helper_damage(venv.site_packages / damaged, text)
    expected = refusal.format(site=venv.site_packages.resolve())

    _helper_assert_refused(installed_dists.run_pivot(project, "repro", "--json"), expected)
    assert not (project.root / _OUTPUT).exists(), f"the refused {_STAGE} must not run"

    (project.root / ".pivot" / "config.yaml").write_text(_UNSAFE_CONFIG)
    _helper_assert_runs_with_warning(project, expected, "unsafe `pivot repro`")
    installed_dists.lock_path(project, _STAGE).unlink()
    _helper_assert_runs_with_warning(project, expected, "with the lock deleted, `pivot repro`")


def test_defects_of_unreached_installs_refuse_nothing_and_a_deleted_listed_file_changes_the_value(
    tmp_path: pathlib.Path,
) -> None:
    """(h) Beside `Fix_Lib`, which the stage calls, `brokenurl` (its `direct_url.json` not a JSON
    object) and `norecord` (its RECORD deleted) are installed from their paths and reached by no
    stage: nothing is refused, and the result is filed in the run cache, so with the lock deleted
    `pivot repro` restores it. Then a file `Fix_Lib`'s RECORD lists, deleted from site-packages,
    makes the stage stale on `dist:fix-lib`, with no refusal."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    sources = [
        installed_dists.write_library(
            src, "Fix_Lib", "fixlib", _FIXLIB_ALONE, files={"fixlib/extra.py": "EXTRA = 1\n"}
        ),
        installed_dists.write_library(src, "brokenurl", "brokenurl", _FIXLIB_ALONE),
        installed_dists.write_library(src, "norecord", "norecord", _FIXLIB_ALONE),
    ]
    installed_dists.install(venv, *(str(source) for source in sources))
    _helper_damage(venv.site_packages / "brokenurl-0.1.dist-info" / "direct_url.json", "[]")
    _helper_damage(venv.site_packages / "norecord-0.1.dist-info" / "RECORD", None)
    project = _helper_project(tmp_path, venv, _helper_pipeline("fixlib"))
    manifest = _helper_ran(project, "value-1")
    assert installed_dists.dist_keys(manifest) == ["dist:fix-lib"], f"the lock: {manifest}"
    installed_dists.lock_path(project, _STAGE).unlink()
    status, shown = installed_dists.repro_status(project, _STAGE)
    assert status == "cached", f"`pivot repro` must restore {_STAGE} from the run cache:\n{shown}"

    (venv.site_packages / "fixlib" / "extra.py").unlink()

    _helper_assert_changes(project, ["dist:fix-lib"])


def test_two_installs_of_one_name_both_reached_give_one_key_that_each_changes(
    tmp_path: pathlib.Path,
) -> None:
    """(i) `Fix_Lib` (module `fixlib_one`) in the venv's site-packages and `fix.lib` (module
    `fixlib_two`), the same PEP 503 name, in a second site-packages directory on the path; the
    stage calls both. The lock holds one key, `dist:fix-lib`, and a byte edited in place in either
    install changes its value and nothing else."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    one = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib_one", 'def value():\n    return "one-1"\n'
    )
    two = installed_dists.write_library(
        tmp_path / "src", "fix.lib", "fixlib_two", 'def value():\n    return "two-1"\n'
    )
    second = tmp_path / "second" / "site-packages"
    installed_dists.install_directory(venv, one)
    installed_dists.install(venv, "--target", str(second), str(two))
    (venv.site_packages / "second.pth").write_text(f"{second}\n")
    found = importlib.metadata.Distribution.discover(
        name="fix-lib", path=[str(venv.site_packages), str(second)]
    )
    assert len(list(found)) == 2, "two distributions named fix-lib must be installed"
    project = _helper_project(tmp_path, venv, _helper_pipeline("fixlib_one", "fixlib_two"))
    before = _helper_ran(project, "one-1two-1")
    assert installed_dists.dist_keys(before) == ["dist:fix-lib"], f"the lock: {before}"

    installed_dists.replace_once(
        venv.site_packages / "fixlib_one" / "__init__.py", '"one-1"', '"one-2"'
    )
    after_one = _helper_ran(project, "one-2two-1")
    installed_dists.replace_once(second / "fixlib_two" / "__init__.py", '"two-1"', '"two-2"')
    after_two = _helper_ran(project, "one-2two-2")

    values = [manifest["dist:fix-lib"] for manifest in (before, after_one, after_two)]
    assert len(set(values)) == 3, f"each edit must change dist:fix-lib: {values}"
    others = [_helper_without(manifest, "dist:fix-lib") for manifest in (after_one, after_two)]
    assert others == [_helper_without(before, "dist:fix-lib")] * 2, (
        f"no other key may change: before {before}, after {after_one}, {after_two}"
    )


def test_a_no_fingerprint_stage_gets_the_key_of_its_output_loaders_distribution(
    tmp_path: pathlib.Path,
) -> None:
    """(j) A `@no_fingerprint` stage whose output loader class comes from `fixloader`, installed
    from its path: the lock holds `dist:fixloader`."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(tmp_path / "src", "fixloader", "fixloader", _LOADER)
    installed_dists.install_directory(venv, source)
    project = _helper_project(tmp_path, venv, _NO_FINGERPRINT_PIPELINE)

    status, shown = installed_dists.repro_status(project, _STAGE)

    assert status == "ran", f"`pivot repro` must run {_STAGE}:\n{shown}"
    written = [path.read_text() for path in (project.root / "data" / "p").iterdir()]
    assert written == ["HELLO"], f"the library's loader must write the output: {written}"
    manifest = installed_dists.lock_manifest(project, _STAGE)
    assert installed_dists.dist_keys(manifest) == ["dist:fixloader"], f"the lock: {manifest}"


def test_a_project_file_shadowing_a_module_reached_through_a_base_class_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """(l) The stage reaches the installed `fixbase` only through its user class's base class, and
    the project root holds `fixbase.py`. A program that imported the installed `fixbase` before it
    started pivot (`python -P -c "import fixbase; ..." repro`) has its stage refused, with the
    worker-import message naming the module and both files. With `fixbase.py` removed, the same
    command runs the stage."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(tmp_path / "src", "fixbase", "fixbase", _FIXBASE)
    installed_dists.install_directory(venv, source)
    project = _helper_project(tmp_path, venv, _BASE_CLASS_PIPELINE)
    shadow = project.root / "fixbase.py"
    shadow.write_text(_SHADOW)
    held = (venv.site_packages / "fixbase" / "__init__.py").resolve()
    expected = (
        f"module 'fixbase' is {held} in this process, but a worker would import "
        + f"{shadow.resolve()}. The stage would run code its fingerprint does not describe "
        + "(Pivot started an interpreter as it starts a worker to confirm this). Fix: rename one "
        + "of the two modules; or, when this program imported its own module of that name "
        + "before it ran Pivot, run Pivot in a new process (the `pivot` command)."
    )

    _helper_assert_refused(installed_dists.run_pivot_holding(project, "fixbase", "repro"), expected)
    assert not (project.root / _OUTPUT).exists(), f"the refused {_STAGE} must not run"

    shadow.unlink()
    ran = installed_dists.run_pivot_holding(project, "fixbase", "repro")

    shown = installed_dists.shown(ran)
    assert ran.returncode == 0, f"`pivot repro` must run {_STAGE}:\n{shown}"
    written = (project.root / _OUTPUT).read_text()
    assert written == "installed", f"{_STAGE} must run the installed fixbase, not {written!r}"


# In process: the module's functions on plain data and on real installs --------------------------

_FORM = installed_distributions.InstallForm
_SITE = "/home/user/venv/lib/python3.13/site-packages"
_PIVOT_DIRS = (str(installed_dists.pivot_package_dir()),)

# `direct_url.json` (PEP 610) as uv and pip write it for each install form, in their own key order
# and spacing (uv's for a local wheel holds no hash).
_UV_DIRECTORY_URL = b'{"url":"file:///home/user/src/fixlib","dir_info":{}}'
_PIP_DIRECTORY_URL = b'{"dir_info": {}, "url": "file:///home/user/src/fixlib"}'
_UV_EDITABLE_URL = b'{"url":"file:///home/user/src/fixlib","dir_info":{"editable":true}}'
_UV_GIT_URL = (
    b'{"url":"file:///home/user/src/fixgit","vcs_info":{"vcs":"git",'
    b'"commit_id":"6964203a125157369c715bd7ba6b130944b8329e",'
    b'"requested_revision":"6964203a125157369c715bd7ba6b130944b8329e"}}'
)
_UV_WHEEL_URL = (
    b'{"url":"file:///home/user/wheels/fixwheel-0.1-py3-none-any.whl","archive_info":{}}'
)
_PIP_WHEEL_HASH = b"935cd5a593c2ac5031e455ed1aa1cdbcd43b2bb26275a8c361e59578b4203e3c"
_PIP_WHEEL_URL = (
    b'{"archive_info": {"hash": "sha256='
    + _PIP_WHEEL_HASH
    + b'", "hashes": {"sha256": "'
    + _PIP_WHEEL_HASH
    + b'"}}, "url": "file:///home/user/wheels/fixwheel-0.1-py3-none-any.whl"}'
)

# METADATA as setuptools writes it, its `Requires-Dist` headers with an extra, a marker and a name
# that is not normalised.
_SETUPTOOLS_METADATA = (
    b"Metadata-Version: 2.4\nName: Fix_Lib\nVersion: 0.1\n"
    b'Requires-Dist: fixdep[cli]>=1.0; python_version < "3"\n'
    b"Requires-Dist: Other.Name__x\n"
)
# A header continued over lines, one of them only white space, as loky's `License` header is,
# before the `Requires-Dist` headers, and a description after the headers that spells one more.
_CONTINUED_METADATA = (
    b"Metadata-Version: 2.4\nName: fix-lib\nVersion: 0.1\nLicense: BSD 3-Clause License\n"
    b"        \n        Copyright (c) 2017.\n        All rights reserved.\n"
    b"Requires-Dist: cloudpickle\nRequires-Dist: fixdep ; extra == 'test'\n"
    b"\nRequires-Dist: in-the-description\n"
)
_CRLF_METADATA = (
    b"Metadata-Version: 2.1\r\nName: fix-lib\r\nRequires-Dist: fixdep (>=1.0)\r\n"
    b"\r\nRequires-Dist: in-the-description\r\n"
)
# `requires.txt` as setuptools 80 writes it into an `.egg-info` directory for `dependencies =
# ["fixdep>=1.0", "Other.Name__x", 'legacy; python_version < "3"']` and the extras `cli =
# ["click"]` and `win = ['pywin32; sys_platform == "win32"']`: a section for a marker, one for each
# extra, and one for an extra with a marker.
_SETUPTOOLS_REQUIRES_TXT = (
    b'fixdep>=1.0\nOther.Name__x\n\n[:python_version < "3"]\nlegacy\n\n[cli]\nclick\n\n[win]\n\n'
    b'[win:sys_platform == "win32"]\npywin32\n'
)

# RECORD as uv writes it, with a console script, and as pip writes it: CR LF line ends, and
# bytecode with no hash.
_UV_RECORD = (
    b"../../../bin/fix-lib,sha256=v-5vszzj0FJZ29t3YnJ4yzhoyL3Na1GiXDBymD6G_K8,331\n"
    b"fix_lib-0.1.dist-info/RECORD,,\n"
    b"fixlib/__init__.py,sha256=Crrh4K5yghbuRJk8Wjp1X4scOH2Uf8TE9yyrDkqEIUs,6\n"
)
_PIP_RECORD = (
    b"fix_lib-0.1.dist-info/INSTALLER,sha256=zuuue4knoyJ-UwPPXg8fezS7VCrXJQrAP7zeNuwvFQg,4\r\n"
    b"fix_lib-0.1.dist-info/RECORD,,\r\n"
    b"fixlib/__init__.py,sha256=Crrh4K5yghbuRJk8Wjp1X4scOH2Uf8TE9yyrDkqEIUs,6\r\n"
    b"fixlib/__pycache__/__init__.cpython-313.pyc,,\r\n"
)

# The refusal `decide` gives the distribution `_helper_installed` describes.
_DECIDE_REFUSAL = (
    f"Distribution 'fix-lib' ({_SITE}/fix_lib-0.1.dist-info), which {{how}}, {{reason}}. Pivot "
    "cannot fingerprint the installed code the stage runs, so the stage could skip a run after "
    "that code changes. Fix: reinstall 'fix-lib'."
)
_REACHES = "this stage reaches"
_DEPENDED_ON = "a distribution this stage reaches depends on"
_FIXLIB_RECORD = installed_distributions.Record(("fixlib/__init__.py",), "")
_NO_RECORD = installed_distributions.Record(None, "has no RECORD file")
_NO_METADATA = installed_distributions.Requirements(frozenset(), "has no METADATA file")
_NO_REQUIREMENT = installed_distributions.Requirements(frozenset(), "")

# A stage that imports the installed `fixlib` only in its body, which no fingerprint runs.
_BODY_IMPORT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def make() -> str:
    import fixlib

    return fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""


def _helper_installed(
    form: installed_distributions.InstallForm, reason: str = ""
) -> installed_distributions.Installed:
    """`fix-lib` in `_SITE`, installed in `form` (`reason`: why it is unclassified)."""
    return installed_distributions.Installed(
        dist_info=f"{_SITE}/fix_lib-0.1.dist-info",
        site_dir=_SITE,
        name="fix-lib",
        classification=installed_distributions.Classification(form, reason),
    )


def _helper_origin(path: pathlib.Path) -> tuple[str, str]:
    """The origin `import_origin.of_module` gives a module loaded from the file at `path`."""
    return ("file", os.path.realpath(path))


def _helper_value(site: pathlib.Path, files: collections.abc.Mapping[str, str | None]) -> str:
    """The value of the key of one install in `site` whose hashed files are `files`: each name in
    the hash with the file's path in `site`, read from disk, or None for a file that cannot be
    read (`content_digest`, `key_value`)."""
    return installed_distributions.key_value(
        [
            installed_distributions.content_digest(
                (
                    name,
                    None if path is None else xxhash.xxh64((site / path).read_bytes()).hexdigest(),
                )
                for name, path in files.items()
            )
        ]
    )


def _helper_delete(path: pathlib.Path) -> None:
    path.unlink()


def _helper_replace_with_directory(path: pathlib.Path) -> None:
    path.unlink()
    path.mkdir()


def _helper_replace_with_fifo(path: pathlib.Path) -> None:
    path.unlink()
    os.mkfifo(path)


def _helper_replace_with_failing_read(path: pathlib.Path) -> None:
    """Make `path` a regular file whose read fails with EIO: this process's `/proc/self/mem`,
    read from address 0, which no process maps."""
    path.unlink()
    path.symlink_to("/proc/self/mem")


def _helper_remove_read_permission(path: pathlib.Path) -> None:
    path.chmod(0)


def _helper_egg_info_project(
    venv: installed_dists.Venv,
    root: pathlib.Path,
    name: str,
    requirement: str,
    *,
    without_requires_dist: bool,
) -> pathlib.Path:
    """Write at `root/<name>` the setuptools project of distribution `name` (package `name`), which
    requires `requirement`, and build it, which leaves `<name>.egg-info` beside the package with
    a PKG-INFO, which names `requirement` in a `Requires-Dist` header, and a `requires.txt`, which
    names it too. `without_requires_dist`: the header is removed from PKG-INFO, which leaves the
    requirement in `requires.txt` alone, where `importlib.metadata` reads it then. Returns the
    project's directory, a source tree to put on the path."""
    source = installed_dists.write_library(
        root, name, name, _FIXLIB_ALONE, dependencies=(requirement,)
    )
    installed_dists.build_wheel(venv, source, root / f"{name}-wheels")
    egg_info = source / f"{name}.egg-info"
    pkg_info = egg_info / "PKG-INFO"
    if without_requires_dist:
        lines = pkg_info.read_text().splitlines(keepends=True)
        pkg_info.write_text(
            "".join(line for line in lines if not line.startswith("Requires-Dist:"))
        )
    headers = importlib.metadata.Distribution.at(egg_info).metadata.get_all("Requires-Dist")
    requires = importlib.metadata.Distribution.at(egg_info).requires
    assert (headers, requires) == (
        None if without_requires_dist else [requirement],
        [requirement],
    ), (
        f"{egg_info} must name {requirement!r} in requires.txt and, unless the header is removed, "
        + f"in PKG-INFO: headers {headers}, requires {requires}"
    )
    return source


@pytest.mark.parametrize(
    ("direct_url", "form", "reason"),
    [
        pytest.param(None, _FORM.INDEX, "", id="none"),
        pytest.param(_UV_DIRECTORY_URL, _FORM.LOCAL_DIRECTORY, "", id="uv-directory"),
        pytest.param(_PIP_DIRECTORY_URL, _FORM.LOCAL_DIRECTORY, "", id="pip-directory"),
        pytest.param(
            _UV_DIRECTORY_URL.replace(b"{}", b'{"editable":false}'),
            _FORM.LOCAL_DIRECTORY,
            "",
            id="editable-false",
        ),
        pytest.param(_UV_EDITABLE_URL, _FORM.EDITABLE, "", id="editable-true"),
        pytest.param(_UV_GIT_URL, _FORM.VCS, "", id="uv-git"),
        pytest.param(_UV_WHEEL_URL, _FORM.LOCAL_ARCHIVE, "", id="uv-local-wheel"),
        pytest.param(_PIP_WHEEL_URL, _FORM.LOCAL_ARCHIVE, "", id="pip-local-wheel"),
        pytest.param(
            _PIP_WHEEL_URL.replace(b"file:///home/user/wheels", b"https://files.example.org"),
            _FORM.REMOTE_ARCHIVE,
            "",
            id="https-archive",
        ),
        pytest.param(
            b"[]", _FORM.UNCLASSIFIED, "its direct_url.json is not a JSON object", id="array"
        ),
        pytest.param(
            _UV_DIRECTORY_URL[:-1],
            _FORM.UNCLASSIFIED,
            "its direct_url.json is not JSON (Expecting ',' delimiter: line 1 column 52 (char 51))",
            id="truncated",
        ),
        pytest.param(
            _UV_DIRECTORY_URL.replace(b"fixlib", b"caf\xe9"),
            _FORM.UNCLASSIFIED,
            "its direct_url.json is not JSON ('utf-8' codec can't decode byte 0xe9 in position "
            + "33: invalid continuation byte)",
            id="not-utf-8",
        ),
        pytest.param(
            b'{"url":"file:///home/user/src/fixlib"}',
            _FORM.UNCLASSIFIED,
            "its direct_url.json names 0 of vcs_info, dir_info and archive_info, where PEP 610 "
            + "requires exactly one",
            id="no-form",
        ),
        pytest.param(
            _UV_GIT_URL.replace(b'"vcs_info"', b'"dir_info":{},"vcs_info"'),
            _FORM.UNCLASSIFIED,
            "its direct_url.json names 2 of vcs_info, dir_info and archive_info, where PEP 610 "
            + "requires exactly one",
            id="two-forms",
        ),
        pytest.param(
            _UV_DIRECTORY_URL.replace(b"{}", b"true"),
            _FORM.UNCLASSIFIED,
            "its direct_url.json's dir_info is not a JSON object",
            id="dir-info-not-an-object",
        ),
        pytest.param(
            b'{"archive_info":{}}',
            _FORM.UNCLASSIFIED,
            "its direct_url.json has archive_info and no url",
            id="archive-without-url",
        ),
        pytest.param(
            _UV_WHEEL_URL.replace(b"file:///home/user/wheels", b"https://[::1"),
            _FORM.UNCLASSIFIED,
            "its direct_url.json's url cannot be parsed (Invalid IPv6 URL)",
            id="unparseable-url",
        ),
        pytest.param(
            b"[" * 100_000 + b"]" * 100_000,
            _FORM.UNCLASSIFIED,
            "its direct_url.json is nested too deeply to parse (maximum recursion depth exceeded "
            + "while decoding a JSON array from a unicode string)",
            id="nested-deeper-than-the-parser-recurses",
        ),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", f"{_SITE}/direct_url.json"),
            _FORM.UNCLASSIFIED,
            "its direct_url.json cannot be read ([Errno 13] Permission denied: "
            + f"'{_SITE}/direct_url.json')",
            id="unreadable",
        ),
    ],
)
def test_classify_gives_each_direct_url_json_its_install_form(
    direct_url: bytes | OSError | None, form: installed_distributions.InstallForm, reason: str
) -> None:
    """Each form uv and pip write gets its form: no file is the index form, `dir_info` is a local
    directory unless `editable` is true, `vcs_info` is version control, and `archive_info` is a
    local archive for a `file:` URL and a remote one for another. A file that cannot be read, is
    not JSON or not a JSON object, names no form or two, has a `dir_info` that is not an object,
    an archive with no URL or a URL that cannot be parsed is unclassified, with that reason."""
    classification = installed_distributions.classify(direct_url)

    assert classification == installed_distributions.Classification(form, reason), classification


@pytest.mark.parametrize(
    ("metadata", "names", "problem"),
    [
        pytest.param(
            _SETUPTOOLS_METADATA, {"fixdep", "other-name-x"}, "", id="extras-markers-unnormalised"
        ),
        pytest.param(
            _CONTINUED_METADATA, {"cloudpickle", "fixdep"}, "", id="after-a-continued-header"
        ),
        pytest.param(_CRLF_METADATA, {"fixdep"}, "", id="crlf"),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fixdep\nVersion: 0.1\n", set[str](), "", id="none"
        ),
        pytest.param(
            b"Name: fix-lib\nVersion: 0.1\nRequires-Dist: fixdep\n",
            set[str](),
            "has a METADATA file with no Metadata-Version header",
            id="no-metadata-version",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fix-lib\nnot a header\nRequires-Dist: fixdep\n",
            set[str](),
            "has a METADATA file whose headers Pivot cannot read "
            + "([MissingHeaderBodySeparatorDefect()])",
            id="header-defect",
        ),
        pytest.param(
            b'Metadata-Version: 2.4\nName: fix-lib\nRequires-Dist: ; python_version < "3"\n',
            set[str](),
            "has a Requires-Dist header that names no distribution ('; python_version < \"3\"')",
            id="no-name",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fix-lib\nAuthor: Jos\xe9 Doe\n",
            set[str](),
            "has a METADATA file that is not UTF-8 ('utf-8' codec can't decode byte 0xe9 in "
            + "position 47: invalid continuation byte)",
            id="not-utf-8",
        ),
        pytest.param(None, set[str](), "has no METADATA file", id="no-file"),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", f"{_SITE}/METADATA"),
            set[str](),
            "has a METADATA file Pivot cannot read ([Errno 13] Permission denied: "
            + f"'{_SITE}/METADATA')",
            id="unreadable",
        ),
    ],
)
def test_requirement_names_reads_the_leading_name_of_each_requires_dist_header(
    metadata: bytes | OSError | None, names: set[str], problem: str
) -> None:
    """Each `Requires-Dist` header gives its leading name, PEP 503 normalised, with no extra or
    marker evaluated, after a continued header and with CR LF line ends; a description after the
    headers is not read. A file that cannot be read, is not UTF-8, has a header defect, no
    `Metadata-Version` or a `Requires-Dist` header that names nothing gives no names and that
    problem."""
    requirements = installed_distributions.requirement_names(metadata)

    assert requirements == installed_distributions.Requirements(frozenset(names), problem), (
        requirements
    )


@pytest.mark.parametrize(
    ("requires", "names", "problem"),
    [
        pytest.param(
            _SETUPTOOLS_REQUIRES_TXT,
            {"fixdep", "other-name-x", "click", "legacy", "pywin32"},
            "",
            id="setuptools-sections",
        ),
        pytest.param(
            b"# pinned by hand\r\n\r\n   fixdep (>=1.0)  \r\n", {"fixdep"}, "", id="comment-crlf"
        ),
        pytest.param(b"", set[str](), "", id="empty"),
        pytest.param(None, set[str](), "", id="no-file"),
        pytest.param(
            b"fixdep\n[cli\nclick\n",
            set[str](),
            "has a requires.txt line that names no distribution ('[cli')",
            id="unclosed-section-header",
        ),
        pytest.param(
            b"fixdep\ncaf\xe9\n",
            set[str](),
            "has a requires.txt file that is not UTF-8 ('utf-8' codec can't decode byte 0xe9 in "
            + "position 10: invalid continuation byte)",
            id="not-utf-8",
        ),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", f"{_SITE}/requires.txt"),
            set[str](),
            "has a requires.txt file Pivot cannot read ([Errno 13] Permission denied: "
            + f"'{_SITE}/requires.txt')",
            id="unreadable",
        ),
    ],
)
def test_requires_txt_names_reads_the_leading_name_of_each_requirement_line(
    requires: bytes | OSError | None, names: set[str], problem: str
) -> None:
    """Each requirement line of an `.egg-info` directory's `requires.txt` gives its leading name,
    PEP 503 normalised, those under an `[extra]`, `[:marker]` or `[extra:marker]` section header
    too, with no extra or marker evaluated; a blank line, a comment and a section header give
    none, whatever the line ends. No file, or an empty one, requires nothing. A file that cannot
    be read, is not UTF-8 or has a line that names nothing gives no names and that problem."""
    requirements = installed_distributions.requires_txt_names(requires)

    assert requirements == installed_distributions.Requirements(frozenset(names), problem), (
        requirements
    )


@pytest.mark.parametrize(
    ("record", "paths", "problem"),
    [
        pytest.param(
            _UV_RECORD,
            ("../../../bin/fix-lib", "fix_lib-0.1.dist-info/RECORD", "fixlib/__init__.py"),
            "",
            id="uv",
        ),
        pytest.param(
            _PIP_RECORD,
            (
                "fix_lib-0.1.dist-info/INSTALLER",
                "fix_lib-0.1.dist-info/RECORD",
                "fixlib/__init__.py",
                "fixlib/__pycache__/__init__.cpython-313.pyc",
            ),
            "",
            id="pip-crlf",
        ),
        pytest.param(
            b'"fixlib/a,b.py",sha256=abc,3\n\n,sha256=abc,3\nfixlib/__init__.py,,\n',
            ("fixlib/a,b.py", "fixlib/__init__.py"),
            "",
            id="quoted-comma-and-empty-rows",
        ),
        pytest.param(
            b"fixlib/caf\xe9.py,,\n",
            None,
            "has a RECORD file Pivot cannot read ('utf-8' codec can't decode byte 0xe9 in "
            + "position 10: invalid continuation byte)",
            id="not-utf-8",
        ),
        pytest.param(
            b"a" * 131073,
            None,
            "has a RECORD file Pivot cannot read (field larger than field limit (131072))",
            id="field-over-the-csv-limit",
        ),
        pytest.param(None, None, "has no RECORD file", id="no-file"),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", f"{_SITE}/RECORD"),
            None,
            f"has a RECORD file Pivot cannot read ([Errno 13] Permission denied: '{_SITE}/RECORD')",
            id="unreadable",
        ),
    ],
)
def test_record_paths_lists_the_first_field_of_each_row(
    record: bytes | OSError | None, paths: tuple[str, ...] | None, problem: str
) -> None:
    """Each row gives its first field, a quoted one with its comma, whatever the line ends; an
    empty line or an empty first field gives none. A file that cannot be read, is not UTF-8 or
    that `csv` cannot read gives no paths and that problem."""
    found = installed_distributions.record_paths(record)

    assert found == installed_distributions.Record(paths, problem), found


def test_hashed_files_keep_each_installed_file_once_but_the_installers_and_bytecode() -> None:
    """Of a RECORD's paths, the key hashes each relative one that stays in the site directory,
    once, under its normalised name, and the distribution's own `entry_points.txt` under one name
    whatever its version: not an absolute path, a path that leaves the directory, the console
    script, the other files of a `.dist-info` directory, another distribution's or a nested
    `.dist-info` directory, nor bytecode. A directory entry is kept, to hash as a file that cannot
    be read."""
    paths = [
        "../../../bin/fix-lib",
        "/usr/lib/python3.13/site-packages/fixlib/__init__.py",
        "fixlib/../../outside.py",
        "fixlib/..",
        "./fixlib/__init__.py",
        "fixlib/__init__.py",
        "fixlib/__init__.py",
        "fixlib/",
        "fixlib/data/table.csv",
        "fixlib/__pycache__/__init__.cpython-313.pyc",
        "fixlib/legacy.pyc",
        "fixlib/__pycache__/notes.txt",
        "fix_lib-0.1.dist-info/entry_points.txt",
        "fix_lib-0.1.dist-info/METADATA",
        "fix_lib-0.1.dist-info/RECORD",
        "other-1.0.dist-info/entry_points.txt",
        "fixlib/_vendor/inner-2.0.dist-info/METADATA",
        "fixlib/_vendor/INNER-2.0.DIST-INFO/entry_points.txt",
        "fixlib/_vendor/inner.py",
    ]

    files = installed_distributions.hashed_files(paths, "fix_lib-0.1.dist-info")

    assert files == [
        (".dist-info/entry_points.txt", "fix_lib-0.1.dist-info/entry_points.txt"),
        ("fixlib", "fixlib"),
        ("fixlib/__init__.py", "fixlib/__init__.py"),
        ("fixlib/_vendor/inner.py", "fixlib/_vendor/inner.py"),
        ("fixlib/data/table.csv", "fixlib/data/table.csv"),
    ], files


def test_content_digest_and_key_value_hash_json_arrays_in_name_order() -> None:
    """A content digest is the xxh64 of the JSON array of `[name, digest]` pairs in name order,
    null for a file that cannot be read, which no digest equals; a key's value is the xxh64 of the
    JSON array of its installs' digests, sorted. JSON keeps a name apart from its digest."""
    first = xxhash.xxh64(b"A = 1\n").hexdigest()
    second = xxhash.xxh64(b"B = 1\n").hexdigest()

    digest = installed_distributions.content_digest(
        [("fixlib/b.py", second), ("fixlib/a.py", first)]
    )
    missing = installed_distributions.content_digest([("fixlib/a.py", None)])
    others = {
        installed_distributions.content_digest([("fixlib/a.py", text)])
        for text in ("null", "", first)
    }
    value = installed_distributions.key_value([second, first])

    stream = f'[["fixlib/a.py","{first}"],["fixlib/b.py","{second}"]]'
    assert digest == xxhash.xxh64(stream.encode()).hexdigest(), digest
    assert missing == xxhash.xxh64(b'[["fixlib/a.py",null]]').hexdigest(), missing
    assert missing not in others and len(others) == 3, (missing, others)
    assert installed_distributions.content_digest([("ab", "c")]) != (
        installed_distributions.content_digest([("a", "bc")])
    ), "a name and its digest must stay apart"
    ordered = sorted([first, second])
    assert value == xxhash.xxh64(f'["{ordered[0]}","{ordered[1]}"]'.encode()).hexdigest(), value


def test_candidates_are_the_seeds_and_every_distribution_that_requires_one() -> None:
    """Each distribution that requires a seed, directly or through others, is a candidate, through
    a cycle too; one that only a candidate requires is not."""
    requires = {
        "fix-lib": {"fix-egg", "requests"},
        "fix-egg": {"fixdep"},
        "fixdep": set[str](),
        "requests": {"urllib3"},
        "urllib3": set[str](),
        "cycle-a": {"cycle-b"},
        "cycle-b": {"cycle-a", "fixdep"},
    }

    found = installed_distributions.candidates(["fixdep"], requires)

    assert found == {"fixdep", "fix-egg", "fix-lib", "cycle-a", "cycle-b"}, found


def test_reach_meets_each_requirement_once_and_follows_none_of_a_stop() -> None:
    """The closure meets each distribution once, in the order met, so a cycle ends; a stop
    (pivot's own distribution) is met, but its requirements are not followed."""
    requires = {
        "fix-lib": ("fix-egg",),
        "fix-egg": ("fixdep", "fix-lib"),
        "fixdep": (),
        "pivot": ("pivot-only",),
        "pivot-only": (),
    }

    assert installed_distributions.reach(["fix-lib"], requires, ()) == [
        "fix-lib",
        "fix-egg",
        "fixdep",
    ]
    assert installed_distributions.reach(["fix-lib"], requires, ("fix-egg",)) == [
        "fix-lib",
        "fix-egg",
    ]
    assert installed_distributions.reach(["pivot", "fix-lib"], requires, ("pivot",)) == [
        "pivot",
        "fix-lib",
        "fix-egg",
        "fixdep",
    ]


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        pytest.param(("file", f"{_SITE}/fixlib/core.py"), {"fix-lib", "fix-lib-extra"}, id="file"),
        pytest.param(
            ("file", f"{_SITE}/company/__init__.py"),
            {"company-core", "company-vendor"},
            id="package-init",
        ),
        pytest.param(
            ("file", f"{_SITE}/company/__init__.cpython-313-x86_64-linux-gnu.so"),
            {"company-core", "company-vendor"},
            id="extension-package-init",
        ),
        pytest.param(("file", f"{_SITE}/stray.py"), set[str](), id="file-no-record-lists"),
        pytest.param(
            ("namespace", f"{_SITE}/nsco", "/home/user/other/nsco"),
            {"nsco-local", "nsco-other"},
            id="namespace",
        ),
        pytest.param(("built-in",), set[str](), id="built-in"),
        pytest.param(("frozen",), set[str](), id="frozen"),
        pytest.param(("no file",), set[str](), id="no-file"),
        pytest.param(("unknown", "ImportError: boom"), set[str](), id="unknown"),
        pytest.param(None, set[str](), id="none"),
    ],
)
def test_owners_maps_an_origin_to_the_candidates_by_file(
    origin: tuple[str, ...] | None, expected: set[str]
) -> None:
    """A file maps to every candidate whose RECORD lists it, and a package's `__init__` of any
    suffix also to every candidate with a file under its directory; a namespace package maps to
    every candidate with a file under one of its directories; any other origin maps to none."""
    file_owners = {
        f"{_SITE}/fixlib/core.py": {"fix-lib", "fix-lib-extra"},
        f"{_SITE}/company/__init__.py": {"company-core"},
        f"{_SITE}/company/vendor.py": {"company-vendor"},
        f"{_SITE}/nsco/local/__init__.py": {"nsco-local"},
        "/home/user/other/nsco/other/__init__.py": {"nsco-other"},
    }
    directory_owners = {
        f"{_SITE}/fixlib": {"fix-lib", "fix-lib-extra"},
        f"{_SITE}/company": {"company-core", "company-vendor"},
        f"{_SITE}/nsco": {"nsco-local"},
        f"{_SITE}/nsco/local": {"nsco-local"},
        "/home/user/other/nsco": {"nsco-other"},
        "/home/user/other/nsco/other": {"nsco-other"},
    }

    found = installed_distributions.owners(origin, file_owners, directory_owners)

    assert found == expected, found


@pytest.mark.parametrize(
    ("installed", "record", "requirements", "pivot_own", "owner", "decision"),
    [
        pytest.param(
            _helper_installed(_FORM.UNCLASSIFIED, "its direct_url.json is not a JSON object"),
            _NO_RECORD,
            _NO_METADATA,
            True,
            True,
            installed_distributions.Decision(keyed=False, refusal=""),
            id="pivots-own-first",
        ),
        pytest.param(
            _helper_installed(_FORM.UNCLASSIFIED, "its direct_url.json is not a JSON object"),
            _FIXLIB_RECORD,
            _NO_METADATA,
            False,
            True,
            installed_distributions.Decision(
                keyed=False,
                refusal=_DECIDE_REFUSAL.format(
                    how=_REACHES,
                    reason="is installed in a form Pivot cannot classify: its direct_url.json "
                    + "is not a JSON object",
                ),
            ),
            id="unclassified-before-requirements",
        ),
        pytest.param(
            _helper_installed(_FORM.VCS),
            _NO_RECORD,
            _NO_METADATA,
            False,
            False,
            installed_distributions.Decision(
                keyed=False,
                refusal=_DECIDE_REFUSAL.format(
                    how=_DEPENDED_ON,
                    reason="is installed from version control (vcs_info) and has no RECORD file, "
                    + "so Pivot cannot read its files",
                ),
            ),
            id="keyed-without-record-before-requirements",
        ),
        pytest.param(
            _helper_installed(_FORM.LOCAL_ARCHIVE),
            _FIXLIB_RECORD,
            _NO_METADATA,
            False,
            True,
            installed_distributions.Decision(
                keyed=True,
                refusal=_DECIDE_REFUSAL.format(
                    how=_REACHES,
                    reason="has no METADATA file, so Pivot cannot tell what it depends on",
                ),
            ),
            id="keyed-requirements-unreadable",
        ),
        pytest.param(
            _helper_installed(_FORM.INDEX),
            _NO_RECORD,
            _NO_METADATA,
            False,
            False,
            installed_distributions.Decision(
                keyed=False,
                refusal=_DECIDE_REFUSAL.format(
                    how=_DEPENDED_ON,
                    reason="has no METADATA file, so Pivot cannot tell what it depends on",
                ),
            ),
            id="index-requirements-unreadable",
        ),
        pytest.param(
            _helper_installed(_FORM.LOCAL_DIRECTORY),
            _FIXLIB_RECORD,
            installed_distributions.Requirements(frozenset({"fixdep"}), ""),
            False,
            True,
            installed_distributions.Decision(keyed=True, refusal=""),
            id="keyed",
        ),
        pytest.param(
            _helper_installed(_FORM.INDEX),
            _NO_RECORD,
            _NO_REQUIREMENT,
            False,
            False,
            installed_distributions.Decision(keyed=False, refusal=""),
            id="index-without-record",
        ),
        pytest.param(
            _helper_installed(_FORM.EDITABLE),
            _FIXLIB_RECORD,
            _NO_REQUIREMENT,
            False,
            True,
            installed_distributions.Decision(keyed=False, refusal=""),
            id="editable",
        ),
        pytest.param(
            _helper_installed(_FORM.REMOTE_ARCHIVE),
            _FIXLIB_RECORD,
            _NO_REQUIREMENT,
            False,
            True,
            installed_distributions.Decision(keyed=False, refusal=""),
            id="remote-archive",
        ),
    ],
)
def test_decide_gives_a_key_and_refusals_in_their_order(
    installed: installed_distributions.Installed,
    record: installed_distributions.Record,
    requirements: installed_distributions.Requirements,
    pivot_own: bool,
    owner: bool,
    decision: installed_distributions.Decision,
) -> None:
    """Pivot's own distribution gives nothing, whatever its defects; then an unclassified one is
    refused, then one of a keyed form with no RECORD, then one whose requirements cannot be read,
    each refusal naming the distribution, its `.dist-info` directory, how the stage meets it, the
    reason and the fix. A distribution of a keyed form with a RECORD gets a key, refused or not;
    an index, editable or remote-archive one gets none."""
    found = installed_distributions.decide(
        installed, record, requirements, pivot_own=pivot_own, owner=owner
    )

    assert found == decision, found


def test_distribution_keys_hash_each_reached_git_path_and_local_archive_install_from_disk(
    tmp_path: pathlib.Path,
) -> None:
    """Over real installs in one venv: `Fix_Lib` from its path (with a console script, requiring
    `fixdep`), `fixdep` from its path, `fixgit` from git and `fixwheel` from a wheel file. The
    stage reaches `fixlib`, `fixgit` and `fixwheel`. Each of the four gets a key, `fixdep` through
    the requirement, whose value hashes the install's module files and, for `Fix_Lib`, its
    `entry_points.txt`, read from disk, and not its console script or the installer's files."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixlib = installed_dists.write_library(
        src,
        "Fix_Lib",
        "fixlib",
        _FIXLIB,
        dependencies=("fixdep",),
        scripts={"fix-lib": "fixlib:main"},
    )
    fixgit = installed_dists.write_library(src, "fixgit", "fixgit", _FIXLIB_ALONE)
    fixwheel = installed_dists.write_library(src, "fixwheel", "fixwheel", _FIXLIB_ALONE)
    wheels = tmp_path / "wheels"
    installed_dists.install_directory(
        venv, installed_dists.write_library(src, "fixdep", "fixdep", _FIXDEP)
    )
    installed_dists.install_directory(venv, fixlib)
    installed_dists.install(
        venv, installed_dists.init_git_requirement(fixgit, tmp_path / "home", "fixgit")
    )
    installed_dists.build_wheel(venv, fixwheel, wheels)
    (wheel,) = wheels.glob("*.whl")
    installed_dists.install(venv, str(wheel))
    site = venv.site_packages
    origins = {
        module: _helper_origin(site / module / "__init__.py")
        for module in ("fixlib", "fixgit", "fixwheel")
    }

    found = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)

    entry_points = "fix_lib-0.1.dist-info/entry_points.txt"
    assert found == installed_distributions.DistributionKeys(
        {
            "dist:fix-lib": _helper_value(
                site,
                {
                    ".dist-info/entry_points.txt": entry_points,
                    "fixlib/__init__.py": "fixlib/__init__.py",
                },
            ),
            "dist:fixdep": _helper_value(site, {"fixdep/__init__.py": "fixdep/__init__.py"}),
            "dist:fixgit": _helper_value(site, {"fixgit/__init__.py": "fixgit/__init__.py"}),
            "dist:fixwheel": _helper_value(site, {"fixwheel/__init__.py": "fixwheel/__init__.py"}),
        },
        [],
    ), found


def test_distribution_keys_read_the_files_once_per_path_and_pivot_dirs_until_reset(
    tmp_path: pathlib.Path,
) -> None:
    """The installed files are read once per process for one path and one set of pivot's
    directories: after an edit in place, a second call gives the value of the first. `reset`
    forgets them, and so do another path and other pivot directories; a call with no module
    reads nothing and keeps them."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(tmp_path / "src", "Fix_Lib", "fixlib", _FIXLIB_ALONE)
    installed_dists.install_directory(venv, source)
    site = venv.site_packages
    module = site / "fixlib" / "__init__.py"
    origins = {"fixlib": _helper_origin(module)}
    path = [str(site)]
    longer_path = [*path, str(tmp_path / "missing")]
    other_dirs = [*_PIVOT_DIRS, str(tmp_path / "other")]
    files = {"fixlib/__init__.py": "fixlib/__init__.py"}
    values = list[str]()

    first = installed_distributions.distribution_keys(origins, path, _PIVOT_DIRS)
    values.append(_helper_value(site, files))
    installed_dists.replace_once(module, '"value-1"', '"value-2"')
    again = installed_distributions.distribution_keys(origins, path, _PIVOT_DIRS)
    installed_distributions.reset()
    after_reset = installed_distributions.distribution_keys(origins, path, _PIVOT_DIRS)
    values.append(_helper_value(site, files))
    installed_dists.replace_once(module, '"value-2"', '"value-3"')
    other_path = installed_distributions.distribution_keys(origins, longer_path, _PIVOT_DIRS)
    values.append(_helper_value(site, files))
    installed_dists.replace_once(module, '"value-3"', '"value-4"')
    other_pivot = installed_distributions.distribution_keys(origins, longer_path, other_dirs)
    values.append(_helper_value(site, files))
    no_module = installed_distributions.distribution_keys({}, path, _PIVOT_DIRS)
    installed_dists.replace_once(module, '"value-4"', '"value-5"')
    kept = installed_distributions.distribution_keys(origins, longer_path, other_dirs)

    assert len(set(values)) == 4, f"each edit must change the file's value: {values}"
    keyed = [
        installed_distributions.DistributionKeys({"dist:fix-lib": value}, []) for value in values
    ]
    found = [first, again, after_reset, other_path, other_pivot, no_module, kept]
    assert found == [
        keyed[0],
        keyed[0],
        keyed[1],
        keyed[2],
        keyed[3],
        installed_distributions.DistributionKeys({}, []),
        keyed[3],
    ], found


def test_an_index_install_whose_metadata_cannot_be_read_refuses_nothing_with_no_keyed_install(
    tmp_path: pathlib.Path,
) -> None:
    """With only an index install and an editable one on the path, no requirement is read: the
    reached `idxlib`, whose METADATA is not UTF-8, refuses nothing, and neither it nor the reached
    editable `fixedit` gets a key."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    idxlib = installed_dists.write_library(src, "idxlib", "idxlib", _FIXLIB_ALONE)
    fixedit = installed_dists.write_library(src, "fixedit", "fixedit", _FIXLIB_ALONE)
    wheels = tmp_path / "wheels"
    installed_dists.build_wheel(venv, idxlib, wheels)
    installed_dists.install_from_wheels(venv, wheels, "idxlib")
    installed_dists.install_directory(venv, fixedit, editable=True)
    site = venv.site_packages
    (site / "idxlib-0.1.dist-info" / "METADATA").write_bytes(
        b"Metadata-Version: 2.4\nName: idx\xe9\n"
    )
    origins = {
        "fixedit": _helper_origin(fixedit / "fixedit" / "__init__.py"),
        "idxlib": _helper_origin(site / "idxlib" / "__init__.py"),
    }

    found = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)

    assert found == installed_distributions.DistributionKeys({}, []), found


def test_the_scan_skips_what_is_no_metadata_directory_and_reads_an_egg_info_directory(
    tmp_path: pathlib.Path,
) -> None:
    """The path holds a missing directory, an entry with a NUL character, the venv's
    site-packages twice, and the source tree of `Fix_Egg`, whose setuptools build left
    `Fix_Egg.egg-info` there with a PKG-INFO and no METADATA. Site-packages also holds a file
    named `*.dist-info` and a symbolic link named `*.dist-info` that points to itself. The reached
    `Fix_Lib` requires `Fix_Egg`, which requires `fixdep`: the keys are those of `Fix_Lib` and
    `fixdep`, and nothing is refused."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixlib = installed_dists.write_library(
        src, "Fix_Lib", "fixlib", _FIXLIB_ALONE, dependencies=("fix-egg",)
    )
    egg = installed_dists.write_library(
        src, "Fix_Egg", "fixegg", _FIXLIB_ALONE, dependencies=("fixdep",)
    )
    installed_dists.install_directory(
        venv, installed_dists.write_library(src, "fixdep", "fixdep", _FIXDEP)
    )
    installed_dists.install_directory(venv, fixlib)
    installed_dists.build_wheel(venv, egg, tmp_path / "wheels")
    (egg_info,) = egg.glob("*.egg-info")
    held = sorted(child.name for child in egg_info.iterdir())
    assert "PKG-INFO" in held and "METADATA" not in held, f"{egg_info} holds {held}"
    site = venv.site_packages
    (site / "notes-1.0.dist-info").write_text("a file, not a metadata directory\n")
    (site / "loop-1.0.dist-info").symlink_to("loop-1.0.dist-info")
    path = [str(tmp_path / "missing"), f"{tmp_path}/nul\0entry", str(site), str(site), str(egg)]

    found = installed_distributions.distribution_keys(
        {"fixlib": _helper_origin(site / "fixlib" / "__init__.py")}, path, _PIVOT_DIRS
    )

    assert found == installed_distributions.DistributionKeys(
        {
            "dist:fix-lib": _helper_value(site, {"fixlib/__init__.py": "fixlib/__init__.py"}),
            "dist:fixdep": _helper_value(site, {"fixdep/__init__.py": "fixdep/__init__.py"}),
        },
        [],
    ), found


def test_a_reached_file_no_record_lists_beside_a_recordless_install_is_refused_per_module(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib`, installed from its path with packages `fixlib`, `company.vendor` and the
    namespace portion `nsco.part`, has lost its RECORD; `company-core`, installed as from an
    index, holds `company/__init__.py`, beside which Python left bytecode; `fixother` is installed
    from its path into a second directory on the path. Each reached module with a file that no
    RECORD in site-packages lists is refused once, naming the first such file: `fixlib`'s and
    `fixlib.helpers`' own files, and under the package `company` and the namespace `nsco` the
    file `Fix_Lib` put there, not the bytecode. `fixother`, outside that directory, gets its key;
    a built-in module and a module nothing finds give nothing."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixlib = installed_dists.write_setuptools_project(
        src / "fixlib",
        "Fix_Lib",
        "0.1",
        packages=["fixlib", "company.vendor", "nsco.part"],
        files={
            "fixlib/__init__.py": _FIXLIB_ALONE,
            "fixlib/helpers.py": "HELP = 1\n",
            "company/vendor/__init__.py": "VENDOR = 1\n",
            "nsco/part/__init__.py": "PART = 1\n",
        },
    )
    company = installed_dists.write_setuptools_project(
        src / "company",
        "company-core",
        "0.1",
        packages=["company"],
        files={"company/__init__.py": "CORE = 1\n"},
    )
    fixother = installed_dists.write_library(src, "fixother", "fixother", _FIXLIB_ALONE)
    wheels = tmp_path / "wheels"
    second = tmp_path / "second"
    installed_dists.install_directory(venv, fixlib)
    installed_dists.build_wheel(venv, company, wheels)
    installed_dists.install_from_wheels(venv, wheels, "company-core")
    installed_dists.install(venv, "--target", str(second), str(fixother))
    site = venv.site_packages
    _helper_damage(site / "fix_lib-0.1.dist-info" / "RECORD", None)
    core = site / "company" / "__init__.py"
    py_compile.compile(str(core), doraise=True)
    py_compile.compile(str(core), cfile=str(site / "company" / "legacy.pyc"), doraise=True)
    origins = {
        "company": _helper_origin(core),
        "fixlib": _helper_origin(site / "fixlib" / "__init__.py"),
        "fixlib.helpers": _helper_origin(site / "fixlib" / "helpers.py"),
        "fixother": _helper_origin(second / "fixother" / "__init__.py"),
        "missing": None,
        "nsco": ("namespace", os.path.realpath(site / "nsco")),
        "sys": ("built-in",),
    }

    found = installed_distributions.distribution_keys(
        origins, [str(site), str(second)], _PIVOT_DIRS
    )

    resolved = site.resolve()
    refused = [
        ("company", "company/vendor/__init__.py"),
        ("fixlib", "fixlib/__init__.py"),
        ("fixlib.helpers", "fixlib/helpers.py"),
        ("nsco", "nsco/part/__init__.py"),
    ]
    assert found == installed_distributions.DistributionKeys(
        {"dist:fixother": _helper_value(second, {"fixother/__init__.py": "fixother/__init__.py"})},
        sorted(
            _UNLISTED_REFUSAL.format(module=module, file=file, site=resolved)
            for module, file in refused
        ),
    ), found


def test_reached_and_required_installs_pivot_cannot_read_are_refused(
    tmp_path: pathlib.Path,
) -> None:
    """The stage reaches `Fix_Lib` and `fixurl`, installed from their paths; `Fix_Lib` requires
    `fixdep`, whose RECORD is deleted, and `fixurl`'s `direct_url.json` is a directory. `fixurl`
    is refused as unclassified, with the error of reading the file, and `fixdep` for its missing
    RECORD; `Fix_Lib` gets its key."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = _helper_fixlib_and_fixdep(tmp_path / "src")
    fixurl = installed_dists.write_library(tmp_path / "src", "fixurl", "fixurl", _FIXLIB_ALONE)
    installed_dists.install(venv, str(fixlib), str(fixdep), str(fixurl))
    site = venv.site_packages
    _helper_damage(site / "fixdep-0.1.dist-info" / "RECORD", None)
    url = site / "fixurl-0.1.dist-info" / "direct_url.json"
    _helper_replace_with_directory(url)
    origins = {
        module: _helper_origin(site / module / "__init__.py") for module in ("fixlib", "fixurl")
    }

    found = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)

    resolved = site.resolve()
    unclassified = (
        f"Distribution 'fixurl' ({resolved}/fixurl-0.1.dist-info), which this stage reaches, is "
        f"installed in a form Pivot cannot classify: its direct_url.json cannot be read ([Errno "
        f"21] Is a directory: '{resolved}/fixurl-0.1.dist-info/direct_url.json'). Pivot cannot "
        "fingerprint the installed code the stage runs, so the stage could skip a run after that "
        "code changes. Fix: reinstall 'fixurl'."
    )
    assert found == installed_distributions.DistributionKeys(
        {"dist:fix-lib": _helper_value(site, {"fixlib/__init__.py": "fixlib/__init__.py"})},
        sorted([_RECORDLESS_DEPENDENCY_REFUSAL.format(site=resolved), unclassified]),
    ), found


def test_pivots_own_distribution_gets_no_key_and_its_requirements_are_not_followed(
    tmp_path: pathlib.Path,
) -> None:
    """A distribution `pivot`, installed from its path, requires `fixdep`, installed from its
    path. With its package directory among pivot's directories, the reached `pivot` gives no key
    and its requirement is not followed; with other pivot directories it is a library like any
    other, and both get keys."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    own = installed_dists.write_library(
        src, "pivot", "pivot", "VERSION = 1\n", dependencies=("fixdep",)
    )
    installed_dists.install_directory(
        venv, installed_dists.write_library(src, "fixdep", "fixdep", _FIXDEP)
    )
    installed_dists.install_directory(venv, own)
    site = venv.site_packages
    origins = {"pivot": _helper_origin(site / "pivot" / "__init__.py")}

    as_pivot = installed_distributions.distribution_keys(
        origins, [str(site)], [os.path.realpath(site / "pivot")]
    )
    as_library = installed_distributions.distribution_keys(
        origins, [str(site)], [str(tmp_path / "other")]
    )

    assert as_pivot == installed_distributions.DistributionKeys({}, []), as_pivot
    assert as_library == installed_distributions.DistributionKeys(
        {
            "dist:fixdep": _helper_value(site, {"fixdep/__init__.py": "fixdep/__init__.py"}),
            "dist:pivot": _helper_value(site, {"pivot/__init__.py": "pivot/__init__.py"}),
        },
        [],
    ), as_library


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(_helper_delete, id="missing"),
        pytest.param(_helper_replace_with_directory, id="directory"),
        pytest.param(_helper_replace_with_fifo, id="fifo"),
        pytest.param(_helper_replace_with_failing_read, id="read-fails"),
        # Permissions stop a user who is not root from reading a file; root reads it.
        *(
            [pytest.param(_helper_remove_read_permission, id="no-read-permission")]
            if os.geteuid() != 0
            else []
        ),
    ],
)
def test_a_listed_file_that_cannot_be_read_hashes_as_null_and_refuses_nothing(
    tmp_path: pathlib.Path, damage: collections.abc.Callable[[pathlib.Path], None]
) -> None:
    """`fixlib/extra.py`, which `Fix_Lib`'s RECORD lists, deleted, made a directory or a FIFO
    (which an open without O_NONBLOCK would wait on), a file whose read fails, or a file the user
    may not read: it is hashed as null, so the value changes, and nothing is refused."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(
        tmp_path / "src",
        "Fix_Lib",
        "fixlib",
        _FIXLIB_ALONE,
        files={"fixlib/extra.py": "EXTRA = 1\n"},
    )
    installed_dists.install_directory(venv, source)
    site = venv.site_packages
    origins = {"fixlib": _helper_origin(site / "fixlib" / "__init__.py")}
    files: dict[str, str | None] = {
        "fixlib/__init__.py": "fixlib/__init__.py",
        "fixlib/extra.py": "fixlib/extra.py",
    }
    intact = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)
    intact_value = _helper_value(site, files)
    damage(site / "fixlib" / "extra.py")
    installed_distributions.reset()

    damaged = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)

    null_value = _helper_value(site, {**files, "fixlib/extra.py": None})
    assert intact_value != null_value, "a file hashed as null must change the value"
    assert [intact, damaged] == [
        installed_distributions.DistributionKeys({"dist:fix-lib": intact_value}, []),
        installed_distributions.DistributionKeys({"dist:fix-lib": null_value}, []),
    ], [intact, damaged]


def test_a_record_path_holding_a_nul_character_hashes_as_null_and_refuses_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib`'s RECORD gains a row whose path holds a NUL character, as a RECORD that lost
    bytes to a crash can, and `fixdep`, beside it, has lost its RECORD, so every RECORD of the
    directory is read for the reached `fixlib`. The path names no file on disk: it owns none, it
    is hashed as null, and nothing is refused."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixlib = installed_dists.write_library(src, "Fix_Lib", "fixlib", _FIXLIB_ALONE)
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", _FIXDEP)
    installed_dists.install(venv, str(fixlib), str(fixdep))
    site = venv.site_packages
    record = site / "fix_lib-0.1.dist-info" / "RECORD"
    record.write_bytes(record.read_bytes() + b"fixlib/\x00.py,,\n")
    _helper_damage(site / "fixdep-0.1.dist-info" / "RECORD", None)
    origins = {"fixlib": _helper_origin(site / "fixlib" / "__init__.py")}

    found = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)

    files = {"fixlib/\x00.py": None, "fixlib/__init__.py": "fixlib/__init__.py"}
    assert found == installed_distributions.DistributionKeys(
        {"dist:fix-lib": _helper_value(site, files)}, []
    ), found


def test_a_refusal_of_a_reached_install_fails_the_fingerprint_through_the_registry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Fix_Lib`, whose `direct_url.json` is not a JSON object, is installed in a site directory
    on this process's path, and the stage imports its `fixlib` in its body, which no fingerprint
    runs. The registry's fingerprint of the stage fails with the refusal, which names the
    distribution, and `fixlib` was not imported."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(tmp_path / "src", "Fix_Lib", "fixlib", _FIXLIB_ALONE)
    installed_dists.install_directory(venv, source)
    _helper_damage(venv.site_packages / "fix_lib-0.1.dist-info" / "direct_url.json", "[]")
    monkeypatch.syspath_prepend(str(venv.site_packages))
    root = tmp_path / "project"
    for directory in (".git", ".pivot"):
        (root / directory).mkdir(parents=True)
    (root / "pipeline.py").write_text(_BODY_IMPORT_PIPELINE)
    monkeypatch.setattr(project, "_project_root_cache", root)
    pipeline = discovery.discover_pipeline(project_root=root)
    assert pipeline is not None, "the project must define a pipeline"

    with pytest.raises(exceptions.PivotError) as raised:
        pipeline.ensure_fingerprint(_STAGE)

    refusal = _UNCLASSIFIED_REFUSAL.format(site=venv.site_packages.resolve())
    assert str(raised.value) == f"Stage '{_STAGE}': fingerprinting failed: {refusal}", raised.value
    assert "fixlib" not in sys.modules, "fingerprinting must not import the stage's library"


def test_a_venv_inside_the_project_root_beside_the_projects_egg_info_gets_its_keys(
    tmp_path: pathlib.Path,
) -> None:
    """The project root, first on the path, is a setuptools project `myproj` that requires
    `fixdep`; its build left `myproj.egg-info` there, with no RECORD. The venv lies inside the root
    (`<root>/.venv`), with `Fix_Lib` and its requirement `fixdep` installed from their paths. The
    reached `fixlib` lies in the venv's site-packages, the deepest entry of the path that holds
    its file, which holds no distribution without a RECORD: the keys are those of `Fix_Lib` and
    `fixdep`, and nothing is refused. A module whose file lies outside every entry of the path
    gives nothing."""
    root = installed_dists.write_setuptools_project(
        tmp_path / "project",
        "myproj",
        "0.1",
        packages=["myproj"],
        files={"myproj/__init__.py": "VALUE = 1\n"},
        dependencies=["fixdep"],
    )
    venv = installed_dists.make_venv(root / ".venv")
    fixlib, fixdep = _helper_fixlib_and_fixdep(tmp_path / "src")
    installed_dists.install(venv, str(fixlib), str(fixdep))
    installed_dists.build_wheel(venv, root, tmp_path / "wheels")
    requires = importlib.metadata.Distribution.at(root / "myproj.egg-info").requires
    assert requires == ["fixdep"], f"the project's egg-info must require fixdep: {requires}"
    site = venv.site_packages
    origins = {
        "elsewhere": _helper_origin(tmp_path / "elsewhere.py"),
        "fixlib": _helper_origin(site / "fixlib" / "__init__.py"),
    }

    found = installed_distributions.distribution_keys(origins, [str(root), str(site)], _PIVOT_DIRS)

    assert found == installed_distributions.DistributionKeys(
        {
            "dist:fix-lib": _helper_value(site, {"fixlib/__init__.py": "fixlib/__init__.py"}),
            "dist:fixdep": _helper_value(site, {"fixdep/__init__.py": "fixdep/__init__.py"}),
        },
        [],
    ), found


@pytest.mark.parametrize(
    "without_requires_dist",
    [
        pytest.param(False, id="requirement-in-pkg-info"),
        pytest.param(True, id="requirement-in-requires-txt-alone"),
    ],
)
def test_a_reached_module_of_an_index_egg_info_with_no_record_that_requires_a_local_install_is_refused(
    tmp_path: pathlib.Path, without_requires_dist: bool
) -> None:
    """The source tree of `mid`, on the path after the venv's site-packages, holds the package
    `mid` and the `mid.egg-info` its setuptools build left, with no RECORD and no
    `direct_url.json` (the package-index form), which requires `fixdep`, installed from its path,
    in a `Requires-Dist` header of its PKG-INFO, or in its `requires.txt` alone. The stage reaches
    `mid`, whose file no RECORD lists: `mid` owns no file, so Pivot cannot tell that the stage
    runs it and depends on `fixdep`, and refuses the stage, naming the module, its file and
    `mid`."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    installed_dists.install_directory(
        venv, installed_dists.write_library(src, "fixdep", "fixdep", _FIXDEP)
    )
    mid = _helper_egg_info_project(
        venv, src, "mid", "fixdep", without_requires_dist=without_requires_dist
    )
    origins = {"mid": _helper_origin(mid / "mid" / "__init__.py")}

    found = installed_distributions.distribution_keys(
        origins, [str(venv.site_packages), str(mid)], _PIVOT_DIRS
    )

    tree = mid.resolve()
    refusal = (
        f"Module 'mid' ({tree}/mid/__init__.py) lies in {tree}, where distribution 'mid' "
        f"({tree}/mid.egg-info), installed with no direct_url.json, as from a package index, has "
        "no RECORD file, and no RECORD file there lists that file. Pivot cannot tell whether the "
        "stage runs that distribution's code, so the stage could skip a run after it changes. "
        "Fix: reinstall 'mid'."
    )
    assert found == installed_distributions.DistributionKeys({}, [refusal]), found


def test_an_egg_info_whose_metadata_names_no_requirement_is_followed_through_its_requires_txt(
    tmp_path: pathlib.Path,
) -> None:
    """The reached `Fix_Lib`, installed from its path, requires `mid`, whose source tree, on the
    path, holds the `mid.egg-info` its setuptools build left; its PKG-INFO has no `Requires-Dist`
    header, and its `requires.txt` names `fixdep`, installed from its path. The closure meets
    `mid` by name and follows its `requires.txt`, as `importlib.metadata` does: the keys are those
    of `Fix_Lib` and `fixdep`, and nothing is refused."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixlib = installed_dists.write_library(
        src, "Fix_Lib", "fixlib", _FIXLIB_ALONE, dependencies=("mid",)
    )
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", _FIXDEP)
    installed_dists.install(venv, str(fixlib), str(fixdep))
    mid = _helper_egg_info_project(venv, src, "mid", "fixdep", without_requires_dist=True)
    site = venv.site_packages

    found = installed_distributions.distribution_keys(
        {"fixlib": _helper_origin(site / "fixlib" / "__init__.py")},
        [str(site), str(mid)],
        _PIVOT_DIRS,
    )

    assert found == installed_distributions.DistributionKeys(
        {
            "dist:fix-lib": _helper_value(site, {"fixlib/__init__.py": "fixlib/__init__.py"}),
            "dist:fixdep": _helper_value(site, {"fixdep/__init__.py": "fixdep/__init__.py"}),
        },
        [],
    ), found


def test_a_record_path_resolves_to_what_realpath_gives(tmp_path: pathlib.Path) -> None:
    """A RECORD path is resolved by walking only the names below the nearest directory already
    resolved, and gives what `os.path.realpath` gives, the first time and again from the memo:
    under a package directory that is a symbolic link, for a file that is a symbolic link, by an
    absolute target or by a relative one through `..`, a dangling link, a link to itself, a missing
    directory, a name under a regular file, a `..`, a `.`, a trailing slash, doubled slashes, a
    console script's `../../../bin` path and the root."""
    package = tmp_path / "elsewhere" / "fixlib"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    target = tmp_path / "other" / "core.py"
    target.parent.mkdir()
    target.write_text("")
    (package / "core.py").symlink_to(target)
    (package / "relative.py").symlink_to(pathlib.Path("..", "..", "other", "core.py"))
    site = tmp_path / "site"
    site.mkdir()
    (site / "fixlib").symlink_to(package, target_is_directory=True)
    (site / "plain.py").write_text("")
    (site / "dangling.py").symlink_to(tmp_path / "missing.py")
    (site / "loop.py").symlink_to("loop.py")
    paths = [
        f"{site}/fixlib/__init__.py",
        f"{site}/fixlib/core.py",
        f"{site}/fixlib/relative.py",
        f"{site}/dangling.py",
        f"{site}/loop.py",
        f"{site}/missing/module.py",
        f"{site}/plain.py/under-a-file",
        f"{site}/fixlib/../plain.py",
        f"{site}/./plain.py",
        f"{site}/fixlib/",
        f"{site}//fixlib//core.py",
        f"{site}/../../../bin/fix-lib",
        "/",
    ]
    real_dirs = dict[str, str]()

    first = [installed_distributions._real_path(path, real_dirs) for path in paths]
    again = [installed_distributions._real_path(path, real_dirs) for path in paths]

    expected = [os.path.realpath(path) for path in paths]
    assert expected[1:3] == [str(target.resolve())] * 2, f"the links must resolve: {expected}"
    assert first == expected, f"the first walk must give realpath's paths:\n{first}\n{expected}"
    assert again == expected, f"the memo must give realpath's paths:\n{again}\n{expected}"


def test_no_origin_is_found_when_no_install_on_the_path_can_give_a_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: pytest_mock.MockerFixture
) -> None:
    """With only `idxlib`, installed as from a package index, on the worker's path,
    `fingerprint.add_distribution_keys` finds the origin of none of the stage's `byname:`
    modules, neither of `json`, which this process holds, nor of `fixlib`, which it does not, and
    adds nothing. With `Fix_Lib` installed from its path there too, it finds the origin of each
    once and adds `dist:fix-lib`."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    wheels = tmp_path / "wheels"
    installed_dists.build_wheel(
        venv, installed_dists.write_library(src, "idxlib", "idxlib", _FIXLIB_ALONE), wheels
    )
    installed_dists.install_from_wheels(venv, wheels, "idxlib")
    site = venv.site_packages
    monkeypatch.setattr(sys, "path", [str(site)])
    of_module = mocker.spy(import_origin, "of_module")
    worker_origin = mocker.spy(loaded_source, "worker_origin")
    stage = {"self:make": "1", "byname:json": "", "byname:fixlib": ""}
    without_keyed = dict(stage)

    fingerprint.add_distribution_keys(without_keyed)
    calls_without_keyed = [of_module.call_args_list[:], worker_origin.call_args_list[:]]
    installed_dists.install_directory(
        venv, installed_dists.write_library(src, "Fix_Lib", "fixlib", _FIXLIB_ALONE)
    )
    installed_distributions.reset()
    with_keyed = dict(stage)
    fingerprint.add_distribution_keys(with_keyed)

    assert without_keyed == stage, f"nothing may be added: {without_keyed}"
    assert calls_without_keyed == [[], []], f"no origin may be found: {calls_without_keyed}"
    value = _helper_value(site, {"fixlib/__init__.py": "fixlib/__init__.py"})
    assert with_keyed == {**stage, "dist:fix-lib": value}, with_keyed
    calls = [of_module.call_args_list, worker_origin.call_args_list]
    assert calls == [[mocker.call(sys.modules["json"])], [mocker.call("fixlib")]], calls
