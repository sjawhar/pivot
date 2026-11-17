"""The decisions behind the `dist:` key of an installed distribution, checked through the `pivot`
CLI on real installs.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`),
installs setuptools fixtures into it the way the decision names (from a path, from git, a local
wheel, as from a package index, editable, by pip), and drives its own pivot project, whose stages
call the installed code. What pivot does is read from its reports and from the stages' locks:
which files the key hashes, which install forms get one, how a stage reaches a distribution and
the distributions it depends on, the refusals of a broken install and their downgrade by unsafe
fingerprinting, and the refusal of a project file that shadows a module the stage reaches.

The in-process tests of the same decisions are in `test_installed_distributions_rules.py`, the
pure functions of `pivot.installed_distributions` on plain data, and in
`test_installed_distributions_snapshot.py`, its `distribution_keys` over real installs.
"""

import importlib.metadata
import json
import pathlib
import subprocess

import pytest
from tests import installed_dists

_UNSAFE_CONFIG = "core:\n  unsafe_fingerprinting: true\n"

# Each refusal of a broken install, and its warning under unsafe fingerprinting, ends with this.
_REFUSAL_FIX = "Fix: reinstall"

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

# The refusal of the reached `Fix_Lib` with no RECORD, `{site}` the venv's resolved site-packages.
_RECORDLESS_REACHED_REFUSAL = installed_dists.UNLISTED_REFUSAL.format(
    module="fixlib", file="fixlib/__init__.py", site="{site}"
)


def _helper_pipeline(*modules: str) -> str:
    """The project whose stage imports each of `modules` and writes their `value()`s joined."""
    return _PIPELINE.format(
        imports="\n".join(f"import {module}" for module in modules),
        expression=" + ".join(f"{module}.value()" for module in modules),
    )


def _helper_flat(text: str) -> str:
    """`text` with each run of white space made one space, as a wrapped message reads."""
    return " ".join(text.split())


def _helper_ran(project: installed_dists.Project, output: str) -> dict[str, str]:
    """`pivot repro` runs the stage, which writes `output` (`assert_runs_and_writes`), and refuses
    and warns of nothing; returns the lock's code manifest."""
    shown = installed_dists.assert_runs_and_writes(project, output)
    assert _REFUSAL_FIX not in shown, f"`pivot repro` must refuse and warn of nothing:\n{shown}"
    return installed_dists.lock_manifest(project, installed_dists.STAGE)


def _helper_assert_changes(project: installed_dists.Project, keys: list[str]) -> None:
    """`pivot status --explain --json` names exactly `keys` changed, and refuses nothing."""
    changed, shown = installed_dists.code_changes(project, installed_dists.STAGE)
    assert changed == keys, f"`pivot status --explain` must name exactly {keys} changed:\n{shown}"
    assert _REFUSAL_FIX not in shown, f"`pivot status` must refuse and warn of nothing:\n{shown}"


def _helper_without(manifest: dict[str, str], key: str) -> dict[str, str]:
    return {name: value for name, value in manifest.items() if name != key}


def _helper_assert_refused(ran: subprocess.CompletedProcess[str], refusal: str) -> None:
    """The pivot command `ran` failed, refusing the stage with `refusal`."""
    shown = installed_dists.shown(ran)
    assert ran.returncode == 1, f"pivot must refuse {installed_dists.STAGE}:\n{shown}"
    assert refusal in _helper_flat(shown), f"pivot must refuse with:\n{refusal}\n{shown}"


def _helper_assert_runs_with_warning(
    project: installed_dists.Project, warning: str, command: str
) -> None:
    """`pivot repro` runs the stage, which writes its value (`assert_runs_and_writes`), not
    restoring it from the run cache, and warns `warning`."""
    shown = installed_dists.assert_runs_and_writes(project, "value-1:base-1", command)
    assert warning in _helper_flat(shown), f"{command} must warn:\n{warning}\n{shown}"


def test_installs_by_uv_and_by_pip_in_two_venvs_give_one_value(tmp_path: pathlib.Path) -> None:
    """(a) `Fix_Lib`, with a console script, installed from its path by uv into one venv and by
    pip into a second venv at another path. Their RECORDs differ: each lists its console script,
    whose first line names its own venv, and pip's lists bytecode. The lock that the first venv's
    pivot writes is current for the second venv's pivot: `dist:fix-lib` has one value."""
    source = installed_dists.write_library(
        tmp_path / "src",
        "Fix_Lib",
        "fixlib",
        installed_dists.FIXLIB_ALONE,
        scripts={"fix-lib": "fixlib:main"},
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
    project = installed_dists.make_project_in(tmp_path, by_uv, _helper_pipeline("fixlib"))
    manifest = _helper_ran(project, "value-1")
    assert installed_dists.dist_keys(manifest) == ["dist:fix-lib"], f"the lock: {manifest}"

    by_pip_project = installed_dists.Project(root=project.root, home=project.home, venv=by_pip)

    status, shown = installed_dists.stage_status(by_pip_project, installed_dists.STAGE)
    assert status == "cached", (
        f"the pip venv's pivot must report {installed_dists.STAGE} cached:\n{shown}"
    )
    _helper_assert_changes(by_pip_project, [])


def test_a_byte_edited_in_place_in_an_installed_file_changes_the_value(
    tmp_path: pathlib.Path,
) -> None:
    """(b) One byte of the literal `value()` returns, edited in place in the installed
    `fixlib/__init__.py` with no reinstall, makes the stage stale on `dist:fix-lib`: the key
    hashes the files read from disk, not the hashes RECORD holds."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
    installed_dists.install_directory(venv, source)
    project = installed_dists.make_project_in(tmp_path, venv, _helper_pipeline("fixlib"))
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
        tmp_path / "src",
        "Fix_Lib",
        "fixlib",
        installed_dists.FIXLIB_ALONE,
        scripts={"fix-lib": "fixlib:main"},
    )
    installed_dists.install_directory(venv, source)
    project = installed_dists.make_project_in(tmp_path, venv, _helper_pipeline("fixlib"))
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
    source = installed_dists.write_library(
        tmp_path / "src", "fixwheel", "fixwheel", installed_dists.FIXLIB_ALONE
    )
    wheels = tmp_path / "wheels"
    installed_dists.build_wheel(venv, source, wheels)
    (wheel,) = wheels.glob("*.whl")
    installed_dists.install(venv, str(wheel))
    url = installed_dists.direct_url(venv, "fixwheel")
    assert url is not None and "archive_info" in url and url["url"].startswith("file:"), (
        f"a wheel file's install must record archive_info with a file: URL: {url!r}"
    )
    project = installed_dists.make_project_in(tmp_path, venv, _helper_pipeline("fixwheel"))
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
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
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
    project = installed_dists.make_project_in(tmp_path, venv, _CLOSURE_PIPELINE)
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
    fixlib, fixdep = installed_dists.write_fixlib_and_fixdep(tmp_path / "src")
    installed_dists.install_directory(venv, fixdep, editable=True)
    installed_dists.install_directory(venv, fixlib)
    url = installed_dists.direct_url(venv, "fixdep")
    assert url is not None and url["dir_info"] == {"editable": True}, (
        f"fixdep must be installed editable: direct_url.json is {url!r}"
    )
    project = installed_dists.make_project_in(tmp_path, venv, _helper_pipeline("fixlib"))

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
    project = installed_dists.make_project_in(
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
        tmp_path / "src",
        "Fix_Lib",
        "fixlib",
        installed_dists.FIXLIB_ALONE,
        dependencies=(requirement,),
    )
    fixdep = installed_dists.write_library(
        tmp_path / "src", "fixdep", "fixdep", installed_dists.FIXDEP
    )
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install_directory(venv, fixlib)
    requires = installed_dists.installed(venv, "Fix_Lib").requires
    assert requires == [requirement], f"Fix_Lib's METADATA must require {requirement!r}: {requires}"
    project = installed_dists.make_project_in(tmp_path, venv, _helper_pipeline("fixlib"))

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
            installed_dists.UNCLASSIFIED_REFUSAL,
            id="reached-unclassified",
        ),
        pytest.param(
            "fixdep-0.1.dist-info/RECORD",
            None,
            installed_dists.RECORDLESS_DEPENDENCY_REFUSAL,
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
    fixlib, fixdep = installed_dists.write_fixlib_and_fixdep(tmp_path / "src")
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install_directory(venv, fixlib)
    project = installed_dists.make_project_in(tmp_path, venv, _helper_pipeline("fixlib"))
    installed_dists.damage(venv.site_packages / damaged, text)
    expected = refusal.format(site=venv.site_packages.resolve())

    _helper_assert_refused(installed_dists.run_pivot(project, "repro", "--json"), expected)
    assert not (project.root / installed_dists.OUTPUT).exists(), (
        f"the refused {installed_dists.STAGE} must not run"
    )

    (project.root / ".pivot" / "config.yaml").write_text(_UNSAFE_CONFIG)
    _helper_assert_runs_with_warning(project, expected, "unsafe `pivot repro`")
    installed_dists.lock_path(project, installed_dists.STAGE).unlink()
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
            src,
            "Fix_Lib",
            "fixlib",
            installed_dists.FIXLIB_ALONE,
            files={"fixlib/extra.py": "EXTRA = 1\n"},
        ),
        installed_dists.write_library(src, "brokenurl", "brokenurl", installed_dists.FIXLIB_ALONE),
        installed_dists.write_library(src, "norecord", "norecord", installed_dists.FIXLIB_ALONE),
    ]
    installed_dists.install(venv, *(str(source) for source in sources))
    installed_dists.damage(venv.site_packages / "brokenurl-0.1.dist-info" / "direct_url.json", "[]")
    installed_dists.damage(venv.site_packages / "norecord-0.1.dist-info" / "RECORD", None)
    project = installed_dists.make_project_in(tmp_path, venv, _helper_pipeline("fixlib"))
    manifest = _helper_ran(project, "value-1")
    assert installed_dists.dist_keys(manifest) == ["dist:fix-lib"], f"the lock: {manifest}"
    installed_dists.lock_path(project, installed_dists.STAGE).unlink()
    status, shown = installed_dists.repro_status(project, installed_dists.STAGE)
    assert status == "cached", (
        f"`pivot repro` must restore {installed_dists.STAGE} from the run cache:\n{shown}"
    )

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
    project = installed_dists.make_project_in(
        tmp_path, venv, _helper_pipeline("fixlib_one", "fixlib_two")
    )
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
    project = installed_dists.make_project_in(tmp_path, venv, _NO_FINGERPRINT_PIPELINE)

    status, shown = installed_dists.repro_status(project, installed_dists.STAGE)

    assert status == "ran", f"`pivot repro` must run {installed_dists.STAGE}:\n{shown}"
    written = [path.read_text() for path in (project.root / "data" / "p").iterdir()]
    assert written == ["HELLO"], f"the library's loader must write the output: {written}"
    manifest = installed_dists.lock_manifest(project, installed_dists.STAGE)
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
    project = installed_dists.make_project_in(tmp_path, venv, _BASE_CLASS_PIPELINE)
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
    assert not (project.root / installed_dists.OUTPUT).exists(), (
        f"the refused {installed_dists.STAGE} must not run"
    )

    shadow.unlink()
    ran = installed_dists.run_pivot_holding(project, "fixbase", "repro")

    shown = installed_dists.shown(ran)
    assert ran.returncode == 0, f"`pivot repro` must run {installed_dists.STAGE}:\n{shown}"
    written = (project.root / installed_dists.OUTPUT).read_text()
    assert written == "installed", (
        f"{installed_dists.STAGE} must run the installed fixbase, not {written!r}"
    )
