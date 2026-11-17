"""The `dist:` key of an installed distribution, checked through the `pivot` CLI: seven cases.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`) and its
own pivot project, whose one stage `p/make` calls an installed library. The project is set up
the same way each time: `git init -q`, `pivot init`, the library installed, `pivot repro` runs the
stage and `pivot status --json` reports it `cached`. Then the library changes, and the test reads
what pivot reports, in this order: "stale" means `pivot status --json` reports the stage `stale`
and `pivot status --explain --json` names exactly one code change, the named key; "marks nothing"
means `cached` and no code change. The stage's lock is read after those reports.

A library installed from a local path or from git, and the library it depends on, each get a key
that hashes its installed files; a package-index install, an editable install and pivot itself
get none.
"""

import importlib.metadata
import pathlib
import shutil

from tests import installed_dists

_STAGE = "p/make"
_OUTPUT = pathlib.Path("data/p/make.txt")

_FIXDEP = 'def base():\n    return "base-1"\n'
_FIXLIB = """\
import fixdep


def value():
    return "value-1:" + fixdep.base()


def main():
    print(value())
"""
_IDXLIB = 'def value():\n    return "idx-{version}"\n'

# The project of tests 1 to 6: its stage writes `<module>.value()` and names nothing else of
# the library, nor anything of the library's dependency.
_PIPELINE = """\
import {module}

from pivot import Pipeline, stage


@stage
def make() -> str:
    return {module}.value()


with Pipeline("p") as pipeline:
    make()
"""

# The project of test 7: its stage reads one module of pivot's by name and no other library.
_PIVOT_PIPELINE = """\
from pivot import Pipeline, names, stage


@stage
def make() -> str:
    return names.display_stage_name("p/make", "p")


with Pipeline("p") as pipeline:
    make()
"""

# Pivot builds with uv_build, which the suite cannot fetch offline: setuptools builds the copy.
_PIVOT_PYPROJECT = """\
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "pivot"
version = "{version}"

[tool.setuptools.packages.find]
include = ["pivot", "pivot.*"]

[tool.setuptools.package-data]
"*" = ["*"]
"""


def _helper_fixtures(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Write under `root` the setuptools projects of `Fix_Lib` (module `fixlib`, a console
    script, requires `fixdep`) and `fixdep` (module `fixdep`); returns their directories."""
    fixlib = installed_dists.write_setuptools_project(
        root / "fixlib",
        "Fix_Lib",
        "0.1",
        packages=["fixlib"],
        files={"fixlib/__init__.py": _FIXLIB},
        dependencies=["fixdep"],
        scripts={"fix-lib": "fixlib:main"},
    )
    fixdep = installed_dists.write_setuptools_project(
        root / "fixdep", "fixdep", "0.1", packages=["fixdep"], files={"fixdep/__init__.py": _FIXDEP}
    )
    return fixlib, fixdep


def _helper_project(
    tmp_path: pathlib.Path, venv: installed_dists.Venv, pipeline: str
) -> installed_dists.Project:
    return installed_dists.make_project(tmp_path / "project", tmp_path / "home", venv, pipeline)


def _helper_ran_then_cached(project: installed_dists.Project, output: str) -> None:
    """`pivot repro` runs the stage, which writes `output`; `pivot status --json` then reports it
    `cached`."""
    status, shown = installed_dists.repro_status(project, _STAGE)
    assert status == "ran", f"the first `pivot repro` must run {_STAGE}:\n{shown}"
    written = (project.root / _OUTPUT).read_text()
    assert written == output, f"{_STAGE} must write {output!r}, not {written!r}"
    status, shown = installed_dists.stage_status(project, _STAGE)
    assert status == "cached", f"`pivot status` must report {_STAGE} cached after a run:\n{shown}"


def _helper_assert_stale_on(project: installed_dists.Project, key: str) -> None:
    status, shown = installed_dists.stage_status(project, _STAGE)
    assert status == "stale", f"`pivot status` must report {_STAGE} stale on {key}:\n{shown}"
    keys, shown = installed_dists.code_changes(project, _STAGE)
    assert keys == [key], f"`pivot status --explain` must name exactly {key} changed:\n{shown}"


def _helper_assert_marks_nothing(project: installed_dists.Project) -> None:
    status, shown = installed_dists.stage_status(project, _STAGE)
    assert status == "cached", f"`pivot status` must report {_STAGE} cached:\n{shown}"
    keys, shown = installed_dists.code_changes(project, _STAGE)
    assert keys == [], f"`pivot status --explain` must name no code change:\n{shown}"


def test_a_changed_byte_in_a_function_the_stage_calls_reruns_the_stage(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` and `fixdep` installed from their paths; one byte of the literal `value()`
    returns changes in the source and `Fix_Lib` is reinstalled from its path. The stage is stale
    on `dist:fix-lib`, reruns and writes the new value, and its lock records a new value."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = _helper_fixtures(tmp_path / "src")
    project = _helper_project(tmp_path, venv, _PIPELINE.format(module="fixlib"))
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install_directory(venv, fixlib)
    _helper_ran_then_cached(project, "value-1:base-1")
    before = installed_dists.lock_manifest(project, _STAGE).get("dist:fix-lib")

    installed_dists.replace_once(fixlib / "fixlib" / "__init__.py", '"value-1:"', '"value-2:"')
    installed_dists.install_directory(venv, fixlib, reinstall=True)

    _helper_assert_stale_on(project, "dist:fix-lib")
    status, shown = installed_dists.repro_status(project, _STAGE)
    assert status == "ran", f"`pivot repro` must rerun {_STAGE}:\n{shown}"
    written = (project.root / _OUTPUT).read_text()
    assert written == "value-2:base-1", f"the rerun must write the new value, not {written!r}"
    status, shown = installed_dists.stage_status(project, _STAGE)
    assert status == "cached", (
        f"`pivot status` must report {_STAGE} cached after the rerun:\n{shown}"
    )
    after = installed_dists.lock_manifest(project, _STAGE)
    assert before is not None and "dist:fix-lib" in after, (
        f"the lock must hold dist:fix-lib before and after the change: before {before!r}, "
        + f"after {after}"
    )
    assert after["dist:fix-lib"] != before, f"dist:fix-lib must change from {before!r}: {after}"


def test_a_reinstall_with_the_same_files_marks_nothing(tmp_path: pathlib.Path) -> None:
    """`Fix_Lib` and `fixdep` installed from their paths; `Fix_Lib` is reinstalled from the same
    tree. Nothing is marked, and the lock holds `dist:fix-lib` with the value it held before."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = _helper_fixtures(tmp_path / "src")
    project = _helper_project(tmp_path, venv, _PIPELINE.format(module="fixlib"))
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install_directory(venv, fixlib)
    _helper_ran_then_cached(project, "value-1:base-1")
    before = installed_dists.lock_manifest(project, _STAGE).get("dist:fix-lib")

    installed_dists.install_directory(venv, fixlib, reinstall=True)

    _helper_assert_marks_nothing(project)
    after = installed_dists.lock_manifest(project, _STAGE)
    assert "dist:fix-lib" in after, f"the lock must hold dist:fix-lib: {after}"
    assert after["dist:fix-lib"] == before, f"dist:fix-lib must stay {before!r}: {after}"


def test_a_git_reinstall_at_another_commit_with_the_same_files_marks_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` installed from commit C1 of a git repository, then from C2, which changes only
    a file its wheel does not hold; `fixdep` from its path. Nothing is marked, and the lock holds
    `dist:fix-lib` with the value it held before."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = _helper_fixtures(tmp_path / "src")
    project = _helper_project(tmp_path, venv, _PIPELINE.format(module="fixlib"))
    notes = fixlib / "NOTES.txt"
    notes.write_text("C1\n")
    installed_dists.git(fixlib, project.home, "init", "-q")
    commit_1 = installed_dists.commit_all(fixlib, project.home, "C1")
    notes.write_text("C2\n")
    commit_2 = installed_dists.commit_all(fixlib, project.home, "C2")
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install(venv, installed_dists.git_requirement("Fix_Lib", fixlib, commit_1))
    _helper_ran_then_cached(project, "value-1:base-1")
    before = installed_dists.lock_manifest(project, _STAGE).get("dist:fix-lib")

    installed_dists.install(
        venv, installed_dists.git_requirement("Fix_Lib", fixlib, commit_2), reinstall=True
    )

    url = installed_dists.direct_url(venv, "Fix_Lib")
    assert url is not None and url["vcs_info"]["commit_id"] == commit_2, (
        f"Fix_Lib must now be installed from C2 {commit_2}: direct_url.json is {url!r}"
    )
    _helper_assert_marks_nothing(project)
    after = installed_dists.lock_manifest(project, _STAGE)
    assert "dist:fix-lib" in after, f"the lock must hold dist:fix-lib: {after}"
    assert after["dist:fix-lib"] == before, f"dist:fix-lib must stay {before!r}: {after}"


def test_a_changed_byte_in_a_dependency_of_the_called_library_reruns_the_stage(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` and `fixdep` installed from their paths; one byte of the literal `fixdep.base()`
    returns changes in the source and `fixdep` is reinstalled. The stage, which names nothing of
    `fixdep`, is stale on `dist:fixdep` alone, reruns and writes the new value."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = _helper_fixtures(tmp_path / "src")
    project = _helper_project(tmp_path, venv, _PIPELINE.format(module="fixlib"))
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install_directory(venv, fixlib)
    _helper_ran_then_cached(project, "value-1:base-1")
    before = installed_dists.lock_manifest(project, _STAGE).get("dist:fixdep")

    installed_dists.replace_once(fixdep / "fixdep" / "__init__.py", '"base-1"', '"base-2"')
    installed_dists.install_directory(venv, fixdep, reinstall=True)

    _helper_assert_stale_on(project, "dist:fixdep")
    status, shown = installed_dists.repro_status(project, _STAGE)
    assert status == "ran", f"`pivot repro` must rerun {_STAGE}:\n{shown}"
    written = (project.root / _OUTPUT).read_text()
    assert written == "value-1:base-2", f"the rerun must write the new value, not {written!r}"
    after = installed_dists.lock_manifest(project, _STAGE)
    assert before is not None and "dist:fixdep" in after, (
        f"the lock must hold dist:fixdep before and after the change: before {before!r}, "
        + f"after {after}"
    )
    assert after["dist:fixdep"] != before, f"dist:fixdep must change from {before!r}: {after}"


def test_a_package_index_bump_behaves_as_before(tmp_path: pathlib.Path) -> None:
    """`idxlib` 1.0 installed from a wheel directory as from a package index, then 1.1, which
    changes the bytes of the function the stage calls. Nothing is marked, as before this key
    existed, and the lock holds no `dist:` key."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    wheels = tmp_path / "wheels"
    for version in ("1.0", "1.1"):
        source = installed_dists.write_setuptools_project(
            tmp_path / "src" / f"idxlib-{version}",
            "idxlib",
            version,
            packages=["idxlib"],
            files={"idxlib/__init__.py": _IDXLIB.format(version=version)},
        )
        installed_dists.build_wheel(venv, source, wheels)
    project = _helper_project(tmp_path, venv, _PIPELINE.format(module="idxlib"))
    installed_dists.install_from_wheels(venv, wheels, "idxlib==1.0")
    _helper_ran_then_cached(project, "idx-1.0")

    installed_dists.install_from_wheels(venv, wheels, "idxlib==1.1")

    version = installed_dists.installed(venv, "idxlib").version
    assert version == "1.1", f"idxlib 1.1 must now be installed, not {version}"
    url = installed_dists.direct_url(venv, "idxlib")
    assert url is None, f"an index install writes no direct_url.json; this one holds {url!r}"
    _helper_assert_marks_nothing(project)
    manifest = installed_dists.lock_manifest(project, _STAGE)
    assert installed_dists.dist_keys(manifest) == [], f"the lock must hold no dist: key: {manifest}"


def test_an_editable_install_adds_no_key(tmp_path: pathlib.Path) -> None:
    """`Fix_Lib` installed editable, `fixdep` from a wheel directory as from a package index.
    `fixlib` is user code: one byte edited in `value()` in the source makes the stage stale on
    `func:fixlib.value` alone, and the lock holds no `dist:` key."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = _helper_fixtures(tmp_path / "src")
    wheels = tmp_path / "wheels"
    project = _helper_project(tmp_path, venv, _PIPELINE.format(module="fixlib"))
    installed_dists.build_wheel(venv, fixdep, wheels)
    installed_dists.install_from_wheels(venv, wheels, "fixdep")
    installed_dists.install_directory(venv, fixlib, editable=True)
    _helper_ran_then_cached(project, "value-1:base-1")

    installed_dists.replace_once(fixlib / "fixlib" / "__init__.py", '"value-1:"', '"value-2:"')

    _helper_assert_stale_on(project, "func:fixlib.value")
    manifest = installed_dists.lock_manifest(project, _STAGE)
    assert installed_dists.dist_keys(manifest) == [], f"the lock must hold no dist: key: {manifest}"


def test_a_reinstall_of_pivot_itself_from_another_commit_changes_no_stage(
    tmp_path: pathlib.Path,
) -> None:
    """A copy of the running pivot's package, installed from commit A of a git repository ahead
    of this environment's pivot, then from commit B, which adds one byte, a comment, to the pivot
    module the stage reads by name. Pivot's own distribution gets no key: nothing is marked, and
    the lock holds no `dist:` key."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    project = _helper_project(tmp_path, venv, _PIVOT_PIPELINE)
    repo = tmp_path / "src" / "pivot"
    shutil.copytree(
        installed_dists.pivot_package_dir(),
        repo / "pivot",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (repo / "pyproject.toml").write_text(
        _PIVOT_PYPROJECT.format(version=importlib.metadata.version("pivot"))
    )
    installed_dists.git(repo, project.home, "init", "-q")
    commit_a = installed_dists.commit_all(repo, project.home, "A")
    names = repo / "pivot" / "names.py"
    names.write_text(names.read_text() + "#")
    commit_b = installed_dists.commit_all(repo, project.home, "B")
    installed_dists.install(venv, installed_dists.git_requirement("pivot", repo, commit_a))
    imported = installed_dists.module_file(project, "pivot")
    expected = (venv.site_packages / "pivot" / "__init__.py").resolve()
    assert imported == expected, (
        f"the venv must import the pivot it installed, {expected}, not {imported}"
    )
    _helper_ran_then_cached(project, "make")

    installed_dists.install(
        venv, installed_dists.git_requirement("pivot", repo, commit_b), reinstall=True
    )

    url = installed_dists.direct_url(venv, "pivot")
    assert url is not None and url["vcs_info"]["commit_id"] == commit_b, (
        f"pivot must now be installed from B {commit_b}: direct_url.json is {url!r}"
    )
    _helper_assert_marks_nothing(project)
    manifest = installed_dists.lock_manifest(project, _STAGE)
    assert installed_dists.dist_keys(manifest) == [], f"the lock must hold no dist: key: {manifest}"
