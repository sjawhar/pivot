"""Project code never meets a rule made for an installed distribution, checked through the `pivot`
CLI on real installs.

The walk hashes user code, or the user names it by name (`@pivot.fingerprint_by_name`), so no
installed distribution's RECORD, nor its lack of one, decides it: a module whose file is user code,
and the user's part of a namespace package shared with a library (in the project, or in pip's
checkout in `<venv>/src`, as `is_user_code` takes it), give no `dist:` key and no refusal,
whatever metadata lies beside them. A project's own metadata (the `<name>.egg-info` a setuptools
build leaves beside the project's code, with no RECORD and no `direct_url.json`) refuses nothing
in a path entry outside every `site-packages` and `dist-packages` directory.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`), installs
`Fix_Lib` and the namespace portion `company.client` from their paths, and drives its own pivot
project, whose stage calls them and the project's code. The setuptools project `myproj`, which
requires `Fix_Lib`, leaves `myproj.egg-info` beside its code when it is built: `uv pip install -e`
with build isolation, as a user runs it, writes it there through setuptools' `egg_info`, and so
does the wheel build these tests run, offline and with no build isolation (after `uv pip install
-e`, where a test installs the project editable).

The in-process test of the same rule, over uv's cache inside the project root, is in
`test_installed_distributions_snapshot.py`.
"""

import collections.abc
import importlib.metadata
import json
import os
import pathlib
import shutil

import pytest
from tests import helpers, installed_dists

# Each refusal of a broken install, and its warning under unsafe fingerprinting, ends with this.
_REFUSAL_FIX = "Fix: reinstall"

# The project's `@pivot.fingerprint_by_name` helper `helpers.stamp`, and `mymod`, whose `value()`
# the walk hashes and whose `stamp()` is such a helper.
_HELPERS = 'import pivot\n\n\n@pivot.fingerprint_by_name\ndef stamp():\n    return "h1"\n'
_MYMOD = (
    'import pivot\n\n\ndef value():\n    return "mine-1"\n\n\n'
    '@pivot.fingerprint_by_name\ndef stamp():\n    return "h1"\n'
)
# The project's part of the namespace package `company`, and the installed `company.client`.
_LOCAL = 'def value():\n    return "local-1"\n'
_CLIENT = 'def value():\n    return "client-1"\n'

_HELPER_PIPELINE = """\
import fixlib
import helpers

from pivot import Pipeline, stage


@stage
def make() -> str:
    return helpers.stamp() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
_NAMESPACE_PIPELINE = """\
import company.client
import fixlib

from pivot import Pipeline, stage


@stage
def make() -> str:
    return company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""
_PROJECT_CODE_PIPELINE = """\
import company.client
import fixlib
import mymod

from pivot import Pipeline, stage


@stage
def make() -> str:
    return mymod.value() + ":" + mymod.stamp() + ":" + company.client.value() + ":" + fixlib.value()


with Pipeline("p") as pipeline:
    make()
"""

# What `mymod` and the project's part of `company` hold, each path relative to the directory
# that holds them, and what the stage that calls them writes.
_PROJECT_CODE = {"mymod/__init__.py": _MYMOD, "company/local.py": _LOCAL}
_PROJECT_CODE_OUTPUT = "mine-1:h1:client-1:value-1"


def _helper_install_libraries(venv: installed_dists.Venv, src: pathlib.Path) -> None:
    """Install from their paths, under `src`, `Fix_Lib` (module `fixlib`,
    `installed_dists.FIXLIB_ALONE`) and `company-client`, whose one package is the namespace
    portion `company.client`."""
    fixlib = installed_dists.write_library(src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE)
    client = installed_dists.write_setuptools_project(
        src / "company-client",
        "company-client",
        "0.1",
        packages=["company.client"],
        files={"company/client/__init__.py": _CLIENT},
    )
    installed_dists.install(venv, str(fixlib), str(client))


def _helper_write_myproj(
    directory: pathlib.Path,
    scratch: pathlib.Path,
    packages: collections.abc.Sequence[str],
    files: collections.abc.Mapping[str, str],
) -> None:
    """Write into the existing directory `directory` the setuptools project `myproj` 0.1, which
    requires `Fix_Lib` and holds `files` in its import `packages`: written at `scratch` by
    `installed_dists.write_setuptools_project`, then copied."""
    source = installed_dists.write_setuptools_project(
        scratch, "myproj", "0.1", packages=packages, files=files, dependencies=["Fix_Lib"]
    )
    shutil.copytree(source, directory, dirs_exist_ok=True)


def _helper_build(
    venv: installed_dists.Venv, source: pathlib.Path, wheels: pathlib.Path, holder: pathlib.Path
) -> pathlib.Path:
    """Build a wheel of `myproj` at `source` into `wheels` (`installed_dists.build_wheel`), which
    leaves `myproj.egg-info` in `holder`, the directory that holds its packages, then delete the
    `build` directory the build leaves in `source`, as `uv pip install -e` leaves none. The test
    fails unless that `.egg-info` requires `Fix_Lib` and has no RECORD and no `direct_url.json`.
    Returns it."""
    installed_dists.build_wheel(venv, source, wheels)
    shutil.rmtree(source / "build")
    egg_info = holder / "myproj.egg-info"
    held = sorted(child.name for child in egg_info.iterdir())
    requires = importlib.metadata.Distribution.at(egg_info).requires
    assert requires == ["Fix_Lib"] and not {"RECORD", "direct_url.json"} & set(held), (
        f"{egg_info} must require Fix_Lib, with no RECORD and no direct_url.json: "
        + f"requires {requires}, holds {held}"
    )
    return egg_info


def _helper_ran(project: installed_dists.Project, output: str) -> dict[str, str]:
    """`pivot repro` runs the stage, which writes `output` (`assert_runs_and_writes`), and refuses
    and warns of nothing; returns the lock's code manifest."""
    shown = installed_dists.assert_runs_and_writes(project, output)
    assert _REFUSAL_FIX not in shown, f"`pivot repro` must refuse and warn of nothing:\n{shown}"
    return installed_dists.lock_manifest(project, installed_dists.STAGE)


def _helper_sys_path(project: installed_dists.Project) -> list[str]:
    """The `sys.path` of the venv's interpreter, started in the project as `run_pivot` starts
    pivot, before pivot puts the project root first."""
    ran = helpers.run_cli(
        project.root,
        project.home,
        str(project.venv.python),
        "-P",
        "-c",
        "import json, sys; print(json.dumps(sys.path))",
    )
    assert ran.returncode == 0, f"the venv's interpreter must start:\n{installed_dists.shown(ran)}"
    return json.loads(ran.stdout)


def _helper_in_the_project_root(
    tmp_path: pathlib.Path,
    venv: installed_dists.Venv,
    project: installed_dists.Project,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """The project root, which pivot puts first on the worker's path: the flat setuptools project
    `myproj` (package `mymod`, and `company/local.py`), built there, which leaves
    `myproj.egg-info` in the root. Returns the root."""
    _helper_write_myproj(project.root, tmp_path / "myproj", ["mymod"], _PROJECT_CODE)
    _helper_build(venv, project.root, tmp_path / "wheels", project.root)
    return project.root


def _helper_in_a_src_directory(
    tmp_path: pathlib.Path,
    venv: installed_dists.Venv,
    project: installed_dists.Project,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """`src` of the project root: the setuptools project `myproj` in the src layout (package
    `mymod`, and `company/local.py`, under `src`), installed editable with `uv pip install -e`,
    whose `.pth` file puts `src` on the path (the test fails otherwise), then built, which leaves
    `myproj.egg-info` in `src`. Returns `src`."""
    src = project.root / "src"
    _helper_write_myproj(
        project.root,
        tmp_path / "myproj",
        ["mymod"],
        {f"src/{relative}": text for relative, text in _PROJECT_CODE.items()},
    )
    with (project.root / "pyproject.toml").open("a") as pyproject:
        pyproject.write('\n[tool.setuptools.package-dir]\n"" = "src"\n')
    installed_dists.install_directory(venv, project.root, editable=True)
    path = [os.path.realpath(entry) for entry in _helper_sys_path(project)]
    assert str(src.resolve()) in path, f"the editable install must put {src} on the path: {path}"
    _helper_build(venv, project.root, tmp_path / "wheels", src)
    return src


def _helper_in_a_pythonpath_entry_with_an_egg_info(
    tmp_path: pathlib.Path,
    venv: installed_dists.Venv,
    project: installed_dists.Project,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """`extra`, which `PYTHONPATH` puts on the path of every pivot this test runs: the setuptools
    project `myproj` (package `mymod`, and `company/local.py`), built there, which leaves
    `myproj.egg-info` in `extra`. Returns `extra`."""
    extra = tmp_path / "extra"
    extra.mkdir()
    _helper_write_myproj(extra, tmp_path / "myproj", ["mymod"], _PROJECT_CODE)
    _helper_build(venv, extra, tmp_path / "wheels", extra)
    monkeypatch.setenv("PYTHONPATH", str(extra), prepend=os.pathsep)
    return extra


def _helper_target_install_without_record(
    venv: installed_dists.Venv, src: pathlib.Path, directory: pathlib.Path
) -> None:
    """Install `fixtarget` (written under `src`) into `directory` from its path with `uv pip
    install --target`, whose `direct_url.json` says so (`dir_info`; the test fails otherwise),
    and delete its RECORD: a distribution with no RECORD that Pivot looks at in that directory,
    not a project's own metadata."""
    source = installed_dists.write_library(
        src, "fixtarget", "fixtarget", installed_dists.FIXLIB_ALONE
    )
    installed_dists.install(venv, "--target", str(directory), str(source))
    dist_info = directory / "fixtarget-0.1.dist-info"
    url = json.loads((dist_info / "direct_url.json").read_text())
    assert url["dir_info"] == {}, f"fixtarget must be installed from its path: {url}"
    installed_dists.damage(dist_info / "RECORD", None)


def _helper_in_a_pythonpath_entry_with_a_target_install(
    tmp_path: pathlib.Path,
    venv: installed_dists.Venv,
    project: installed_dists.Project,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """`extra`, which `PYTHONPATH` puts on the path of every pivot this test runs: `mymod` and
    `company/local.py`, beside `fixtarget`, installed there with `uv pip install --target`, which
    has lost its RECORD (`_helper_target_install_without_record`). Returns `extra`."""
    extra = tmp_path / "extra"
    _helper_target_install_without_record(venv, tmp_path / "src", extra)
    for relative, text in _PROJECT_CODE.items():
        (extra / relative).parent.mkdir(parents=True, exist_ok=True)
        (extra / relative).write_text(text)
    monkeypatch.setenv("PYTHONPATH", str(extra), prepend=os.pathsep)
    return extra


@pytest.mark.parametrize(
    ("pipeline_text", "packages", "files", "output"),
    [
        pytest.param(
            _HELPER_PIPELINE,
            ["helpers"],
            {"helpers/__init__.py": _HELPERS},
            "h1:value-1",
            id="by-name-helper",
        ),
        pytest.param(
            _NAMESPACE_PIPELINE,
            [],
            {"company/local.py": _LOCAL},
            "client-1:value-1",
            id="shared-namespace",
        ),
    ],
)
def test_a_flat_project_installed_editable_runs_as_with_its_egg_info_moved_out(
    tmp_path: pathlib.Path,
    pipeline_text: str,
    packages: list[str],
    files: dict[str, str],
    output: str,
) -> None:
    """The project root is the flat setuptools project `myproj`, which requires `Fix_Lib`,
    installed editable with `uv pip install -e`; its build leaves `myproj.egg-info` in the root,
    with no RECORD and no `direct_url.json`, as `uv pip install -e` with build isolation does. The
    stage calls `fixlib.value()` and the project's `@pivot.fingerprint_by_name` helper
    `helpers.stamp()` (`by-name-helper`), or `company.client.value()` of the namespace package
    `company`, which the project shares with `company-client` through `company/local.py` in its
    root (`shared-namespace`). `pivot repro` runs the stage, which writes its output, and refuses
    nothing; with `myproj.egg-info` moved out of the root, `pivot repro --force` writes the same
    lock, its `dist:` keys included. Pivot 0.10.4 runs both projects so."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    _helper_install_libraries(venv, tmp_path / "src")
    project = installed_dists.make_project_in(tmp_path, venv, pipeline_text)
    _helper_write_myproj(project.root, tmp_path / "myproj", packages, files)
    installed_dists.install_directory(venv, project.root, editable=True)
    url = installed_dists.direct_url(venv, "myproj")
    assert url is not None and url["dir_info"] == {"editable": True}, (
        f"myproj must be installed editable: direct_url.json is {url!r}"
    )
    egg_info = _helper_build(venv, project.root, tmp_path / "wheels", project.root)

    beside = _helper_ran(project, output)
    egg_info.rename(tmp_path / egg_info.name)
    installed_dists.pivot_ok(project, "repro", "--force")

    written = (project.root / installed_dists.OUTPUT).read_text()
    assert written == output, f"{installed_dists.STAGE} must write {output!r}, not {written!r}"
    moved_out = installed_dists.lock_manifest(project, installed_dists.STAGE)
    assert beside == moved_out, (
        f"the lock beside myproj.egg-info must be the one without it: {beside} != {moved_out}"
    )


@pytest.mark.parametrize(
    "place",
    [
        pytest.param(_helper_in_the_project_root, id="project-root-egg-info"),
        pytest.param(_helper_in_a_src_directory, id="src-directory-egg-info"),
        pytest.param(_helper_in_a_pythonpath_entry_with_an_egg_info, id="pythonpath-egg-info"),
        pytest.param(
            _helper_in_a_pythonpath_entry_with_a_target_install,
            id="pythonpath-target-install-without-record",
        ),
    ],
)
def test_project_code_beside_metadata_with_no_record_outside_site_packages_is_walked(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    place: collections.abc.Callable[
        [pathlib.Path, installed_dists.Venv, installed_dists.Project, pytest.MonkeyPatch],
        pathlib.Path,
    ],
) -> None:
    """Each kind of path entry outside every `site-packages` and `dist-packages` directory (the
    project root, a `src/` directory, a `PYTHONPATH` entry) holds `mymod`, whose `value()` the
    walk hashes and whose `stamp()` is a `@pivot.fingerprint_by_name` helper, and the project's
    part of the namespace package `company`, beside a distribution with no RECORD there:
    `myproj.egg-info`, which requires `Fix_Lib` and which a setuptools build of the project left
    there, or `fixtarget`, which `uv pip install --target` installed there from its path, and
    which has lost its RECORD. The stage calls `mymod.value()`, `mymod.stamp()`,
    `company.client.value()` and `fixlib.value()`. `pivot repro` runs it, refusing nothing, and
    its lock holds the keys of `company-client` and `Fix_Lib`. After one byte of `mymod.value()`
    changes, the stage is stale on `func:mymod.value` alone, and `pivot repro` writes the new
    value: the project's code is walked as user code, and no distribution's RECORD decides it."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    _helper_install_libraries(venv, tmp_path / "src")
    project = installed_dists.make_project_in(tmp_path, venv, _PROJECT_CODE_PIPELINE)
    directory = place(tmp_path, venv, project, monkeypatch)

    manifest = _helper_ran(project, _PROJECT_CODE_OUTPUT)
    assert installed_dists.dist_keys(manifest) == ["dist:company-client", "dist:fix-lib"], (
        f"the lock: {manifest}"
    )
    installed_dists.replace_once(directory / "mymod" / "__init__.py", '"mine-1"', '"mine-2"')

    installed_dists.assert_stale_on(project, installed_dists.STAGE, "func:mymod.value")
    _helper_ran(project, _PROJECT_CODE_OUTPUT.replace("mine-1", "mine-2"))


def test_the_part_of_a_namespace_package_in_a_checkout_under_the_venv_is_left_out(
    tmp_path: pathlib.Path,
) -> None:
    """`<venv>/src/checkout`, which a `.pth` file in site-packages puts on the path, as pip's
    `install -e git+...` puts its checkout in `<venv>/src` there, holds `company/checkout.py`, the
    checkout's part of the namespace package `company`, beside `fixtarget`, installed there with
    `uv pip install --target`, which has lost its RECORD. The stage calls `company.client.value()`
    and `fixlib.value()`. The checkout lies under the interpreter's prefix but in no
    `site-packages` or `dist-packages` directory, so its directory is the user's part of
    `company`, as `is_user_code` takes it, and is left out of the `dist:` keys' check: `pivot
    repro` runs the stage, refusing nothing, and its lock holds the keys of `company-client` and
    `Fix_Lib`. Taken as a library's directory, as a path under the prefix is for a file, it would
    be walked beside `fixtarget`, and the stage refused."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    _helper_install_libraries(venv, tmp_path / "src")
    checkout = tmp_path / "venv" / "src" / "checkout"
    _helper_target_install_without_record(venv, tmp_path / "src", checkout)
    (checkout / "company").mkdir()
    (checkout / "company" / "checkout.py").write_text(_LOCAL)
    (venv.site_packages / "checkout.pth").write_text(f"{checkout}\n")
    project = installed_dists.make_project_in(tmp_path, venv, _NAMESPACE_PIPELINE)

    manifest = _helper_ran(project, "client-1:value-1")

    assert installed_dists.dist_keys(manifest) == ["dist:company-client", "dist:fix-lib"], (
        f"the lock: {manifest}"
    )
