"""Helpers for the CLI tests of the `dist:` keys of installed distributions: a venv that uv makes
over this test environment, setuptools projects installed into it, and what the `pivot` that the
venv runs reports.

The venv's site-packages holds a `.pth` file that adds this test environment's purelib directory,
so pivot, its dependencies and setuptools are importable there and every install of a fixture,
by uv or by the venv's own pip, runs offline without build isolation. Pivot runs as
`<venv>/bin/python -P -c "<_console_main>"` through `helpers.run_cli`: `-P` keeps the working
directory off `sys.path`, as the `pivot` console script does. `make_standalone_venv` makes instead
a venv without that `.pth` file, into which uv installs a released pivot from the package index and
fixtures it builds with build isolation. This module imports only the standard library, pytest,
PyYAML, `tests.helpers`, the `pivot` package and its `fingerprint` module (the
`installed_distributions` types only for type checking), so the tests that use it still collect
when `packages/pivot/src` is pivot 0.10.4's.

It also holds what the test files of those keys share: the stage `STAGE` of their projects, the
file `OUTPUT` it writes and the `pipeline.py` of a project whose stage calls installed modules
(`pipeline`), the sources of the fixture libraries `Fix_Lib` and `fixdep`, the project's part
`LOCAL` of the namespace package `company` and the module `CLIENT` of the library's part of it,
the two refusals of an
install Pivot cannot read, word for word, as templates (`DISTRIBUTION_REFUSAL`, and
`UNLISTED_FILE_REFUSAL`, which `unlisted_refusal` fills; each ends with the fix `REFUSAL_FIX`
starts) and as those of a broken install of the fixtures, with the start pivot gives each for
`STAGE`, in an error (`STAGE_FAILED`) and in a warning (`STAGE_PREFIX`), a run that refuses and
warns of nothing (`assert_runs_with_no_refusal`), the layout flit's `install --symlink` gives an
editable install (`flit_symlink_install`), built by hand since CI has no flit, the layout uv
gives an install from a package index, its files copied or linked into a cache
(`index_install_by_hand`), built by hand for the tests that start no process, and what those
tests give `installed_distributions.distribution_keys` of a module loaded from a file
(`module_loaded_from`).
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import sysconfig
from typing import TYPE_CHECKING, Any

import pytest
import yaml

import pivot
from pivot import fingerprint
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pivot import installed_distributions

# The code `python -c` runs to start pivot's CLI, as the `pivot` console script does.
_PIVOT_MAIN = "from pivot.cli import _console_main; _console_main()"

# The stage of most projects the tests make, `make` in `Pipeline("p")` of `pipeline.py`, and the
# file, relative to the project's root, that it writes.
STAGE = "p/make"
OUTPUT = pathlib.Path("data/p/make.txt")

# The `pipeline.py` that `pipeline` writes: `{imports}` imports the modules whose `value()`s
# `{expression}` joins.
_PIPELINE = """\
{imports}

from pivot import Pipeline, stage


@stage
def make() -> str:
    return {expression}


with Pipeline("p") as pipeline:
    make()
"""


def pipeline(*modules: str) -> str:
    """The `pipeline.py` of a project whose stage `STAGE` imports each of `modules` and writes
    their `value()`s joined, and names nothing else of them."""
    return _PIPELINE.format(
        imports="\n".join(f"import {module}" for module in modules),
        expression=" + ".join(f"{module}.value()" for module in modules),
    )


# The source of `fixdep`'s module, and of `Fix_Lib`'s module `fixlib`, whose `value()` adds
# `fixdep.base()` to a literal and whose `main()` a console script can run
# (`write_fixlib_and_fixdep`).
FIXDEP = 'def base():\n    return "base-1"\n'
_FIXLIB = """\
import fixdep


def value():
    return "value-1:" + fixdep.base()


def main():
    print(value())
"""
# A `fixlib` that imports nothing.
FIXLIB_ALONE = """\
def value():
    return "value-1"


def main():
    print(value())
"""
# The project's part of the namespace package `company` it shares with `company-client`, the module
# `company.local`, and the module of `company-client`'s namespace portion `company.client`, which
# imports its package, as a library module that reads a sibling through it does, so
# `company.client` holds `company`.
LOCAL = 'def value():\n    return "local-1"\n'
CLIENT = 'import company\n\n\ndef value():\n    return "client-1"\n'

# The two refusals pivot gives of an install it cannot read, word for word, each a template for
# `str.format`, so that a change to the words of either is made here alone: a distribution's
# (`installed_distributions.decide`), `{how}` being `REACHED` or `DEPENDED_ON`, and that of a
# reached module whose file no RECORD lists beside a distribution with no RECORD
# (`installed_distributions.distribution_keys`, which `unlisted_refusal` fills), `{form}` saying
# how that distribution is installed (an `InstallForm`'s value) and `{problem}` what is wrong
# with its RECORD. Each ends with the fix `REFUSAL_FIX` starts, and so does the warning that
# unsafe fingerprinting gives in its place.
REFUSAL_FIX = "Fix: reinstall"
DISTRIBUTION_REFUSAL = (
    "Distribution '{name}' ({dist_info}), which {how}, {reason}. Pivot cannot fingerprint the "
    "installed code the stage runs, so the stage could skip a run after that code changes. "
    f"{REFUSAL_FIX} '{{name}}'."
)
REACHED = "this stage reaches"
DEPENDED_ON = "a distribution this stage reaches depends on"
UNLISTED_FILE_REFUSAL = (
    "Module '{module}' ({file}) lies in {site}, where distribution '{name}' ({dist_info}), "
    "installed {form}, {problem}, and no RECORD file there lists that file. Pivot cannot tell "
    "whether the stage runs that distribution's code, so the stage could skip a run after it "
    f"changes. {REFUSAL_FIX} '{{name}}'."
)

# How `InstallForm.LOCAL_DIRECTORY` and `InstallForm.INDEX` complete "installed ..." in a refusal
# (each one's value), written out for the refusals this module builds when it is imported, since
# it imports nothing of `installed_distributions` at run time (above).
_LOCAL_DIRECTORY_FORM = "from a local directory (dir_info)"  # InstallForm.LOCAL_DIRECTORY.value
_INDEX_FORM = "with no direct_url.json, as from a package index"  # InstallForm.INDEX.value


def unlisted_refusal(
    module: str,
    file: str,
    site: str,
    name: str,
    form: installed_distributions.InstallForm,
    dist_info: str | None = None,
) -> str:
    """The refusal of reached module `module`, whose file `file` lies in site directory `site`,
    where distribution `name`, installed as `form` says, has no RECORD file, and no RECORD there
    lists that file (`UNLISTED_FILE_REFUSAL`): `name`'s metadata directory is `dist_info`, else
    `{site}/{name}-0.1.dist-info`."""
    return _unlisted_file_refusal(module, file, site, name, form.value, dist_info)


def _unlisted_file_refusal(
    module: str, file: str, site: str, name: str, form: str, dist_info: str | None
) -> str:
    """`UNLISTED_FILE_REFUSAL` filled as `unlisted_refusal` says, `form` being an `InstallForm`'s
    value."""
    return UNLISTED_FILE_REFUSAL.format(
        module=module,
        file=file,
        site=site,
        name=name,
        dist_info=f"{site}/{name}-0.1.dist-info" if dist_info is None else dist_info,
        form=form,
        problem="has no RECORD file",
    )


# The reason `DISTRIBUTION_REFUSAL` gives for a distribution whose `direct_url.json` is no JSON
# object (`InstallForm.UNCLASSIFIED`, `installed_distributions.classify`).
NOT_A_JSON_OBJECT_REASON = (
    "is installed in a form Pivot cannot classify: its direct_url.json is not a JSON object"
)
# The refusals of a broken install of `Fix_Lib` and `fixdep`, `{site}` left in each for the venv's
# resolved site-packages.
UNCLASSIFIED_REFUSAL = DISTRIBUTION_REFUSAL.format(
    name="fix-lib",
    dist_info="{site}/fix_lib-0.1.dist-info",
    how=REACHED,
    reason=NOT_A_JSON_OBJECT_REASON,
)
RECORDLESS_DEPENDENCY_REFUSAL = DISTRIBUTION_REFUSAL.format(
    name="fixdep",
    dist_info="{site}/fixdep-0.1.dist-info",
    how=DEPENDED_ON,
    reason=f"is installed {_LOCAL_DIRECTORY_FORM} and has no RECORD file, so Pivot cannot read "
    + "its files",
)
# A reached module `{module}`, whose file `{file}` no RECORD lists, beside `Fix_Lib` with no RECORD:
# `{module}` and `{file}`, relative to `{site}`, left in it too.
UNLISTED_REFUSAL = _unlisted_file_refusal(
    "{module}",
    "{site}/{file}",
    "{site}",
    "fix-lib",
    _LOCAL_DIRECTORY_FORM,
    "{site}/fix_lib-0.1.dist-info",
)
# A reached module `{module}`, whose file `{file}` no RECORD lists, beside `{name}` 0.1, installed
# into `{site}` from a local directory, which has lost its RECORD: `{module}`, `{name}` and
# `{file}`, relative to `{site}`, left in it too.
UNLISTED_LOCAL_DIRECTORY_REFUSAL = _unlisted_file_refusal(
    "{module}", "{site}/{file}", "{site}", "{name}", _LOCAL_DIRECTORY_FORM, None
)
# A reached module `{module}`, whose file `{file}` no RECORD lists, beside `{name}.egg-info` in
# `{site}`, the metadata a setuptools build or develop install leaves beside its code, with no
# RECORD and no `direct_url.json`: `{module}`, `{name}` and `{file}`, relative to `{site}`, left in
# it too.
UNLISTED_EGG_INFO_REFUSAL = _unlisted_file_refusal(
    "{module}", "{site}/{file}", "{site}", "{name}", _INDEX_FORM, "{site}/{name}.egg-info"
)
# How pivot starts the warning unsafe fingerprinting gives of `STAGE` in place of each of those
# refusals, as it starts every refusal it can downgrade: with the stage's name.
STAGE_PREFIX = f"Stage '{STAGE}': "
# How the error of a refused `STAGE` starts: the stage's name once, before its first refusal.
STAGE_FAILED = f"{STAGE_PREFIX}fingerprinting failed: "

# The `.dist-info` directory of the `flitlib` that `flit_symlink_install` installs, and the
# METADATA flit 4.2.0 wrote in it.
FLIT_DIST_INFO = "flitlib-1.0.dist-info"
_FLIT_METADATA = (
    b"Metadata-Version: 2.5\nName: flitlib\nVersion: 1.0\nSummary: A tiny flit package\n"
    + b"Import-Name: flitlib\n"
)

# `direct_url.json` (PEP 610) as uv 0.13.5 writes it for a local directory and a git install.
UV_DIRECTORY_URL = b'{"url":"file:///home/user/src/fixlib","dir_info":{}}'
UV_GIT_URL = (
    b'{"url":"file:///home/user/src/fixgit","vcs_info":{"vcs":"git",'
    b'"commit_id":"6964203a125157369c715bd7ba6b130944b8329e",'
    b'"requested_revision":"6964203a125157369c715bd7ba6b130944b8329e"}}'
)


@dataclasses.dataclass(frozen=True)
class Venv:
    """A venv `make_venv` made: its interpreter and its site-packages directory."""

    python: pathlib.Path
    site_packages: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Project:
    """A pivot project at `root`, run by the `pivot` of `venv`, with `home` as `HOME`."""

    root: pathlib.Path
    home: pathlib.Path
    venv: Venv


def make_venv(venv: pathlib.Path, *, seed: bool = False) -> Venv:
    """Make a venv at `venv` (`_new_venv`; `seed`: with pip, `uv venv --seed`) whose
    site-packages holds a `.pth` file that adds this test environment's purelib directory to
    `sys.path` after it."""
    made = _new_venv(venv, seed=seed)
    (made.site_packages / "_parent.pth").write_text(
        f"import site; site.addsitedir({sysconfig.get_path('purelib')!r})\n"
    )
    return made


def make_standalone_venv(venv: pathlib.Path, *specs: str) -> Venv:
    """Make a venv at `venv` (`_new_venv`) that does not see this test environment, and install
    `specs` into it with `uv pip install`, from the package index and with build isolation, so it
    needs network access; the test fails unless the install succeeds."""
    made = _new_venv(venv, seed=False)
    _run_uv(venv.parent, "pip", "install", "--python", str(made.python), *specs)
    return made


def write_setuptools_project(
    root: pathlib.Path,
    name: str,
    version: str,
    *,
    packages: Sequence[str],
    files: Mapping[str, str],
    dependencies: Sequence[str] = (),
    scripts: Mapping[str, str] | None = None,
    package_data: Mapping[str, Sequence[str]] | None = None,
) -> pathlib.Path:
    """Write at `root` a setuptools project: `files` (each path relative to `root`, with its text)
    and a `pyproject.toml` for distribution `name` at `version` that names its import `packages`,
    its `dependencies`, its console `scripts` (command, then `module:function`) and, for each
    package of `package_data`, the glob patterns of the other files its wheel holds. Returns
    `root`."""
    # A JSON string, or array of strings, that `json.dumps` writes is also valid TOML.
    lines = [
        "[build-system]",
        'requires = ["setuptools"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f"name = {json.dumps(name)}",
        f"version = {json.dumps(version)}",
        f"dependencies = {json.dumps(list(dependencies))}",
        "",
    ]
    if scripts:
        lines.append("[project.scripts]")
        lines.extend(
            f"{json.dumps(command)} = {json.dumps(target)}" for command, target in scripts.items()
        )
        lines.append("")
    lines.extend(["[tool.setuptools]", f"packages = {json.dumps(list(packages))}", ""])
    if package_data:
        lines.append("[tool.setuptools.package-data]")
        lines.extend(
            f"{json.dumps(package)} = {json.dumps(list(patterns))}"
            for package, patterns in package_data.items()
        )
        lines.append("")
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("\n".join(lines))
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def write_library(
    root: pathlib.Path,
    name: str,
    module: str,
    source: str,
    *,
    version: str = "0.1",
    dependencies: Sequence[str] = (),
    scripts: Mapping[str, str] | None = None,
    files: Mapping[str, str] | None = None,
    package_data: Sequence[str] = (),
) -> pathlib.Path:
    """Write at `root/<module>` the setuptools project (`write_setuptools_project`) of
    distribution `name`, whose package `module` has `source` as its `__init__.py`, with `files`
    beside it, and whose wheel also holds the files of `module` that the glob patterns
    `package_data` match. Returns the project's directory."""
    return write_setuptools_project(
        root / module,
        name,
        version,
        packages=[module],
        files={f"{module}/__init__.py": source, **(files or {})},
        dependencies=dependencies,
        scripts=scripts,
        package_data={module: package_data} if package_data else None,
    )


def write_fixlib_and_fixdep(
    root: pathlib.Path, *, console_script: bool = False
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write under `root` the projects (`write_library`) of `Fix_Lib` (module `fixlib` of
    `_FIXLIB`, requires `fixdep`; `console_script`: with the console script `fix-lib`, which runs
    `fixlib:main`) and `fixdep` (module `fixdep` of `FIXDEP`); returns their directories."""
    fixlib = write_library(
        root,
        "Fix_Lib",
        "fixlib",
        _FIXLIB,
        dependencies=("fixdep",),
        scripts={"fix-lib": "fixlib:main"} if console_script else None,
    )
    fixdep = write_library(root, "fixdep", "fixdep", FIXDEP)
    return fixlib, fixdep


def replace_once(path: pathlib.Path, old: str, new: str) -> None:
    """Replace the one occurrence of `old` in the file at `path` with `new`; the test fails when
    the file does not hold `old` exactly once."""
    text = path.read_text()
    assert text.count(old) == 1, f"{path} must hold {old!r} once:\n{text}"
    path.write_text(text.replace(old, new))


def damage(path: pathlib.Path, text: str | None) -> None:
    """Write `text` over the installed file at `path`, or delete it when `text` is None."""
    assert path.is_file(), f"{path} must be installed before the test damages it"
    if text is None:
        path.unlink()
    else:
        path.write_text(text)


def install(venv: Venv, *spec: str, reinstall: bool = False, link_mode: str | None = None) -> str:
    """`uv pip install --offline --python <venv>/bin/python --no-build-isolation --no-deps
    [--reinstall] [--link-mode <link_mode>] <spec>`; the test fails unless it succeeds. Returns
    what uv printed. With no `link_mode`, uv links the files as `UV_LINK_MODE` says, else as it
    does by default: the pod that runs the tests by hand sets `copy`, CI sets nothing."""
    return _run_uv(
        venv.python.parent,
        "pip",
        "install",
        "--offline",
        "--python",
        str(venv.python),
        "--no-build-isolation",
        "--no-deps",
        *(["--reinstall"] if reinstall else []),
        *(["--link-mode", link_mode] if link_mode is not None else []),
        *spec,
    )


def install_directory(
    venv: Venv,
    source: pathlib.Path,
    *,
    reinstall: bool = False,
    editable: bool = False,
    link_mode: str | None = None,
) -> str:
    """Install the project at `source` from its path (`editable`: with `-e`; `link_mode`: as
    `install` takes it), after deleting what an earlier build left in it
    (`_remove_build_outputs`)."""
    _remove_build_outputs(source)
    return install(
        venv,
        *(["-e"] if editable else []),
        str(source),
        reinstall=reinstall,
        link_mode=link_mode,
    )


def build_wheel(venv: Venv, source: pathlib.Path, wheels: pathlib.Path) -> str:
    """Build a wheel of the project at `source` into the directory `wheels` with `uv build
    --offline --no-build-isolation --wheel` on the venv's interpreter, after deleting what an
    earlier build left in it."""
    _remove_build_outputs(source)
    return _run_uv(
        source.parent,
        "build",
        "--offline",
        "--python",
        str(venv.python),
        "--no-build-isolation",
        "--wheel",
        "--out-dir",
        str(wheels),
        str(source),
    )


def install_from_wheels(
    venv: Venv, wheels: pathlib.Path, requirement: str, *, link_mode: str | None = None
) -> str:
    """Install `requirement` from the wheels in `wheels` as from a package index
    (`--no-index --find-links`), which writes no `direct_url.json` (`link_mode`: as `install`
    takes it)."""
    return install(
        venv, "--no-index", "--find-links", str(wheels), requirement, link_mode=link_mode
    )


def pip_install_directory(venv: Venv, source: pathlib.Path) -> str:
    """Install the project at `source` from its path with the venv's own pip (`python -m pip
    install --no-index --no-build-isolation --no-deps`; the venv needs pip, `make_venv(...,
    seed=True)`), after deleting what an earlier build left in it; the test fails unless it
    succeeds. Returns what pip printed."""
    _remove_build_outputs(source)
    return _run_checked(
        venv.python.parent,
        [
            str(venv.python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-index",
            "--no-build-isolation",
            "--no-deps",
            str(source),
        ],
    )


def flit_symlink_install(site: pathlib.Path, checkout: pathlib.Path, link: str) -> None:
    """Install `flitlib` 1.0 from the checkout at `checkout` into the site directory `site` as
    flit 4.2.0's `flit install --symlink` did on 2026-11-17 (a flit_core project `flitlib` with
    one package `flitlib`, or one module `flitlib.py`): `site/<link>` a symbolic link to
    `checkout/<link>` (the package's directory, or the module's file), and `FLIT_DIST_INFO`
    beside it with flit's INSTALLER, METADATA, REQUESTED, a `direct_url.json` that names the
    checkout and says editable, and a RECORD, its rows ended by CR LF and with flit's hex
    `sha256=` digests, that lists the link, not the files under it. flit lists a package's link
    first and a module's after the metadata files; no reader depends on the order, which is a
    package's here. The test writes the checkout's files."""
    (site / link).symlink_to(checkout / link)
    dist_info = site / FLIT_DIST_INFO
    dist_info.mkdir()
    direct_url = {"url": checkout.as_uri(), "dir_info": {"editable": True}}
    files = {
        "INSTALLER": b"flit",
        "METADATA": _FLIT_METADATA,
        "REQUESTED": b"",
        "direct_url.json": json.dumps(direct_url).encode(),
    }
    rows = [f"{link},,"]
    for name, data in files.items():
        (dist_info / name).write_bytes(data)
        rows.append(
            f"{FLIT_DIST_INFO}/{name},sha256={hashlib.sha256(data).hexdigest()},{len(data)}"
        )
    rows.append(f"{FLIT_DIST_INFO}/RECORD,,")
    (dist_info / "RECORD").write_bytes("".join(f"{row}\r\n" for row in rows).encode())


def index_install_by_hand(
    site: pathlib.Path, name: str, files: Mapping[str, str], *, cache: pathlib.Path | None
) -> None:
    """Install distribution `name` 0.1 into the site directory `site` with no installer, as uv
    0.13.5 laid out an install from a package index on 2026-11-17: `files` (each path in `site`,
    with its text), and `<name>-0.1.dist-info` with the wheel's METADATA and WHEEL, uv's
    INSTALLER and REQUESTED, a RECORD of uv's rows (`sha256=` in URL-safe base64 with no padding,
    then the size) and no `direct_url.json`. With no `cache`, every file is a copy, as
    `--link-mode copy` leaves it. With one, each file the wheel holds (`files`, METADATA and
    WHEEL) is a symbolic link, from a real directory of `site`, to the file under
    `cache/<name>`, as `--link-mode symlink` links it into uv's cache of unpacked wheels, and
    uv's own files are real."""
    dist_info = f"{name}-0.1.dist-info"
    wheel = {
        **{relative: text.encode() for relative, text in files.items()},
        f"{dist_info}/METADATA": f"Metadata-Version: 2.4\nName: {name}\nVersion: 0.1\n".encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    written = {f"{dist_info}/INSTALLER": b"uv", f"{dist_info}/REQUESTED": b""}
    rows = list[str]()
    for relative, data in sorted({**wheel, **written}.items()):
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if cache is None or relative in written:
            path.write_bytes(data)
        else:
            cached = cache / name / relative
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
            path.symlink_to(cached)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{relative},sha256={digest},{len(data)}")
    rows.append(f"{dist_info}/RECORD,,")
    (site / dist_info / "RECORD").write_text("".join(f"{row}\n" for row in sorted(rows)))


def module_loaded_from(found: pathlib.Path) -> installed_distributions.ReachedModule:
    """What `fingerprint.add_distribution_keys` gives of a module this process loaded from the
    file the import found at `found` (`fingerprint.reached_module_of`): the origin
    `import_origin.of_module` gives it, the file resolved, and the places that file lies in: the
    file Pivot judges it by (`fingerprint.judged_file`), then every later file of its chain of
    links. Pivot 0.10.4 has no `fingerprint.reached_module_of`, which is looked up only when this
    is called, so the tests that use this module still collect there."""
    return fingerprint.reached_module_of(("file", os.path.realpath(found)), str(found))


def git(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> str:
    """Run `git <args>` in `cwd` as `helpers.run_cli` runs a command, with a fixed author and no
    signing; the test fails unless it exits 0. Returns its standard output."""
    ran = helpers.run_cli(
        cwd,
        home,
        "git",
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "-c",
        "commit.gpgsign=false",
        *args,
    )
    assert ran.returncode == 0, f"git {shlex.join(args)} exited {ran.returncode}:\n{shown(ran)}"
    return ran.stdout


def commit_all(repo: pathlib.Path, home: pathlib.Path, message: str) -> str:
    """Commit every file of the git repository `repo`; returns the new commit's id."""
    git(repo, home, "add", "-A")
    git(repo, home, "commit", "-q", "-m", message)
    return git(repo, home, "rev-parse", "HEAD").strip()


def git_requirement(name: str, repo: pathlib.Path, commit: str) -> str:
    """The requirement that installs distribution `name` from `commit` of the local git
    repository `repo`."""
    return f"{name} @ git+{repo.as_uri()}@{commit}"


def init_git_requirement(source: pathlib.Path, home: pathlib.Path, name: str) -> str:
    """Make the project at `source` a git repository of one commit, `C1`; the requirement that
    installs distribution `name` from that commit (`git_requirement`)."""
    git(source, home, "init", "-q")
    commit = commit_all(source, home, "C1")
    return git_requirement(name, source, commit)


def installed(venv: Venv, name: str) -> importlib.metadata.Distribution:
    """The one distribution called `name` (as `importlib.metadata` matches names) in the venv's
    site-packages; the test fails when it holds none or more than one."""
    found = list(
        importlib.metadata.Distribution.discover(name=name, path=[str(venv.site_packages)])
    )
    assert len(found) == 1, (
        f"{venv.site_packages} must hold one distribution {name!r}; it holds {len(found)}: "
        + f"{sorted(os.listdir(venv.site_packages))}"
    )
    return found[0]


def direct_url(venv: Venv, name: str) -> dict[str, Any] | None:
    """The parsed `direct_url.json` (PEP 610) of distribution `name` in the venv, None when its
    metadata directory holds none."""
    text = installed(venv, name).read_text("direct_url.json")
    return None if text is None else json.loads(text)


def reinstall_from_commit(
    venv: Venv, name: str, repo: pathlib.Path, commit: str, label: str
) -> None:
    """Reinstall distribution `name` from `commit`, which a failure calls `label`, of the local git
    repository `repo` (`git_requirement`); the test fails unless its `direct_url.json` then names
    that commit."""
    install(venv, git_requirement(name, repo, commit), reinstall=True)
    url = direct_url(venv, name)
    assert url is not None and url["vcs_info"]["commit_id"] == commit, (
        f"{name} must now be installed from {label} {commit}: direct_url.json is {url!r}"
    )


def make_project_in(tmp_path: pathlib.Path, venv: Venv, pipeline_text: str) -> Project:
    """`git init -q` and `pivot init` a new project at `tmp_path/project`, with `tmp_path/home` as
    `HOME`, then write `pipeline_text` as its `pipeline.py`."""
    root = tmp_path / "project"
    root.mkdir(parents=True)
    project = Project(root=root, home=tmp_path / "home", venv=venv)
    git(root, project.home, "init", "-q")
    pivot_ok(project, "init")
    (root / "pipeline.py").write_text(pipeline_text)
    return project


def run_pivot(project: Project, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the venv's `pivot <args>` in the project; the exit code is not checked."""
    return helpers.run_cli(
        project.root, project.home, str(project.venv.python), "-P", "-c", _PIVOT_MAIN, *args
    )


def run_pivot_holding(
    project: Project, module: str, *args: str
) -> subprocess.CompletedProcess[str]:
    """Run the venv's `pivot <args>` in the project from a program that imports `module` before
    it starts pivot's CLI (`python -P -c "import <module>; <_console_main>"`), so the program
    holds the module `-P` lets it find, the installed one, when pivot's import session starts;
    the exit code is not checked."""
    return helpers.run_cli(
        project.root,
        project.home,
        str(project.venv.python),
        "-P",
        "-c",
        f"import {module}; {_PIVOT_MAIN}",
        *args,
    )


def pivot_ok(project: Project, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the venv's `pivot <args>` in the project; the test fails unless it exits 0."""
    ran = run_pivot(project, *args)
    assert ran.returncode == 0, f"pivot {shlex.join(args)} exited {ran.returncode}:\n{shown(ran)}"
    return ran


def module_file(project: Project, module: str) -> pathlib.Path:
    """The resolved `__file__` of `module` as the venv's interpreter imports it in the project,
    started as `run_pivot` starts pivot."""
    ran = helpers.run_cli(
        project.root,
        project.home,
        str(project.venv.python),
        "-P",
        "-c",
        f"import {module}; print({module}.__file__)",
    )
    assert ran.returncode == 0, f"the venv cannot import {module}:\n{shown(ran)}"
    return pathlib.Path(ran.stdout.strip()).resolve()


def repro_statuses(project: Project) -> tuple[dict[str, str], str]:
    """What one `pivot repro --json` did to each stage (`ran`, `cached`, ...), by the stage's
    name, from the stage's one `stage_complete` event, and the command's output."""
    ran = pivot_ok(project, "repro", "--json")
    completed = [
        (event["stage"], event["status"])
        for line in ran.stdout.splitlines()
        if line.startswith("{") and (event := _parsed(line, ran))["type"] == "stage_complete"
    ]
    statuses = dict(completed)
    assert len(statuses) == len(completed), (
        f"pivot repro must complete each stage once:\n{shown(ran)}"
    )
    return statuses, shown(ran)


def repro_status(project: Project, stage: str) -> tuple[str, str]:
    """What `pivot repro --json` did to `stage` (`repro_statuses`), and the command's output."""
    statuses, output = repro_statuses(project)
    assert stage in statuses, f"pivot repro must complete {stage} once:\n{output}"
    return statuses[stage], output


def stage_statuses(project: Project) -> tuple[dict[str, str], str]:
    """Each stage's status in one `pivot status --json`, by the stage's name, and the command's
    output."""
    entries, output = _stage_entries(project, "status", "--json")
    return {name: entry["status"] for name, entry in entries.items()}, output


def stage_status(project: Project, stage: str) -> tuple[str, str]:
    """`stage`'s status in `pivot status --json` (`stage_statuses`), and the command's output."""
    statuses, output = stage_statuses(project)
    assert stage in statuses, f"pivot status must report {stage} once:\n{output}"
    return statuses[stage], output


def stage_code_changes(project: Project) -> tuple[dict[str, list[str]], str]:
    """The key of each of each stage's `code_changes` in one `pivot status --explain --json`, in
    the order it reports them, by the stage's name, and the command's output."""
    entries, output = _stage_entries(project, "status", "--explain", "--json")
    changes = {
        name: [change["key"] for change in entry["code_changes"]] for name, entry in entries.items()
    }
    return changes, output


def code_changes(project: Project, stage: str) -> tuple[list[str], str]:
    """The key of each of `stage`'s `code_changes` in `pivot status --explain --json`
    (`stage_code_changes`), in the order it reports them, and the command's output."""
    changes, output = stage_code_changes(project)
    assert stage in changes, f"pivot status must report {stage} once:\n{output}"
    return changes[stage], output


def assert_runs_and_writes(project: Project, output: str, command: str = "`pivot repro`") -> str:
    """`pivot repro --json` (`repro_status`), which `command` names in a failure, runs `STAGE`,
    which writes `output` to `OUTPUT`; returns the command's output."""
    status, shown = repro_status(project, STAGE)
    assert status == "ran", f"{command} must run {STAGE}:\n{shown}"
    written = (project.root / OUTPUT).read_text()
    assert written == output, f"{STAGE} must write {output!r}, not {written!r}"
    return shown


def assert_runs_with_no_refusal(project: Project, output: str) -> dict[str, str]:
    """`pivot repro` runs `STAGE`, which writes `output` (`assert_runs_and_writes`), and refuses
    and warns of nothing (its output holds no `REFUSAL_FIX`); returns the lock's code manifest
    (`lock_manifest`)."""
    shown = assert_runs_and_writes(project, output)
    assert REFUSAL_FIX not in shown, f"`pivot repro` must refuse and warn of nothing:\n{shown}"
    return lock_manifest(project, STAGE)


def assert_ran_then_cached(project: Project, output: str) -> None:
    """The first `pivot repro` runs `STAGE`, which writes `output` (`assert_runs_and_writes`);
    `pivot status --json` then reports it `cached`."""
    assert_runs_and_writes(project, output, "the first `pivot repro`")
    status, shown = stage_status(project, STAGE)
    assert status == "cached", f"`pivot status` must report {STAGE} cached after a run:\n{shown}"


def assert_stale_on(project: Project, stage: str, key: str) -> None:
    """`pivot status --json` reports `stage` stale, and `pivot status --explain --json` names
    exactly one code change of it, `key`."""
    status, shown = stage_status(project, stage)
    assert status == "stale", f"`pivot status` must report {stage} stale on {key}:\n{shown}"
    keys, shown = code_changes(project, stage)
    assert keys == [key], f"`pivot status --explain` must name exactly {key} changed:\n{shown}"


def assert_marks_nothing(project: Project, stage: str) -> None:
    """`pivot status --json` reports `stage` cached, and `pivot status --explain --json` names no
    code change of it."""
    status, shown = stage_status(project, stage)
    assert status == "cached", f"`pivot status` must report {stage} cached:\n{shown}"
    keys, shown = code_changes(project, stage)
    assert keys == [], f"`pivot status --explain` must name no code change:\n{shown}"


def lock_path(project: Project, stage: str) -> pathlib.Path:
    """The lock of `stage`, named `<pipeline>/<stage>`: `.pivot/stages/<pipeline>/<stage>.lock`."""
    return project.root / ".pivot" / "stages" / f"{stage}.lock"


def lock_manifest(project: Project, stage: str) -> dict[str, str]:
    """The `code_manifest` of `stage`'s lock (`lock_path`)."""
    path = lock_path(project, stage)
    document = yaml.safe_load(path.read_text())
    assert isinstance(document, dict) and "code_manifest" in document, f"{path}: {document!r}"
    manifest = document["code_manifest"]
    assert isinstance(manifest, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in manifest.items()
    ), f"{path}: code_manifest {manifest!r} must map strings to strings"
    return manifest


def dist_keys(manifest: Mapping[str, str]) -> list[str]:
    """The `dist:` keys of a code manifest, sorted."""
    return sorted(key for key in manifest if key.startswith("dist:"))


def pivot_package_dir() -> pathlib.Path:
    """The directory of the `pivot` package that this test process imported."""
    file = pivot.__file__
    assert file is not None, "the pivot package this process imported has no __file__"
    return pathlib.Path(file).resolve().parent


def shown(ran: subprocess.CompletedProcess[str]) -> str:
    """A command's standard output and standard error, for an assertion message."""
    return f"stdout:\n{ran.stdout}\nstderr:\n{ran.stderr}"


def _new_venv(venv: pathlib.Path, *, seed: bool) -> Venv:
    """Make a venv at `venv` with `uv venv --python <this interpreter>` (`seed`: with pip, `uv
    venv --seed`)."""
    _run_uv(
        venv.parent, "venv", *(["--seed"] if seed else []), "--python", sys.executable, str(venv)
    )
    python = venv / "bin" / "python"
    ran = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        env=helpers.clean_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert ran.returncode == 0, f"{python} could not name its purelib:\n{ran.stdout}{ran.stderr}"
    return Venv(python=python, site_packages=pathlib.Path(ran.stdout.strip()))


def _stage_entries(project: Project, *args: str) -> tuple[dict[str, dict[str, Any]], str]:
    """Run `pivot <args>`, a form of `pivot status --json`, in the project; each stage's one entry
    in the `stages` of the JSON document it printed, by the stage's name, and the command's
    output."""
    ran = pivot_ok(project, *args)
    stages = _parsed(ran.stdout, ran)["stages"]
    entries = {entry["name"]: entry for entry in stages}
    assert len(entries) == len(stages), f"pivot status must report each stage once:\n{shown(ran)}"
    return entries, shown(ran)


def _parsed(text: str, ran: subprocess.CompletedProcess[str]) -> Any:
    """The JSON value `text`, which command `ran` printed; the test fails, showing the command's
    output, when `text` is not JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        pytest.fail(f"pivot printed {text!r}, which is not JSON ({error}):\n{shown(ran)}")


def _remove_build_outputs(source: pathlib.Path) -> None:
    """Delete what a setuptools build leaves in the project at `source` (`build/` and each
    `*.egg-info` directory), so that its next build starts from a clean tree and no file of an
    earlier build reaches a RECORD."""
    for leftover in [source / "build", *source.glob("*.egg-info")]:
        if leftover.exists():
            shutil.rmtree(leftover)


def _uv() -> str:
    """The uv executable: the one `PIVOT_TEST_UV` names, else the `uv` on PATH. The test fails
    when neither names an executable file."""
    found = os.environ.get("PIVOT_TEST_UV") or shutil.which("uv")
    if found is None or not (pathlib.Path(found).is_file() and os.access(found, os.X_OK)):
        pytest.fail(
            "these tests need uv: set PIVOT_TEST_UV to the uv executable or put uv on PATH "
            + f"(PIVOT_TEST_UV is {os.environ.get('PIVOT_TEST_UV')!r}, uv on PATH is "
            + f"{shutil.which('uv')!r})",
            pytrace=False,
        )
    return found


def _run_uv(cwd: pathlib.Path, *args: str) -> str:
    """Run `uv <args>` in `cwd` with `helpers.clean_env()`; the test fails unless it exits 0.
    Returns what it printed."""
    return _run_checked(cwd, [_uv(), *args])


def _run_checked(cwd: pathlib.Path, command: Sequence[str]) -> str:
    """Run `command` in `cwd` with `helpers.clean_env()`; the test fails unless it exits 0.
    Returns what it printed."""
    ran = subprocess.run(
        command,
        cwd=cwd,
        env=helpers.clean_env(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    output = ran.stdout + ran.stderr
    assert ran.returncode == 0, f"`{shlex.join(command)}` exited {ran.returncode}:\n{output}"
    return output
