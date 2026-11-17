"""`pivot.installed_distributions.distribution_keys` over real installs, called in this process.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`),
installs setuptools fixtures into it (from a path, from git, a local wheel, as from a package
index, editable) and gives each reached module's origin as plain data: which files each key
hashes, read from disk once per process; what the scan of the path reads; the refusals of an
install Pivot cannot read, and which reached package directories the check of an install with no
RECORD walks, and how far, once per process. Then `fingerprint.add_distribution_keys`, which
looks up no module's origin when no install on the path can give a key, and one fingerprint,
through the registry, of a stage that imports its installed library only in its body. No fixture
is imported into this process.

The CLI tests of the same decisions are in `test_installed_distributions.py`, and the pure
functions on plain data in `test_installed_distributions_rules.py`.
"""

import collections.abc
import importlib.metadata
import importlib.util
import os
import pathlib
import py_compile
import sys

import pytest
import pytest_mock
import xxhash
from tests import helpers, installed_dists

from pivot import exceptions, fingerprint, import_origin, installed_distributions, loaded_source

_PIVOT_DIRS = (str(installed_dists.pivot_package_dir()),)

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

# A module name of 197 letters: its file, with `.py`, has a name of 200 bytes.
_LONG_MODULE = "m" * 197


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


def _helper_change_last_byte(path: pathlib.Path) -> None:
    """Change one byte of the file at `path`, its last, to another value."""
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))


def _helper_padding(length: int) -> pathlib.Path:
    """A relative path of `length` bytes: names of 254 letters, then one of 1 to 255 letters
    (NAME_MAX, the most bytes one name can hold)."""
    full, last = divmod(length - 1, 255)
    return pathlib.Path(*["d" * 254] * full, "d" * (last + 1))


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
        root, name, name, installed_dists.FIXLIB_ALONE, dependencies=(requirement,)
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
    fixlib, fixdep = installed_dists.write_fixlib_and_fixdep(src, console_script=True)
    fixgit = installed_dists.write_library(src, "fixgit", "fixgit", installed_dists.FIXLIB_ALONE)
    fixwheel = installed_dists.write_library(
        src, "fixwheel", "fixwheel", installed_dists.FIXLIB_ALONE
    )
    wheels = tmp_path / "wheels"
    installed_dists.install_directory(venv, fixdep)
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


def test_a_pyc_file_outside_pycache_is_hashed_and_the_bytecode_pip_compiles_is_not(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib`'s module `fixlib.core` ships as bytecode alone: its wheel holds `fixlib/core.pyc`
    and no `core.py`. pip installs it from its path, compiles `fixlib/__init__.py` into
    `__pycache__`, and lists both `.pyc` files in its RECORD. The key hashes `fixlib/__init__.py`
    and `fixlib/core.pyc`, read from disk: one byte of the bytecode pip compiled changed changes
    nothing, and one byte of `core.pyc` changed changes the value, so a change to a sourceless
    module is seen."""
    venv = installed_dists.make_venv(tmp_path / "venv", seed=True)
    source = installed_dists.write_library(
        tmp_path / "src",
        "Fix_Lib",
        "fixlib",
        installed_dists.FIXLIB_ALONE,
        files={"fixlib/core.py": "CORE = 1\n"},
        package_data=["*.pyc"],
    )
    core_source = source / "fixlib" / "core.py"
    py_compile.compile(str(core_source), cfile=str(source / "fixlib" / "core.pyc"), doraise=True)
    core_source.unlink()
    installed_dists.pip_install_directory(venv, source)
    site = venv.site_packages
    compiled = importlib.util.cache_from_source("fixlib/__init__.py")
    listed = sorted(
        str(path)
        for path in installed_dists.installed(venv, "Fix_Lib").files or []
        if str(path).startswith("fixlib/")
    )
    assert listed == sorted(["fixlib/__init__.py", compiled, "fixlib/core.pyc"]), (
        f"pip's RECORD must list fixlib's two modules and the bytecode pip compiled: {listed}"
    )
    origins = {"fixlib": _helper_origin(site / "fixlib" / "__init__.py")}
    hashed = {name: name for name in ("fixlib/__init__.py", "fixlib/core.pyc")}
    values = [_helper_value(site, hashed)]

    intact = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)
    _helper_change_last_byte(site / compiled)
    installed_distributions.reset()
    compiled_changed = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)
    _helper_change_last_byte(site / "fixlib" / "core.pyc")
    values.append(_helper_value(site, hashed))
    installed_distributions.reset()
    core_changed = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)

    assert values[0] != values[1], f"one byte of core.pyc must change its value: {values}"
    keyed = [
        installed_distributions.DistributionKeys({"dist:fix-lib": value}, []) for value in values
    ]
    found = [intact, compiled_changed, core_changed]
    assert found == [keyed[0], keyed[0], keyed[1]], found


def test_distribution_keys_read_the_files_once_per_path_and_pivot_dirs_until_reset(
    tmp_path: pathlib.Path,
) -> None:
    """The installed files are read once per process for one path and one set of pivot's
    directories: after an edit in place, a second call gives the value of the first. `reset`
    forgets them, and so do another path and other pivot directories; a call with no module
    reads nothing and keeps them."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
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
    idxlib = installed_dists.write_library(src, "idxlib", "idxlib", installed_dists.FIXLIB_ALONE)
    fixedit = installed_dists.write_library(src, "fixedit", "fixedit", installed_dists.FIXLIB_ALONE)
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
        src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE, dependencies=("fix-egg",)
    )
    egg = installed_dists.write_library(
        src, "Fix_Egg", "fixegg", installed_dists.FIXLIB_ALONE, dependencies=("fixdep",)
    )
    installed_dists.install_directory(
        venv, installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
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
    index, holds `company/__init__.py`, beside which Python left bytecode outside `__pycache__`
    (`legacy.pyc`, as `compileall -b` writes it); `fixother` is installed from its path into a
    second directory on the path. Each reached module with a file that no RECORD in site-packages
    lists is refused once, naming the first such file: `fixlib`'s and `fixlib.helpers`' own files,
    under the package `company` that bytecode, which the walk reads as any file outside
    `__pycache__`, at the top of the package before its subpackage `vendor`, and under the
    namespace `nsco` the file `Fix_Lib` put there. `fixother`, outside that directory, gets its
    key; a built-in module and a module nothing finds give nothing."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixlib = installed_dists.write_setuptools_project(
        src / "fixlib",
        "Fix_Lib",
        "0.1",
        packages=["fixlib", "company.vendor", "nsco.part"],
        files={
            "fixlib/__init__.py": installed_dists.FIXLIB_ALONE,
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
    fixother = installed_dists.write_library(
        src, "fixother", "fixother", installed_dists.FIXLIB_ALONE
    )
    wheels = tmp_path / "wheels"
    second = tmp_path / "second"
    installed_dists.install_directory(venv, fixlib)
    installed_dists.build_wheel(venv, company, wheels)
    installed_dists.install_from_wheels(venv, wheels, "company-core")
    installed_dists.install(venv, "--target", str(second), str(fixother))
    site = venv.site_packages
    installed_dists.damage(site / "fix_lib-0.1.dist-info" / "RECORD", None)
    core = site / "company" / "__init__.py"
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
        ("company", "company/legacy.pyc"),
        ("fixlib", "fixlib/__init__.py"),
        ("fixlib.helpers", "fixlib/helpers.py"),
        ("nsco", "nsco/part/__init__.py"),
    ]
    assert found == installed_distributions.DistributionKeys(
        {"dist:fixother": _helper_value(second, {"fixother/__init__.py": "fixother/__init__.py"})},
        sorted(
            installed_dists.UNLISTED_REFUSAL.format(module=module, file=file, site=resolved)
            for module, file in refused
        ),
    ), found


@pytest.mark.parametrize(
    "holds_the_venv",
    [
        pytest.param(False, id="beside-the-venv"),
        pytest.param(True, id="holding-the-venv"),
    ],
)
def test_a_reached_package_is_not_walked_for_a_recordless_install_in_another_path_entry(
    tmp_path: pathlib.Path, mocker: pytest_mock.MockerFixture, holds_the_venv: bool
) -> None:
    """`Fix_Lib`, installed from its path with the package `fixlib` and the namespace portion
    `nsco.part`, is reached through both. `fixother`, installed from its path into another
    directory on the path, beside the venv or holding it, has lost its RECORD. Every file under
    `fixlib` and `nsco` lies in the venv's site-packages, the deepest entry of the path above
    each, which holds no install without a RECORD, and no entry lies under either: neither
    directory is walked, `Fix_Lib` gets its key and nothing is refused."""
    other = tmp_path / "other"
    other.mkdir()
    venv = installed_dists.make_venv((other if holds_the_venv else tmp_path) / "venv")
    src = tmp_path / "src"
    fixlib = installed_dists.write_setuptools_project(
        src / "fixlib",
        "Fix_Lib",
        "0.1",
        packages=["fixlib", "nsco.part"],
        files={
            "fixlib/__init__.py": installed_dists.FIXLIB_ALONE,
            "nsco/part/__init__.py": "PART = 1\n",
        },
    )
    fixother = installed_dists.write_library(
        src, "fixother", "fixother", installed_dists.FIXLIB_ALONE
    )
    installed_dists.install_directory(venv, fixlib)
    installed_dists.install(venv, "--target", str(other), str(fixother))
    installed_dists.damage(other / "fixother-0.1.dist-info" / "RECORD", None)
    site = venv.site_packages
    files_under = mocker.spy(installed_distributions, "_files_under")
    origins = {
        "fixlib": _helper_origin(site / "fixlib" / "__init__.py"),
        "nsco": ("namespace", os.path.realpath(site / "nsco")),
    }

    found = installed_distributions.distribution_keys(origins, [str(site), str(other)], _PIVOT_DIRS)

    hashed = {name: name for name in ("fixlib/__init__.py", "nsco/part/__init__.py")}
    assert found == installed_distributions.DistributionKeys(
        {"dist:fix-lib": _helper_value(site, hashed)}, []
    ), found
    walked = [call.args[0] for call in files_under.call_args_list]
    assert walked == [], f"no reached directory may be walked: {walked}"


def test_a_reached_package_is_walked_once_and_only_until_each_recordless_entry_has_its_file(
    tmp_path: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """`Fix_Lib`, installed from its path with the namespace portion `nsco.part`, has lost its
    RECORD, and a stray `nsco/stray.py` lies beside `nsco/part`. `company-core`, installed as
    from an index, holds `company/__init__.py`, and its directory `company/vendored` is on the
    path too, with `fixother`, installed there from its path, which has lost its RECORD. The walk
    of the reached package `company`, whose own file a RECORD lists, finds uv's `.lock`, the one
    file at the top of `company/vendored`, an entry of the path under `company`. The walk of the
    reached namespace `nsco` stops at `stray.py`, the one file at its top, which no RECORD lists,
    and resolves nothing under `part`. Each module is refused once, and a second call gives the
    same refusals and walks nothing again."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixlib = installed_dists.write_setuptools_project(
        src / "fixlib",
        "Fix_Lib",
        "0.1",
        packages=["nsco.part"],
        files={"nsco/part/__init__.py": "PART = 1\n"},
    )
    company = installed_dists.write_setuptools_project(
        src / "company",
        "company-core",
        "0.1",
        packages=["company"],
        files={"company/__init__.py": "CORE = 1\n"},
    )
    fixother = installed_dists.write_library(
        src, "fixother", "fixother", installed_dists.FIXLIB_ALONE
    )
    wheels = tmp_path / "wheels"
    installed_dists.install_directory(venv, fixlib)
    installed_dists.build_wheel(venv, company, wheels)
    installed_dists.install_from_wheels(venv, wheels, "company-core")
    site = venv.site_packages
    vendored = site / "company" / "vendored"
    installed_dists.install(venv, "--target", str(vendored), str(fixother))
    installed_dists.damage(site / "fix_lib-0.1.dist-info" / "RECORD", None)
    installed_dists.damage(vendored / "fixother-0.1.dist-info" / "RECORD", None)
    (site / "nsco" / "stray.py").write_text("STRAY = 1\n")
    top_files = sorted(child.name for child in vendored.iterdir() if child.is_file())
    assert top_files == [".lock"], f"uv must leave .lock alone at the top of {vendored}"
    files_under = mocker.spy(installed_distributions, "_files_under")
    real_path = mocker.spy(installed_distributions, "_real_path")
    nsco = os.path.realpath(site / "nsco")
    origins = {
        "company": _helper_origin(site / "company" / "__init__.py"),
        "nsco": ("namespace", nsco),
    }
    path = [str(site), str(vendored)]

    first = installed_distributions.distribution_keys(origins, path, _PIVOT_DIRS)
    walked_first = [call.args[0] for call in files_under.call_args_list]
    again = installed_distributions.distribution_keys(origins, path, _PIVOT_DIRS)

    inner = vendored.resolve()
    unlisted_lock = (
        f"Module 'company' ({inner}/.lock) lies in {inner}, where distribution 'fixother' "
        f"({inner}/fixother-0.1.dist-info), installed from a local directory (dir_info), has no "
        "RECORD file, and no RECORD file there lists that file. Pivot cannot tell whether the "
        "stage runs that distribution's code, so the stage could skip a run after it changes. "
        "Fix: reinstall 'fixother'."
    )
    unlisted_stray = installed_dists.UNLISTED_REFUSAL.format(
        module="nsco", file="nsco/stray.py", site=site.resolve()
    )
    expected = installed_distributions.DistributionKeys({}, sorted([unlisted_lock, unlisted_stray]))
    assert [first, again] == [expected, expected], [first, again]
    assert walked_first == [os.path.realpath(site / "company"), nsco], walked_first
    walked = [call.args[0] for call in files_under.call_args_list]
    assert walked == walked_first, f"a second call must walk nothing: {walked}"
    under_nsco = [
        call.args[0]
        for call in real_path.call_args_list
        if pathlib.Path(call.args[0]).is_relative_to(nsco)
    ]
    assert under_nsco == [f"{nsco}/stray.py"], f"the walk must stop at stray.py: {under_nsco}"


def test_reached_and_required_installs_pivot_cannot_read_are_refused(
    tmp_path: pathlib.Path,
) -> None:
    """The stage reaches `Fix_Lib` and `fixurl`, installed from their paths; `Fix_Lib` requires
    `fixdep`, whose RECORD is deleted, and `fixurl`'s `direct_url.json` is a directory. `fixurl`
    is refused as unclassified, with the error of reading the file, and `fixdep` for its missing
    RECORD; `Fix_Lib` gets its key."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = installed_dists.write_fixlib_and_fixdep(tmp_path / "src")
    fixurl = installed_dists.write_library(
        tmp_path / "src", "fixurl", "fixurl", installed_dists.FIXLIB_ALONE
    )
    installed_dists.install(venv, str(fixlib), str(fixdep), str(fixurl))
    site = venv.site_packages
    installed_dists.damage(site / "fixdep-0.1.dist-info" / "RECORD", None)
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
        sorted([installed_dists.RECORDLESS_DEPENDENCY_REFUSAL.format(site=resolved), unclassified]),
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
        venv, installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
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
        installed_dists.FIXLIB_ALONE,
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
    fixlib = installed_dists.write_library(src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE)
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
    installed_dists.install(venv, str(fixlib), str(fixdep))
    site = venv.site_packages
    record = site / "fix_lib-0.1.dist-info" / "RECORD"
    record.write_bytes(record.read_bytes() + b"fixlib/\x00.py,,\n")
    installed_dists.damage(site / "fixdep-0.1.dist-info" / "RECORD", None)
    origins = {"fixlib": _helper_origin(site / "fixlib" / "__init__.py")}

    found = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)

    files = {"fixlib/\x00.py": None, "fixlib/__init__.py": "fixlib/__init__.py"}
    assert found == installed_distributions.DistributionKeys(
        {"dist:fix-lib": _helper_value(site, files)}, []
    ), found


@pytest.mark.parametrize(
    ("module", "file"),
    [
        pytest.param("fixlib", "fixlib/__init__.py", id="package-walked"),
        pytest.param(f"fixlib.{_LONG_MODULE}", f"fixlib/{_LONG_MODULE}.py", id="module-own-file"),
    ],
)
def test_a_listed_file_of_path_max_bytes_or_more_is_owned_and_not_refused(
    tmp_path: pathlib.Path, module: str, file: str
) -> None:
    """`Fix_Lib`, installed from its path with the module `fixlib.<197 letters>`, and `fixdep`,
    which has lost its RECORD, lie in a venv that then moves into a directory so deep that the
    module's file, which `Fix_Lib`'s RECORD lists, is PATH_MAX bytes long in site-packages: a
    name of 200 bytes in a directory shorter than PATH_MAX, as a moved tree can leave it. The
    reached package `fixlib`, whose directory is walked for `fixdep` and holds that file, and the
    reached module whose own file it is, are not refused for a file that no RECORD lists:
    `Fix_Lib` owns the file and gets its key, which hashes the file as null, as `os.open` refuses
    a path of PATH_MAX bytes."""
    path_max = os.pathconf("/", "PC_PATH_MAX")
    root = tmp_path.resolve()
    venv = installed_dists.make_venv(root / "venv")
    src = root / "src"
    listed = f"fixlib/{_LONG_MODULE}.py"
    fixlib = installed_dists.write_library(
        src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE, files={listed: "LONG = 1\n"}
    )
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
    installed_dists.install(venv, str(fixlib), str(fixdep))
    installed_dists.damage(venv.site_packages / "fixdep-0.1.dist-info" / "RECORD", None)
    inside = venv.site_packages.relative_to(root / "venv")
    tail = f"/venv/{inside}/{listed}"  # the file's path below the padding
    padding = _helper_padding(path_max - len(os.fsencode(f"{root}/")) - len(os.fsencode(tail)))
    moved = root / padding / "venv"
    moved.parent.mkdir(parents=True)
    (root / "venv").rename(moved)
    site = moved / inside
    lengths = [len(os.fsencode(f"{site}/{path}")) for path in (listed, "fixlib")]
    assert lengths[0] == path_max > lengths[1], f"only {listed} may be PATH_MAX long: {lengths}"
    origins = {module: _helper_origin(site / file)}

    found = installed_distributions.distribution_keys(origins, [str(site)], _PIVOT_DIRS)

    files = {"fixlib/__init__.py": "fixlib/__init__.py", listed: None}
    assert found == installed_distributions.DistributionKeys(
        {"dist:fix-lib": _helper_value(site, files)}, []
    ), found


def test_a_refusal_of_a_reached_install_fails_the_fingerprint_through_the_registry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Fix_Lib`, whose `direct_url.json` is not a JSON object, is installed in a site directory
    on this process's path, and the stage imports its `fixlib` in its body, which no fingerprint
    runs. The registry's fingerprint of the stage fails with the refusal, which names the
    distribution after the stage's name, wrapped as every refusal is (`fingerprinting failed`, so
    the stage is named twice, as for a non-scalar capture), and `fixlib` was not imported."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
    installed_dists.install_directory(venv, source)
    installed_dists.damage(venv.site_packages / "fix_lib-0.1.dist-info" / "direct_url.json", "[]")
    monkeypatch.syspath_prepend(str(venv.site_packages))
    pipeline = helpers.discover_project(tmp_path, monkeypatch, {}, _BODY_IMPORT_PIPELINE)

    with pytest.raises(exceptions.PivotError) as raised:
        pipeline.ensure_fingerprint(installed_dists.STAGE)

    refusal = installed_dists.UNCLASSIFIED_REFUSAL.format(site=venv.site_packages.resolve())
    assert str(raised.value) == (
        f"Stage '{installed_dists.STAGE}': fingerprinting failed: "
        + f"{installed_dists.STAGE_PREFIX}{refusal}"
    ), raised.value
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
    fixlib, fixdep = installed_dists.write_fixlib_and_fixdep(tmp_path / "src")
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
        venv, installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
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
        src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE, dependencies=("mid",)
    )
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
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
        venv,
        installed_dists.write_library(src, "idxlib", "idxlib", installed_dists.FIXLIB_ALONE),
        wheels,
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
        venv, installed_dists.write_library(src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE)
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
