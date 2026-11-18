"""`pivot.installed_distributions.distribution_keys` over real installs, called in this process.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`),
installs setuptools fixtures into it (from a path, from git, a local wheel, as from a package
index, editable) and gives each reached module's origin and file as its import located it as
plain data: which files each key hashes, read from disk once per process; what the scan of the
path reads; the refusals of an install Pivot cannot read, and which reached package directories
the check of an install with no RECORD walks, and how far, once per process. Then
`fingerprint.add_distribution_keys`, which looks up no module's origin when no install on the
path can give a key, and the fingerprint, through the registry, of a stage that imports its
installed library only in its body, whose error names the stage once before its refusals. Last,
over files uv links into its cache (`--link-mode symlink`) and the editable layouts uv and flit's
`install --symlink` make:
`installed_distributions.listed_by_editable`, which reads only the RECORDs of one directory's
editable installs and builds no index of the candidates; `fingerprint.is_library_file`, which
asks it only about a file linked under a library path; the keys of a namespace package and of a
package that holds another install's file, which are those of a venv whose files are copies; and
the keys and refusals of module and package files linked by hand into `site-packages`, from the
project root or from another entry of the path, laid out by hand with those files copied or
linked into a cache, each module given as `fingerprint.reached_module_of` gives it. No fixture
is imported into this process.

The CLI tests of the same decisions are in `test_installed_distributions.py` and
`test_installed_distributions_link_mode.py`, and the pure functions on plain data in
`test_installed_distributions_rules.py`.
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

# A stage that imports, in its body, the installed `fixlib` and the project's module `unloaded`,
# which discovery does not load, and the refusal of that import, which does not start with the
# stage's name.
_UNLOADED_BODY_IMPORT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def make() -> str:
    import fixlib
    import unloaded

    return fixlib.value() + unloaded.VALUE


with Pipeline("p") as pipeline:
    make()
"""
_UNLOADED_BODY_IMPORT_REFUSAL = (
    "'make' imports user module 'unloaded' in its body, but pipeline discovery did not load it, "
    "so Pivot cannot fingerprint it without running it. Fix: import it at module level (or in "
    "pipeline.py) so discovery loads it, or move the value into the stage's StageParams."
)

# A module name of 197 letters: its file, with `.py`, has a name of 200 bytes.
_LONG_MODULE = "m" * 197


def _helper_namespace(directory: pathlib.Path) -> installed_distributions.ReachedModule:
    """What `fingerprint.add_distribution_keys` gives of a namespace package whose one directory
    is `directory`: the origin `import_origin.of_module` gives it, and no file."""
    return installed_distributions.ReachedModule(("namespace", os.path.realpath(directory)), ())


def _helper_real(path: pathlib.Path) -> str:
    """`path` with its directory resolved and its last name kept
    (`fingerprint.with_real_directory`): for a file in a site directory, the file
    `fingerprint.judged_file` gives, in which form `fingerprint.is_library_file` asks
    `listed_by_editable` about it and `fingerprint.add_distribution_keys` gives the first place of
    a module's file."""
    return fingerprint.with_real_directory(str(path))


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


# The two layouts of a site directory laid out by hand that a test compares: its installed files
# copies, as `--link-mode copy` leaves them, and links into a cache, as `--link-mode symlink` does.
_LAYOUTS = ("copied", "linked-into-a-cache")


def _helper_recordless_refusal(module: str, form: installed_distributions.InstallForm) -> str:
    """The refusal of a reached module `module` whose file `{site}/<module>.py` no RECORD lists
    beside distribution `module`, installed by hand `form`, which has lost its RECORD
    (`_helper_install_by_hand`); `{site}` stays in the message, as `_helper_written_site` writes
    it."""
    site = "{site}"
    return installed_dists.UNLISTED_FILE_REFUSAL.format(
        module=module,
        file=f"{site}/{module}.py",
        site=site,
        name=module,
        dist_info=f"{site}/{module}-0.1.dist-info",
        form=form.value,
        problem="has no RECORD file",
    )


# The refusal of `gitmod`, installed by hand as from git, which has lost its RECORD.
_RECORDLESS_GITMOD_REFUSAL = _helper_recordless_refusal(
    "gitmod", installed_distributions.InstallForm.VCS
)

# The refusal of the reached module `mod`, whose file `{site}/mod.py` no RECORD lists, beside
# `myproj.egg-info` in `{site}` (`_helper_project_egg_info`), as `_helper_written_site` writes it.
_MYPROJ_MOD_REFUSAL = installed_dists.UNLISTED_FILE_REFUSAL.format(
    module="mod",
    file="{site}/mod.py",
    site="{site}",
    name="myproj",
    dist_info="{site}/myproj.egg-info",
    form=installed_distributions.InstallForm.INDEX.value,
    problem="has no RECORD file",
)


def _helper_install_by_hand(
    root: pathlib.Path,
    layout: str,
    name: str,
    files: collections.abc.Mapping[str, str],
    direct_url: bytes | None,
    *,
    entry: str = "venv/site-packages",
) -> pathlib.Path:
    """Install distribution `name` 0.1 by hand into the directory `entry` under `root` as uv
    installs one from a package index (`installed_dists.index_install_by_hand`), its files copies
    in the `copied` layout and links into `root/cache` in the `linked-into-a-cache` one, with
    `direct_url` written as its `direct_url.json`, a file, as uv writes it, when given (for a git
    install, `installed_dists.UV_GIT_URL`). Returns that directory, its site directory."""
    site = root / entry
    cache = None if layout == "copied" else root / "cache"
    installed_dists.index_install_by_hand(site, name, files, cache=cache)
    if direct_url is not None:
        (site / f"{name}-0.1.dist-info" / "direct_url.json").write_bytes(direct_url)
    return site


def _helper_link_by_hand(
    directory: pathlib.Path, site: pathlib.Path, name: str, layout: str
) -> None:
    """Make `directory/<name>` a link by hand to `site/<name>`, which is a copy in the `copied`
    layout and a link into the cache in the other (the test fails otherwise); a directory
    `<name>` lies in is a real directory under `directory`."""
    (directory / name).parent.mkdir(parents=True, exist_ok=True)
    (directory / name).symlink_to(site / name)
    linked = (site / name).is_symlink()
    assert linked == (layout != "copied"), f"{site / name} in the {layout} layout: link {linked}"


def _helper_project_egg_info(project: pathlib.Path, requirement: str) -> None:
    """Write `project/myproj.egg-info` as a setuptools build leaves it beside a project's code,
    with no RECORD: a PKG-INFO that names `requirement` in a `Requires-Dist` header, and a
    `requires.txt` that names it too. The test fails unless `importlib.metadata` reads that one
    requirement there."""
    egg_info = project / "myproj.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text(
        f"Metadata-Version: 2.4\nName: myproj\nVersion: 0.1\nRequires-Dist: {requirement}\n"
    )
    (egg_info / "requires.txt").write_text(f"{requirement}\n")
    requires = importlib.metadata.Distribution.at(egg_info).requires
    assert requires == [requirement], f"{egg_info} must require {requirement}: {requires}"


def _helper_recordless_root_install(root: pathlib.Path, layout: str) -> None:
    """Install `projlib` (`projlib.py`) by hand into `root/project`, the project root, in the
    `layout` (`_helper_install_by_hand`), from a local directory (its `direct_url.json` is
    `installed_dists.UV_DIRECTORY_URL`), and delete its RECORD: a distribution with no RECORD in
    the project root that Pivot looks at, which `myproj.egg-info`, the project's own metadata,
    is not."""
    entry = _helper_install_by_hand(
        root,
        layout,
        "projlib",
        {"projlib.py": "PROJ = 1\n"},
        installed_dists.UV_DIRECTORY_URL,
        entry="project",
    )
    installed_dists.damage(entry / "projlib-0.1.dist-info" / "RECORD", None)


def _helper_written_site(
    found: installed_distributions.DistributionKeys, site: pathlib.Path
) -> installed_distributions.DistributionKeys:
    """`found` with the resolved path of the site directory `site` written `{site}` in each
    refusal, so two layouts laid out in two directories compare."""
    real = str(site.resolve())
    return installed_distributions.DistributionKeys(
        found.keys, [refusal.replace(real, "{site}") for refusal in found.refusals]
    )


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
    modules = {
        module: installed_dists.module_loaded_from(site / module / "__init__.py")
        for module in ("fixlib", "fixgit", "fixwheel")
    }

    found = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)

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
    modules = {"fixlib": installed_dists.module_loaded_from(site / "fixlib" / "__init__.py")}
    hashed = {name: name for name in ("fixlib/__init__.py", "fixlib/core.pyc")}
    values = [_helper_value(site, hashed)]

    intact = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)
    _helper_change_last_byte(site / compiled)
    installed_distributions.reset()
    compiled_changed = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)
    _helper_change_last_byte(site / "fixlib" / "core.pyc")
    values.append(_helper_value(site, hashed))
    installed_distributions.reset()
    core_changed = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)

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
    modules = {"fixlib": installed_dists.module_loaded_from(module)}
    path = [str(site)]
    longer_path = [*path, str(tmp_path / "missing")]
    other_dirs = [*_PIVOT_DIRS, str(tmp_path / "other")]
    files = {"fixlib/__init__.py": "fixlib/__init__.py"}
    values = list[str]()

    first = installed_distributions.distribution_keys(modules, path, _PIVOT_DIRS)
    values.append(_helper_value(site, files))
    installed_dists.replace_once(module, '"value-1"', '"value-2"')
    again = installed_distributions.distribution_keys(modules, path, _PIVOT_DIRS)
    installed_distributions.reset()
    after_reset = installed_distributions.distribution_keys(modules, path, _PIVOT_DIRS)
    values.append(_helper_value(site, files))
    installed_dists.replace_once(module, '"value-2"', '"value-3"')
    other_path = installed_distributions.distribution_keys(modules, longer_path, _PIVOT_DIRS)
    values.append(_helper_value(site, files))
    installed_dists.replace_once(module, '"value-3"', '"value-4"')
    other_pivot = installed_distributions.distribution_keys(modules, longer_path, other_dirs)
    values.append(_helper_value(site, files))
    no_module = installed_distributions.distribution_keys({}, path, _PIVOT_DIRS)
    installed_dists.replace_once(module, '"value-4"', '"value-5"')
    kept = installed_distributions.distribution_keys(modules, longer_path, other_dirs)

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
    modules = {
        "fixedit": installed_dists.module_loaded_from(fixedit / "fixedit" / "__init__.py"),
        "idxlib": installed_dists.module_loaded_from(site / "idxlib" / "__init__.py"),
    }

    found = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)

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
        {"fixlib": installed_dists.module_loaded_from(site / "fixlib" / "__init__.py")},
        path,
        _PIVOT_DIRS,
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
    modules = {
        "company": installed_dists.module_loaded_from(core),
        "fixlib": installed_dists.module_loaded_from(site / "fixlib" / "__init__.py"),
        "fixlib.helpers": installed_dists.module_loaded_from(site / "fixlib" / "helpers.py"),
        "fixother": installed_dists.module_loaded_from(second / "fixother" / "__init__.py"),
        "missing": installed_distributions.ReachedModule(None, ()),
        "nsco": _helper_namespace(site / "nsco"),
        "sys": installed_distributions.ReachedModule(("built-in",), ()),
    }

    found = installed_distributions.distribution_keys(
        modules, [str(site), str(second)], _PIVOT_DIRS
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
    modules = {
        "fixlib": installed_dists.module_loaded_from(site / "fixlib" / "__init__.py"),
        "nsco": _helper_namespace(site / "nsco"),
    }

    found = installed_distributions.distribution_keys(modules, [str(site), str(other)], _PIVOT_DIRS)

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
    modules = {
        "company": installed_dists.module_loaded_from(site / "company" / "__init__.py"),
        "nsco": _helper_namespace(site / "nsco"),
    }
    path = [str(site), str(vendored)]

    first = installed_distributions.distribution_keys(modules, path, _PIVOT_DIRS)
    walked_first = [call.args[0] for call in files_under.call_args_list]
    again = installed_distributions.distribution_keys(modules, path, _PIVOT_DIRS)

    inner = vendored.resolve()
    unlisted_lock = installed_dists.UNLISTED_FILE_REFUSAL.format(
        module="company",
        file=f"{inner}/.lock",
        site=inner,
        name="fixother",
        dist_info=f"{inner}/fixother-0.1.dist-info",
        form=installed_distributions.InstallForm.LOCAL_DIRECTORY.value,
        problem="has no RECORD file",
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
    modules = {
        module: installed_dists.module_loaded_from(site / module / "__init__.py")
        for module in ("fixlib", "fixurl")
    }

    found = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)

    resolved = site.resolve()
    dist_info = f"{resolved}/fixurl-0.1.dist-info"
    unclassified = installed_dists.DISTRIBUTION_REFUSAL.format(
        name="fixurl",
        dist_info=dist_info,
        how=installed_dists.REACHED,
        reason="is installed in a form Pivot cannot classify: its direct_url.json cannot be read "
        + f"([Errno 21] Is a directory: '{dist_info}/direct_url.json')",
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
    modules = {"pivot": installed_dists.module_loaded_from(site / "pivot" / "__init__.py")}

    as_pivot = installed_distributions.distribution_keys(
        modules, [str(site)], [os.path.realpath(site / "pivot")]
    )
    as_library = installed_distributions.distribution_keys(
        modules, [str(site)], [str(tmp_path / "other")]
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
    modules = {"fixlib": installed_dists.module_loaded_from(site / "fixlib" / "__init__.py")}
    files: dict[str, str | None] = {
        "fixlib/__init__.py": "fixlib/__init__.py",
        "fixlib/extra.py": "fixlib/extra.py",
    }
    intact = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)
    intact_value = _helper_value(site, files)
    damage(site / "fixlib" / "extra.py")
    installed_distributions.reset()

    damaged = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)

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
    modules = {"fixlib": installed_dists.module_loaded_from(site / "fixlib" / "__init__.py")}

    found = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)

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
    modules = {module: installed_dists.module_loaded_from(site / file)}

    found = installed_distributions.distribution_keys(modules, [str(site)], _PIVOT_DIRS)

    files = {"fixlib/__init__.py": "fixlib/__init__.py", listed: None}
    assert found == installed_distributions.DistributionKeys(
        {"dist:fix-lib": _helper_value(site, files)}, []
    ), found


@pytest.mark.parametrize(
    ("pipeline_text", "damaged", "refusals"),
    [
        pytest.param(
            _BODY_IMPORT_PIPELINE,
            {"fix_lib-0.1.dist-info/direct_url.json": "[]"},
            [installed_dists.UNCLASSIFIED_REFUSAL],
            id="one-refusal",
        ),
        pytest.param(
            _BODY_IMPORT_PIPELINE,
            {"fix_lib-0.1.dist-info/direct_url.json": "[]", "fixdep-0.1.dist-info/RECORD": None},
            [installed_dists.UNCLASSIFIED_REFUSAL, installed_dists.RECORDLESS_DEPENDENCY_REFUSAL],
            id="two-refusals",
        ),
        pytest.param(
            _UNLOADED_BODY_IMPORT_PIPELINE,
            {"fix_lib-0.1.dist-info/direct_url.json": "[]"},
            [_UNLOADED_BODY_IMPORT_REFUSAL, installed_dists.UNCLASSIFIED_REFUSAL],
            id="a-refusal-that-does-not-name-the-stage-first",
        ),
    ],
)
def test_a_refusal_of_a_reached_install_fails_the_fingerprint_through_the_registry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_text: str,
    damaged: dict[str, str | None],
    refusals: list[str],
) -> None:
    """`Fix_Lib` and its requirement `fixdep` are installed in a site directory on this process's
    path, and the stage imports `fixlib` in its body, which no fingerprint runs, and in
    `_UNLOADED_BODY_IMPORT_PIPELINE` the project's module `unloaded`, which discovery did not
    load. Then `damaged` (each file, relative to site-packages, overwritten with its text or
    deleted): `Fix_Lib`'s `direct_url.json` made `[]`, which is no JSON object, and in
    `two-refusals` `fixdep`'s RECORD deleted too. The registry's fingerprint of the stage fails
    with an error that names the stage once, first (`STAGE_FAILED`), then each of `refusals` in
    order, a blank line between two. A refusal of a distribution starts with the stage's name, as
    the warning unsafe fingerprinting gives in its place does, and is given here without it; the
    refusal of the body import, which does not start with it, keeps all of its text. `fixlib` was
    not imported."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    fixlib, fixdep = installed_dists.write_fixlib_and_fixdep(tmp_path / "src")
    installed_dists.install_directory(venv, fixdep)
    installed_dists.install_directory(venv, fixlib)
    for path, text in damaged.items():
        installed_dists.damage(venv.site_packages / path, text)
    monkeypatch.syspath_prepend(str(venv.site_packages))
    pipeline = helpers.discover_project(
        tmp_path, monkeypatch, {"unloaded.py": 'VALUE = ":unloaded"\n'}, pipeline_text
    )

    with pytest.raises(exceptions.PivotError) as raised:
        pipeline.ensure_fingerprint(installed_dists.STAGE)

    site = venv.site_packages.resolve()
    expected = installed_dists.STAGE_FAILED + "\n\n".join(
        refusal.format(site=site) for refusal in refusals
    )
    assert str(raised.value) == expected, raised.value
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
    modules = {
        "elsewhere": installed_dists.module_loaded_from(tmp_path / "elsewhere.py"),
        "fixlib": installed_dists.module_loaded_from(site / "fixlib" / "__init__.py"),
    }

    found = installed_distributions.distribution_keys(modules, [str(root), str(site)], _PIVOT_DIRS)

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
def test_a_reached_module_of_a_source_tree_beside_its_egg_info_with_no_record_is_not_refused(
    tmp_path: pathlib.Path, without_requires_dist: bool
) -> None:
    """The source tree of `mid`, on the path after the venv's site-packages, holds the package
    `mid` and the `mid.egg-info` its setuptools build left, with no RECORD and no
    `direct_url.json` (the package-index form), which requires `fixdep`, installed from its path,
    in a `Requires-Dist` header of its PKG-INFO, or in its `requires.txt` alone. The stage reaches
    `mid`, whose file no RECORD lists. The source tree is no `site-packages` or `dist-packages`
    directory, so `mid.egg-info` is the project's own metadata, which refuses nothing there, and
    `mid` owns no file, so it gives no key. (`fingerprint.add_distribution_keys` gives no module
    whose file is user code, as `mid/__init__.py` in a source tree is: the walk hashes it.)"""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    installed_dists.install_directory(
        venv, installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
    )
    mid = _helper_egg_info_project(
        venv, src, "mid", "fixdep", without_requires_dist=without_requires_dist
    )
    modules = {"mid": installed_dists.module_loaded_from(mid / "mid" / "__init__.py")}

    found = installed_distributions.distribution_keys(
        modules, [str(venv.site_packages), str(mid)], _PIVOT_DIRS
    )

    assert found == installed_distributions.DistributionKeys({}, []), found


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
        {"fixlib": installed_dists.module_loaded_from(site / "fixlib" / "__init__.py")},
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
    worker_spec = mocker.spy(loaded_source, "worker_spec")
    stage = {"self:make": "1", "byname:json": "", "byname:fixlib": ""}
    without_keyed = dict(stage)

    fingerprint.add_distribution_keys(without_keyed)
    calls_without_keyed = [of_module.call_args_list[:], worker_spec.call_args_list[:]]
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
    calls = [of_module.call_args_list, worker_spec.call_args_list]
    assert calls == [[mocker.call(sys.modules["json"])], [mocker.call("fixlib")]], calls


def test_listed_by_editable_reads_only_the_records_of_editable_installs_that_list_a_linked_file(
    tmp_path: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Over real installs that uv links into its cache (`--link-mode symlink`): `idxlib`,
    installed as from a package index; `Fix_Lib` from its path, with a console script, whose
    RECORD row leaves the site directory; `fixbad` from its path with its `direct_url.json` made
    `[1, 2]` (unclassified); `fixedit` installed editable, with a console script too, whose
    setuptools finder module uv links into its cache; `flitlib` as flit's `install --symlink`
    links an editable module file; a link `stray.py` that no RECORD there lists; and another
    entry on the path that holds `Fix_Lib` too, its `direct_url.json` made `[1, 2]`, and a
    `stray.py` that flit links there as an editable module file. Only an editable install's RECORD
    in a file's own entry lists it: `fixedit`'s finder module, `flitlib.py` and the other entry's
    `stray.py`; no other linked file, not site-packages' `stray.py`, whose path only the other
    entry's editable RECORD names, nor `fixlib`'s file where its link points, which no entry of
    the path holds. Those questions read the RECORD of each editable install of the files' entries
    once, however many files are asked about, and no other RECORD; they read no requirement and
    build no index of the candidates. `needs_origins` then builds the index, which reads the
    RECORDs of the candidates, of both entries, and none a second time."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    wheels = tmp_path / "wheels"
    installed_dists.build_wheel(
        venv,
        installed_dists.write_library(src, "idxlib", "idxlib", installed_dists.FIXLIB_ALONE),
        wheels,
    )
    installed_dists.install_from_wheels(venv, wheels, "idxlib", link_mode="symlink")
    site = venv.site_packages
    fixlib = installed_dists.write_library(
        src, "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE, scripts={"fix-lib": "fixlib:main"}
    )
    fixbad = installed_dists.write_library(src, "fixbad", "fixbad", installed_dists.FIXLIB_ALONE)
    fixedit = installed_dists.write_library(
        src, "fixedit", "fixedit", installed_dists.FIXLIB_ALONE, scripts={"fixedit": "fixedit:main"}
    )
    installed_dists.install_directory(venv, fixlib, link_mode="symlink")
    installed_dists.install_directory(venv, fixbad, link_mode="symlink")
    installed_dists.install_directory(venv, fixedit, editable=True, link_mode="symlink")
    installed_dists.damage(site / "fixbad-0.1.dist-info" / "direct_url.json", "[1, 2]")
    other = tmp_path / "other"
    installed_dists.install(venv, "--target", str(other), str(fixlib))
    installed_dists.damage(other / "fix_lib-0.1.dist-info" / "direct_url.json", "[1, 2]")
    path = [str(site), str(other)]
    checkout = tmp_path / "flitlib"
    checkout.mkdir()
    (checkout / "flitlib.py").write_text(installed_dists.FIXLIB_ALONE)
    installed_dists.flit_symlink_install(site, checkout, "flitlib.py")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "stray.py").write_text("STRAY = 1\n")
    (site / "stray.py").symlink_to(tmp_path / "elsewhere" / "stray.py")
    other_checkout = tmp_path / "otherflit"
    other_checkout.mkdir()
    (other_checkout / "stray.py").write_text("STRAY = 2\n")
    installed_dists.flit_symlink_install(other, other_checkout, "stray.py")
    finders = sorted(site.glob("__editable___fixedit_*_finder.py"))
    assert len(finders) == 1, f"setuptools must write one finder module for fixedit: {finders}"
    files = {
        "idxlib": site / "idxlib" / "__init__.py",
        "fixlib": site / "fixlib" / "__init__.py",
        "fixbad": site / "fixbad" / "__init__.py",
        "fixedit-finder": finders[0],
        "flitlib": site / "flitlib.py",
        "stray": site / "stray.py",
        "other-stray": other / "stray.py",
    }
    unlinked = sorted(name for name, file in files.items() if not file.is_symlink())
    assert unlinked == [], f"each file asked about must be a symbolic link: {unlinked}"
    installed_distributions.reset()
    read_record = mocker.spy(installed_distributions, "_read_record")
    read_requirements = mocker.spy(installed_distributions, "_read_requirements")
    build_index = mocker.spy(installed_distributions, "_build_index")

    found = {
        name: installed_distributions.listed_by_editable(_helper_real(file), path, _PIVOT_DIRS)
        for name, file in files.items()
    }
    found["fixlib-target"] = installed_distributions.listed_by_editable(
        os.path.realpath(files["fixlib"]), path, _PIVOT_DIRS
    )
    real_tmp = os.path.realpath(tmp_path)
    read_for_files = sorted(
        os.path.relpath(call.args[0].dist_info, real_tmp) for call in read_record.call_args_list
    )
    calls_for_files = (read_requirements.call_count, build_index.call_count)
    needs_origins = installed_distributions.needs_origins(path, _PIVOT_DIRS)

    assert found == {
        "idxlib": False,
        "fixlib": False,
        "fixbad": False,
        "fixedit-finder": True,
        "flitlib": True,
        "stray": False,
        "other-stray": True,
        "fixlib-target": False,
    }, found
    in_site = os.path.relpath(os.path.realpath(site), real_tmp)
    assert read_for_files == sorted(
        [
            f"{in_site}/fixedit-0.1.dist-info",
            f"{in_site}/{installed_dists.FLIT_DIST_INFO}",
            f"other/{installed_dists.FLIT_DIST_INFO}",
        ]
    ), (
        "only the RECORD of each editable install of the files' entries may be read, once: "
        + f"{read_for_files}"
    )
    assert calls_for_files == (0, 0), (
        f"no requirement may be read and no index built for a linked file: {calls_for_files}"
    )
    assert needs_origins is True, "Fix_Lib, installed from its path, can give a key"
    assert build_index.call_count == 1, f"the index must be built once: {build_index.call_count}"
    read = sorted(
        os.path.relpath(call.args[0].dist_info, real_tmp) for call in read_record.call_args_list
    )
    assert read == sorted(
        [
            f"{in_site}/fix_lib-0.1.dist-info",
            f"{in_site}/fixbad-0.1.dist-info",
            f"{in_site}/fixedit-0.1.dist-info",
            f"{in_site}/{installed_dists.FLIT_DIST_INFO}",
            "other/fix_lib-0.1.dist-info",
            f"other/{installed_dists.FLIT_DIST_INFO}",
        ]
    ), f"the index must read each candidate's RECORD and no RECORD a second time: {read}"


def test_is_library_file_asks_the_records_only_about_a_file_linked_under_a_library_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: pytest_mock.MockerFixture
) -> None:
    """`fixcopy` installed from its path with `--link-mode copy`, `fixlink` with `--link-mode
    symlink`, `flitlib` as flit's `install --symlink` links a package's directory, `fixtarget`
    with `--link-mode symlink` into `--target` directory `target`, not named site-packages, and
    `stray.py`, a link made by hand in site-packages to a project file. The venv is reached
    through a symbolic link to its directory, and the worker's path is its site-packages, so
    reached, and `target`. `fingerprint.is_library_file` takes `fixcopy`'s file as library code by
    its resolved path, and `fixlink`'s, a link into uv's cache, and `stray.py` as library code,
    since no editable install's RECORD in their site directory lists them, as copies there would
    be; it takes as user code `flitlib`'s `__init__.py`, which is no link (its directory is) and
    resolves into the checkout, `fixtarget`'s file, a link whose path is no library path, and the
    project file; and a code object compiled from each is user source (`_is_user_source_file`)
    exactly where the file is user code. It asks the RECORDs only about `fixlink`'s file and
    `stray.py`, with their directory resolved, which the scan's resolved entry of the path
    holds."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    src = tmp_path / "src"
    fixcopy = installed_dists.write_library(src, "fixcopy", "fixcopy", installed_dists.FIXLIB_ALONE)
    fixlink = installed_dists.write_library(src, "fixlink", "fixlink", installed_dists.FIXLIB_ALONE)
    fixtarget = installed_dists.write_library(
        src, "fixtarget", "fixtarget", installed_dists.FIXLIB_ALONE
    )
    installed_dists.install_directory(venv, fixcopy, link_mode="copy")
    installed_dists.install_directory(venv, fixlink, link_mode="symlink")
    target = tmp_path / "target"
    installed_dists.install(venv, "--target", str(target), str(fixtarget), link_mode="symlink")
    (tmp_path / "venv-link").symlink_to(tmp_path / "venv")
    site = tmp_path / "venv-link" / venv.site_packages.relative_to(tmp_path / "venv")
    checkout = tmp_path / "flitlib"
    (checkout / "flitlib").mkdir(parents=True)
    (checkout / "flitlib" / "__init__.py").write_text(installed_dists.FIXLIB_ALONE)
    installed_dists.flit_symlink_install(site, checkout, "flitlib")
    project_file = tmp_path / "project" / "helpers.py"
    project_file.parent.mkdir()
    project_file.write_text("HELP = 1\n")
    (site / "stray.py").symlink_to(project_file)
    files = {
        "fixcopy": site / "fixcopy" / "__init__.py",
        "fixlink": site / "fixlink" / "__init__.py",
        "flitlib": site / "flitlib" / "__init__.py",
        "fixtarget": target / "fixtarget" / "__init__.py",
        "project": project_file,
        "stray": site / "stray.py",
    }
    links = {name: file.is_symlink() for name, file in files.items()}
    assert links == {
        "fixcopy": False,
        "fixlink": True,
        "flitlib": False,
        "fixtarget": True,
        "project": False,
        "stray": True,
    }, links
    monkeypatch.setattr(sys, "path", [str(site), str(target)])
    asked_editable = mocker.spy(installed_distributions, "listed_by_editable")

    found = {name: fingerprint.is_library_file(str(file)) for name, file in files.items()}
    user_source = {
        name: fingerprint._is_user_source_file(str(file)) for name, file in files.items()
    }

    assert found == {
        "fixcopy": True,
        "fixlink": True,
        "flitlib": False,
        "fixtarget": False,
        "project": False,
        "stray": True,
    }, found
    assert user_source == {
        "fixcopy": False,
        "fixlink": False,
        "flitlib": True,
        "fixtarget": True,
        "project": True,
        "stray": False,
    }, user_source
    asked = [call.args[0] for call in asked_editable.call_args_list]
    real_site = os.path.realpath(venv.site_packages)
    expected = [
        os.path.join(real_site, "fixlink", "__init__.py"),
        os.path.join(real_site, "stray.py"),
    ]
    assert asked == expected, f"only fixlink's file and stray.py may be asked about: {asked}"


def test_a_namespace_package_uv_links_into_its_cache_gives_the_key_of_a_copied_one(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Ns`, whose one package is the namespace portion `nsco.part`, installed from its path
    into one venv with `--link-mode copy` and into another with `--link-mode symlink`, where each
    of its files is a link into uv's cache from a real directory. A stage that reaches the
    namespace package `nsco`, whose origin names its directory in site-packages, gets
    `dist:fix-ns` in both, with one value."""
    source = installed_dists.write_setuptools_project(
        tmp_path / "src" / "fixns",
        "Fix_Ns",
        "0.1",
        packages=["nsco.part"],
        files={"nsco/part/__init__.py": "PART = 1\n"},
    )
    copied = installed_dists.make_venv(tmp_path / "copy")
    linked = installed_dists.make_venv(tmp_path / "symlink")
    installed_dists.install_directory(copied, source, link_mode="copy")
    installed_dists.install_directory(linked, source, link_mode="symlink")
    links = [
        (venv.site_packages / "nsco" / "part" / "__init__.py").is_symlink()
        for venv in (copied, linked)
    ]
    assert links == [False, True], f"nsco/part/__init__.py must be a copy, then a link: {links}"

    found = [
        installed_distributions.distribution_keys(
            {"nsco": _helper_namespace(venv.site_packages / "nsco")},
            [str(venv.site_packages)],
            _PIVOT_DIRS,
        )
        for venv in (copied, linked)
    ]

    value = _helper_value(copied.site_packages, {"nsco/part/__init__.py": "nsco/part/__init__.py"})
    keyed = installed_distributions.DistributionKeys({"dist:fix-ns": value}, [])
    assert found == [keyed, keyed], found


def test_a_file_one_install_puts_in_anothers_package_is_reached_in_a_venv_uv_links_into_its_cache(
    tmp_path: pathlib.Path,
) -> None:
    """`company-core`, whose package is `company`, installed as from a package index, and
    `Fix_Lib`, whose one file is `company/vendor/__init__.py`, inside that package, installed
    from its path, both into one venv with `--link-mode copy` and into another with `--link-mode
    symlink`, where each file is a link into its own distribution's entry of uv's cache, from a
    real directory. A stage that reaches the package `company` (the walk records the top name of
    `import company.vendor`), whose `__init__` file the import locates in site-packages, gets
    `dist:fix-lib` in both venvs, with one value: the package's directory is the one the import
    searched, where `Fix_Lib`'s RECORD names its file, not the directory of the resolved
    `__init__` file, which lies in `company-core`'s entry of the cache. Once `Fix_Lib` has lost
    its RECORD, the stage is refused in both venvs for the file `Fix_Lib` put under the package,
    which no RECORD lists, named where it lies in site-packages."""
    src = tmp_path / "src"
    company = installed_dists.write_library(src, "company-core", "company", "CORE = 1\n")
    fixlib = installed_dists.write_setuptools_project(
        src / "fixlib",
        "Fix_Lib",
        "0.1",
        packages=["company.vendor"],
        files={"company/vendor/__init__.py": "VENDOR = 1\n"},
    )
    wheels = tmp_path / "wheels"
    link_modes = ("copy", "symlink")
    venvs = [installed_dists.make_venv(tmp_path / link_mode) for link_mode in link_modes]
    installed_dists.build_wheel(venvs[0], company, wheels)
    for venv, link_mode in zip(venvs, link_modes, strict=True):
        installed_dists.install_from_wheels(venv, wheels, "company-core", link_mode=link_mode)
        installed_dists.install_directory(venv, fixlib, link_mode=link_mode)
    vendor = "company/vendor/__init__.py"
    links = [
        [(venv.site_packages / file).is_symlink() for file in ("company/__init__.py", vendor)]
        for venv in venvs
    ]
    assert links == [[False, False], [True, True]], (
        f"the two files must be copies in the first venv and links in the second: {links}"
    )

    intact = [
        installed_distributions.distribution_keys(
            {
                "company": installed_dists.module_loaded_from(
                    venv.site_packages / "company" / "__init__.py"
                )
            },
            [str(venv.site_packages)],
            _PIVOT_DIRS,
        )
        for venv in venvs
    ]
    for venv in venvs:
        installed_dists.damage(venv.site_packages / "fix_lib-0.1.dist-info" / "RECORD", None)
    installed_distributions.reset()
    recordless = [
        installed_distributions.distribution_keys(
            {
                "company": installed_dists.module_loaded_from(
                    venv.site_packages / "company" / "__init__.py"
                )
            },
            [str(venv.site_packages)],
            _PIVOT_DIRS,
        )
        for venv in venvs
    ]

    value = _helper_value(venvs[0].site_packages, {vendor: vendor})
    keyed = installed_distributions.DistributionKeys({"dist:fix-lib": value}, [])
    assert intact == [keyed, keyed], intact
    refused = [
        installed_distributions.DistributionKeys(
            {},
            [
                installed_dists.UNLISTED_REFUSAL.format(
                    module="company", file=vendor, site=venv.site_packages.resolve()
                )
            ],
        )
        for venv in venvs
    ]
    assert recordless == refused, recordless


@pytest.mark.parametrize("egg_info", ["alone", "beside-a-recordless-egg-info"])
def test_a_project_file_linked_to_a_recordless_git_install_is_refused_as_in_a_copied_venv(
    tmp_path: pathlib.Path, egg_info: str
) -> None:
    """`gitmod`, installed by hand into `site-packages` as uv installs from git, in each layout
    (its `gitmod.py` copied, or linked into a cache), has lost its RECORD; `project/gitmod.py` is a
    link made by hand to `site-packages/gitmod.py`, and the path is the project root, then
    `site-packages`; `beside-a-recordless-egg-info`: the project root also holds
    `myproj.egg-info`, with no RECORD, which requires `gitmod`. The stage reaches `gitmod`
    through the project file, as `fingerprint.reached_module_of` gives it. Both layouts refuse it
    with one refusal, naming the module's file in `site-packages`, where Pivot judges the project
    file in both (`fingerprint.judged_file`: its link reaches `site-packages/gitmod.py`, a library
    path); in the linked layout its resolved path lies in the cache, outside every entry of the
    path. No refusal names `myproj`, since the project root is neither where Pivot judges the file
    nor where it resolves, nor where its chain of links reaches a library file, and since
    `myproj.egg-info` there is the project's own metadata, which refuses nothing in a path entry
    outside every `site-packages` and `dist-packages` directory. Judged as the import found it, in
    the project root, the linked layout would refuse nothing."""
    found = dict[str, installed_distributions.DistributionKeys]()
    for layout in _LAYOUTS:
        root = tmp_path / layout
        site = _helper_install_by_hand(
            root, layout, "gitmod", {"gitmod.py": "VALUE = 1\n"}, installed_dists.UV_GIT_URL
        )
        installed_dists.damage(site / "gitmod-0.1.dist-info" / "RECORD", None)
        project = root / "project"
        _helper_link_by_hand(project, site, "gitmod.py", layout)
        if egg_info != "alone":
            _helper_project_egg_info(project, "gitmod")
        keys = installed_distributions.distribution_keys(
            {"gitmod": installed_dists.module_loaded_from(project / "gitmod.py")},
            [str(project), str(site)],
            _PIVOT_DIRS,
        )
        found[layout] = _helper_written_site(keys, site)

    refused = installed_distributions.DistributionKeys({}, [_RECORDLESS_GITMOD_REFUSAL])
    assert found == {"copied": refused, "linked-into-a-cache": refused}, found


@pytest.mark.parametrize("record", ["lost", "kept"])
def test_a_library_file_linked_from_another_path_entry_is_checked_where_its_link_reaches(
    tmp_path: pathlib.Path, record: str
) -> None:
    """`gitmod`, installed by hand into `B/venv/site-packages` as uv installs from git, in each
    layout (its `gitmod.py` copied, or linked into a cache), with its RECORD `kept` or `lost`;
    `A/site-packages/gitmod.py` is a link made by hand to that file, and the path is
    `A/site-packages`, then `B/venv/site-packages`. The stage reaches `gitmod` through `A`'s file,
    as `fingerprint.reached_module_of` gives it. Pivot judges the file in `A`, its first library
    file, and its chain of links reaches a second one in `B`, where it lies too: in the copied
    layout `B`'s file is its resolved path, and in the linked one that lies in the cache, outside
    every entry of the path. So both layouts give the same: `kept`, the key of `gitmod` and no
    refusal; `lost`, one refusal naming the file in `B`. With only the file Pivot judged checked,
    the linked layout would find the file in `A` alone, where no distribution lacks a RECORD, and
    refuse nothing."""
    found = dict[str, installed_distributions.DistributionKeys]()
    for layout in _LAYOUTS:
        root = tmp_path / layout
        site = _helper_install_by_hand(
            root / "B", layout, "gitmod", {"gitmod.py": "VALUE = 1\n"}, installed_dists.UV_GIT_URL
        )
        if record == "lost":
            installed_dists.damage(site / "gitmod-0.1.dist-info" / "RECORD", None)
        entry = root / "A" / "site-packages"
        _helper_link_by_hand(entry, site, "gitmod.py", layout)
        keys = installed_distributions.distribution_keys(
            {"gitmod": installed_dists.module_loaded_from(entry / "gitmod.py")},
            [str(entry), str(site)],
            _PIVOT_DIRS,
        )
        found[layout] = _helper_written_site(keys, site)

    if record == "lost":
        expected = installed_distributions.DistributionKeys({}, [_RECORDLESS_GITMOD_REFUSAL])
    else:
        copied_site = tmp_path / "copied" / "B" / "venv" / "site-packages"
        value = _helper_value(copied_site, {"gitmod.py": "gitmod.py"})
        expected = installed_distributions.DistributionKeys({"dist:gitmod": value}, [])
    assert found == {"copied": expected, "linked-into-a-cache": expected}, found


def test_a_library_file_linked_to_a_recordless_install_outside_every_library_path_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """`wlib`, installed by hand from a local directory into `P/libs`, a path entry that is no
    library path (as a `--target` directory is), in each layout (its `wlib.py` copied, or linked
    into a cache), has lost its RECORD; `A/site-packages/wlib.py` is a link made by hand to that
    file, and the path is `A/site-packages`, then `P/libs`. The stage reaches `wlib` through `A`'s
    file, as `fingerprint.reached_module_of` gives it. Pivot judges the file in `A`, the first
    library file of its chain of links, and the chain then reaches `P/libs/wlib.py`, where the
    file lies too, though that is no library path: in the copied layout it is the resolved path,
    and in the linked one that lies in the cache, outside every entry of the path. So both layouts
    refuse the stage with one refusal, naming the file in `P/libs`. With only the library files
    of the chain as its places, the linked layout would find the file in `A` alone, where no
    distribution lacks a RECORD, and refuse nothing."""
    found = dict[str, installed_distributions.DistributionKeys]()
    for layout in _LAYOUTS:
        root = tmp_path / layout
        libs = _helper_install_by_hand(
            root / "P",
            layout,
            "wlib",
            {"wlib.py": "W = 1\n"},
            installed_dists.UV_DIRECTORY_URL,
            entry="libs",
        )
        installed_dists.damage(libs / "wlib-0.1.dist-info" / "RECORD", None)
        entry = root / "A" / "site-packages"
        _helper_link_by_hand(entry, libs, "wlib.py", layout)
        keys = installed_distributions.distribution_keys(
            {"wlib": installed_dists.module_loaded_from(entry / "wlib.py")},
            [str(entry), str(libs)],
            _PIVOT_DIRS,
        )
        found[layout] = _helper_written_site(keys, libs)

    refusal = _helper_recordless_refusal(
        "wlib", installed_distributions.InstallForm.LOCAL_DIRECTORY
    )
    refused = installed_distributions.DistributionKeys({}, [refusal])
    assert found == {"copied": refused, "linked-into-a-cache": refused}, found


def test_a_package_file_linked_to_a_file_another_entrys_record_lists_is_not_refused_by_the_walk(
    tmp_path: pathlib.Path,
) -> None:
    """In each layout (the installed files copied, or linked into a cache), `pkg`, whose one file
    is `pkg/__init__.py`, and `alib`, which has lost its RECORD, are installed by hand as from git
    into `A/venv/site-packages`, and `pkgplugin`, whose one file is `pkg/plugin.py`, into
    `B/venv/site-packages`, with its RECORD. `A`'s `pkg/plugin.py` is a link made by hand to
    `B`'s, and the path is `A`'s site-packages, then `B`'s. The stage reaches the package `pkg`
    through `A`'s `pkg/__init__.py`, so the check of a file that no RECORD lists walks `A`'s
    `pkg`, beside `alib`. `plugin.py` lies in `A`, where the walk lists it, and in `B`, where its
    chain of links reaches, and `pkgplugin`'s RECORD there lists it, so neither layout refuses the
    stage, and both give the key of `pkg` alone. With only where the walk lists it as its place,
    the linked layout, where the file resolves into the cache, would find it in `A` alone and
    refuse the stage, as `pkg.plugin` reached as its own module is not."""
    found = dict[str, installed_distributions.DistributionKeys]()
    for layout in _LAYOUTS:
        root = tmp_path / layout
        first = _helper_install_by_hand(
            root / "A", layout, "pkg", {"pkg/__init__.py": "PKG = 1\n"}, installed_dists.UV_GIT_URL
        )
        _helper_install_by_hand(
            root / "A", layout, "alib", {"alib.py": "ALIB = 1\n"}, installed_dists.UV_GIT_URL
        )
        installed_dists.damage(first / "alib-0.1.dist-info" / "RECORD", None)
        second = _helper_install_by_hand(
            root / "B",
            layout,
            "pkgplugin",
            {"pkg/plugin.py": "PLUGIN = 1\n"},
            installed_dists.UV_GIT_URL,
        )
        _helper_link_by_hand(first, second, "pkg/plugin.py", layout)
        found[layout] = installed_distributions.distribution_keys(
            {"pkg": installed_dists.module_loaded_from(first / "pkg" / "__init__.py")},
            [str(first), str(second)],
            _PIVOT_DIRS,
        )

    copied_site = tmp_path / "copied" / "A" / "venv" / "site-packages"
    value = _helper_value(copied_site, {"pkg/__init__.py": "pkg/__init__.py"})
    keyed = installed_distributions.DistributionKeys({"dist:pkg": value}, [])
    assert found == {"copied": keyed, "linked-into-a-cache": keyed}, found


def test_a_project_file_linked_to_an_index_file_refuses_nothing_beside_recordless_root_installs(
    tmp_path: pathlib.Path,
) -> None:
    """In each layout (the installed files copied, or linked into a cache), `site-packages` holds
    `idxmod`, installed by hand as from a package index, whose RECORD lists `mod.py`, and
    `gitlib`, installed by hand as from git, with its RECORD. The project root, first on the path,
    holds `myproj.egg-info`, laid out as a setuptools build leaves it, with no RECORD, which
    requires `gitlib` (a `Requires-Dist` header and `requires.txt`), `projlib`, installed there by
    hand from a local directory, which has lost its RECORD (`_helper_recordless_root_install`),
    and `project/mod.py`, a link made by hand to `site-packages/mod.py`. The stage reaches `mod`
    through the project file, as `fingerprint.reached_module_of` gives it. Neither layout refuses
    it, and both give the same keys, none: `idxmod`'s RECORD in `site-packages` lists the file
    Pivot judged the module by (`fingerprint.judged_file`), which lies in `site-packages` in
    both, and the project root is neither where Pivot judged it nor where it resolves. With the
    project file as the import found it in its place, the linked layout, whose resolved path lies
    in the cache outside every entry of the path, would find the file only in the project root,
    which no RECORD there lists, beside `projlib`, and would refuse the stage."""
    found = dict[str, installed_distributions.DistributionKeys]()
    for layout in _LAYOUTS:
        root = tmp_path / layout
        site = _helper_install_by_hand(root, layout, "idxmod", {"mod.py": "MOD = 1\n"}, None)
        _helper_install_by_hand(
            root, layout, "gitlib", {"gitlib.py": "LIB = 1\n"}, installed_dists.UV_GIT_URL
        )
        project = root / "project"
        _helper_link_by_hand(project, site, "mod.py", layout)
        _helper_project_egg_info(project, "gitlib")
        _helper_recordless_root_install(root, layout)
        found[layout] = installed_distributions.distribution_keys(
            {"mod": installed_dists.module_loaded_from(project / "mod.py")},
            [str(project), str(site)],
            _PIVOT_DIRS,
        )

    nothing = installed_distributions.DistributionKeys({}, [])
    assert found == {"copied": nothing, "linked-into-a-cache": nothing}, found


def test_a_project_file_linked_to_an_unlisted_file_refuses_nothing_beside_recordless_root_installs(
    tmp_path: pathlib.Path,
) -> None:
    """In each layout (the installed files copied, or linked into a cache), `site-packages` holds
    `gitlib`, installed by hand as from git, with its RECORD, and `orphan.py`, which no RECORD
    lists (a copy, or a link into the cache). The project root, first on the path, holds
    `myproj.egg-info`, with no RECORD, which requires `gitlib`, `projlib`, installed there by hand
    from a local directory, which has lost its RECORD (`_helper_recordless_root_install`), and
    `project/orphan.py`, a link made by hand to `site-packages/orphan.py`. The stage reaches
    `orphan` through the project file. Neither layout refuses it, and both give the same keys,
    none: Pivot judges the project file at `site-packages/orphan.py` in both
    (`fingerprint.judged_file`), where no distribution lacks a RECORD, and its resolved path lies
    in `site-packages` or in the cache, neither of which is the project root. Judged at the
    project file, each layout would find the file in the project root, which no RECORD lists,
    beside `projlib`, and would refuse the stage: the copied layout where the file is a copy,
    whose resolved path lies in `site-packages`, too."""
    found = dict[str, installed_distributions.DistributionKeys]()
    for layout in _LAYOUTS:
        root = tmp_path / layout
        site = _helper_install_by_hand(
            root, layout, "gitlib", {"gitlib.py": "LIB = 1\n"}, installed_dists.UV_GIT_URL
        )
        if layout == "copied":
            (site / "orphan.py").write_text("ORPHAN = 1\n")
        else:
            cached = root / "cache" / "orphan.py"
            cached.write_text("ORPHAN = 1\n")
            (site / "orphan.py").symlink_to(cached)
        project = root / "project"
        _helper_link_by_hand(project, site, "orphan.py", layout)
        _helper_project_egg_info(project, "gitlib")
        _helper_recordless_root_install(root, layout)
        found[layout] = installed_distributions.distribution_keys(
            {"orphan": installed_dists.module_loaded_from(project / "orphan.py")},
            [str(project), str(site)],
            _PIVOT_DIRS,
        )

    nothing = installed_distributions.DistributionKeys({}, [])
    assert found == {"copied": nothing, "linked-into-a-cache": nothing}, found


@pytest.mark.parametrize(
    ("egg_info_in", "refusals"),
    [
        pytest.param("project", [], id="egg-info-in-the-project-root"),
        pytest.param("venv/site-packages", [_MYPROJ_MOD_REFUSAL], id="egg-info-in-site-packages"),
    ],
)
def test_a_projects_egg_info_refuses_no_library_file_uv_links_into_a_cache_under_the_root(
    tmp_path: pathlib.Path, egg_info_in: str, refusals: list[str]
) -> None:
    """In each layout (the installed files copied, or linked into a cache that lies under the
    project root, as `UV_CACHE_DIR` there puts uv's cache), `site-packages` holds `idxmod`,
    installed by hand as from a package index, which has lost its RECORD, and `gitlib`,
    installed by hand as from git, with its RECORD. `myproj.egg-info`, laid out as a setuptools
    build leaves it, with no RECORD and no `direct_url.json`, which requires `gitlib`, lies in
    the project root or in `site-packages` (`egg_info_in`, under the layout's directory), and
    the path is the project root, then `site-packages`. The stage reaches `mod`, whose file no
    RECORD lists, and `gitlib`. Both layouts give the key of `gitlib` and the same refusals:
    none beside the project's own `.egg-info` in the project root, which is no `site-packages`
    or `dist-packages` directory, though in the linked layout `mod.py` resolves into the cache
    there; and in `site-packages`, where such an `.egg-info` is a distribution that can own a
    file no RECORD lists, one refusal naming `mod.py` and `myproj`."""
    found = dict[str, installed_distributions.DistributionKeys]()
    for layout in _LAYOUTS:
        root = tmp_path / layout
        project = root / "project"
        project.mkdir(parents=True)
        site = root / "venv" / "site-packages"
        cache = None if layout == "copied" else project / ".uv-cache"
        installed_dists.index_install_by_hand(site, "idxmod", {"mod.py": "MOD = 1\n"}, cache=cache)
        installed_dists.damage(site / "idxmod-0.1.dist-info" / "RECORD", None)
        installed_dists.index_install_by_hand(
            site, "gitlib", {"gitlib.py": "LIB = 1\n"}, cache=cache
        )
        (site / "gitlib-0.1.dist-info" / "direct_url.json").write_bytes(installed_dists.UV_GIT_URL)
        _helper_project_egg_info(root / egg_info_in, "gitlib")
        resolved = os.path.realpath(site / "mod.py")
        assert pathlib.Path(resolved).is_relative_to(project.resolve()) == (cache is not None), (
            f"mod.py must resolve under the project root in the linked layout alone: {resolved}"
        )
        keys = installed_distributions.distribution_keys(
            {
                name: installed_dists.module_loaded_from(site / f"{name}.py")
                for name in ("gitlib", "mod")
            },
            [str(project), str(site)],
            _PIVOT_DIRS,
        )
        found[layout] = _helper_written_site(keys, site)

    copied_site = tmp_path / "copied" / "venv" / "site-packages"
    expected = installed_distributions.DistributionKeys(
        {"dist:gitlib": _helper_value(copied_site, {"gitlib.py": "gitlib.py"})}, refusals
    )
    assert found == {"copied": expected, "linked-into-a-cache": expected}, found


def test_a_project_package_linked_file_by_file_gets_the_keys_of_each_install_in_its_directory(
    tmp_path: pathlib.Path,
) -> None:
    """In each layout (the installed files copied, or linked into a cache), `pkg`, whose one file
    is `pkg/__init__.py`, and `pkgplugin`, whose one file is `pkg/plugin.py`, are installed by hand
    into `site-packages` as uv installs from git, with their RECORDs. `project/pkg` is a real
    directory whose two files are links made by hand to the files of `site-packages/pkg`, and the
    path is the project root, then `site-packages`. The stage reaches the package `pkg` through
    `project/pkg/__init__.py`. Both layouts give the keys of `pkg` and of `pkgplugin`, with one
    value each: Pivot judges the `__init__` file at `site-packages/pkg/__init__.py` in both
    (`fingerprint.judged_file`), so the package's directory is `site-packages/pkg`, where
    `pkgplugin` has a file. Judged at the project file where it is a copy, the copied layout
    alone would take `project/pkg` as the package's directory and give the key of `pkg` alone."""
    found = dict[str, installed_distributions.DistributionKeys]()
    for layout in _LAYOUTS:
        root = tmp_path / layout
        site = _helper_install_by_hand(
            root, layout, "pkg", {"pkg/__init__.py": "PKG = 1\n"}, installed_dists.UV_GIT_URL
        )
        _helper_install_by_hand(
            root, layout, "pkgplugin", {"pkg/plugin.py": "PLUGIN = 1\n"}, installed_dists.UV_GIT_URL
        )
        project = root / "project"
        _helper_link_by_hand(project, site, "pkg/__init__.py", layout)
        _helper_link_by_hand(project, site, "pkg/plugin.py", layout)
        found[layout] = installed_distributions.distribution_keys(
            {"pkg": installed_dists.module_loaded_from(project / "pkg" / "__init__.py")},
            [str(project), str(site)],
            _PIVOT_DIRS,
        )

    copied_site = tmp_path / "copied" / "venv" / "site-packages"
    keyed = installed_distributions.DistributionKeys(
        {
            "dist:pkg": _helper_value(copied_site, {"pkg/__init__.py": "pkg/__init__.py"}),
            "dist:pkgplugin": _helper_value(copied_site, {"pkg/plugin.py": "pkg/plugin.py"}),
        },
        [],
    )
    assert found == {"copied": keyed, "linked-into-a-cache": keyed}, found
