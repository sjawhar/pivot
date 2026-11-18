"""The refusal of a reached module's file that no RECORD lists beside an install with no RECORD
(`installed_distributions.distribution_keys`), called in this process on site directories this
test lays out by hand: a module file linked from one entry of the path into another that holds
such an install, a project file linked into such an install's package, and a project file,
beside such an install, linked to a file an index install's RECORD lists. A file lies in the
entry where the import finds it and in the one it resolves into, and a RECORD of either lists
it. No test starts a process.

The pure functions of `pivot.installed_distributions` are in
`test_installed_distributions_rules.py`, and the same check over real installs in
`test_installed_distributions_snapshot.py`.
"""

import os
import pathlib

from tests import installed_dists

from pivot import fingerprint, installed_distributions

_FORM = installed_distributions.InstallForm

# The refusal of reached module `{module}`, whose file `{file}` lies in site directory `{site}`,
# where `{name}` 0.1, installed `{form}`, has no RECORD (`_helper_recordless_install`), and no
# RECORD lists that file.
_UNLISTED_REFUSAL = (
    "Module '{module}' ({file}) lies in {site}, where distribution '{name}' "
    "({site}/{name}-0.1.dist-info), installed {form}, has no RECORD file, and no RECORD file "
    "there lists that file. Pivot cannot tell whether the stage runs that distribution's code, "
    "so the stage could skip a run after it changes. Fix: reinstall '{name}'."
)


def _helper_recordless_install(site: pathlib.Path, name: str, direct_url: bytes) -> None:
    """Install distribution `name` 0.1 into the site directory `site` by hand, with `direct_url`
    as its `direct_url.json`, and lose its RECORD: `<name>-0.1.dist-info` holds only METADATA,
    which names no requirement, and `direct_url.json`."""
    dist_info = site / f"{name}-0.1.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: 0.1\n")
    (dist_info / "direct_url.json").write_bytes(direct_url)


def _helper_reached(found: pathlib.Path) -> installed_distributions.ReachedModule:
    """What `fingerprint.add_distribution_keys` gives of a module this process loaded from the
    file the import found at `found`: the origin `import_origin.of_module` gives it, the file
    resolved, and that file with its directory resolved (`fingerprint.with_real_directory`)."""
    return installed_distributions.ReachedModule(
        ("file", os.path.realpath(found)), fingerprint.with_real_directory(str(found))
    )


def test_a_module_file_linked_into_another_path_entry_is_refused_beside_its_recordless_install(
    tmp_path: pathlib.Path,
) -> None:
    """`xlib`, installed from a local directory into `s1/site-packages`, has lost its RECORD, and
    `x.py` lies beside it. `s2/site-packages`, before it on the path, holds `x.py` as a link to
    that file, where the import of `x` finds it. The site directory where the import finds the
    file holds no install with no RECORD, but the one the file resolves into does, and no RECORD
    of either lists it: the stage that reaches `x` is refused, naming the file where it lies in
    `s1/site-packages`, as when the import finds it there."""
    first = tmp_path / "s1" / "site-packages"
    second = tmp_path / "s2" / "site-packages"
    _helper_recordless_install(first, "xlib", installed_dists.UV_DIRECTORY_URL)
    (first / "x.py").write_text("X = 1\n")
    second.mkdir(parents=True)
    (second / "x.py").symlink_to(first / "x.py")
    path = [str(second), str(first)]

    linked = installed_distributions.distribution_keys(
        {"x": _helper_reached(second / "x.py")}, path, ()
    )
    found_there = installed_distributions.distribution_keys(
        {"x": _helper_reached(first / "x.py")}, path, ()
    )

    site = os.path.realpath(first)
    refusal = _UNLISTED_REFUSAL.format(
        module="x", file=f"{site}/x.py", site=site, name="xlib", form=_FORM.LOCAL_DIRECTORY.value
    )
    expected = installed_distributions.DistributionKeys({}, [refusal])
    assert [linked, found_there] == [expected, expected], [linked, found_there]


def test_a_project_file_linked_into_a_package_of_a_recordless_install_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """The project root, first on the path, holds `helper.py` as a link to `x/helper.py` in the
    site-packages of a `.venv` inside it, where `x`, installed from git, has lost its RECORD. The
    import of `helper` finds the file in the root, which holds no install with no RECORD, but the
    file resolves into site-packages, which does, and no RECORD of either lists it: the stage
    that reaches `helper` is refused, naming the file where it lies in site-packages."""
    root = tmp_path / "project"
    site = root / ".venv" / "lib" / "python3.13" / "site-packages"
    _helper_recordless_install(site, "x", installed_dists.UV_GIT_URL)
    (site / "x").mkdir()
    (site / "x" / "__init__.py").write_text("")
    (site / "x" / "helper.py").write_text("HELP = 1\n")
    (root / "helper.py").symlink_to(site / "x" / "helper.py")

    found = installed_distributions.distribution_keys(
        {"helper": _helper_reached(root / "helper.py")}, [str(root), str(site)], ()
    )

    real_site = os.path.realpath(site)
    refusal = _UNLISTED_REFUSAL.format(
        module="helper",
        file=f"{real_site}/x/helper.py",
        site=real_site,
        name="x",
        form=_FORM.VCS.value,
    )
    assert found == installed_distributions.DistributionKeys({}, [refusal]), found


def test_a_project_file_linked_to_a_file_an_index_record_lists_is_not_refused(
    tmp_path: pathlib.Path,
) -> None:
    """The project root, first on the path, holds `proj`, installed from a local directory, whose
    RECORD is lost, and `y.py` as a link to `y.py` in `site-packages`, which the RECORD of `ylib`,
    installed as from a package index, lists. The import of `y` finds the file in the root,
    beside the install with no RECORD, but a RECORD of the site directory the file resolves into
    lists it: nothing is refused, as when the import finds it in `site-packages`."""
    root = tmp_path / "project"
    site = tmp_path / "venv" / "site-packages"
    _helper_recordless_install(root, "proj", installed_dists.UV_DIRECTORY_URL)
    installed_dists.index_install_by_hand(site, "ylib", {"y.py": "Y = 1\n"}, cache=None)
    (root / "y.py").symlink_to(site / "y.py")
    path = [str(root), str(site)]

    linked = installed_distributions.distribution_keys(
        {"y": _helper_reached(root / "y.py")}, path, ()
    )
    found_there = installed_distributions.distribution_keys(
        {"y": _helper_reached(site / "y.py")}, path, ()
    )

    expected = installed_distributions.DistributionKeys({}, [])
    assert [linked, found_there] == [expected, expected], [linked, found_there]
