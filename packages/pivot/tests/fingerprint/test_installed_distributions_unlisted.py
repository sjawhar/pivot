"""The refusal of a reached module's file that no RECORD lists beside an install with no RECORD
(`installed_distributions.distribution_keys`), called in this process on site directories this
test lays out by hand: a module file linked from one entry of the path into another that holds
such an install, a module file linked to another file of the entry that holds one, a project
file linked into such an install's package, and a module file, beside such an install, linked to
a file of another entry that an index install's RECORD lists, a copy or a link into a cache. A
file lies in the entry where Pivot judges it (`fingerprint.judged_file`: where the import finds
it, or, for a project file linked into `site-packages`, where its link reaches), in each entry
that holds a later file of its chain of links, and in the one it resolves into, and a RECORD of
any of them lists it; where two lie in one entry, the refusal names the file where Pivot judged
it, else the first file of its chain that lies there. No test starts a process.

The pure functions of `pivot.installed_distributions` are in
`test_installed_distributions_rules.py`, and the same check over real installs in
`test_installed_distributions_snapshot.py`.
"""

import os
import pathlib

import pytest
from tests import installed_dists

from pivot import installed_distributions

_FORM = installed_distributions.InstallForm


def _helper_recordless_install(site: pathlib.Path, name: str, direct_url: bytes) -> None:
    """Install distribution `name` 0.1 into the site directory `site` by hand, with `direct_url`
    as its `direct_url.json`, and lose its RECORD: `<name>-0.1.dist-info` holds only METADATA,
    which names no requirement, and `direct_url.json`."""
    dist_info = site / f"{name}-0.1.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: 0.1\n")
    (dist_info / "direct_url.json").write_bytes(direct_url)


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
        {"x": installed_dists.module_loaded_from(second / "x.py")}, path, ()
    )
    found_there = installed_distributions.distribution_keys(
        {"x": installed_dists.module_loaded_from(first / "x.py")}, path, ()
    )

    site = os.path.realpath(first)
    refusal = installed_dists.unlisted_refusal(
        module="x", file=f"{site}/x.py", site=site, name="xlib", form=_FORM.LOCAL_DIRECTORY
    )
    expected = installed_distributions.DistributionKeys({}, [refusal])
    assert [linked, found_there] == [expected, expected], [linked, found_there]


def test_a_module_file_linked_to_another_file_of_its_path_entry_is_refused_as_found(
    tmp_path: pathlib.Path,
) -> None:
    """`xlib`, installed from a local directory into `site-packages`, has lost its RECORD, and
    `x.py` beside it is a link to `sub/y.py`. The file lies in `site-packages` both where the
    import of `x` finds it and where it resolves, and no RECORD there lists it: the stage that
    reaches `x` is refused once, naming the file where the import found it, `x.py`, not
    `sub/y.py`."""
    site = tmp_path / "site-packages"
    _helper_recordless_install(site, "xlib", installed_dists.UV_DIRECTORY_URL)
    (site / "sub").mkdir()
    (site / "sub" / "y.py").write_text("Y = 1\n")
    (site / "x.py").symlink_to(site / "sub" / "y.py")

    found = installed_distributions.distribution_keys(
        {"x": installed_dists.module_loaded_from(site / "x.py")}, [str(site)], ()
    )

    real_site = os.path.realpath(site)
    refusal = installed_dists.unlisted_refusal(
        module="x",
        file=f"{real_site}/x.py",
        site=real_site,
        name="xlib",
        form=_FORM.LOCAL_DIRECTORY,
    )
    assert found == installed_distributions.DistributionKeys({}, [refusal]), found


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
        {"helper": installed_dists.module_loaded_from(root / "helper.py")},
        [str(root), str(site)],
        (),
    )

    real_site = os.path.realpath(site)
    refusal = installed_dists.unlisted_refusal(
        module="helper",
        file=f"{real_site}/x/helper.py",
        site=real_site,
        name="x",
        form=_FORM.VCS,
    )
    assert found == installed_distributions.DistributionKeys({}, [refusal]), found


@pytest.mark.parametrize("layout", ["copied", "linked-into-a-cache"])
def test_a_module_file_linked_to_a_file_an_index_record_lists_in_another_entry_is_not_refused(
    tmp_path: pathlib.Path, layout: str
) -> None:
    """`s2/site-packages`, first on the path, holds `wlib`, installed from a local directory,
    whose RECORD is lost, and `y.py` as a link to `y.py` in `s1/site-packages`, which the RECORD
    of `ylib`, installed as from a package index, lists; in the `linked-into-a-cache` layout that
    file is itself a link into a cache, as uv's symlink link mode leaves it. The file lies in
    `s2/site-packages`, where Pivot judges it, beside the install with no RECORD, and in
    `s1/site-packages`, where its chain of links reaches a second library file, or where it
    resolves: a RECORD of a site directory it lies in lists it, though none in
    `s2/site-packages` does, so nothing is refused in either layout."""
    first = tmp_path / "s1" / "site-packages"
    second = tmp_path / "s2" / "site-packages"
    cache = None if layout == "copied" else tmp_path / "cache"
    installed_dists.index_install_by_hand(first, "ylib", {"y.py": "Y = 1\n"}, cache=cache)
    _helper_recordless_install(second, "wlib", installed_dists.UV_DIRECTORY_URL)
    (second / "y.py").symlink_to(first / "y.py")
    linked = (first / "y.py").is_symlink()
    assert linked == (layout != "copied"), f"{first / 'y.py'} in the {layout} layout: {linked}"

    found = installed_distributions.distribution_keys(
        {"y": installed_dists.module_loaded_from(second / "y.py")},
        [str(second), str(first)],
        (),
    )

    assert found == installed_distributions.DistributionKeys({}, []), found
