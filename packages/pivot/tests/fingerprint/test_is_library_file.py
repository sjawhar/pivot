"""`fingerprint.is_library_file`, called in this process on site directories this test lays out by
hand as uv and flit lay them out (`tests.installed_dists`): a library package linked into a
project root, a path entry that is a link to `site-packages`, flit's editable links to a
checkout, and chains of links made by hand from a project file (to a file of `site-packages`,
copied or linked into a cache, to an editable install's link there, to a file outside every
library path, a loop, and chains of 40 and 41 links). No test starts a process.

`installed_distributions.linked_by_editable` and `fingerprint.is_library_file` over real installs
that uv links into its cache are in `test_installed_distributions_snapshot.py`, and the pure
functions of `pivot.installed_distributions` in `test_installed_distributions_rules.py`.
"""

import os
import pathlib
import sys

import pytest
import pytest_mock
from tests import installed_dists

from pivot import fingerprint, installed_distributions


@pytest.mark.parametrize(
    "linked",
    [
        pytest.param(False, id="copied"),
        pytest.param(True, id="linked-into-a-cache"),
    ],
)
def test_is_library_file_takes_a_library_file_reached_through_a_link_to_its_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, linked: bool
) -> None:
    """`idxlib`, laid out by hand in `site-packages` as uv installs from a package index
    (`installed_dists.index_install_by_hand`), its files copies or, as `--link-mode symlink`
    leaves them, links into a cache from real directories. `project/idxlib` is a link to its
    package's directory, as a library package linked into a project root, and `libs` a link to
    `site-packages`. `fingerprint.is_library_file` takes the module's file as library code where
    it is installed, through `project/idxlib` with `site-packages` on the path, and through
    `libs` with `libs` on the path in its place, whether the file is a copy or a link: a linked
    file is judged where its directory resolves, in `site-packages`, as a copy is by its resolved
    path. A project file beside the link stays user code."""
    site = tmp_path / "venv" / "site-packages"
    installed_dists.index_install_by_hand(
        site,
        "idxlib",
        {"idxlib/__init__.py": "LIB = 1\n"},
        cache=tmp_path / "cache" if linked else None,
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "idxlib").symlink_to(site / "idxlib", target_is_directory=True)
    (project / "helpers.py").write_text("HELP = 1\n")
    libs = tmp_path / "libs"
    libs.symlink_to(site, target_is_directory=True)
    installed = site / "idxlib" / "__init__.py"
    assert installed.is_symlink() is linked, f"{installed} must be a link only when linked"
    assert not installed.parent.is_symlink(), f"{installed.parent} must be a real directory"

    monkeypatch.setattr(sys, "path", [str(project), str(site)])
    found = {
        "in site-packages": fingerprint.is_library_file(str(installed)),
        "through project/idxlib": fingerprint.is_library_file(
            str(project / "idxlib" / "__init__.py")
        ),
        "project file": fingerprint.is_library_file(str(project / "helpers.py")),
    }
    monkeypatch.setattr(sys, "path", [str(project), str(libs)])
    found["through libs"] = fingerprint.is_library_file(str(libs / "idxlib" / "__init__.py"))

    assert found == {
        "in site-packages": True,
        "through project/idxlib": True,
        "project file": False,
        "through libs": True,
    }, found


@pytest.mark.parametrize(
    ("link", "source", "asked"),
    [
        pytest.param("flitlib", "flitlib/__init__.py", False, id="package-directory"),
        pytest.param("flitlib.py", "flitlib.py", True, id="module-file"),
    ],
)
def test_is_library_file_leaves_flits_editable_link_user_code_through_a_linked_path_entry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
    link: str,
    source: str,
    asked: bool,
) -> None:
    """`flitlib`, laid out as flit's `install --symlink` installs it
    (`installed_dists.flit_symlink_install`): a link in `site-packages` to the checkout's package
    directory, or to its module file, which the RECORD of an editable install lists. `libs`, a
    link to `site-packages`, is the one entry of the path. `fingerprint.is_library_file` takes
    the checkout's source as user code, in `site-packages` and through `libs`. A file under the
    linked package directory is judged in the checkout, where its directory resolves, and the
    RECORDs are not asked about it; the linked module file is judged in `site-packages`, and the
    RECORDs are asked about it there, where the one that lists it is an editable install's."""
    site = tmp_path / "venv" / "site-packages"
    site.mkdir(parents=True)
    checkout = tmp_path / "flitlib"
    (checkout / source).parent.mkdir(parents=True, exist_ok=True)
    (checkout / source).write_text(installed_dists.FIXLIB_ALONE)
    installed_dists.flit_symlink_install(site, checkout, link)
    libs = tmp_path / "libs"
    libs.symlink_to(site, target_is_directory=True)
    monkeypatch.setattr(sys, "path", [str(libs)])
    asked_editable = mocker.spy(installed_distributions, "linked_by_editable")

    found = [fingerprint.is_library_file(str(entry / source)) for entry in (site, libs)]

    assert found == [False, False], f"flit's source must be user code in both: {found}"
    files_asked = [call.args[0] for call in asked_editable.call_args_list]
    in_site = os.path.join(os.path.realpath(site), source)
    assert files_asked == ([in_site, in_site] if asked else []), files_asked


def _helper_site(root: pathlib.Path) -> pathlib.Path:
    """`root/venv/site-packages`, made empty: a library path by its name."""
    site = root / "venv" / "site-packages"
    site.mkdir(parents=True)
    return site


def _helper_cache_link(root: pathlib.Path, site: pathlib.Path, name: str) -> pathlib.Path:
    """`site/<name>`, a symbolic link to `root/cache/<name>`, as `--link-mode symlink` links a
    file into uv's cache, that no RECORD lists."""
    cache = root / "cache"
    cache.mkdir()
    (cache / name).write_text("MOD = 1\n")
    (site / name).symlink_to(cache / name)
    return site / name


def _helper_project_link(root: pathlib.Path, name: str, target: pathlib.Path) -> pathlib.Path:
    """`root/project/<name>`, a symbolic link made by hand to `target`."""
    project = root / "project"
    project.mkdir(exist_ok=True)
    (project / name).symlink_to(target)
    return project / name


def test_is_library_file_judges_a_project_file_linked_into_site_packages_as_in_a_copied_venv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`project/mod.py`, a link made by hand to `mod.py` in `site-packages`, which is a copy in
    one layout, as `--link-mode copy` leaves a file, and in another a link into a cache that no
    RECORD lists, as `--link-mode symlink` leaves one. `fingerprint.is_library_file` takes the
    project file as library code in both: in the copied layout its resolved path lies in
    `site-packages`; in the linked one its resolved path lies in the cache, and Pivot follows its
    link to `site-packages/mod.py`, a link under a library path that no editable install's RECORD
    lists, and decides there, as a copied venv decides."""
    resolved_in_site = dict[str, bool]()
    found = dict[str, bool]()
    for layout in ("copied", "linked"):
        site = _helper_site(tmp_path / layout)
        if layout == "copied":
            (site / "mod.py").write_text("MOD = 1\n")
        else:
            _helper_cache_link(tmp_path / layout, site, "mod.py")
        linked = _helper_project_link(tmp_path / layout, "mod.py", site / "mod.py")
        monkeypatch.setattr(sys, "path", [str(linked.parent), str(site)])
        resolved_in_site[layout] = "site-packages" in pathlib.Path(os.path.realpath(linked)).parts
        found[layout] = fingerprint.is_library_file(str(linked))

    assert resolved_in_site == {"copied": True, "linked": False}, resolved_in_site
    assert found == {"copied": True, "linked": True}, found


def test_is_library_file_leaves_a_project_file_linked_to_an_editable_installs_link_user_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: pytest_mock.MockerFixture
) -> None:
    """`project/flitlib.py`, a link made by hand to `flitlib.py` in `site-packages`, which flit's
    `install --symlink` links to a checkout's module file and the RECORD of that editable install
    lists (`installed_dists.flit_symlink_install`). Pivot follows the project file's link to
    `site-packages/flitlib.py` and decides there: the editable install's RECORD lists that file,
    so its resolved path decides, in the checkout, and the project file is user code, as
    `site-packages/flitlib.py` is. The RECORDs are asked about that file alone."""
    site = _helper_site(tmp_path)
    checkout = tmp_path / "flitlib"
    checkout.mkdir()
    (checkout / "flitlib.py").write_text(installed_dists.FIXLIB_ALONE)
    installed_dists.flit_symlink_install(site, checkout, "flitlib.py")
    linked = _helper_project_link(tmp_path, "flitlib.py", site / "flitlib.py")
    monkeypatch.setattr(sys, "path", [str(linked.parent), str(site)])
    asked_editable = mocker.spy(installed_distributions, "linked_by_editable")

    found = [fingerprint.is_library_file(str(file)) for file in (linked, site / "flitlib.py")]

    assert found == [False, False], f"both files must be user code: {found}"
    asked = [call.args[0] for call in asked_editable.call_args_list]
    in_site = os.path.join(os.path.realpath(site), "flitlib.py")
    assert asked == [in_site, in_site], f"only {in_site} may be asked about: {asked}"


def test_is_library_file_takes_a_project_file_linked_outside_every_library_path_as_user_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: pytest_mock.MockerFixture
) -> None:
    """`project/helpers.py`, a link made by hand to `shared/helpers.py`, a file outside every
    library path: `fingerprint.is_library_file` follows the link to that file, which is no
    library path and no link, and takes the project file as user code, asking no RECORD."""
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "helpers.py").write_text("HELP = 1\n")
    linked = _helper_project_link(tmp_path, "helpers.py", shared / "helpers.py")
    monkeypatch.setattr(sys, "path", [str(linked.parent), str(shared)])
    asked_editable = mocker.spy(installed_distributions, "linked_by_editable")

    found = fingerprint.is_library_file(str(linked))

    assert found is False, "a link to a file outside every library path must be user code"
    assert asked_editable.call_count == 0, asked_editable.call_args_list


def test_is_library_file_takes_a_loop_of_links_in_the_project_root_as_user_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`project/a.py`, a link to `b.py`, and `project/b.py`, a link to `a.py`: following the links
    never reaches a library path or a file, so `fingerprint.is_library_file` stops at its bound on
    the links it follows and takes the file as user code."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").symlink_to("b.py")
    (project / "b.py").symlink_to("a.py")
    monkeypatch.setattr(sys, "path", [str(project)])

    found = fingerprint.is_library_file(str(project / "a.py"))

    assert found is False, "a loop of links in the project root must be user code"


@pytest.mark.parametrize(
    ("links", "library"),
    [
        pytest.param(40, True, id="40-links"),
        pytest.param(41, False, id="41-links"),
    ],
)
def test_is_library_file_follows_at_most_40_links_to_a_library_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, links: int, library: bool
) -> None:
    """A chain of `links` links made by hand, from `project/p0.py` through `project/p1.py` and
    on to `mod.py` in `site-packages`, a link into a cache that no RECORD lists. Pivot follows
    at most 40 links, as Linux's path lookup does: through 40 the project file is judged in
    `site-packages`, library code; through 41 it is user code."""
    site = _helper_site(tmp_path)
    _helper_cache_link(tmp_path, site, "mod.py")
    project = tmp_path / "project"
    project.mkdir()
    for index in range(links - 1):
        (project / f"p{index}.py").symlink_to(f"p{index + 1}.py")
    (project / f"p{links - 1}.py").symlink_to(site / "mod.py")
    monkeypatch.setattr(sys, "path", [str(project), str(site)])

    found = fingerprint.is_library_file(str(project / "p0.py"))

    assert found is library, f"through {links} links library code must be {library}, not {found}"
