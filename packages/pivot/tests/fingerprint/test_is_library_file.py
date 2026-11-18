"""`fingerprint.is_library_file`, called in this process on site directories this test lays out by
hand as uv and flit lay them out (`tests.installed_dists`): a library package linked into a
project root, a path entry that is a link to `site-packages`, and flit's editable links to a
checkout. No test starts a process.

`installed_distributions.listed_as_library` and `fingerprint.is_library_file` over real installs
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
    listed = mocker.spy(installed_distributions, "listed_as_library")

    found = [fingerprint.is_library_file(str(entry / source)) for entry in (site, libs)]

    assert found == [False, False], f"flit's source must be user code in both: {found}"
    files_asked = [call.args[0] for call in listed.call_args_list]
    in_site = os.path.join(os.path.realpath(site), source)
    assert files_asked == ([in_site, in_site] if asked else []), files_asked
