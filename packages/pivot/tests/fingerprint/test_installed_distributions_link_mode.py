"""Library code installed as a symbolic link, checked through the `pivot` CLI.

uv's `--link-mode symlink` (`UV_LINK_MODE=symlink`) installs each file of a distribution as a link
into uv's cache, inside real directories of `site-packages`: such a venv fingerprints as one whose
files uv copied, and gives the same `dist:` keys, refusals and warnings
(`fingerprint.is_library_file`: a link there is library code unless an editable install's RECORD
lists it). An editable install's code stays user code, as in pivot 0.10.4, so an edit to it reruns
the stage: flit's `install --symlink` links a checkout's package directory, or its module file,
into `site-packages`, and uv links the `.pth` file and the finder module that setuptools writes
for an editable install into its cache.

Each test makes its venvs with uv over this test environment (`tests.installed_dists`) and its own
pivot project in each. A test that installs with uv passes the link mode to uv itself (the pod
that runs the tests by hand sets `UV_LINK_MODE=copy`, CI sets nothing). The flit test builds
flit's layout by hand, as flit 4.2.0 wrote it, since CI has no flit. The file imports nothing that
pivot 0.10.4 lacks, and so runs with `packages/pivot/src` at that release too.
"""

import pathlib

import pytest
from tests import installed_dists

# `Fix_Lib`'s module `fixlib`: the function the stage calls, which adds `fixdep.base()` to a
# literal, and a class whose metaclass defines `__getattribute__`, as pydantic's `BaseModel` has
# `ModelMetaclass`, which a project's class subclasses.
_FIXLIB_WITH_METACLASS = """\
import fixdep


class Meta(type):
    def __getattribute__(cls, name):
        return super().__getattribute__(name)


class Model(metaclass=Meta):
    pass


def value():
    return "value-1:" + fixdep.base()
"""

# `fixlate`'s module, which nothing imports but the stage's body.
_FIXLATE = 'def value():\n    return ":late"\n'

# A stage that calls `fixlib.value()`, reads a constant of a project class that subclasses
# `fixlib.Model`, and imports `fixlate` in its body, which discovery does not load.
_METACLASS_PIPELINE = """\
import fixlib

from pivot import Pipeline, stage


class Thing(fixlib.Model):
    SUFFIX = ":thing"


@stage
def make() -> str:
    import fixlate

    return fixlib.value() + Thing.SUFFIX + fixlate.value()


with Pipeline("p") as pipeline:
    make()
"""

# The refusal of `fixdep`, which `Fix_Lib` requires, when its `direct_url.json` is no JSON object,
# word for word, `{site}` the venv's resolved site-packages.
_UNCLASSIFIED_DEPENDENCY_REFUSAL = (
    "Distribution 'fixdep' ({site}/fixdep-0.1.dist-info), which a distribution this stage "
    "reaches depends on, is installed in a form Pivot cannot classify: its direct_url.json is "
    "not a JSON object. Pivot cannot fingerprint the installed code the stage runs, so the stage "
    "could skip a run after that code changes. Fix: reinstall 'fixdep'."
)


def _helper_linked_project(
    root: pathlib.Path, sources: tuple[pathlib.Path, ...], link_mode: str, pipeline_text: str
) -> installed_dists.Project:
    """Make a venv at `root/venv`, install the projects at `sources` into it from their paths in
    that order with uv's `link_mode`, and make a project at `root/project` whose pipeline is
    `pipeline_text`. The test fails unless `fixlib`'s installed module file is a symbolic link
    exactly when `link_mode` is `symlink`."""
    root.mkdir()
    venv = installed_dists.make_venv(root / "venv")
    for source in sources:
        installed_dists.install_directory(venv, source, link_mode=link_mode)
    linked = (venv.site_packages / "fixlib" / "__init__.py").is_symlink()
    assert linked == (link_mode == "symlink"), (
        f"fixlib/__init__.py must be a link only with --link-mode symlink: {link_mode} {linked}"
    )
    return installed_dists.make_project_in(root, venv, pipeline_text)


def _helper_assert_same_lock(
    copied: installed_dists.Project, linked: installed_dists.Project, dist_keys: list[str]
) -> None:
    """The lock of `STAGE` in the copied venv's project `copied` holds exactly the `dist:` keys
    `dist_keys`, and the lock in the linked venv's project `linked` records the same code
    manifest."""
    copied_manifest = installed_dists.lock_manifest(copied, installed_dists.STAGE)
    linked_manifest = installed_dists.lock_manifest(linked, installed_dists.STAGE)
    assert installed_dists.dist_keys(copied_manifest) == dist_keys, (
        f"the copied venv's lock must hold the dist: keys {dist_keys}: {copied_manifest}"
    )
    assert linked_manifest == copied_manifest, (
        f"the linked venv's lock must equal the copied one's:\nlinked {linked_manifest}\n"
        + f"copied {copied_manifest}"
    )


def test_a_venv_uv_links_into_its_cache_fingerprints_as_one_it_copied_into(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` (a function the stage calls, and a class with a metaclass that defines
    `__getattribute__`, which a project class subclasses), its dependency `fixdep`, and `fixlate`,
    which the stage imports in its body alone, all from their paths, installed into one venv with
    `--link-mode copy` and into another with `--link-mode symlink`, each with its own project and
    the same pipeline. In the second venv the installed module is a link into uv's cache, in the
    first it is not. `pivot repro` runs the stage in both, refusing nothing, and the two locks
    record the same code manifest, with the `dist:` keys of the three distributions. Once they
    are reinstalled into the first venv with `--link-mode symlink`, its stage is still cached,
    with no code change."""
    src = tmp_path / "src"
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
    fixlib = installed_dists.write_library(
        src, "Fix_Lib", "fixlib", _FIXLIB_WITH_METACLASS, dependencies=("fixdep",)
    )
    fixlate = installed_dists.write_library(src, "fixlate", "fixlate", _FIXLATE)
    sources = (fixdep, fixlib, fixlate)
    copied = _helper_linked_project(tmp_path / "copy", sources, "copy", _METACLASS_PIPELINE)
    linked = _helper_linked_project(tmp_path / "symlink", sources, "symlink", _METACLASS_PIPELINE)

    installed_dists.assert_ran_then_cached(copied, "value-1:base-1:thing:late")
    installed_dists.assert_ran_then_cached(linked, "value-1:base-1:thing:late")

    _helper_assert_same_lock(copied, linked, ["dist:fix-lib", "dist:fixdep", "dist:fixlate"])

    for source in sources:
        installed_dists.install_directory(copied.venv, source, reinstall=True, link_mode="symlink")
    relinked = (copied.venv.site_packages / "fixlib" / "__init__.py").is_symlink()
    assert relinked, "fixlib/__init__.py must be a link once reinstalled with --link-mode symlink"
    installed_dists.assert_marks_nothing(copied, installed_dists.STAGE)


@pytest.mark.parametrize(
    ("link", "source"),
    [
        pytest.param("flitlib", "flitlib/__init__.py", id="package-directory"),
        pytest.param("flitlib.py", "flitlib.py", id="module-file"),
    ],
)
def test_an_edit_to_the_checkout_flit_links_into_site_packages_reruns_the_stage(
    tmp_path: pathlib.Path, link: str, source: str
) -> None:
    """`flitlib`, installed as flit's `install --symlink` installs it
    (`installed_dists.flit_symlink_install`): a link in `site-packages` to the checkout's package
    directory, or to its module file, which the RECORD of an editable install lists. The stage
    calls `flitlib.value()`, which the walk hashes as user code. After one byte of the checkout's
    source changes, the stage is stale on `func:flitlib.value` alone, and `pivot repro` runs it
    and writes the new value."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    checkout = tmp_path / "flitlib"
    (checkout / source).parent.mkdir(parents=True, exist_ok=True)
    (checkout / source).write_text(installed_dists.FIXLIB_ALONE)
    installed_dists.flit_symlink_install(venv.site_packages, checkout, link)
    project = installed_dists.make_project_in(tmp_path, venv, installed_dists.pipeline("flitlib"))
    installed_dists.assert_ran_then_cached(project, "value-1")

    installed_dists.replace_once(checkout / source, '"value-1"', '"value-2"')

    installed_dists.assert_stale_on(project, installed_dists.STAGE, "func:flitlib.value")
    installed_dists.assert_runs_and_writes(project, "value-2", "the second `pivot repro`")


def _helper_unsafe_warning(project: installed_dists.Project, command: str) -> str:
    """`pivot repro --json`, which `command` names in a failure, runs `STAGE` in `project`, which
    writes `value-1` (`installed_dists.assert_runs_and_writes`), and gives one warning, its line
    started with the stage's name (`STAGE_PREFIX`), as unsafe fingerprinting starts each warning
    it gives in place of a refusal; returns that line with the venv's resolved site-packages
    written `{site}`."""
    shown = installed_dists.assert_runs_and_writes(project, "value-1", command)
    warnings = [
        line for line in shown.splitlines() if line.startswith(installed_dists.STAGE_PREFIX)
    ]
    assert len(warnings) == 1, (
        f"{command} must give one warning that names {installed_dists.STAGE}:\n{shown}"
    )
    return warnings[0].replace(str(project.venv.site_packages.resolve()), "{site}")


def test_a_linked_library_whose_record_is_lost_runs_with_the_copied_ones_warning_when_unsafe(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` installed from its path into one venv with `--link-mode copy` and into another
    with `--link-mode symlink`, each with its own project and the same pipeline, each with its
    RECORD lost, and `core.unsafe_fingerprinting` set in each project (`pivot config set`). The
    linked venv's refusal turns into a warning as the copied venv's does: `pivot repro` runs the
    stage in both, and both give the same warning, the refusal of a reached file that no RECORD
    lists beside `Fix_Lib`, which has none. In the linked venv, with the lock deleted, the next
    `pivot repro` runs the stage again, with the same warning: the run cache keeps no result that
    unsafe fingerprinting let run, where it would restore one."""
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
    projects = dict[str, installed_dists.Project]()
    warnings = dict[str, str]()
    for link_mode in ("copy", "symlink"):
        project = _helper_linked_project(
            tmp_path / link_mode, (source,), link_mode, installed_dists.pipeline("fixlib")
        )
        site = project.venv.site_packages
        installed_dists.damage(site / "fix_lib-0.1.dist-info" / "RECORD", None)
        installed_dists.pivot_ok(project, "config", "set", "core.unsafe_fingerprinting", "true")
        projects[link_mode] = project
        warnings[link_mode] = _helper_unsafe_warning(project, f"unsafe `pivot repro` ({link_mode})")
    installed_dists.lock_path(projects["symlink"], installed_dists.STAGE).unlink()
    again = _helper_unsafe_warning(projects["symlink"], "with the lock deleted, `pivot repro`")

    expected = installed_dists.STAGE_PREFIX + installed_dists.UNLISTED_REFUSAL.format(
        module="fixlib", file="fixlib/__init__.py", site="{site}"
    )
    assert warnings == {"copy": expected, "symlink": expected}, (
        f"both venvs must warn with:\n{expected}\n{warnings}"
    )
    assert again == expected, f"the run with the lock deleted must warn with:\n{expected}\n{again}"


def _helper_refusal(project: installed_dists.Project, site: pathlib.Path) -> str:
    """The refusal `pivot repro --json` exits 1 with in `project`, from its `Error:` line, with
    the venv's resolved site-packages `site` written `{site}`; the test fails unless it exits 1."""
    ran = installed_dists.run_pivot(project, "repro", "--json")
    shown = installed_dists.shown(ran)
    assert ran.returncode == 1, f"pivot must refuse {installed_dists.STAGE}:\n{shown}"
    errors = [line for line in ran.stderr.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1, f"pivot must print one Error: line:\n{shown}"
    return errors[0].replace(str(site.resolve()), "{site}")


def test_a_linked_library_whose_record_is_lost_is_refused_as_a_copied_one_is(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` installed from its path into one venv with `--link-mode copy` and into another
    with `--link-mode symlink`, each with its own project and the same pipeline, and each with its
    RECORD lost. A venv uv links into its cache fingerprints as one it copied into (BET-1194 spec
    v9, "uv's symlink link mode"): the copied venv refuses the stage, since Pivot cannot tell
    which files the reached distribution owns, so the linked venv refuses it with the same
    refusal. Walking the linked library as user code instead hashes its Python source only: a
    change to a package data file it reads, or to any other file it ships that is not Python
    source, leaves the stage cached with the old output."""
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
    refusals = dict[str, str]()
    for link_mode in ("copy", "symlink"):
        root = tmp_path / link_mode
        root.mkdir()
        venv = installed_dists.make_venv(root / "venv")
        installed_dists.install_directory(venv, source, link_mode=link_mode)
        linked = (venv.site_packages / "fixlib" / "__init__.py").is_symlink()
        assert linked == (link_mode == "symlink"), (
            f"fixlib/__init__.py must be a link only with --link-mode symlink: {link_mode} {linked}"
        )
        installed_dists.damage(venv.site_packages / "fix_lib-0.1.dist-info" / "RECORD", None)
        project = installed_dists.make_project_in(root, venv, installed_dists.pipeline("fixlib"))
        refusals[link_mode] = _helper_refusal(project, venv.site_packages)

    assert refusals["symlink"] == refusals["copy"], (
        f"the linked venv must refuse as the copied one does:\nlinked {refusals['symlink']}\n"
        + f"copied {refusals['copy']}"
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
            "fixdep-0.1.dist-info/direct_url.json",
            "[]",
            _UNCLASSIFIED_DEPENDENCY_REFUSAL,
            id="dependency-unclassified",
        ),
    ],
)
def test_a_broken_install_refuses_the_stage_with_one_error_in_a_copied_and_a_linked_venv(
    tmp_path: pathlib.Path, damaged: str, text: str | None, refusal: str
) -> None:
    """`Fix_Lib`, which the stage calls, and its requirement `fixdep`, installed from their paths
    into one venv with `--link-mode copy` and into another with `--link-mode symlink`, each with
    its own project and the same pipeline, then one installed file damaged in each (`damaged`,
    relative to site-packages, overwritten with `text` or deleted): the reached `Fix_Lib`'s
    `direct_url.json` made `[]`, which is no JSON object; `fixdep`'s RECORD deleted; `fixdep`'s
    `direct_url.json` made `[]`. The copied venv refuses the stage with `refusal`, after the
    stage's name (`STAGE_PREFIX`), and the linked venv refuses it with the same `Error:` line.
    Only `reached-unclassified` depends on `fingerprint.is_library_file` taking as library code a
    linked module that only the RECORD of a distribution Pivot cannot classify lists.
    The two `dependency-` cases guard the parity of the two venvs on a path the narrowed rule
    does not decide: the reached module is a link that `Fix_Lib`'s own RECORD, of a distribution
    that is not editable, lists, and the refusal comes from the dependency closure of `Fix_Lib`
    (`installed_distributions.reach`)."""
    fixlib, fixdep = installed_dists.write_fixlib_and_fixdep(tmp_path / "src")
    refusals = dict[str, str]()
    for link_mode in ("copy", "symlink"):
        project = _helper_linked_project(
            tmp_path / link_mode, (fixdep, fixlib), link_mode, installed_dists.pipeline("fixlib")
        )
        installed_dists.damage(project.venv.site_packages / damaged, text)
        refusals[link_mode] = _helper_refusal(project, project.venv.site_packages)

    expected = installed_dists.STAGE_PREFIX + refusal
    assert refusals["copy"].endswith(expected), (
        f"the copied venv must refuse with:\n{expected}\n{refusals['copy']}"
    )
    assert refusals["symlink"] == refusals["copy"], (
        f"the linked venv must refuse as the copied one does:\nlinked {refusals['symlink']}\n"
        + f"copied {refusals['copy']}"
    )


def test_a_linked_module_its_record_does_not_list_is_library_code_as_a_copied_one_is(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` installed from its path into one venv with `--link-mode copy` and into another
    with `--link-mode symlink`, each with its own project and the same pipeline, and in each the
    row of `fixlib/__init__.py` taken out of its RECORD, which it keeps, so no distribution on the
    path lacks a RECORD. In both venvs the module file is library code that no RECORD lists:
    `pivot repro` runs the stage, refusing nothing, and the two locks record the same code
    manifest, which holds no `dist:` key, since no distribution owns the module. Walked as user
    code where its link points, the linked module would put its `func:` entries in the linked
    venv's lock alone."""
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
    projects = dict[str, installed_dists.Project]()
    for link_mode in ("copy", "symlink"):
        project = _helper_linked_project(
            tmp_path / link_mode, (source,), link_mode, installed_dists.pipeline("fixlib")
        )
        record = project.venv.site_packages / "fix_lib-0.1.dist-info" / "RECORD"
        rows = record.read_text().splitlines(keepends=True)
        kept = [row for row in rows if not row.startswith("fixlib/__init__.py,")]
        assert len(kept) == len(rows) - 1, f"{record} must list fixlib/__init__.py once: {rows}"
        installed_dists.damage(record, "".join(kept))
        installed_dists.assert_ran_then_cached(project, "value-1")
        projects[link_mode] = project

    _helper_assert_same_lock(projects["copy"], projects["symlink"], [])


def test_an_editable_install_whose_finder_uv_links_into_its_cache_reruns_as_a_copied_one(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` installed editable (`-e`) from one checkout into one venv with `--link-mode
    copy` and into another with `--link-mode symlink`, each with its own project and the same
    pipeline. setuptools writes a finder module, which imports `fixlib` from the checkout, and a
    `.pth` file that installs it; the editable install's RECORD lists both, and in the second venv
    uv links both into its cache. The checkout's code is user code in both venvs, as in pivot
    0.10.4: `pivot repro` runs the stage in both, refusing nothing, and the two locks record the
    same code manifest, which holds no `dist:` key. After one byte of the checkout's source
    changes, the stage is stale on `func:fixlib.value` alone in both, and `pivot repro` runs it
    in both and writes the new value. This guards the parity of the two venvs on a path that
    `fingerprint.is_library_file`'s rule for links does not decide: the checkout's files are no
    links, and no stage reaches the finder module, the module file the editable install's RECORD
    lists that uv links into its cache."""
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
    projects = dict[str, installed_dists.Project]()
    for link_mode in ("copy", "symlink"):
        root = tmp_path / link_mode
        root.mkdir()
        venv = installed_dists.make_venv(root / "venv")
        installed_dists.install_directory(venv, source, editable=True, link_mode=link_mode)
        installed = sorted(venv.site_packages.glob("__editable__*"))
        kinds = sorted(path.suffix for path in installed)
        assert kinds == [".pth", ".py"], f"setuptools must write a .pth and a finder: {installed}"
        links = [path.is_symlink() for path in installed]
        assert links == [link_mode == "symlink"] * 2, (
            f"uv must link the .pth and the finder only with --link-mode symlink: {link_mode} "
            + f"{dict(zip(installed, links, strict=True))}"
        )
        projects[link_mode] = installed_dists.make_project_in(
            root, venv, installed_dists.pipeline("fixlib")
        )
        installed_dists.assert_ran_then_cached(projects[link_mode], "value-1")
    _helper_assert_same_lock(projects["copy"], projects["symlink"], [])

    installed_dists.replace_once(source / "fixlib" / "__init__.py", '"value-1"', '"value-2"')

    for project in projects.values():
        installed_dists.assert_stale_on(project, installed_dists.STAGE, "func:fixlib.value")
        installed_dists.assert_runs_and_writes(project, "value-2", "the second `pivot repro`")
