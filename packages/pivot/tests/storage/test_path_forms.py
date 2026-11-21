from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot.storage import path_forms

if TYPE_CHECKING:
    import pathlib


def _helper_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """``tmp_path`` resolved, so that only the symlinks a test makes tell the forms apart."""
    return tmp_path.resolve()


def _helper_stored(root: pathlib.Path, relative: str | None) -> str | None:
    """The stored path ``relative`` names under ``root``; None for None."""
    return None if relative is None else str(root / relative)


def test_forms_are_the_path_and_its_real_path_through_a_missing_paths_existing_ancestors(
    tmp_path: pathlib.Path,
) -> None:
    """The resolved form of a missing path under a symlinked directory resolves that directory."""
    root = _helper_root(tmp_path)
    (root / "real").mkdir()
    (root / "link").symlink_to("real")

    found = path_forms.forms(root / "link" / "missing.txt")

    assert found == (root / "link" / "missing.txt", root / "real" / "missing.txt")


def test_enclosing_and_held_ask_each_form_given_where_a_symlinked_directory_splits_them(
    tmp_path: pathlib.Path,
) -> None:
    """``link`` links to ``target/sub``: by its normalized form ``link`` holds the stored path
    ``link/a``, and by its resolved form it lies inside the stored path ``target``, whose
    resolved form also holds ``link/a``'s resolved form."""
    root = _helper_root(tmp_path)
    (root / "target" / "sub").mkdir(parents=True)
    (root / "link").symlink_to("target/sub")
    a, b, query = str(root / "link" / "a"), str(root / "target"), root / "link"
    trie = path_forms.forms_trie([a, b])

    assert path_forms.enclosing(trie, query) is None
    assert path_forms.enclosing(trie, *path_forms.forms(query)) == b
    assert path_forms.held(trie, query) == [a]
    assert path_forms.held(trie, *path_forms.forms(query)) == [a]
    assert path_forms.held(trie, path_forms.resolved(query)) == [a]


def test_overlapping_both_forms_at_once_and_form_by_form_name_different_stored_paths(
    tmp_path: pathlib.Path,
) -> None:
    """With ``link`` linking to ``target/sub``, ``link`` holds ``link/a`` by its normalized form
    and lies inside ``target`` by its resolved form. Asked of both forms at once (a trie keyed by
    both forms, as ``engine.graph.tracked_overlap`` asks), "is or lies inside" comes first and
    names ``target``; asked form by form, each against the stored paths' same form (as
    ``ownership.tracked_output_refusal`` asks), the normalized form comes first and names
    ``link/a``."""
    root = _helper_root(tmp_path)
    (root / "target" / "sub").mkdir(parents=True)
    (root / "link").symlink_to("target/sub")
    a, b, query = root / "link" / "a", root / "target", root / "link"
    lexical = path_forms.parts_trie([(a, a), (b, b)])
    resolved = path_forms.parts_trie([(path_forms.resolved(a), a), (path_forms.resolved(b), b)])

    both = path_forms.overlapping(path_forms.forms_trie([str(a), str(b)]), *path_forms.forms(query))

    assert both == str(b)
    assert path_forms.overlapping(lexical, query) == a
    assert path_forms.overlapping(resolved, path_forms.resolved(query)) == b


def test_a_path_equal_to_a_stored_path_in_one_form_only_is_it_in_that_form(
    tmp_path: pathlib.Path,
) -> None:
    """``link`` links to ``real``: ``link/f`` is the stored ``real/f`` by its resolved form only,
    and ``real/g`` is the stored ``link/g`` by that stored path's resolved form only."""
    root = _helper_root(tmp_path)
    (root / "real").mkdir()
    (root / "link").symlink_to("real")
    stored_real, stored_link = root / "real" / "f", root / "link" / "g"
    query = root / "link" / "f"
    trie = path_forms.forms_trie([str(stored_real), str(stored_link)])
    lexical = path_forms.parts_trie([(stored_real, stored_real)])

    assert path_forms.enclosing(trie, query) is None
    assert path_forms.enclosing(trie, *path_forms.forms(query)) == str(stored_real)
    assert path_forms.overlapping(trie, *path_forms.forms(query)) == str(stored_real)
    assert path_forms.enclosing(trie, root / "real" / "g") == str(stored_link)
    assert path_forms.enclosing(lexical, query) is None
    assert path_forms.enclosing(lexical, *path_forms.forms(query)) == stored_real


@pytest.mark.parametrize(
    ("stored", "query", "enclosing", "held"),
    [
        pytest.param("data", "data-backup/x", None, [], id="a-longer-name-is-not-inside"),
        pytest.param("data-backup/x", "data", None, [], id="a-shorter-name-does-not-hold"),
        pytest.param("data/x", "data", None, ["data/x"], id="a-directory-holds"),
        pytest.param("data", "data/x", "data", [], id="a-path-lies-inside"),
        pytest.param("data", "data", "data", [], id="a-path-is"),
    ],
)
def test_path_parts_decide_containment_not_a_string_prefix(
    tmp_path: pathlib.Path, stored: str, query: str, enclosing: str | None, held: list[str]
) -> None:
    """``data-backup/x`` is not inside ``data``, nor does ``data`` hold it."""
    root = _helper_root(tmp_path)
    trie = path_forms.forms_trie([str(root / stored)])
    forms = path_forms.forms(root / query)

    assert path_forms.enclosing(trie, *forms) == _helper_stored(root, enclosing)
    assert path_forms.held(trie, *forms) == [str(root / path) for path in held]


def test_enclosing_names_the_outermost_stored_path(tmp_path: pathlib.Path) -> None:
    """Of the stored ``d`` and ``d/e``, both of which hold ``d/e/f``, ``d`` is named."""
    root = _helper_root(tmp_path)
    trie = path_forms.forms_trie([str(root / "d" / "e"), str(root / "d")])

    assert path_forms.enclosing(trie, root / "d" / "e" / "f") == str(root / "d")
    assert path_forms.enclosing(trie, root / "d" / "e") == str(root / "d")


def test_held_sorts_strings_by_character_and_paths_by_their_parts(tmp_path: pathlib.Path) -> None:
    """``a-1`` sorts before ``a/x`` as a string ('-' before '/') and after it as a path (``a``
    before ``a-1``), so a ``str`` trie names ``a-1`` first and a ``pathlib.Path`` trie names
    ``a/x`` first."""
    root = _helper_root(tmp_path)
    dash, nested = root / "a-1", root / "a" / "x"
    strings = path_forms.forms_trie([str(nested), str(dash)])
    paths = path_forms.parts_trie([(nested, nested), (dash, dash)])

    assert path_forms.held(strings, root) == [str(dash), str(nested)]
    assert path_forms.overlapping(strings, root) == str(dash)
    assert path_forms.held(paths, root) == [nested, dash]
    assert path_forms.overlapping(paths, root) == nested


@pytest.mark.parametrize(
    ("order", "named"),
    [
        pytest.param(("link", "real"), "real", id="the-real-path-last"),
        pytest.param(("real", "link"), "link", id="the-link-last"),
    ],
)
def test_forms_trie_keeps_the_later_path_where_two_share_a_form(
    tmp_path: pathlib.Path, order: tuple[str, str], named: str
) -> None:
    """``link`` links to the directory ``real``, so ``link``'s resolved form is ``real``: the
    stored path named for ``real`` is the later of the two, and ``link`` stays ``link``'s."""
    root = _helper_root(tmp_path)
    (root / "real").mkdir()
    (root / "link").symlink_to("real")
    trie = path_forms.forms_trie([str(root / name) for name in order])

    assert path_forms.enclosing(trie, root / "real") == str(root / named)
    assert path_forms.enclosing(trie, root / "link") == str(root / "link")
