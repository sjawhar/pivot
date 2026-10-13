import pytest

from pivot import loaders, outputs


def test_directory_out_valid_path() -> None:
    """DirectoryOut should accept path ending with '/'."""
    dir_out = outputs.DirectoryOut(path="output/dir/", loader=loaders.JSON[dict[str, int]]())
    assert dir_out.path == "output/dir/"
    assert isinstance(dir_out.loader, loaders.JSON)


def test_directory_out_invalid_path_no_trailing_slash() -> None:
    """DirectoryOut should raise ValueError if path doesn't end with '/'."""
    with pytest.raises(ValueError, match="must end with '/'"):
        outputs.DirectoryOut(path="output/dir", loader=loaders.JSON[dict[str, int]]())


def test_directory_out_non_string_path_raises_error() -> None:
    """DirectoryOut should raise an error if path is not a string."""
    # Passing a list will fail when __post_init__ tries to call .endswith() on it
    with pytest.raises(AttributeError):
        outputs.DirectoryOut(
            path=["a/", "b/"],  # pyright: ignore[reportArgumentType]
            loader=loaders.JSON[dict[str, int]](),
        )


def test_directory_out_frozen() -> None:
    """DirectoryOut should be immutable (frozen dataclass)."""
    dir_out = outputs.DirectoryOut(path="output/", loader=loaders.JSON[dict[str, int]]())
    with pytest.raises(AttributeError):
        dir_out.path = "other/"  # pyright: ignore[reportAttributeAccessIssue]


def test_directory_out_cache_default_true() -> None:
    """DirectoryOut should have cache=True by default."""
    dir_out = outputs.DirectoryOut(path="output/", loader=loaders.JSON[dict[str, int]]())
    assert dir_out.cache is True
