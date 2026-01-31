import pytest

from pivot import loaders, outputs


def test_out_cache_default_true() -> None:
    """Out should have cache=True by default."""
    out = outputs.Out(path="file.txt", loader=loaders.PathOnly())
    assert out.cache is True


def test_metric_cache_default_false() -> None:
    """Metric should have cache=False by default (git-tracked)."""
    metric = outputs.Metric(path="metrics.json")
    assert metric.cache is False


def test_plot_cache_default_true() -> None:
    """Plot should have cache=True by default."""
    plot = outputs.Plot(path="loss.csv", loader=loaders.PathOnly())
    assert plot.cache is True


def test_plot_options() -> None:
    """Plot should store x, y, template options."""
    plot = outputs.Plot(
        path="loss.csv", loader=loaders.PathOnly(), x="epoch", y="loss", template="linear"
    )
    assert plot.x == "epoch"
    assert plot.y == "loss"
    assert plot.template == "linear"


def test_all_outputs_frozen() -> None:
    """All output types should be immutable (frozen dataclasses)."""
    out = outputs.Out(path="file.txt", loader=loaders.PathOnly())
    metric = outputs.Metric(path="metrics.json")
    plot = outputs.Plot(path="loss.csv", loader=loaders.PathOnly())

    with pytest.raises(AttributeError):
        out.path = "other.txt"  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AttributeError):
        metric.cache = True  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AttributeError):
        plot.x = "step"  # pyright: ignore[reportAttributeAccessIssue]


def test_normalize_out_string() -> None:
    """String should become Out object."""
    result = outputs.normalize_out("file.txt")
    assert isinstance(result, outputs.Out)
    assert result.path == "file.txt"
    assert result.cache is True


def test_normalize_out_passthrough() -> None:
    """Out subclasses should pass through unchanged."""
    out = outputs.Out(path="file.txt", loader=loaders.PathOnly(), cache=False)
    metric = outputs.Metric(path="metrics.json")
    plot = outputs.Plot(path="loss.csv", loader=loaders.PathOnly(), x="epoch")

    assert outputs.normalize_out(out) is out
    assert outputs.normalize_out(metric) is metric
    assert outputs.normalize_out(plot) is plot


def test_out_with_explicit_cache_false() -> None:
    """Out can explicitly set cache=False."""
    out = outputs.Out(path="file.txt", loader=loaders.PathOnly(), cache=False)
    assert out.cache is False


def test_metric_with_explicit_cache_true() -> None:
    """Metric can explicitly override cache to True."""
    metric = outputs.Metric(path="metrics.json", cache=True)
    assert metric.cache is True


def test_plot_with_no_options() -> None:
    """Plot without visualization options should have None defaults."""
    plot = outputs.Plot(path="loss.csv", loader=loaders.PathOnly())
    assert plot.x is None
    assert plot.y is None
    assert plot.template is None


def test_out_subclass_hierarchy() -> None:
    """Metric, Plot should inherit from Out."""
    assert issubclass(outputs.Metric, outputs.Out)
    assert issubclass(outputs.Plot, outputs.Out)
    # Note: IncrementalOut and DirectoryOut are standalone classes (not Out subclasses)
    # to avoid dataclass field ordering issues. They implement the BaseOut protocol.


def test_out_instances_are_out() -> None:
    """Instances should be recognizable as Out."""
    out = outputs.Out(path="file.txt", loader=loaders.PathOnly())
    metric = outputs.Metric(path="metrics.json")
    plot = outputs.Plot(path="loss.csv", loader=loaders.PathOnly())

    assert isinstance(out, outputs.Out)
    assert isinstance(metric, outputs.Out)
    assert isinstance(plot, outputs.Out)


# IncrementalOut tests


def test_incremental_out_cache_default_true() -> None:
    """IncrementalOut should have cache=True by default."""
    inc = outputs.IncrementalOut(path="database.csv", loader=loaders.PathOnly())
    assert inc.cache is True


def test_incremental_out_frozen() -> None:
    """IncrementalOut should be immutable (frozen dataclass)."""
    inc = outputs.IncrementalOut(path="database.csv", loader=loaders.PathOnly())
    with pytest.raises(AttributeError):
        inc.path = "other.csv"  # pyright: ignore[reportAttributeAccessIssue]


def test_incremental_out_implements_base_out_protocol() -> None:
    """IncrementalOut should implement BaseOut protocol (not inherit from Out)."""
    inc = outputs.IncrementalOut(path="database.csv", loader=loaders.PathOnly())
    # IncrementalOut is a standalone class that implements BaseOut protocol
    assert isinstance(inc, outputs.BaseOut)
    assert hasattr(inc, "path")
    assert hasattr(inc, "cache")
    assert hasattr(inc, "loader")


def test_normalize_out_incremental_passthrough() -> None:
    """IncrementalOut should pass through normalize_out unchanged."""
    inc = outputs.IncrementalOut(path="database.csv", loader=loaders.PathOnly())
    assert outputs.normalize_out(inc) is inc


# DirectoryOut tests


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


def test_directory_out_implements_base_out_protocol() -> None:
    """DirectoryOut should implement BaseOut protocol (not inherit from Out)."""
    dir_out = outputs.DirectoryOut(path="output/", loader=loaders.JSON[dict[str, int]]())
    # DirectoryOut is a standalone class that implements BaseOut protocol
    assert isinstance(dir_out, outputs.BaseOut)
    assert hasattr(dir_out, "path")
    assert hasattr(dir_out, "cache")
    assert hasattr(dir_out, "loader")


def test_directory_out_frozen() -> None:
    """DirectoryOut should be immutable (frozen dataclass)."""
    dir_out = outputs.DirectoryOut(path="output/", loader=loaders.JSON[dict[str, int]]())
    with pytest.raises(AttributeError):
        dir_out.path = "other/"  # pyright: ignore[reportAttributeAccessIssue]


def test_directory_out_cache_default_true() -> None:
    """DirectoryOut should have cache=True by default."""
    dir_out = outputs.DirectoryOut(path="output/", loader=loaders.JSON[dict[str, int]]())
    assert dir_out.cache is True


def test_normalize_out_directory_passthrough() -> None:
    """DirectoryOut should pass through normalize_out unchanged."""
    dir_out = outputs.DirectoryOut(path="output/", loader=loaders.JSON[dict[str, int]]())
    assert outputs.normalize_out(dir_out) is dir_out


# TypeGuard tests


def test_is_directory_out_true_for_directory_out() -> None:
    """is_directory_out should return True for DirectoryOut."""
    dir_out = outputs.DirectoryOut(path="output/", loader=loaders.JSON[dict[str, int]]())
    assert outputs.is_directory_out(dir_out) is True


def test_is_directory_out_false_for_other_types() -> None:
    """is_directory_out should return False for Out and IncrementalOut."""
    out = outputs.Out(path="file.txt", loader=loaders.PathOnly())
    inc = outputs.IncrementalOut(path="cache.json", loader=loaders.JSON())
    assert outputs.is_directory_out(out) is False
    assert outputs.is_directory_out(inc) is False


def test_is_incremental_out_true_for_incremental_out() -> None:
    """is_incremental_out should return True for IncrementalOut."""
    inc = outputs.IncrementalOut(path="cache.json", loader=loaders.JSON())
    assert outputs.is_incremental_out(inc) is True


def test_is_incremental_out_false_for_other_types() -> None:
    """is_incremental_out should return False for Out and DirectoryOut."""
    out = outputs.Out(path="file.txt", loader=loaders.PathOnly())
    dir_out = outputs.DirectoryOut(path="output/", loader=loaders.JSON[dict[str, int]]())
    assert outputs.is_incremental_out(out) is False
    assert outputs.is_incremental_out(dir_out) is False
