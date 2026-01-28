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
    assert issubclass(outputs.IncrementalOut, outputs.Out)


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


def test_incremental_out_is_out_subclass() -> None:
    """IncrementalOut should inherit from Out."""
    assert issubclass(outputs.IncrementalOut, outputs.Out)
    inc = outputs.IncrementalOut(path="database.csv", loader=loaders.PathOnly())
    assert isinstance(inc, outputs.Out)


def test_normalize_out_incremental_passthrough() -> None:
    """IncrementalOut should pass through normalize_out unchanged."""
    inc = outputs.IncrementalOut(path="database.csv", loader=loaders.PathOnly())
    assert outputs.normalize_out(inc) is inc
