# pyright: reportAttributeAccessIssue=false

import pytest

from pivot import outputs


def test_out_cache_default_true() -> None:
    """Out should have cache=True by default."""
    out = outputs.Out(path="file.txt")
    assert out.cache is True


def test_metric_cache_default_false() -> None:
    """Metric should have cache=False by default (git-tracked)."""
    metric = outputs.Metric(path="metrics.json")
    assert metric.cache is False


def test_plot_cache_default_true() -> None:
    """Plot should have cache=True by default."""
    plot = outputs.Plot(path="loss.csv")
    assert plot.cache is True


def test_plot_options() -> None:
    """Plot should store x, y, template options."""
    plot = outputs.Plot(path="loss.csv", x="epoch", y="loss", template="linear")
    assert plot.x == "epoch"
    assert plot.y == "loss"
    assert plot.template == "linear"


def test_all_outputs_frozen() -> None:
    """All output types should be immutable (frozen dataclasses)."""
    out = outputs.Out(path="file.txt")
    metric = outputs.Metric(path="metrics.json")
    plot = outputs.Plot(path="loss.csv")

    with pytest.raises(AttributeError):
        out.path = "other.txt"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        metric.cache = True  # type: ignore[misc]

    with pytest.raises(AttributeError):
        plot.x = "step"  # type: ignore[misc]


def test_normalize_out_string() -> None:
    """String should become Out object."""
    result = outputs.normalize_out("file.txt")
    assert isinstance(result, outputs.Out)
    assert result.path == "file.txt"
    assert result.cache is True


def test_normalize_out_passthrough() -> None:
    """BaseOut subclasses should pass through unchanged."""
    out = outputs.Out(path="file.txt", cache=False)
    metric = outputs.Metric(path="metrics.json")
    plot = outputs.Plot(path="loss.csv", x="epoch")

    assert outputs.normalize_out(out) is out
    assert outputs.normalize_out(metric) is metric
    assert outputs.normalize_out(plot) is plot


def test_out_persist_option() -> None:
    """All output types should support persist option."""
    out = outputs.Out(path="file.txt", persist=True)
    metric = outputs.Metric(path="metrics.json", persist=True)
    plot = outputs.Plot(path="loss.csv", persist=True)

    assert out.persist is True
    assert metric.persist is True
    assert plot.persist is True


def test_out_with_explicit_cache_false() -> None:
    """Out can explicitly set cache=False."""
    out = outputs.Out(path="file.txt", cache=False)
    assert out.cache is False


def test_metric_with_explicit_cache_true() -> None:
    """Metric can explicitly override cache to True."""
    metric = outputs.Metric(path="metrics.json", cache=True)
    assert metric.cache is True


def test_plot_with_no_options() -> None:
    """Plot without visualization options should have None defaults."""
    plot = outputs.Plot(path="loss.csv")
    assert plot.x is None
    assert plot.y is None
    assert plot.template is None


def test_baseout_is_base_class() -> None:
    """Out, Metric, Plot should all inherit from BaseOut."""
    assert issubclass(outputs.Out, outputs.BaseOut)
    assert issubclass(outputs.Metric, outputs.BaseOut)
    assert issubclass(outputs.Plot, outputs.BaseOut)


def test_out_instances_are_baseout() -> None:
    """Instances should be recognizable as BaseOut."""
    out = outputs.Out(path="file.txt")
    metric = outputs.Metric(path="metrics.json")
    plot = outputs.Plot(path="loss.csv")

    assert isinstance(out, outputs.BaseOut)
    assert isinstance(metric, outputs.BaseOut)
    assert isinstance(plot, outputs.BaseOut)


# IncrementalOut tests


def test_incremental_out_cache_default_true() -> None:
    """IncrementalOut should have cache=True by default."""
    inc = outputs.IncrementalOut(path="database.csv")
    assert inc.cache is True


def test_incremental_out_persist_default_false() -> None:
    """IncrementalOut should have persist=False by default."""
    inc = outputs.IncrementalOut(path="database.csv")
    assert inc.persist is False


def test_incremental_out_frozen() -> None:
    """IncrementalOut should be immutable (frozen dataclass)."""
    inc = outputs.IncrementalOut(path="database.csv")
    with pytest.raises(AttributeError):
        inc.path = "other.csv"  # type: ignore[misc]


def test_incremental_out_is_base_out() -> None:
    """IncrementalOut should inherit from BaseOut."""
    assert issubclass(outputs.IncrementalOut, outputs.BaseOut)
    inc = outputs.IncrementalOut(path="database.csv")
    assert isinstance(inc, outputs.BaseOut)


def test_normalize_out_incremental_passthrough() -> None:
    """IncrementalOut should pass through normalize_out unchanged."""
    inc = outputs.IncrementalOut(path="database.csv")
    assert outputs.normalize_out(inc) is inc
