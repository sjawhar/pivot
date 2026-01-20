import pydantic
import pytest

from pivot import config, exceptions

# --- PivotConfig defaults tests ---


def test_config_defaults_has_cache_section() -> None:
    defaults = config.PivotConfig.get_default()
    assert defaults.cache.dir == ".pivot/cache"
    assert defaults.cache.checkout_mode == [
        config.CheckoutMode.HARDLINK,
        config.CheckoutMode.SYMLINK,
        config.CheckoutMode.COPY,
    ]


def test_config_defaults_has_core_section() -> None:
    defaults = config.PivotConfig.get_default()
    assert defaults.core.max_workers == -2
    assert defaults.core.state_dir == ".pivot"


def test_config_defaults_has_remote_section() -> None:
    defaults = config.PivotConfig.get_default()
    assert defaults.remote.jobs == 20
    assert defaults.remote.retries == 10
    assert defaults.remote.connect_timeout == 30


def test_config_defaults_has_watch_section() -> None:
    defaults = config.PivotConfig.get_default()
    assert defaults.watch.debounce == 300


def test_config_defaults_has_display_section() -> None:
    defaults = config.PivotConfig.get_default()
    assert defaults.display.precision == 5


def test_config_defaults_has_diff_section() -> None:
    defaults = config.PivotConfig.get_default()
    assert defaults.diff.max_rows == 10000


def test_config_defaults_has_remotes_section() -> None:
    defaults = config.PivotConfig.get_default()
    assert defaults.remotes == {}


def test_config_defaults_has_default_remote() -> None:
    defaults = config.PivotConfig.get_default()
    assert defaults.default_remote == ""


def test_pivot_config_rejects_invalid_remote_name() -> None:
    """PivotConfig Pydantic validator rejects invalid remote names."""
    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.PivotConfig(remotes={"has space": "s3://bucket"})
    assert "invalid remote name" in str(exc_info.value).lower()


def test_pivot_config_accepts_valid_remote_names() -> None:
    """PivotConfig Pydantic validator accepts valid remote names."""
    cfg = config.PivotConfig(remotes={"my-remote": "s3://bucket", "my_remote": "s3://bucket2"})
    assert cfg.remotes == {"my-remote": "s3://bucket", "my_remote": "s3://bucket2"}


# --- validate_config_value tests ---


def test_validate_cache_dir_accepts_string() -> None:
    result = config.validate_config_value("cache.dir", "/tmp/cache")
    assert result == "/tmp/cache"


def test_validate_cache_dir_accepts_relative_path() -> None:
    result = config.validate_config_value("cache.dir", ".pivot/cache")
    assert result == ".pivot/cache"


def test_validate_cache_checkout_mode_accepts_valid_list() -> None:
    result = config.validate_config_value("cache.checkout_mode", ["hardlink", "copy"])
    assert result == ["hardlink", "copy"]


def test_validate_cache_checkout_mode_accepts_comma_separated_string() -> None:
    result = config.validate_config_value("cache.checkout_mode", "symlink,copy")
    assert result == ["symlink", "copy"]


def test_validate_cache_checkout_mode_accepts_single_mode() -> None:
    result = config.validate_config_value("cache.checkout_mode", "hardlink")
    assert result == ["hardlink"]


def test_validate_cache_checkout_mode_rejects_invalid_mode() -> None:
    with pytest.raises(exceptions.ConfigValidationError) as exc_info:
        config.validate_config_value("cache.checkout_mode", "invalid")
    assert "hardlink" in str(exc_info.value), "Error should mention valid options"


def test_validate_core_max_workers_accepts_positive_int() -> None:
    result = config.validate_config_value("core.max_workers", 4)
    assert result == 4


def test_validate_core_max_workers_accepts_negative_int() -> None:
    result = config.validate_config_value("core.max_workers", -2)
    assert result == -2


def test_validate_core_max_workers_accepts_string_int() -> None:
    result = config.validate_config_value("core.max_workers", "8")
    assert result == 8


def test_validate_core_max_workers_rejects_zero() -> None:
    with pytest.raises(exceptions.ConfigValidationError) as exc_info:
        config.validate_config_value("core.max_workers", 0)
    assert "cannot be 0" in str(exc_info.value) or "zero" in str(exc_info.value).lower()


def test_validate_core_max_workers_rejects_non_int() -> None:
    with pytest.raises(exceptions.ConfigValidationError) as exc_info:
        config.validate_config_value("core.max_workers", "not_a_number")
    assert "integer" in str(exc_info.value).lower()


def test_validate_core_state_dir_accepts_string() -> None:
    result = config.validate_config_value("core.state_dir", "/tmp/state")
    assert result == "/tmp/state"


def test_validate_remote_jobs_accepts_positive_int() -> None:
    result = config.validate_config_value("remote.jobs", 10)
    assert result == 10


def test_validate_remote_jobs_rejects_zero() -> None:
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("remote.jobs", 0)


def test_validate_remote_jobs_rejects_negative() -> None:
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("remote.jobs", -1)


def test_validate_remote_retries_accepts_zero() -> None:
    result = config.validate_config_value("remote.retries", 0)
    assert result == 0


def test_validate_remote_retries_accepts_positive() -> None:
    result = config.validate_config_value("remote.retries", 5)
    assert result == 5


def test_validate_remote_retries_rejects_negative() -> None:
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("remote.retries", -1)


def test_validate_remote_connect_timeout_accepts_positive() -> None:
    result = config.validate_config_value("remote.connect_timeout", 60)
    assert result == 60


def test_validate_remote_connect_timeout_rejects_zero() -> None:
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("remote.connect_timeout", 0)


def test_validate_watch_debounce_accepts_positive() -> None:
    result = config.validate_config_value("watch.debounce", 500)
    assert result == 500


def test_validate_watch_debounce_accepts_zero() -> None:
    result = config.validate_config_value("watch.debounce", 0)
    assert result == 0


def test_validate_watch_debounce_rejects_negative() -> None:
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("watch.debounce", -1)


def test_validate_display_precision_accepts_valid_range() -> None:
    result = config.validate_config_value("display.precision", 3)
    assert result == 3


def test_validate_display_precision_accepts_zero() -> None:
    result = config.validate_config_value("display.precision", 0)
    assert result == 0


def test_validate_display_precision_rejects_negative() -> None:
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("display.precision", -1)


def test_validate_diff_max_rows_accepts_positive() -> None:
    result = config.validate_config_value("diff.max_rows", 5000)
    assert result == 5000


def test_validate_diff_max_rows_rejects_zero() -> None:
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("diff.max_rows", 0)


def test_validate_remotes_accepts_valid_s3_url() -> None:
    result = config.validate_config_value("remotes.origin", "s3://my-bucket/prefix")
    assert result == "s3://my-bucket/prefix"


def test_validate_remotes_accepts_s3_url_without_prefix() -> None:
    result = config.validate_config_value("remotes.backup", "s3://bucket")
    assert result == "s3://bucket"


def test_validate_remotes_rejects_invalid_url() -> None:
    with pytest.raises(exceptions.ConfigValidationError) as exc_info:
        config.validate_config_value("remotes.origin", "not-an-s3-url")
    assert "s3://" in str(exc_info.value).lower()


def test_validate_remotes_rejects_http_url() -> None:
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("remotes.origin", "http://example.com")


def test_validate_remotes_rejects_invalid_remote_name_with_dot() -> None:
    # A dot in the remote name creates a 3-part key, which is invalid
    with pytest.raises(exceptions.ConfigValidationError):
        config.validate_config_value("remotes.has.dot", "s3://bucket")


def test_validate_remotes_rejects_invalid_remote_name_with_space() -> None:
    with pytest.raises(exceptions.ConfigValidationError) as exc_info:
        config.validate_config_value("remotes.has space", "s3://bucket")
    assert "invalid remote name" in str(exc_info.value).lower()


def test_validate_remotes_accepts_valid_remote_names() -> None:
    # alphanumeric, hyphens, underscores are valid
    assert config.validate_config_value("remotes.origin", "s3://bucket") == "s3://bucket"
    assert config.validate_config_value("remotes.my-remote", "s3://bucket") == "s3://bucket"
    assert config.validate_config_value("remotes.my_remote", "s3://bucket") == "s3://bucket"
    assert config.validate_config_value("remotes.Remote123", "s3://bucket") == "s3://bucket"


def test_validate_default_remote_accepts_string() -> None:
    result = config.validate_config_value("default_remote", "origin")
    assert result == "origin"


def test_validate_default_remote_accepts_empty_string() -> None:
    result = config.validate_config_value("default_remote", "")
    assert result == ""


def test_validate_unknown_key_raises_error() -> None:
    with pytest.raises(exceptions.ConfigValidationError) as exc_info:
        config.validate_config_value("unknown.key", "value")
    assert "unknown" in str(exc_info.value).lower()


# --- get_config_default tests ---


def test_get_config_default_returns_default_for_cache_dir() -> None:
    result = config.get_config_default("cache.dir")
    assert result == ".pivot/cache"


def test_get_config_default_returns_default_for_core_max_workers() -> None:
    result = config.get_config_default("core.max_workers")
    assert result == -2


def test_get_config_default_returns_none_for_unknown_key() -> None:
    result = config.get_config_default("unknown.key")
    assert result is None


def test_get_config_default_returns_none_for_remotes() -> None:
    result = config.get_config_default("remotes.origin")
    assert result is None


# --- is_valid_key tests ---


def test_is_valid_key_returns_true_for_known_keys() -> None:
    assert config.is_valid_key("cache.dir") is True
    assert config.is_valid_key("core.max_workers") is True
    assert config.is_valid_key("remote.jobs") is True


def test_is_valid_key_returns_true_for_remotes_pattern() -> None:
    assert config.is_valid_key("remotes.origin") is True
    assert config.is_valid_key("remotes.backup") is True
    assert config.is_valid_key("remotes.my-remote") is True


def test_is_valid_key_returns_false_for_unknown_keys() -> None:
    assert config.is_valid_key("unknown.key") is False
    assert config.is_valid_key("foo.bar.baz") is False
