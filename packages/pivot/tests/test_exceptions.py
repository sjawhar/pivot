# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

from pivot import exceptions


def test_dependency_not_found_error_formats_fuzzy_suggestion() -> None:
    err = exceptions.DependencyNotFoundError(
        "train",
        "dat/raw.csv",
        available_outputs=["data/raw.csv", "features/train.parquet"],
    )

    message = err.format_user_message()
    assert "depends on 'dat/raw.csv'" in message
    assert "Did you mean: 'data/raw.csv'?" in message
    assert "produced by any stage" in message


def test_stage_not_found_error_limits_suggestion_count() -> None:
    err = exceptions.StageNotFoundError(
        ["trian", "evla", "predcit", "extra"],
        ["train", "eval", "predict", "score"],
    )

    message = err.format_user_message()
    assert "Unknown stage(s): trian, evla, predcit, extra" in message
    assert "'trian' -> 'train'" in message
    assert "(showing first 3 of 4 unknown stages)" in message


def test_tracked_file_missing_error_messages_and_suggestions() -> None:
    missing_err = exceptions.TrackedFileMissingError(["data/raw.csv", "data/extra.csv"])
    attempted_err = exceptions.TrackedFileMissingError(["data/raw.csv"], checkout_attempted=True)

    assert "don't exist on disk" in str(missing_err)
    assert "Run 'pivot checkout --only-missing'" in missing_err.get_suggestion()

    assert "not in local cache" in str(attempted_err)
    assert attempted_err.get_suggestion() == "Run 'pivot pull' to fetch from remote storage"


def test_get_suggestion_overrides_cover_remote_and_config_errors() -> None:
    assert exceptions.CyclicGraphError().get_suggestion() == (
        "Check stage dependencies for circular references"
    )
    assert exceptions.RemoteNotConfiguredError().get_suggestion() == (
        "Run 'pivot config set remotes.<name> <url>' to configure a remote"
    )
    assert exceptions.ConfigKeyError().get_suggestion() == (
        "Run 'pivot config list' to see available config keys"
    )


def test_reduce_round_trip_for_picklable_errors() -> None:
    dep_err = exceptions.DependencyNotFoundError("train", "raw", ["data/raw"])
    stage_err = exceptions.StageNotFoundError(["trian"], ["train"])
    tracked_err = exceptions.TrackedFileMissingError(["data.csv"], checkout_attempted=True)

    dep_ctor, dep_args = dep_err.__reduce__()
    stage_ctor, stage_args = stage_err.__reduce__()
    tracked_ctor, tracked_args = tracked_err.__reduce__()

    restored_dep = dep_ctor(*dep_args)
    restored_stage = stage_ctor(*stage_args)
    restored_tracked = tracked_ctor(*tracked_args)

    assert isinstance(restored_dep, exceptions.DependencyNotFoundError)
    assert isinstance(restored_stage, exceptions.StageNotFoundError)
    assert isinstance(restored_tracked, exceptions.TrackedFileMissingError)
