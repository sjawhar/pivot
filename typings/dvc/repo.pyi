"""Type stubs for dvc.repo."""

from typing import Any

class Index:
    """DVC repository index."""

    stages: list[Any]  # Can be PipelineStage or other stage types

class Repo:
    """DVC repository."""

    def __init__(self, path: str) -> None: ...

    index: Index
