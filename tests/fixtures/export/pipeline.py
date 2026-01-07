"""Sample pipeline stages for export CLI tests.

These are module-level functions that can be exported to DVC YAML format.
They are registered manually in tests to avoid side effects at import time.
"""

from pydantic import BaseModel


class TrainParams(BaseModel):
    """Parameters for training stage."""

    learning_rate: float = 0.01
    epochs: int = 100


def preprocess() -> None:
    """Preprocess data stage."""
    pass


def train(params: TrainParams) -> None:
    """Train model stage with Pydantic parameters."""
    pass


def evaluate() -> None:
    """Evaluate model stage."""
    pass
