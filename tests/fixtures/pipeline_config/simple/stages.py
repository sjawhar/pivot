"""Simple stage functions for testing pivot.yaml loading."""

from __future__ import annotations


def preprocess() -> None:
    """Read raw data and write clean data."""
    with open("data/raw.csv") as f:
        content = f.read()
    with open("data/clean.csv", "w") as f:
        f.write(content.upper())


def train() -> None:
    """Read clean data and write model."""
    with open("data/clean.csv") as f:
        content = f.read()
    with open("models/model.pkl", "w") as f:
        f.write(f"MODEL:{len(content)}")
