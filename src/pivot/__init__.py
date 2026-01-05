"""Pivot: High-Performance Python Pipeline Tool.

Public API for defining and running pipelines.

Example:
    >>> from pivot import stage
    >>>
    >>> @stage(deps=['data.csv'], outs=['processed.parquet'])
    >>> def preprocess(input_file: str = 'data.csv'):
    ...     import pandas as pd
    ...     df = pd.read_csv(input_file)
    ...     df = df.dropna()
    ...     df.to_parquet('processed.parquet')
    >>>
    >>> @stage(deps=['processed.parquet'], outs=['model.pkl'])
    >>> def train(data_file: str = 'processed.parquet', lr: float = 0.01):
    ...     import pandas as pd
    ...     df = pd.read_parquet(data_file)
    ...     # Train model...
"""

from pivot.registry import REGISTRY as REGISTRY
from pivot.registry import stage as stage

__version__ = "0.1.0-dev"
