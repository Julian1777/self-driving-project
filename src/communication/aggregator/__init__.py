"""Perception Aggregator Package"""

from src.communication.aggregator import (
    PerceptionAggregator,
    AggregationResult,
    create_aggregator
)

__all__ = [
    'PerceptionAggregator',
    'AggregationResult',
    'create_aggregator'
]

__version__ = '0.1.0'
