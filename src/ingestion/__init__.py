"""
Data ingestion module - abstract and extensible.
"""
from .base import (
    Record,
    DataConnector,
    EntityExtractor,
    SentimentClassifier,
    TopicClassifier
)

from .connectors import (
    CSVConnector,
    JSONLinesConnector,
    DirectoryConnector
)

from .classifiers import (
    KeywordEntityExtractor,
    SpacyEntityExtractor,
    VADERSentimentClassifier,
    TransformerSentimentClassifier,
    ZeroShotTopicClassifier,
    KeywordTopicClassifier
)

from .pipeline import IngestionPipeline

__all__ = [
    # Base classes
    'Record',
    'DataConnector',
    'EntityExtractor',
    'SentimentClassifier',
    'TopicClassifier',

    # Connectors
    'CSVConnector',
    'JSONLinesConnector',
    'DirectoryConnector',

    # Classifiers
    'KeywordEntityExtractor',
    'SpacyEntityExtractor',
    'VADERSentimentClassifier',
    'TransformerSentimentClassifier',
    'ZeroShotTopicClassifier',
    'KeywordTopicClassifier',

    # Pipeline
    'IngestionPipeline',
]
