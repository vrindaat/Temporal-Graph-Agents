"""
Base classes for data ingestion abstraction.
Allows the framework to work with any data source and domain.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional, Dict, Any, List
from enum import Enum


@dataclass
class Record:
    """
    Universal record format for any data source.
    All connectors must convert their data to this format.
    """
    entity: str  # Brand, product, person, organization, etc.
    text: str  # Review text, feedback, comment, etc.
    date: datetime  # Timestamp
    sentiment: Optional[str] = None  # POSITIVE, NEGATIVE, NEUTRAL (can be auto-detected)
    topic: Optional[str] = None  # Can be auto-classified
    metadata: Dict[str, Any] = None  # Additional fields (rating, user_id, etc.)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataConnector(ABC):
    """
    Abstract base class for data connectors.
    Implement this to add support for new data sources.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Connector-specific configuration
                e.g., {"path": "data.csv", "entity_col": "brand", ...}
        """
        self.config = config

    @abstractmethod
    def load_records(self) -> Iterator[Record]:
        """
        Yield records from the data source.
        Must convert source data to Record format.
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that required config fields are present"""
        pass


class EntityExtractor(ABC):
    """
    Abstract base class for entity extraction.
    Replaces hardcoded brand lists.
    """

    @abstractmethod
    def extract(self, text: str) -> Optional[str]:
        """Extract the main entity from text (brand, product, person, etc.)"""
        pass


class SentimentClassifier(ABC):
    """Abstract base class for sentiment classification"""

    @abstractmethod
    def classify(self, text: str) -> str:
        """Return: POSITIVE, NEGATIVE, or NEUTRAL"""
        pass


class TopicClassifier(ABC):
    """Abstract base class for topic classification"""

    @abstractmethod
    def classify(self, text: str) -> str:
        """Return topic label (domain-specific)"""
        pass

    @abstractmethod
    def get_available_topics(self) -> List[str]:
        """Return list of possible topic labels"""
        pass
