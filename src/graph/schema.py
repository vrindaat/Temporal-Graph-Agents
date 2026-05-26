from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class MarketingTopic(Enum):
    QUALITY = "Product Quality & Durability"
    PRICE = "Price & Value"
    SERVICE = "Customer Service & Shipping"
    PERFORMANCE = "Performance & Reliability"
    USABILITY = "Ease of Use & Design"
    GENERAL = "General Experience"


class Sentiment(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass
class Node:
    id: str
    type: str  # "Brand" or "Review"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalEdge:
    source: str
    target: str
    relation: str
    topic: MarketingTopic
    sentiment: Sentiment
    start_date: datetime
    end_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def exists_at(self, query_date: datetime) -> bool:
        """Check if this edge is valid at the given date."""
        if self.start_date > query_date:
            return False
        if self.end_date is not None and self.end_date < query_date:
            return False
        return True

    def exists_in_range(self, range_start: datetime, range_end: datetime) -> bool:
        """Check if this edge overlaps with a date range."""
        if self.start_date > range_end:
            return False
        if self.end_date is not None and self.end_date < range_start:
            return False
        return True
