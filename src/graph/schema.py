from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum
from datetime import datetime

# 1. THE VOCABULARY
class MarketingTopic(Enum):
    # Updated for Amazon/Product context
    QUALITY = "Product Quality, Durability, and Build"
    PRICE = "Price, Value for Money, and Cost"
    SERVICE = "Customer Service, Shipping, and Returns"
    PERFORMANCE = "Performance, Speed, and Reliability"
    USABILITY = "Ease of Use and Design"
    GENERAL = "General Experience"

class Sentiment(Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

# 2. THE STRUCTURES
@dataclass
class Node:
    id: str       # e.g., "Apple"
    type: str     # e.g., "Brand" or "Review"
    properties: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class TemporalEdge:
    source: str   # "Apple"
    target: str   # "Review_123"
    relation: str # "REVIEWED_IN"
    topic: MarketingTopic
    sentiment: Sentiment
    start_date: datetime
    end_date: Optional[datetime] = None
    
    # 3. THE TIME MACHINE LOGIC (Stays the same)
    def exists_at(self, query_date: datetime) -> bool:
        if self.start_date > query_date:
            return False 
        return True