"""
Universal data ingestion pipeline.
Coordinates connectors, classifiers, and graph building.
"""
from typing import Optional, Dict, Any
from tqdm import tqdm

from src.graph.engine import TemporalGraphEngine
from src.graph.schema import Node, TemporalEdge, MarketingTopic, Sentiment
from .base import DataConnector, EntityExtractor, SentimentClassifier, TopicClassifier


# Map string sentiments to enum
SENTIMENT_MAP = {
    "POSITIVE": Sentiment.POSITIVE,
    "NEGATIVE": Sentiment.NEGATIVE,
    "NEUTRAL": Sentiment.NEUTRAL,
    "positive": Sentiment.POSITIVE,
    "negative": Sentiment.NEGATIVE,
    "neutral": Sentiment.NEUTRAL,
}

# Map string topics to enum
TOPIC_MAP = {
    "Product Quality & Durability": MarketingTopic.QUALITY,
    "Price & Value": MarketingTopic.PRICE,
    "Customer Service & Delivery": MarketingTopic.SERVICE,
    "Performance & Reliability": MarketingTopic.PERFORMANCE,
    "Ease of Use & Design": MarketingTopic.USABILITY,
    "General Experience": MarketingTopic.GENERAL,
    "General": MarketingTopic.GENERAL,
    # Lowercase variants
    "product quality and durability": MarketingTopic.QUALITY,
    "price and value for money": MarketingTopic.PRICE,
    "customer service and delivery": MarketingTopic.SERVICE,
    "performance and speed": MarketingTopic.PERFORMANCE,
    "ease of use and design": MarketingTopic.USABILITY,
    "general opinion": MarketingTopic.GENERAL,
}


class IngestionPipeline:
    """
    Universal ingestion pipeline that works with any data source.

    Example usage:
        pipeline = IngestionPipeline(
            connector=CSVConnector({...}),
            entity_extractor=KeywordEntityExtractor(entities=["Apple", "Samsung"]),
            sentiment_classifier=VADERSentimentClassifier(),
            topic_classifier=ZeroShotTopicClassifier(topics=[...])
        )
        pipeline.run(graph_engine, limit=10000)
    """

    def __init__(self,
                 connector: DataConnector,
                 entity_extractor: Optional[EntityExtractor] = None,
                 sentiment_classifier: Optional[SentimentClassifier] = None,
                 topic_classifier: Optional[TopicClassifier] = None):
        """
        Args:
            connector: Data source connector
            entity_extractor: Entity extraction (if None, uses record.entity field)
            sentiment_classifier: Sentiment classification (if None, uses record.sentiment field)
            topic_classifier: Topic classification (if None, uses record.topic field)
        """
        self.connector = connector
        self.entity_extractor = entity_extractor
        self.sentiment_classifier = sentiment_classifier
        self.topic_classifier = topic_classifier

        self.stats = {
            "processed": 0,
            "added": 0,
            "skipped_no_entity": 0,
            "skipped_short_text": 0,
            "skipped_no_date": 0,
            "errors": 0,
        }

    def run(self, graph: TemporalGraphEngine, limit: Optional[int] = None, batch_size: int = 32):
        """
        Run the ingestion pipeline.

        Args:
            graph: Graph engine to populate
            limit: Maximum number of records to process (None for all)
            batch_size: Number of records to batch for topic classification
        """
        print(f"[Pipeline] Starting ingestion...")
        print(f"  Connector: {self.connector.__class__.__name__}")
        print(f"  Entity Extractor: {self.entity_extractor.__class__.__name__ if self.entity_extractor else 'Field-based'}")
        print(f"  Sentiment Classifier: {self.sentiment_classifier.__class__.__name__ if self.sentiment_classifier else 'Field-based'}")
        print(f"  Topic Classifier: {self.topic_classifier.__class__.__name__ if self.topic_classifier else 'Field-based'}")

        records = []
        for record in tqdm(self.connector.load_records(), desc="Loading records"):
            self.stats["processed"] += 1

            if limit and self.stats["processed"] > limit:
                break

            # Extract entity (if extractor provided AND record doesn't already have entity)
            if not record.entity and self.entity_extractor:
                entity = self.entity_extractor.extract(record.text)
                if not entity:
                    self.stats["skipped_no_entity"] += 1
                    continue
                record.entity = entity

            # Validate entity exists
            if not record.entity:
                self.stats["skipped_no_entity"] += 1
                continue

            # Validate text length
            if len(record.text) < 10:
                self.stats["skipped_short_text"] += 1
                continue

            # Validate date
            if not record.date:
                self.stats["skipped_no_date"] += 1
                continue

            records.append(record)

            # Process in batches (for efficiency with topic classification)
            if len(records) >= batch_size:
                self._process_batch(graph, records)
                records = []

        # Process remaining records
        if records:
            self._process_batch(graph, records)

        print(f"\n[Pipeline] Complete!")
        print(f"  Processed: {self.stats['processed']}")
        print(f"  Added to graph: {self.stats['added']}")
        print(f"  Skipped (no entity): {self.stats['skipped_no_entity']}")
        print(f"  Skipped (short text): {self.stats['skipped_short_text']}")
        print(f"  Skipped (no date): {self.stats['skipped_no_date']}")
        print(f"  Errors: {self.stats['errors']}")

        return self.stats

    def _process_batch(self, graph: TemporalGraphEngine, records: list):
        """Process a batch of records (classify and add to graph)"""

        # Batch sentiment classification
        if self.sentiment_classifier:
            for record in records:
                if not record.sentiment:
                    try:
                        record.sentiment = self.sentiment_classifier.classify(record.text)
                    except Exception as e:
                        self.stats["errors"] += 1
                        continue

        # Batch topic classification
        if self.topic_classifier:
            texts = [r.text for r in records if not r.topic]
            if texts:
                try:
                    # For classifiers that support batch processing
                    if hasattr(self.topic_classifier, 'classify_batch'):
                        topics = self.topic_classifier.classify_batch(texts)
                    else:
                        topics = [self.topic_classifier.classify(text) for text in texts]

                    topic_idx = 0
                    for record in records:
                        if not record.topic:
                            record.topic = topics[topic_idx]
                            topic_idx += 1
                except Exception as e:
                    # Fallback: classify individually
                    for record in records:
                        if not record.topic:
                            try:
                                record.topic = self.topic_classifier.classify(record.text)
                            except:
                                record.topic = "General"

        # Add to graph
        for record in records:
            try:
                # Map sentiment string to enum
                sentiment_enum = SENTIMENT_MAP.get(record.sentiment, Sentiment.NEUTRAL)

                # Map topic string to enum
                topic_enum = TOPIC_MAP.get(record.topic, MarketingTopic.GENERAL)

                # Create nodes
                brand_node = Node(
                    id=record.entity,
                    type="Brand",
                    properties={"name": record.entity}
                )

                review_id = f"{record.entity}_{record.date.isoformat()}_{hash(record.text) % 100000}"
                review_node = Node(
                    id=review_id,
                    type="Review",
                    properties={
                        "text": record.text,
                        "rating": record.metadata.get("rating") if record.metadata else None
                    }
                )

                # Create edge
                edge = TemporalEdge(
                    source=record.entity,
                    target=review_id,
                    relation="HAS_REVIEW",
                    topic=topic_enum,
                    sentiment=sentiment_enum,
                    start_date=record.date,
                    end_date=None,
                    metadata=record.metadata or {}
                )

                graph.add_data(brand_node, review_node, edge)
                self.stats["added"] += 1

            except Exception as e:
                self.stats["errors"] += 1
                continue
