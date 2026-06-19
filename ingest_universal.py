"""
Universal data ingestion script.
Works with any data source via the abstracted ingestion pipeline.

Examples:
  # CSV with automatic classification
  python ingest_universal.py --config configs/amazon_reviews.yaml

  # JSON with pre-labeled sentiment
  python ingest_universal.py --config configs/support_tickets.yaml

  # Directory of files
  python ingest_universal.py --config configs/social_media.yaml
"""
import argparse
import yaml
from pathlib import Path

from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.ingestion import (
    IngestionPipeline,
    CSVConnector,
    JSONLinesConnector,
    DirectoryConnector,
    KeywordEntityExtractor,
    SpacyEntityExtractor,
    VADERSentimentClassifier,
    TransformerSentimentClassifier,
    ZeroShotTopicClassifier,
    KeywordTopicClassifier,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_connector(config: dict):
    """Factory function to create connector from config"""
    connector_type = config['connector']['type']
    connector_config = config['connector']['config']

    if connector_type == 'csv':
        return CSVConnector(connector_config)
    elif connector_type == 'jsonl':
        return JSONLinesConnector(connector_config)
    elif connector_type == 'directory':
        return DirectoryConnector(connector_config)
    else:
        raise ValueError(f"Unknown connector type: {connector_type}")


def create_entity_extractor(config: dict):
    """Factory function to create entity extractor from config"""
    if 'entity_extractor' not in config:
        return None

    extractor_config = config['entity_extractor']
    extractor_type = extractor_config['type']

    if extractor_type == 'keyword':
        entities = set(extractor_config['entities'])
        case_sensitive = extractor_config.get('case_sensitive', False)
        return KeywordEntityExtractor(entities, case_sensitive)
    elif extractor_type == 'spacy':
        entity_types = extractor_config['entity_types']
        model = extractor_config.get('model', 'en_core_web_sm')
        return SpacyEntityExtractor(entity_types, model)
    elif extractor_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown entity extractor type: {extractor_type}")


def create_sentiment_classifier(config: dict):
    """Factory function to create sentiment classifier from config"""
    if 'sentiment_classifier' not in config:
        return None

    classifier_config = config['sentiment_classifier']
    classifier_type = classifier_config['type']

    if classifier_type == 'vader':
        pos_threshold = classifier_config.get('pos_threshold', 0.05)
        neg_threshold = classifier_config.get('neg_threshold', -0.05)
        return VADERSentimentClassifier(pos_threshold, neg_threshold)
    elif classifier_type == 'transformer':
        model = classifier_config.get('model', 'distilbert-base-uncased-finetuned-sst-2-english')
        return TransformerSentimentClassifier(model)
    elif classifier_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown sentiment classifier type: {classifier_type}")


def create_topic_classifier(config: dict):
    """Factory function to create topic classifier from config"""
    if 'topic_classifier' not in config:
        return None

    classifier_config = config['topic_classifier']
    classifier_type = classifier_config['type']

    if classifier_type == 'zero-shot':
        topics = classifier_config['topics']
        model = classifier_config.get('model', 'facebook/bart-large-mnli')
        confidence_threshold = classifier_config.get('confidence_threshold', 0.4)
        return ZeroShotTopicClassifier(topics, model, confidence_threshold)
    elif classifier_type == 'keyword':
        keyword_map = classifier_config['keyword_map']
        default_topic = classifier_config.get('default_topic', 'General')
        return KeywordTopicClassifier(keyword_map, default_topic)
    elif classifier_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown topic classifier type: {classifier_type}")


def main():
    parser = argparse.ArgumentParser(description="Universal data ingestion")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--output', default=None, help='Output graph file path')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records')
    args = parser.parse_args()

    print("=" * 80)
    print("   UNIVERSAL DATA INGESTION")
    print("=" * 80)

    # Load configuration
    print(f"\n[1/5] Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Create components
    print("[2/5] Creating pipeline components...")
    connector = create_connector(config)
    entity_extractor = create_entity_extractor(config)
    sentiment_classifier = create_sentiment_classifier(config)
    topic_classifier = create_topic_classifier(config)

    # Create pipeline
    print("[3/5] Initializing pipeline...")
    pipeline = IngestionPipeline(
        connector=connector,
        entity_extractor=entity_extractor,
        sentiment_classifier=sentiment_classifier,
        topic_classifier=topic_classifier
    )

    # Create graph
    print("[4/5] Creating graph engine...")
    graph = TemporalGraphEngine()

    # Run ingestion
    print("[5/5] Running ingestion...")
    limit = args.limit or config.get('limit')
    batch_size = config.get('batch_size', 32)
    stats = pipeline.run(graph, limit=limit, batch_size=batch_size)

    # Validate
    if graph.graph.number_of_edges() == 0:
        print("\n[ERROR] Graph is empty. Check your data and configuration.")
        return

    # Save graph
    output_path = args.output or config.get('output', settings.graph_path)
    print(f"\n[Saving] Writing graph to {output_path}...")
    graph.save_to_disk(output_path)

    # Print summary
    graph_stats = graph.stats()
    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)
    print(f"  Total edges: {graph_stats['total_edges']}")
    print(f"  Entities: {graph_stats['brand_count']}")
    print(f"  Entity list: {', '.join(graph_stats['brands'][:10])}")
    if len(graph_stats['brands']) > 10:
        print(f"  ... and {len(graph_stats['brands']) - 10} more")
    print("=" * 80)


if __name__ == "__main__":
    main()
