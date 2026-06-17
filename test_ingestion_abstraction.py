#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the abstracted ingestion system.
Verifies all components work correctly.
"""
import sys
import tempfile

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
import csv
from datetime import datetime
from pathlib import Path

from src.graph.engine import TemporalGraphEngine
from src.ingestion import (
    CSVConnector,
    KeywordEntityExtractor,
    VADERSentimentClassifier,
    KeywordTopicClassifier,
    IngestionPipeline
)


def create_test_csv():
    """Create a test CSV file"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
    writer = csv.writer(temp_file)

    # Write header
    writer.writerow(['date', 'company', 'feedback', 'rating'])

    # Write test data
    test_data = [
        ('2023-01-15', 'TechCorp', 'The product quality is excellent and very durable', '5'),
        ('2023-02-20', 'TechCorp', 'Great customer service, fast delivery', '5'),
        ('2023-03-10', 'GadgetCo', 'Too expensive for what you get', '2'),
        ('2023-04-05', 'GadgetCo', 'Product broke after one week, very disappointing', '1'),
        ('2023-05-12', 'TechCorp', 'Average performance, nothing special', '3'),
        ('2023-06-18', 'GadgetCo', 'Love the design, easy to use', '5'),
    ]

    for row in test_data:
        writer.writerow(row)

    temp_file.close()
    return temp_file.name


def test_ingestion():
    print("=" * 80)
    print("TESTING ABSTRACTED INGESTION SYSTEM")
    print("=" * 80)

    # Step 1: Create test data
    print("\n[1/6] Creating test CSV...")
    csv_path = create_test_csv()
    print(f"  Created: {csv_path}")

    # Step 2: Configure connector
    print("\n[2/6] Configuring CSV connector...")
    connector = CSVConnector({
        'path': csv_path,
        'entity_col': 'company',
        'text_col': 'feedback',
        'date_col': 'date',
        'date_format': '%Y-%m-%d'
    })

    # Step 3: Configure extractors/classifiers
    print("\n[3/6] Configuring classifiers...")

    entity_extractor = KeywordEntityExtractor(
        entities={'TechCorp', 'GadgetCo'},
        case_sensitive=False
    )

    sentiment_classifier = VADERSentimentClassifier()

    topic_classifier = KeywordTopicClassifier(
        keyword_map={
            'Quality': ['quality', 'durable', 'broke', 'broken'],
            'Service': ['service', 'delivery', 'support'],
            'Price': ['expensive', 'cheap', 'cost', 'price'],
            'Design': ['design', 'easy to use', 'user-friendly'],
            'Performance': ['performance', 'speed', 'fast', 'slow']
        }
    )

    # Step 4: Create pipeline
    print("\n[4/6] Creating ingestion pipeline...")
    pipeline = IngestionPipeline(
        connector=connector,
        entity_extractor=entity_extractor,
        sentiment_classifier=sentiment_classifier,
        topic_classifier=topic_classifier
    )

    # Step 5: Run ingestion
    print("\n[5/6] Running ingestion...")
    graph = TemporalGraphEngine()
    stats = pipeline.run(graph, limit=None, batch_size=3)

    # Step 6: Verify results
    print("\n[6/6] Verifying results...")

    assert stats['added'] > 0, "No records were added to graph"
    assert graph.graph.number_of_edges() > 0, "Graph is empty"

    # Test querying
    brands = graph.get_brands()
    print(f"\n  Extracted brands: {brands}")
    assert len(brands) >= 2, "Expected at least 2 brands"

    # Query for TechCorp in 2023
    facts = graph.get_snapshot_for_year('TechCorp', 2023)
    print(f"  TechCorp 2023 facts: {len(facts)}")
    assert len(facts) > 0, "No facts found for TechCorp"

    # Check sentiment distribution
    sentiments = [f.sentiment.value for f in facts]
    print(f"  TechCorp sentiments: {sentiments}")

    # Check topics
    topics = [f.topic.value for f in facts]
    print(f"  TechCorp topics: {topics}")

    # Cleanup
    Path(csv_path).unlink()

    print("\n" + "=" * 80)
    print("[PASS] ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe abstracted ingestion system is working correctly.")
    print("You can now use it with your own data sources!")


if __name__ == "__main__":
    try:
        test_ingestion()
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
