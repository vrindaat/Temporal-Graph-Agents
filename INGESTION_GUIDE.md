# Data Ingestion Guide

This guide explains how to use the abstracted data ingestion system to work with your own data.

## Overview

The ingestion system is **fully abstracted** and can work with any data source (CSV, JSON, databases, APIs) and any domain (reviews, support tickets, social media, surveys, etc.).

## Architecture

```
Data Source → Connector → Entity Extractor → Sentiment Classifier → Topic Classifier → Knowledge Graph
```

### Components

1. **Data Connector**: Loads data from source (CSV, JSON, database, API)
2. **Entity Extractor**: Identifies the main entity in each record (brand, product, person, etc.)
3. **Sentiment Classifier**: Classifies sentiment (POSITIVE, NEGATIVE, NEUTRAL)
4. **Topic Classifier**: Categorizes content into topics
5. **Knowledge Graph**: Stores temporal relationships

## Quick Start

### Option 1: Use Existing Configuration

```bash
# For Amazon reviews (original dataset)
python ingest_universal.py --config configs/amazon_reviews.yaml

# Limit to 1000 records for testing
python ingest_universal.py --config configs/amazon_reviews.yaml --limit 1000
```

### Option 2: Create Custom Configuration

1. Copy a template config:
   ```bash
   cp configs/custom_csv_template.yaml configs/my_data.yaml
   ```

2. Edit `configs/my_data.yaml` to match your data format

3. Run ingestion:
   ```bash
   python ingest_universal.py --config configs/my_data.yaml
   ```

## Configuration Format

### Basic Structure

```yaml
# Data source
connector:
  type: csv  # or jsonl, directory
  config:
    path: ./data/my_data.csv
    entity_col: brand
    text_col: review_text
    date_col: timestamp
    date_format: "%Y-%m-%d"

# Entity extraction (optional if entity is in data)
entity_extractor:
  type: keyword
  entities: [Apple, Samsung, Google]

# Sentiment classification
sentiment_classifier:
  type: vader

# Topic classification
topic_classifier:
  type: zero-shot
  topics: [Quality, Service, Price]

# Processing options
limit: null
batch_size: 32
output: ./output_graph.pkl
```

## Data Connectors

### CSV Connector

For single CSV files:

```yaml
connector:
  type: csv
  config:
    path: ./data/reviews.csv
    entity_col: brand          # Column with entity name
    text_col: review_text      # Column with text content
    date_col: timestamp        # Column with date/timestamp
    sentiment_col: sentiment   # Optional: pre-labeled sentiment
    topic_col: category        # Optional: pre-labeled topic
    date_format: "%Y-%m-%d"    # Optional: date format
```

### JSONL Connector

For newline-delimited JSON files:

```yaml
connector:
  type: jsonl
  config:
    path: ./data/feedback.jsonl
    entity_field: product.name    # Supports nested fields
    text_field: comment
    date_field: created_at
```

### Directory Connector

For multiple files in a directory:

```yaml
connector:
  type: directory
  config:
    directory: ./data/reviews/
    pattern: "*.csv"              # File pattern to match
    connector_type: csv           # Type of files (csv or jsonl)
    connector_config:
      entity_col: brand
      text_col: text
      date_col: date
```

## Entity Extractors

### Keyword Entity Extractor

Fast, deterministic matching against known entities:

```yaml
entity_extractor:
  type: keyword
  entities:
    - Apple
    - Samsung
    - Microsoft
  case_sensitive: false
```

### spaCy NER Entity Extractor

More flexible, uses Named Entity Recognition:

```yaml
entity_extractor:
  type: spacy
  entity_types:
    - ORG      # Organizations
    - PRODUCT  # Products
    - PERSON   # People
  model: en_core_web_sm
```

### No Entity Extractor

If entity is already in your data:

```yaml
entity_extractor:
  type: none
```

## Sentiment Classifiers

### VADER (Rule-based)

Fast, works well for social media and reviews:

```yaml
sentiment_classifier:
  type: vader
  pos_threshold: 0.05   # Score >= this is POSITIVE
  neg_threshold: -0.05  # Score <= this is NEGATIVE
```

### Transformer-based

More accurate but slower:

```yaml
sentiment_classifier:
  type: transformer
  model: distilbert-base-uncased-finetuned-sst-2-english
```

### No Sentiment Classifier

If sentiment is already in your data:

```yaml
sentiment_classifier:
  type: none
```

## Topic Classifiers

### Zero-Shot (Transformer-based)

No training required, just specify topic labels:

```yaml
topic_classifier:
  type: zero-shot
  topics:
    - Product Quality
    - Customer Service
    - Pricing
    - User Experience
  model: facebook/bart-large-mnli
  confidence_threshold: 0.4
```

### Keyword-based

Fast, interpretable, rule-based:

```yaml
topic_classifier:
  type: keyword
  keyword_map:
    Quality:
      - broken
      - defective
      - durable
      - build quality
    Price:
      - expensive
      - cheap
      - value
      - cost
    Service:
      - support
      - customer service
      - delivery
  default_topic: General
```

### No Topic Classifier

If topic is already in your data:

```yaml
topic_classifier:
  type: none
```

## Example Use Cases

### 1. Product Reviews (CSV with pre-labeled data)

```yaml
connector:
  type: csv
  config:
    path: ./data/reviews.csv
    entity_col: product_name
    text_col: review_text
    date_col: review_date
    sentiment_col: star_rating  # 1-5 stars (will be auto-converted)
    date_format: "%Y-%m-%d"

entity_extractor:
  type: none  # Entity already in data

sentiment_classifier:
  type: vader  # Override star ratings with sentiment analysis

topic_classifier:
  type: zero-shot
  topics: [Quality, Performance, Design, Price]
```

### 2. Support Tickets (JSONL)

```yaml
connector:
  type: jsonl
  config:
    path: ./data/tickets.jsonl
    entity_field: product_id
    text_field: description
    date_field: created_at

entity_extractor:
  type: none

sentiment_classifier:
  type: vader

topic_classifier:
  type: keyword
  keyword_map:
    Bug: [bug, error, crash, broken]
    Feature: [feature, request, suggest]
    Account: [account, login, billing]
```

### 3. Social Media Mentions (Directory of CSVs)

```yaml
connector:
  type: directory
  config:
    directory: ./data/twitter/
    pattern: "mentions_*.csv"
    connector_type: csv
    connector_config:
      entity_col: mentioned_brand
      text_col: tweet_text
      date_col: posted_at

entity_extractor:
  type: keyword
  entities: [BrandA, BrandB, CompetitorX]

sentiment_classifier:
  type: vader

topic_classifier:
  type: zero-shot
  topics: [Product Launch, Customer Service, Brand Perception]
```

## Adding New Data Sources

To add support for a new data source:

1. Create a new connector class in `src/ingestion/connectors.py`:

```python
from .base import DataConnector, Record

class MyCustomConnector(DataConnector):
    def validate_config(self) -> bool:
        # Check required config fields
        return "api_key" in self.config

    def load_records(self) -> Iterator[Record]:
        # Load from your data source
        api_key = self.config["api_key"]
        # ... fetch data ...
        for item in data:
            yield Record(
                entity=item["brand"],
                text=item["text"],
                date=item["timestamp"],
                metadata={"source": "api"}
            )
```

2. Register in factory function in `ingest_universal.py`

3. Use in config:

```yaml
connector:
  type: my_custom
  config:
    api_key: your-key-here
```

## Troubleshooting

### No entities extracted
- Check entity names match your data (case-sensitive option)
- Try `entity_extractor: type: none` if entity is already in data
- Verify `entity_col` points to correct column

### Low sentiment accuracy
- Try switching between VADER and transformer
- Adjust thresholds for your domain
- VADER works best for social media, transformer for formal text

### Wrong topics assigned
- Increase `confidence_threshold` to make classification stricter
- Add more specific keywords for keyword-based classifier
- Use domain-specific topic labels

### Date parsing errors
- Specify `date_format` explicitly
- Check date column has valid values
- Supported formats: ISO 8601, YYYY-MM-DD, MM/DD/YYYY, etc.

## Performance Tips

- Use `limit` parameter for testing before full ingestion
- Keyword extractors are faster than NER
- VADER is faster than transformer models
- Increase `batch_size` if you have more memory
- Use GPU for transformer-based classifiers (set `device=0`)

## Next Steps

After ingestion, use the knowledge graph for temporal analysis:

```python
from src.graph.engine import TemporalGraphEngine

graph = TemporalGraphEngine()
graph.load_from_disk("./my_graph.pkl")

# Query data
facts = graph.get_snapshot_for_year("Apple", 2023)
print(f"Found {len(facts)} reviews for Apple in 2023")
```

See `main.py` for full querying and reporting examples.
