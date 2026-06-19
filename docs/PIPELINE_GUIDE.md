# Pipeline Configuration Guide

This guide explains how to use the unified pipeline system for end-to-end temporal knowledge graph analysis.

## Overview

The pipeline system provides a **single entry point** for the entire workflow:

```
Data → Ingestion → Knowledge Graph → LLM Agents → Fact-Checked Reports
```

All components are configured via YAML files, making it easy to:
- Switch between different data sources
- Change LLM backends
- Adjust agent parameters
- Run complete workflows or query-only mode

## Quick Start

### 1. Query Existing Graph

If you already have a knowledge graph:

```bash
# Single query
python run_pipeline.py \
  --config configs/pipeline_query_only.yaml \
  --skip-ingestion \
  --entity Apple \
  --year 2021

# Interactive mode
python run_pipeline.py \
  --config configs/pipeline_query_only.yaml \
  --skip-ingestion \
  --interactive
```

### 2. Complete Pipeline (Ingest + Query)

To ingest new data and build a graph:

```bash
# Run full pipeline
python run_pipeline.py --config configs/pipeline_amazon_reviews.yaml

# Then query
python run_pipeline.py \
  --config configs/pipeline_amazon_reviews.yaml \
  --skip-ingestion \
  --interactive
```

### 3. Custom Data

1. Copy template config:
   ```bash
   cp configs/pipeline_custom_template.yaml configs/my_project.yaml
   ```

2. Edit `configs/my_project.yaml` to match your data

3. Run:
   ```bash
   python run_pipeline.py --config configs/my_project.yaml
   ```

## Configuration Structure

### Complete Configuration File

```yaml
# Project metadata
project_name: "My Analysis Project"
description: "Optional description"

# Output paths
graph_path: ./my_graph.pkl
output_dir: ./output/my_project

# LLM backend
llm:
  backend: bedrock  # or 'local'
  model_id: us.meta.llama3-1-8b-instruct-v1:0
  region: us-east-2

# Data ingestion (optional)
ingestion:
  connector:
    type: csv
    config: {...}
  entity_extractor: {...}
  sentiment_classifier: {...}
  topic_classifier: {...}
  limit: 5000
  batch_size: 32

# Historian configuration
historian:
  type: default
  max_tokens: 1024
  temperature: 0.4

# Critic configuration
critic:
  type: robust
  max_tokens: 512
  temperature: 0.1

# Date range
date_range:
  min: 2014
  max: 2024
```

## Configuration Sections

### 1. LLM Configuration

```yaml
llm:
  backend: bedrock  # 'bedrock' or 'local'
  model_id: us.meta.llama3-1-8b-instruct-v1:0
  region: us-east-2  # For Bedrock only
```

**Available backends:**
- `bedrock`: AWS Bedrock (requires AWS credentials)
- `local`: Local GPU inference (requires GPU + model files)

**Model IDs:**
- Bedrock: Use inference profile IDs (e.g., `us.meta.llama3-1-8b-instruct-v1:0`)
- Local: Use HuggingFace model paths (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)

### 2. Ingestion Configuration

Set to `null` if you already have a knowledge graph.

```yaml
ingestion:
  # Data connector (required)
  connector:
    type: csv  # csv, jsonl, or directory
    config:
      path: ./data/reviews.csv
      entity_col: brand
      text_col: review_text
      date_col: timestamp
      date_format: "%Y-%m-%d"

  # Entity extraction (optional)
  entity_extractor:
    type: keyword  # keyword, spacy, or none
    entities: [Apple, Samsung, Google]
    case_sensitive: false

  # Sentiment classification (optional)
  sentiment_classifier:
    type: vader  # vader, transformer, or none

  # Topic classification (optional)
  topic_classifier:
    type: zero-shot  # zero-shot, keyword, or none
    topics: [Quality, Service, Price]

  # Processing options
  limit: null  # null for unlimited
  batch_size: 32
```

See [INGESTION_GUIDE.md](INGESTION_GUIDE.md) for detailed ingestion configuration.

### 3. Agent Configuration

#### Historian

Generates temporal reports from knowledge graph data.

```yaml
historian:
  type: default  # Which historian implementation
  max_tokens: 1024  # Maximum response length
  temperature: 0.4  # Sampling temperature (0=deterministic, 1=creative)
```

**Parameters:**
- `max_tokens`: 512-2048 (shorter = faster, longer = more detail)
- `temperature`: 0.1-0.7 (lower = more factual, higher = more varied)

#### Critic

Verifies reports against ground truth.

```yaml
critic:
  type: robust  # Which critic implementation
  max_tokens: 512
  temperature: 0.1  # Lower is better for verification
```

**Parameters:**
- `max_tokens`: 256-1024
- `temperature`: 0.05-0.2 (should be low for consistency)

## Usage Modes

### Mode 1: Single Query

Get a report for one entity and year:

```bash
python run_pipeline.py \
  --config configs/pipeline_query_only.yaml \
  --skip-ingestion \
  --entity Samsung \
  --year 2022
```

**Output:**
```
REPORT: Samsung (2022)
================================================================================
[Full report text...]

VERIFICATION: PASS
================================================================================
Reasoning: Passed all checks
```

### Mode 2: Interactive

Continuously query different entities and years:

```bash
python run_pipeline.py \
  --config configs/pipeline_query_only.yaml \
  --skip-ingestion \
  --interactive
```

**Usage:**
```
>> Entity (or 'q' to quit): Samsung
>> Baseline year: 2020
>> Comparison year: 2022

[Reports for both years generated...]
```

### Mode 3: Complete Pipeline

Ingest data, build graph, and prepare for querying:

```bash
python run_pipeline.py --config configs/pipeline_amazon_reviews.yaml
```

This will:
1. Load and process your data
2. Extract entities, classify sentiment/topics
3. Build temporal knowledge graph
4. Save graph to disk
5. Initialize agents
6. Show available entities

Then run in interactive or query mode with `--skip-ingestion`.

## Configuration Examples

### Example 1: Amazon Reviews (Original Dataset)

```yaml
project_name: "Amazon Product Reviews Analysis"
graph_path: ./thesis_graph.pkl

llm:
  backend: bedrock
  model_id: us.meta.llama3-1-8b-instruct-v1:0
  region: us-east-2

ingestion:
  connector:
    type: directory
    config:
      directory: ./data/amazon_data
      pattern: "*.csv"
      connector_type: csv
      connector_config:
        entity_col: brand
        text_col: review_text
        date_col: timestamp

  entity_extractor:
    type: keyword
    entities: [Apple, Samsung, Sony, ...]

  sentiment_classifier:
    type: vader

  topic_classifier:
    type: zero-shot
    topics:
      - product quality and durability
      - customer service and delivery
      - performance and speed
```

### Example 2: Custom CSV Data

```yaml
project_name: "Customer Feedback Analysis"
graph_path: ./customer_feedback_graph.pkl

llm:
  backend: bedrock
  model_id: us.meta.llama3-1-8b-instruct-v1:0

ingestion:
  connector:
    type: csv
    config:
      path: ./data/feedback.csv
      entity_col: product_name
      text_col: feedback_text
      date_col: submission_date

  entity_extractor:
    type: none  # Entity already in CSV

  sentiment_classifier:
    type: vader

  topic_classifier:
    type: keyword
    keyword_map:
      Bug: [bug, error, crash, broken]
      Feature: [feature, request, add, implement]
      Performance: [slow, lag, speed, fast]
```

### Example 3: Query-Only (No Ingestion)

```yaml
project_name: "Query Existing Graph"
graph_path: ./thesis_graph.pkl

llm:
  backend: bedrock
  model_id: us.meta.llama3-1-8b-instruct-v1:0

ingestion: null  # Skip ingestion

historian:
  max_tokens: 2048  # Longer reports
  temperature: 0.3

critic:
  max_tokens: 512
  temperature: 0.05  # Stricter verification
```

## Advanced Usage

### Custom Agent Implementations

You can create custom Historian or Critic implementations:

```python
from src.agents.base import HistorianBase

class MyCustomHistorian(HistorianBase):
    def conduct_audit(self, entity: str, year: int) -> str:
        # Custom implementation
        ...

# Register it
from src.agents.factory import register_historian
register_historian('custom', MyCustomHistorian)
```

Then use in config:
```yaml
historian:
  type: custom
  max_tokens: 1024
```

### Environment Variables

Instead of hardcoding credentials, use environment variables:

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-2

# Run pipeline
python run_pipeline.py --config configs/my_project.yaml
```

### Programmatic Usage

You can also use the pipeline in your Python code:

```python
from src.config.pipeline_config import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator

# Load config
config = PipelineConfig.from_yaml('configs/my_project.yaml')

# Create and initialize pipeline
pipeline = PipelineOrchestrator(config)
pipeline.initialize(skip_ingestion=True)

# Run queries
result = pipeline.run_audit('Apple', 2021)
print(result['report'])
print(result['status'])
```

## Troubleshooting

### "Graph file not found"
- Run with ingestion first (remove `--skip-ingestion`)
- Or check `graph_path` in your config

### "Unknown historian/critic type"
- Check available types in `src/agents/factory.py`
- Make sure you registered custom types

### LLM errors
- Verify AWS credentials are set
- Check model ID is correct for your region
- Ensure sufficient AWS quotas/permissions

### Ingestion failures
- See [INGESTION_GUIDE.md](INGESTION_GUIDE.md) for detailed troubleshooting
- Check data file paths and column names
- Verify date format matches your data

## Next Steps

1. **Try the examples:**
   ```bash
   python run_pipeline.py --config configs/pipeline_query_only.yaml --skip-ingestion --interactive
   ```

2. **Create your own config:**
   ```bash
   cp configs/pipeline_custom_template.yaml configs/my_project.yaml
   # Edit my_project.yaml
   python run_pipeline.py --config configs/my_project.yaml
   ```

3. **Deploy as API:** See API documentation for exposing the pipeline as a REST service

4. **Integrate in your app:** Import and use `PipelineOrchestrator` programmatically
