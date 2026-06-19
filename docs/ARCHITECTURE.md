# System Architecture

## Overview

The Temporal Graph Agent is a **modular, extensible framework** for temporal knowledge graph analysis with fact-checking capabilities. It's designed to work with any data source, any LLM backend, and any domain.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  CLI (run_pipeline.py) │ API (FastAPI) │ Web UI (Gradio)       │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│               PIPELINE ORCHESTRATION LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  PipelineOrchestrator                                           │
│  - Configuration management                                      │
│  - Component initialization                                      │
│  - Workflow coordination                                         │
└───────┬─────────────┬───────────────┬─────────────┬────────────┘
        │             │               │             │
        ▼             ▼               ▼             ▼
┌──────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐
│  INGESTION   │ │   GRAPH    │ │   LLM    │ │   AGENTS   │
│    LAYER     │ │   ENGINE   │ │ BACKENDS │ │   LAYER    │
└──────────────┘ └────────────┘ └──────────┘ └────────────┘
```

## Core Components

### 1. Data Ingestion Layer (`src/ingestion/`)

**Purpose:** Abstract data source handling

**Key Classes:**
- `DataConnector` (base): Load data from any source
- `CSVConnector`, `JSONLinesConnector`, `DirectoryConnector`
- `EntityExtractor`: Extract entities from text
- `SentimentClassifier`: Classify sentiment
- `TopicClassifier`: Categorize content
- `IngestionPipeline`: Coordinate ingestion workflow

**Extensibility:**
```python
class MyConnector(DataConnector):
    def load_records(self) -> Iterator[Record]:
        # Custom data source logic
        yield Record(entity, text, date, ...)
```

### 2. Graph Engine (`src/graph/`)

**Purpose:** Temporal knowledge graph storage and querying

**Key Classes:**
- `TemporalGraphEngine`: Core graph operations
- `SnapshotFact`: Temporal fact representation
- `Node`, `TemporalEdge`: Graph primitives
- `MarketingTopic`, `Sentiment`: Domain enums

**Features:**
- Temporal range queries
- Entity-centric snapshots
- Efficient serialization

### 3. LLM Backends (`src/llm/`)

**Purpose:** Abstract LLM interface

**Key Classes:**
- `LLMBackend` (base): Abstract LLM interface
- `BedrockBackend`: AWS Bedrock integration
- `LocalHFBackend`: Local HuggingFace models

**Extensibility:**
```python
class MyLLMBackend(LLMBackend):
    def generate(self, prompt, max_tokens, temperature):
        # Custom LLM logic
        return response_text
```

### 4. Agent Layer (`src/agents/`)

**Purpose:** Intelligent analysis and verification

**Key Classes:**
- `HistorianBase`: Generate temporal reports
- `HistorianAgent`: Default implementation
- `CriticBase`: Verify reports against ground truth
- `RobustCriticAgent`: Rule-based + LLM verification

**Architecture:**
```
HistorianAgent → LLM → Generate Report
                  ↓
CriticAgent → Compare with Ground Truth → Verification Result
```

**Verification Strategy:**
1. **Rule-based checks** (fast, deterministic)
   - Sentiment count validation
   - Percentage sanity checks
   - Data availability checks

2. **LLM spot-check** (for small datasets)
   - Claim extraction
   - Fabrication detection

### 5. Pipeline Orchestration (`src/pipeline/`)

**Purpose:** End-to-end workflow management

**Key Classes:**
- `PipelineOrchestrator`: Coordinate complete workflow
- `PipelineConfig`: Configuration management

**Workflow:**
```
1. Load Configuration (YAML)
2. Initialize Graph Engine
3. Run Ingestion (optional)
4. Initialize LLM Backend
5. Create Agents
6. Execute Queries
```

### 6. Configuration System (`src/config/`)

**Purpose:** Unified configuration management

**Key Classes:**
- `PipelineConfig`: Top-level config
- `LLMConfig`, `IngestionConfig`, `HistorianConfig`, `CriticConfig`

**Benefits:**
- Single source of truth
- YAML-based (human-readable)
- Validation and error checking
- Serialization support

## Data Flow

### Ingestion Flow

```
Raw Data (CSV/JSON/etc.)
    ↓
DataConnector.load_records()
    ↓
Record(entity, text, date, ...)
    ↓
EntityExtractor.extract() → entity
SentimentClassifier.classify() → sentiment
TopicClassifier.classify() → topic
    ↓
TemporalGraphEngine.add_data()
    ↓
Knowledge Graph (NetworkX MultiDiGraph)
    ↓
Serialized to disk (.pkl)
```

### Query Flow

```
User Query (entity, year)
    ↓
TemporalGraphEngine.get_snapshot_for_year()
    ↓
List[SnapshotFact]
    ↓
HistorianAgent.conduct_audit()
    ↓
LLM.generate(prompt + facts)
    ↓
Report Text
    ↓
CriticAgent.verify_audit()
    ↓
Rule-based checks + LLM verification
    ↓
VerificationResult(status, issues, reasoning)
    ↓
User sees fact-checked report
```

## Key Design Principles

### 1. Abstraction & Modularity

Every major component has an abstract base class:
- Easy to extend
- Swap implementations without changing code
- Clear interfaces

### 2. Configuration-Driven

All behavior controlled via YAML configs:
- No hardcoded values
- Easy to replicate experiments
- Version-controllable

### 3. Domain-Agnostic

Works with any:
- Data source (CSV, JSON, databases, APIs)
- Domain (reviews, tickets, social media, surveys)
- Entity type (brands, products, people, organizations)

### 4. LLM-Agnostic

Supports any LLM backend:
- Cloud (Bedrock, OpenAI, Anthropic)
- Local (HuggingFace, vLLM, Ollama)
- Just implement `LLMBackend` interface

### 5. Fact-Checking First

Unique selling point:
- Critic agent prevents hallucinations
- Rule-based + LLM verification
- Configurable strictness

## Extension Points

### Add New Data Connector

```python
from src.ingestion.base import DataConnector

class DatabaseConnector(DataConnector):
    def load_records(self):
        # Connect to database
        # Yield Record objects
        pass
```

### Add New LLM Backend

```python
from src.llm.base import LLMBackend

class OpenAIBackend(LLMBackend):
    def generate(self, prompt, max_tokens, temperature):
        # Call OpenAI API
        return response
```

### Add Custom Agent

```python
from src.agents.base import HistorianBase

class DetailedHistorian(HistorianBase):
    def conduct_audit(self, entity, year):
        # Custom report generation logic
        return detailed_report
```

### Add New Classifier

```python
from src.ingestion.base import TopicClassifier

class BERTTopicClassifier(TopicClassifier):
    def classify(self, text):
        # BERT-based classification
        return topic
```

## Technology Stack

**Core:**
- Python 3.10+
- NetworkX (graph engine)
- PyYAML (configuration)

**NLP:**
- spaCy (NER)
- VADER (sentiment)
- Transformers (classification)

**LLMs:**
- AWS Bedrock SDK
- HuggingFace Transformers
- PyTorch

**Utilities:**
- Pydantic (validation)
- tqdm (progress bars)
- dataclasses (structures)

## Performance Characteristics

**Ingestion:**
- ~100-1000 records/second (CSV)
- Bottleneck: Topic classification (LLM)
- Batch processing for efficiency

**Querying:**
- Graph query: < 100ms
- Report generation: 2-10 seconds (LLM-dependent)
- Verification: < 5 seconds

**Memory:**
- Graph: ~1MB per 1000 edges
- LLM: 4-16GB depending on model

## Deployment Options

### 1. CLI Tool
```bash
python run_pipeline.py --config config.yaml --interactive
```

### 2. Python Library
```python
from src.pipeline import PipelineOrchestrator
pipeline = PipelineOrchestrator(config)
result = pipeline.run_audit('Apple', 2021)
```

### 3. REST API
```python
# FastAPI service (api/main.py)
POST /audit
{
  "entity": "Apple",
  "year": 2021
}
```

### 4. Web UI
```python
# Gradio interface (hf_space/app.py)
gradio.launch()
```

### 5. Docker Container
```bash
docker run temporal-graph-agent --config config.yaml
```

## Security Considerations

**Data Privacy:**
- Graphs stored locally (not cloud)
- No data sent to external services (unless using cloud LLM)

**LLM Security:**
- Prompt injection protection (input validation)
- Rate limiting on API calls
- Credential management via environment variables

**Access Control:**
- File-based permissions
- API authentication (if deployed)

## Future Extensions

**Planned:**
1. More data connectors (PostgreSQL, MongoDB, APIs)
2. More LLM backends (OpenAI, Anthropic, Cohere)
3. GraphQL API
4. Real-time ingestion (streaming)
5. Multi-modal support (images, audio)
6. Distributed graph storage
7. Advanced visualizations

**Community Contributions:**
- Custom domain adaptations
- New classifier implementations
- Specialized agents
- Integration plugins

## Documentation

- **INGESTION_GUIDE.md** - Data ingestion details
- **PIPELINE_GUIDE.md** - End-to-end workflow
- **README.md** - Getting started
- **API_DOCS.md** - REST API reference (coming soon)

## Getting Help

1. Check documentation guides
2. Review example configs in `configs/`
3. Run test scripts: `python test_ingestion_abstraction.py`
4. Open GitHub issues for bugs/questions
