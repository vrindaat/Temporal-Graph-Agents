# Temporal-Graph-Agent (TGA)

### A Modular, Extensible Framework for Temporal Knowledge Graph Analysis with Fact-Checking

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![NetworkX](https://img.shields.io/badge/Graph-NetworkX-green)
![AWS Bedrock](https://img.shields.io/badge/LLM-Bedrock-yellow)

**Temporal-Graph-Agent** is a production-ready framework for building temporal knowledge graphs from any data source and generating fact-checked historical reports using LLMs. Unlike traditional RAG systems, TGA enforces strict temporal constraints and verifies all outputs against ground truth.

**Key Features:**
- ✅ **Data-source agnostic** - Works with CSV, JSON, databases, APIs
- ✅ **Domain agnostic** - Reviews, tickets, social media, surveys, etc.
- ✅ **LLM agnostic** - AWS Bedrock, local models, or any LLM backend
- ✅ **Fact-checking built-in** - Critic agent prevents hallucinations
- ✅ **Configuration-driven** - YAML configs for entire pipeline
- ✅ **Modular & extensible** - Easy to add custom components

Originally developed for longitudinal brand auditing, now **generalized for enterprise use**.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/temporal-graph-agent.git
cd temporal-graph-agent
pip install -r requirements.txt
```

### Run Example

```bash
# Query existing knowledge graph
python run_pipeline.py \
  --config configs/pipeline_query_only.yaml \
  --skip-ingestion \
  --entity Apple \
  --year 2021
```

### Interactive Mode

```bash
python run_pipeline.py \
  --config configs/pipeline_query_only.yaml \
  --skip-ingestion \
  --interactive
```

---

## 🏗️ Architecture

The framework consists of **5 modular layers**:

```
┌──────────────────────────────────────────┐
│  User Interface (CLI / API / Web UI)    │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  Pipeline Orchestration                  │
│  (Configuration + Workflow Management)   │
└──┬────────┬────────┬────────┬────────────┘
   │        │        │        │
   ▼        ▼        ▼        ▼
┌────────┬──────┬─────────┬────────┐
│Ingestion│Graph │LLM      │Agents  │
│Layer    │Engine│Backends │Layer   │
└─────────┴──────┴─────────┴────────┘
```

### 1. Ingestion Layer
Abstract data connectors for any source:
- CSV, JSONL, Directory connectors
- Keyword or NER entity extraction
- VADER or transformer sentiment analysis
- Zero-shot or keyword topic classification

### 2. Graph Engine
Temporal knowledge graph with:
- NetworkX-based storage
- Temporal range queries
- Entity-centric snapshots
- Efficient serialization

### 3. LLM Backends
Pluggable LLM interface:
- AWS Bedrock (Llama, Claude, etc.)
- Local HuggingFace models
- Easy to add OpenAI, Anthropic, etc.

### 4. Agents Layer
Intelligent analysis:
- **Historian**: Generates temporal reports
- **Critic**: Verifies against ground truth (rule-based + LLM)

### 5. Pipeline Orchestration
End-to-end workflow management:
- YAML-based configuration
- Component initialization
- Workflow coordination

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

---

## 📖 Documentation

- **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)** - Complete pipeline usage guide
- **[INGESTION_GUIDE.md](INGESTION_GUIDE.md)** - Data ingestion details
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture deep-dive

---

## 🎯 Use Cases

### 1. Brand Intelligence
Track sentiment evolution over time:
```yaml
# Example: Track Apple brand perception 2014-2024
entities: [Apple, Samsung, Google]
topics: [Product Quality, Customer Service, Innovation]
```

### 2. Customer Support Analytics
Identify recurring issues:
```yaml
# Example: Support ticket analysis
topics: [Bug Reports, Feature Requests, Account Issues]
sentiment: vader
```

### 3. Social Media Monitoring
Monitor brand mentions:
```yaml
# Example: Twitter/Reddit brand monitoring
connector: api
entities: [YourBrand, Competitor1, Competitor2]
```

### 4. Product Reviews
Aggregate cross-platform reviews:
```yaml
# Example: App Store + Google Play reviews
connector: directory
topics: [UI/UX, Performance, Features]
```

---

## ⚙️ Configuration System

All behavior controlled via YAML configs:

```yaml
project_name: "My Analysis Project"
graph_path: ./my_graph.pkl

# LLM Backend
llm:
  backend: bedrock
  model_id: us.meta.llama3-1-8b-instruct-v1:0
  region: us-east-2

# Data Ingestion (optional)
ingestion:
  connector:
    type: csv
    config:
      path: ./data/reviews.csv
      entity_col: brand
      text_col: review_text
      date_col: timestamp

  entity_extractor:
    type: keyword
    entities: [Apple, Samsung, Google]

  sentiment_classifier:
    type: vader

  topic_classifier:
    type: zero-shot
    topics: [Quality, Service, Price]

# Agent Configuration
historian:
  max_tokens: 1024
  temperature: 0.4

critic:
  max_tokens: 512
  temperature: 0.1
```

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for complete configuration options.

---

## 🔧 Extensibility

### Add Custom Data Connector

```python
from src.ingestion.base import DataConnector, Record

class DatabaseConnector(DataConnector):
    def load_records(self) -> Iterator[Record]:
        # Your database logic
        yield Record(entity, text, date, ...)
```

### Add Custom LLM Backend

```python
from src.llm.base import LLMBackend

class OpenAIBackend(LLMBackend):
    def generate(self, prompt, max_tokens, temperature):
        # Your LLM logic
        return response_text
```

### Add Custom Agent

```python
from src.agents.base import HistorianBase

class DetailedHistorian(HistorianBase):
    def conduct_audit(self, entity, year):
        # Your report logic
        return detailed_report
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for extension points.

---

## 📊 The Problem TGA Solves

| Traditional RAG | Temporal-Graph-Agent |
|-----------------|---------------------|
| ❌ Temporal collapse - conflates past/present | ✅ Strict temporal boundaries |
| ❌ No fact-checking - accepts hallucinations | ✅ Built-in verification system |
| ❌ Black-box retrieval | ✅ Transparent graph queries |
| ❌ Domain-specific implementations | ✅ Works with any domain/data |
| ❌ Hardcoded logic | ✅ Configuration-driven |

**TGA enforces temporal integrity while maintaining LLM fluency.**

---

## 🎓 Research Context

Originally developed as a thesis project for longitudinal brand auditing using Amazon reviews:

- **Problem**: LLMs conflate events across time, causing "temporal hallucinations"
- **Solution**: Temporal knowledge graph + adversarial verification
- **Results**: ~100% hallucination detection, 0% temporal leakage

**Now generalized** for enterprise use with any data source and domain.

---

## 🛠️ Requirements

```txt
Python 3.10+
networkx>=3.0
transformers>=4.30
spacy>=3.5
vaderSentiment>=3.3
boto3>=1.28  # For AWS Bedrock
PyYAML>=6.0
pydantic>=2.0
tqdm>=4.60
```

---

## 🚢 Deployment Options

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
```bash
# FastAPI service
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Web UI
```bash
# Gradio interface
python hf_space/app.py
```

### 5. Docker (Coming Soon)
```bash
docker run temporal-graph-agent --config config.yaml
```

---

## 📝 Examples

### Example 1: Analyze Customer Feedback

```bash
# 1. Create config
cp configs/pipeline_custom_template.yaml configs/customer_feedback.yaml

# 2. Edit config to point to your CSV

# 3. Run ingestion
python run_pipeline.py --config configs/customer_feedback.yaml

# 4. Query
python run_pipeline.py \
  --config configs/customer_feedback.yaml \
  --skip-ingestion \
  --entity "Product X" \
  --year 2023
```

### Example 2: Compare Two Time Periods

```bash
python run_pipeline.py \
  --config configs/my_project.yaml \
  --skip-ingestion \
  --interactive

>> Entity: Samsung
>> Baseline year: 2020
>> Comparison year: 2023

[Shows reports for both years with fact-checking]
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- New data connectors (PostgreSQL, MongoDB, Twitter API, etc.)
- New LLM backends (OpenAI, Anthropic, Cohere, etc.)
- Domain-specific agents
- Visualization tools
- Performance optimizations

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🔗 Links

- **Documentation**: See `docs/` folder
- **Examples**: See `configs/` folder
- **Tests**: `python test_ingestion_abstraction.py`

---

## 🙏 Acknowledgments

- Built on NetworkX, Transformers, spaCy
- Original research for BU thesis project
- Now maintained as open-source framework

---

**Built with ❤️ for temporal data analysis and fact-checked LLM outputs**
