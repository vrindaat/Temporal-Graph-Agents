# TGA Complete Rebuild — Implementation Plan

> **Purpose**: Self-contained instructions to rebuild Temporal-Graph-Agent from scratch.
> Every core component is broken. This plan fixes the foundations first, then adds metrics and deployment.
> A model like Claude Sonnet should follow this step-by-step in order.

---

## What's Wrong (Diagnosis Summary)

| Layer | Issue | Impact |
|-------|-------|--------|
| **Imports** | All `__init__.py` renamed to `src_init.py`, `llm_init.py`, etc. | Code can't import; nothing runs |
| **Ingestion** | SpaCy NER extracts wrong brands from reviews | Graph has wrong brand-review edges |
| **Ingestion** | Date parsing silently defaults to 2021-01-01 on failure | Temporal analysis completely broken |
| **Ingestion** | Only 200 reviews per file (hardcoded test limit) | 99%+ of data discarded |
| **Ingestion** | BART-MNLI zero-shot with complex labels = random topics | Topic assignments meaningless |
| **Ingestion** | Hardcoded SCC cluster paths | Won't run on any other machine |
| **Graph Engine** | `exists_at()` method never called, ignores end_date | Temporal firewall doesn't exist |
| **Graph Engine** | `get_snapshot()` returns 100-char truncated text, no metadata | Agents can't reason on data |
| **Graph Engine** | No date range queries (only point-in-time) | Year audits compare single days |
| **Graph Engine** | Substring brand matching causes false positives | "Apple" matches "Pineapple" |
| **Historian** | Vague prompt, no structure enforcement, no grounding | Generic/hallucinated reports |
| **Critic** | Can't verify — sees same truncated data as Historian | Always passes or random |
| **Critic** | No output parsing — returns raw LLM text | Downstream can't use results |
| **LLM Wrapper** | Single temperature (0.1) for both agents | Too rigid for Historian |
| **LLM Wrapper** | 512 token limit truncates multi-section reports | Incomplete outputs |
| **Security** | AWS credentials committed in plaintext | Must be purged |

---

## Rebuild Order

```
Phase 1: Project Structure & Imports (make it run)
Phase 2: Config System (make it portable)
Phase 3: Graph Engine Rebuild (make temporal logic work)
Phase 4: Ingestion Pipeline Rebuild (make data correct)
Phase 5: LLM Backend (make inference flexible)
Phase 6: Agent Rewrite (make outputs good)
Phase 7: Metrics & Evaluation (prove it works)
Phase 8: FastAPI Layer (make it accessible)
Phase 9: HuggingFace Deployment (demo + dataset)
Phase 10: AWS Deployment (scalable inference)
```

---

## PHASE 1: Project Structure & Imports

### Step 1.1: Create `.gitignore`

Create `.gitignore` in project root:

```gitignore
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.env
*.env
models_cache/
*.pkl
bedrock.txt
.vscode/
.idea/
Thumbs.db
.DS_Store
results/
data/hf_dataset/
```

### Step 1.2: Fix all `__init__.py` files

**Delete** these incorrectly-named files:
- `src/graph/src_init.py`
- `src/llm/llm_init.py`
- `src/srcc_init.py`

**Create** proper `__init__.py` files (all initially empty except `src/llm/__init__.py`):

**`src/__init__.py`** — empty file

**`src/graph/__init__.py`** — empty file

**`src/agents/__init__.py`** — empty file

**`src/utils/__init__.py`** — empty file

**`src/llm/__init__.py`**:
```python
from .base import LLMBackend, GenerationMetrics
from .factory import create_llm_backend
```

**`evaluation/__init__.py`** — empty file

**`api/__init__.py`** — empty file

### Step 1.3: Remove secrets from git history

```bash
git rm --cached bedrock.txt 2>/dev/null || true
git rm --cached thesis_graph.pkl 2>/dev/null || true
git rm -r --cached __pycache__/ 2>/dev/null || true

# Purge from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch bedrock.txt" \
  --prune-empty --tag-name-filter cat -- --all

git add .gitignore
git commit -m "Fix project structure: add .gitignore, remove secrets from tracking"
```

### Step 1.4: Delete old broken init files

```bash
rm -f "src/graph/src_init.py"
rm -f "src/llm/llm_init.py"
rm -f "src/srcc_init.py"
```

Then create the correct `__init__.py` files per Step 1.2.

---

## PHASE 2: Configuration System

### Step 2.1: Install new dependencies

Add to `requirements.txt`:
```
pydantic-settings>=2.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
boto3>=1.28.0
python-dotenv>=1.0.0
rouge-score>=0.1.2
bert-score>=0.3.13
gradio>=4.0
```

### Step 2.2: Rewrite `config/settings.py`

Replace the entire file:

```python
import os
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_backend: Literal["local", "bedrock"] = "local"
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_home: str = "./models_cache"
    historian_max_tokens: int = 1024
    historian_temperature: float = 0.4
    critic_max_tokens: int = 512
    critic_temperature: float = 0.1

    # Graph
    graph_path: str = "./thesis_graph.pkl"

    # Ingestion
    ingest_limit: int = 5000
    ingest_batch_size: int = 32
    data_dir: str = "./data/amazon_data"

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "meta.llama3-8b-instruct-v1:0"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Evaluation
    eval_output_dir: str = "./results"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
os.environ["HF_HOME"] = settings.hf_home
```

### Step 2.3: Create `.env.example`

```env
# LLM Backend: "local" (GPU) or "bedrock" (AWS API)
LLM_BACKEND=local
MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
HF_HOME=./models_cache
HISTORIAN_MAX_TOKENS=1024
HISTORIAN_TEMPERATURE=0.4
CRITIC_MAX_TOKENS=512
CRITIC_TEMPERATURE=0.1

# Graph
GRAPH_PATH=./thesis_graph.pkl

# Ingestion
INGEST_LIMIT=5000
INGEST_BATCH_SIZE=32
DATA_DIR=./data/amazon_data

# AWS (only if LLM_BACKEND=bedrock)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=meta.llama3-8b-instruct-v1:0

# API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## PHASE 3: Graph Engine Rebuild

The current graph engine has broken temporal filtering, substring matching, and returns useless 100-char snippets. Rebuild completely.

### Step 3.1: Rewrite `src/graph/schema.py`

```python
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
```

### Step 3.2: Rewrite `src/graph/engine.py`

```python
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

import networkx as nx

from .schema import Node, TemporalEdge, MarketingTopic, Sentiment


@dataclass
class SnapshotFact:
    """A single fact from a graph query — rich enough for agents to reason on."""
    brand: str
    review_id: str
    topic: MarketingTopic
    sentiment: Sentiment
    date: datetime
    review_text: str
    rating: Optional[float] = None


class TemporalGraphEngine:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.last_query_ms: float = 0.0

    def add_node(self, node: Node):
        self.graph.add_node(node.id, type=node.type, **node.properties)

    def add_edge(self, edge: TemporalEdge):
        self.graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            topic=edge.topic.value if isinstance(edge.topic, MarketingTopic) else edge.topic,
            sentiment=edge.sentiment.value if isinstance(edge.sentiment, Sentiment) else edge.sentiment,
            start_date=edge.start_date,
            end_date=edge.end_date,
            **edge.metadata,
        )

    def add_data(self, brand_node: Node, review_node: Node, edge: TemporalEdge):
        self.add_node(brand_node)
        self.add_node(review_node)
        self.add_edge(edge)

    def get_brands(self) -> List[str]:
        return sorted([n for n, d in self.graph.nodes(data=True) if d.get("type") == "Brand"])

    def get_snapshot_for_year(self, brand: str, year: int) -> List[SnapshotFact]:
        """Get all facts for a brand within a specific year (date RANGE query)."""
        start = time.perf_counter()
        range_start = datetime(year, 1, 1)
        range_end = datetime(year, 12, 31, 23, 59, 59)

        facts = []
        for u, v, data in self.graph.edges(data=True):
            # Exact brand match (case-insensitive)
            if u.lower() != brand.lower() and v.lower() != brand.lower():
                continue

            edge_date = data.get("start_date")
            if edge_date is None:
                continue
            if not (range_start <= edge_date <= range_end):
                continue

            review_node_id = v if u.lower() == brand.lower() else u
            node_data = self.graph.nodes.get(review_node_id, {})

            topic_str = data.get("topic", "General Experience")
            sentiment_str = data.get("sentiment", "NEUTRAL")
            try:
                topic = MarketingTopic(topic_str)
            except ValueError:
                topic = MarketingTopic.GENERAL
            try:
                sentiment = Sentiment(sentiment_str)
            except ValueError:
                sentiment = Sentiment.NEUTRAL

            facts.append(SnapshotFact(
                brand=brand,
                review_id=review_node_id,
                topic=topic,
                sentiment=sentiment,
                date=edge_date,
                review_text=node_data.get("text", ""),
                rating=node_data.get("rating"),
            ))

        facts.sort(key=lambda f: f.date)
        self.last_query_ms = (time.perf_counter() - start) * 1000
        return facts

    def format_facts_for_prompt(self, facts: List[SnapshotFact], max_facts: int = 80) -> str:
        """Format facts into structured text for LLM prompts."""
        if not facts:
            return "No data available for this brand and time period."

        selected = facts[:max_facts]
        lines = []
        lines.append(f"Total reviews: {len(facts)} (showing {len(selected)})")
        lines.append(f"Date range: {facts[0].date.strftime('%Y-%m-%d')} to {facts[-1].date.strftime('%Y-%m-%d')}")
        lines.append("")

        by_topic = defaultdict(list)
        for fact in selected:
            by_topic[fact.topic.value].append(fact)

        for topic, topic_facts in by_topic.items():
            pos = sum(1 for f in topic_facts if f.sentiment == Sentiment.POSITIVE)
            neg = sum(1 for f in topic_facts if f.sentiment == Sentiment.NEGATIVE)
            neu = sum(1 for f in topic_facts if f.sentiment == Sentiment.NEUTRAL)
            lines.append(f"## {topic} ({pos} positive, {neg} negative, {neu} neutral)")

            for fact in topic_facts[:10]:
                sent_marker = {"POSITIVE": "+", "NEGATIVE": "-", "NEUTRAL": "~"}[fact.sentiment.value]
                date_str = fact.date.strftime("%Y-%m-%d")
                text = fact.review_text[:300] if fact.review_text else "(no text)"
                lines.append(f"  [{sent_marker}] ({date_str}) {text}")
            lines.append("")

        return "\n".join(lines)

    def save_to_disk(self, path: str = None):
        path = path or "thesis_graph.pkl"
        print(f"[Engine] Saving graph ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges) to {path}")
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load_from_disk(self, path: str = None) -> bool:
        path = path or "thesis_graph.pkl"
        try:
            with open(path, "rb") as f:
                self.graph = pickle.load(f)
            print(f"[Engine] Loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return True
        except FileNotFoundError:
            print(f"[Engine] File not found: {path}")
            return False
        except Exception as e:
            print(f"[Engine] Load failed: {e}")
            return False

    def stats(self) -> Dict:
        brands = self.get_brands()
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "brand_count": len(brands),
            "brands": brands[:20],
        }
```

---

## PHASE 4: Ingestion Pipeline Rebuild

### Step 4.1: Rewrite `src/utils/loader.py`

```python
import csv
import glob
import json
import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Generator

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline

from config.settings import settings
from src.graph.schema import Node, TemporalEdge, MarketingTopic, Sentiment
from src.graph.engine import TemporalGraphEngine


KNOWN_BRANDS = {
    "apple", "samsung", "sony", "nintendo", "microsoft", "google", "lg",
    "bose", "jbl", "anker", "logitech", "razer", "corsair", "asus",
    "dell", "hp", "lenovo", "acer", "msi", "alienware",
    "nike", "adidas", "under armour", "puma", "reebok",
    "kitchenaid", "instant pot", "ninja", "cuisinart", "dyson",
    "philips", "panasonic", "canon", "nikon", "fujifilm",
    "xbox", "playstation", "oculus", "valve",
}

SKIP_ENTITIES = {
    "amazon", "amazon.com", "seller", "usa", "china", "india",
    "ebay", "walmart", "target", "best buy", "costco",
    "ups", "usps", "fedex", "dhl",
    "the", "this", "that", "very", "great", "good", "bad",
}

TOPIC_LABELS = [
    "product quality and durability",
    "price and value for money",
    "customer service and delivery",
    "performance and speed",
    "ease of use and design",
    "general opinion",
]

TOPIC_MAP = {
    "product quality and durability": MarketingTopic.QUALITY,
    "price and value for money": MarketingTopic.PRICE,
    "customer service and delivery": MarketingTopic.SERVICE,
    "performance and speed": MarketingTopic.PERFORMANCE,
    "ease of use and design": MarketingTopic.USABILITY,
    "general opinion": MarketingTopic.GENERAL,
}


class ReviewLoader:
    def __init__(self, graph_engine: TemporalGraphEngine):
        self.graph = graph_engine
        self.stats = {"processed": 0, "added": 0, "skipped_no_brand": 0, "skipped_short": 0, "skipped_no_date": 0, "errors": 0}

        print("[Loader] Initializing NLP models...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.vader = SentimentIntensityAnalyzer()

        print("[Loader] Loading topic classifier (CPU)...")
        self.classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
        print("[Loader] Ready.")

    def extract_brand(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for brand in KNOWN_BRANDS:
            if re.search(r'\b' + re.escape(brand) + r'\b', text_lower):
                return brand.title()

        doc = self.nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "ORG" and len(ent.text) > 2:
                ent_lower = ent.text.lower().strip()
                if ent_lower not in SKIP_ENTITIES and len(ent_lower) < 30:
                    return ent.text.strip()
        return None

    def classify_topic(self, texts: List[str]) -> List[MarketingTopic]:
        try:
            results = self.classifier(texts, candidate_labels=TOPIC_LABELS, multi_label=False)
            if isinstance(results, dict):
                results = [results]
            topics = []
            for result in results:
                top_label = result["labels"][0]
                confidence = result["scores"][0]
                if confidence > 0.4:
                    topics.append(TOPIC_MAP.get(top_label, MarketingTopic.GENERAL))
                else:
                    topics.append(MarketingTopic.GENERAL)
            return topics
        except Exception as e:
            print(f"[Loader] Classification error: {e}")
            return [MarketingTopic.GENERAL] * len(texts)

    def analyze_sentiment(self, text: str, rating: Optional[float] = None) -> Sentiment:
        vader_scores = self.vader.polarity_scores(text[:1000])
        compound = vader_scores["compound"]

        if rating is not None:
            if rating >= 4.0 and compound >= -0.1:
                return Sentiment.POSITIVE
            elif rating <= 2.0 and compound <= 0.1:
                return Sentiment.NEGATIVE
            elif rating >= 4.0 and compound < -0.3:
                return Sentiment.NEUTRAL
            elif rating <= 2.0 and compound > 0.3:
                return Sentiment.NEGATIVE

        if compound >= 0.2:
            return Sentiment.POSITIVE
        elif compound <= -0.2:
            return Sentiment.NEGATIVE
        return Sentiment.NEUTRAL

    def parse_date(self, raw_date) -> Optional[datetime]:
        if raw_date is None or raw_date == "":
            return None
        try:
            ts = float(raw_date)
            if ts > 1e12:
                ts = ts / 1000
            if 946684800 < ts < 1893456000:
                return datetime.fromtimestamp(ts)
        except (ValueError, TypeError, OSError):
            pass
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"]:
            try:
                return datetime.strptime(str(raw_date), fmt)
            except ValueError:
                continue
        return None

    def load_directory(self, data_dir: str = None):
        data_dir = data_dir or settings.data_dir
        if not os.path.exists(data_dir):
            print(f"[Loader] ERROR: Directory not found: {data_dir}")
            return

        files = glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.json"))
        if not files:
            print(f"[Loader] No files found in {data_dir}")
            return

        print(f"[Loader] Found {len(files)} files. Limit: {settings.ingest_limit}/file")
        for fp in files:
            try:
                self._process_file(fp)
            except Exception as e:
                print(f"[Loader] Error in {os.path.basename(fp)}: {e}")
                self.stats["errors"] += 1

        print(f"\n[Loader] Complete. {self.stats}")

    def _process_file(self, file_path: str):
        filename = os.path.basename(file_path)
        print(f"  {filename}...", end=" ", flush=True)

        if file_path.endswith(".json"):
            iterator = self._read_json(file_path)
        else:
            iterator = self._read_csv(file_path)

        batch = []
        file_added = 0

        for i, row in enumerate(iterator):
            if i >= settings.ingest_limit:
                break
            self.stats["processed"] += 1

            row_lower = {k.lower(): v for k, v in row.items()}
            text = row_lower.get("reviewtext", row_lower.get("text", ""))
            title = row_lower.get("summary", row_lower.get("title", ""))
            full_text = f"{title}. {text}" if title else text

            if len(full_text.strip()) < 20:
                self.stats["skipped_short"] += 1
                continue

            try:
                rating = float(row_lower.get("overall", row_lower.get("rating", 0)))
            except (ValueError, TypeError):
                rating = None

            raw_date = row_lower.get("unixreviewtime", row_lower.get("timestamp", None))
            date = self.parse_date(raw_date)
            if date is None:
                self.stats["skipped_no_date"] += 1
                continue

            brand = self.extract_brand(full_text)
            if brand is None:
                self.stats["skipped_no_brand"] += 1
                continue

            batch.append({"brand": brand, "text": full_text, "rating": rating, "date": date, "id": f"Rev_{filename}_{i}"})

            if len(batch) >= settings.ingest_batch_size:
                file_added += self._process_batch(batch)
                batch = []

        if batch:
            file_added += self._process_batch(batch)
        print(f"{file_added} added")

    def _process_batch(self, batch: List[Dict]) -> int:
        texts = [item["text"][:512] for item in batch]
        topics = self.classify_topic(texts)

        count = 0
        for item, topic in zip(batch, topics):
            sentiment = self.analyze_sentiment(item["text"], item.get("rating"))
            brand_node = Node(item["brand"], "Brand")
            review_node = Node(item["id"], "Review", {"text": item["text"][:500], "rating": item.get("rating")})
            edge = TemporalEdge(
                source=item["brand"], target=item["id"], relation="REVIEWED_IN",
                topic=topic, sentiment=sentiment, start_date=item["date"],
            )
            self.graph.add_data(brand_node, review_node, edge)
            self.stats["added"] += 1
            count += 1
        return count

    def _read_json(self, path: str) -> Generator:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def _read_csv(self, path: str) -> Generator:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
```

### Step 4.2: Rewrite `ingest.py`

```python
"""
Phase 1: Data Ingestion & Graph Building.
Usage: python ingest.py [--data-dir ./data/amazon_data] [--output thesis_graph.pkl] [--limit 5000]
"""
import argparse
import sys

from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.utils.loader import ReviewLoader


def run_ingestion(data_dir=None, output=None, limit=None):
    print("=" * 60)
    print("   PHASE 1: DATA INGESTION & GRAPH BUILDING")
    print("=" * 60)

    if limit:
        settings.ingest_limit = limit
    if data_dir:
        settings.data_dir = data_dir

    graph = TemporalGraphEngine()
    loader = ReviewLoader(graph)
    loader.load_directory(settings.data_dir)

    if graph.graph.number_of_edges() == 0:
        print("[ERROR] Graph is empty. Check your data directory.")
        sys.exit(1)

    stats = graph.stats()
    print(f"\n[Success] {stats['total_edges']} edges, {stats['brand_count']} brands")
    print(f"  Brands: {stats['brands'][:10]}")

    output_path = output or settings.graph_path
    graph.save_to_disk(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_ingestion(args.data_dir, args.output, args.limit)
```

---

## PHASE 5: LLM Backend

### Step 5.1: Create `src/llm/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    prompt_tokens: int = 0
    response_tokens: int = 0
    latency_ms: float = 0.0


class LLMBackend(ABC):
    def __init__(self):
        self.last_metrics: GenerationMetrics = GenerationMetrics()

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        ...

    @abstractmethod
    def token_count(self, text: str) -> int:
        ...
```

### Step 5.2: Create `src/llm/local_backend.py`

```python
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from config.settings import settings
from .base import LLMBackend, GenerationMetrics


class LocalHFBackend(LLMBackend):
    def __init__(self):
        super().__init__()
        print(f"[LLM] Loading {settings.model_id} (4-bit)...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id, cache_dir=settings.hf_home)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=settings.hf_home,
        )
        print("[LLM] Ready.")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        start = time.perf_counter()
        pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer,
            max_new_tokens=max_tokens, temperature=max(temperature, 0.01),
            do_sample=temperature > 0.05, return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        result = pipe(prompt)
        text = result[0]["generated_text"].strip()
        self.last_metrics = GenerationMetrics(
            prompt_tokens=self.token_count(prompt),
            response_tokens=self.token_count(text),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return text

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
```

### Step 5.3: Create `src/llm/bedrock_backend.py`

```python
import time
import boto3
from config.settings import settings
from .base import LLMBackend, GenerationMetrics


class BedrockBackend(LLMBackend):
    def __init__(self):
        super().__init__()
        self.client = boto3.client(
            "bedrock-runtime", region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        self.model_id = settings.bedrock_model_id
        print(f"[LLM] Bedrock ready ({self.model_id})")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        start = time.perf_counter()
        resp = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": max(temperature, 0.01)},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        self.last_metrics = GenerationMetrics(
            prompt_tokens=self.token_count(prompt),
            response_tokens=self.token_count(text),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return text

    def token_count(self, text: str) -> int:
        return int(len(text.split()) * 1.3)
```

### Step 5.4: Create `src/llm/factory.py`

```python
from config.settings import settings
from .base import LLMBackend


def create_llm_backend() -> LLMBackend:
    if settings.llm_backend == "local":
        from .local_backend import LocalHFBackend
        return LocalHFBackend()
    elif settings.llm_backend == "bedrock":
        from .bedrock_backend import BedrockBackend
        return BedrockBackend()
    raise ValueError(f"Unknown backend: {settings.llm_backend}")
```

### Step 5.5: Create `src/llm/__init__.py`

```python
from .base import LLMBackend, GenerationMetrics
from .factory import create_llm_backend
```

### Step 5.6: Delete old `src/llm/wrapper.py`

This file is fully replaced by the backend system. Delete it.

---

## PHASE 6: Agent Rewrite

### Step 6.1: Rewrite `src/agents/historian.py`

```python
from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.llm.base import LLMBackend


class HistorianAgent:
    def __init__(self, llm: LLMBackend, graph: TemporalGraphEngine):
        self.llm = llm
        self.graph = graph

    def conduct_audit(self, brand: str, year: int) -> str:
        print(f"  [Historian] Generating report for '{brand}' ({year})...")

        facts = self.graph.get_snapshot_for_year(brand, year)
        if not facts:
            return f"No data available for {brand} in {year}."
        if len(facts) < 3:
            return f"Insufficient data for {brand} in {year} (only {len(facts)} reviews)."

        context = self.graph.format_facts_for_prompt(facts, max_facts=80)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Brand Health Analyst generating a report from customer review data.

STRICT RULES:
- Use ONLY the data below. Do NOT add information from your training data.
- Every claim must be supported by at least one review from the data.
- If a topic has no data, say "No data available" — do not speculate.
- Report sentiment counts as they appear (e.g., "7 of 10 reviews were negative").
- Use specific quotes or paraphrases from reviews as evidence.

CUSTOMER REVIEW DATA FOR {brand} ({year}):
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Generate a Brand Health Report for {brand} ({year}, {len(facts)} reviews).

Format:
## Executive Summary
(2-3 sentences: sentiment breakdown, key finding)

## Critical Issues
- [topic]: [description with evidence from reviews]

## Strengths
- [topic]: [description with evidence from reviews]

## Sentiment Breakdown
- Positive: X reviews (Y%)
- Negative: X reviews (Y%)
- Neutral: X reviews (Y%)

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return self.llm.generate(prompt, max_tokens=settings.historian_max_tokens, temperature=settings.historian_temperature)
```

### Step 6.2: Rewrite `src/agents/critic.py`

```python
import re
from collections import Counter
from typing import Dict, List

from config.settings import settings
from src.graph.engine import TemporalGraphEngine, SnapshotFact
from src.llm.base import LLMBackend


class CriticAgent:
    def __init__(self, llm: LLMBackend, graph: TemporalGraphEngine):
        self.llm = llm
        self.graph = graph

    def verify_audit(self, brand: str, audit_draft: str, year: int) -> Dict:
        print(f"  [Critic] Verifying report for '{brand}' ({year})...")

        if not audit_draft or len(audit_draft.strip()) < 20:
            return {"status": "ERROR", "reasoning": "Draft too short.", "issues_found": [], "raw_output": ""}

        facts = self.graph.get_snapshot_for_year(brand, year)
        if not facts:
            return {"status": "ERROR", "reasoning": f"No ground truth for {brand} in {year}.", "issues_found": [], "raw_output": ""}

        ground_truth = self._build_ground_truth(facts)
        prompt = self._build_prompt(brand, year, audit_draft, ground_truth)

        raw = self.llm.generate(prompt, max_tokens=settings.critic_max_tokens, temperature=settings.critic_temperature)
        return self.parse_verdict(raw)

    def _build_ground_truth(self, facts: List[SnapshotFact]) -> str:
        pos = sum(1 for f in facts if f.sentiment.value == "POSITIVE")
        neg = sum(1 for f in facts if f.sentiment.value == "NEGATIVE")
        neu = sum(1 for f in facts if f.sentiment.value == "NEUTRAL")
        topics = Counter(f.topic.value for f in facts)

        lines = [
            f"GROUND TRUTH ({len(facts)} reviews):",
            f"Sentiment: {pos} positive, {neg} negative, {neu} neutral",
            "",
            "Topics present:",
        ]
        for topic, count in topics.most_common():
            lines.append(f"  - {topic}: {count} reviews")
        lines.append("")
        lines.append("Sample reviews:")
        for f in facts[:25]:
            lines.append(f"  [{f.sentiment.value}] [{f.topic.value}] {f.review_text[:200]}")
        return "\n".join(lines)

    def _build_prompt(self, brand: str, year: int, draft: str, ground_truth: str) -> str:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Fact-Checking Critic. Verify a brand report against ground truth data.

CHECK FOR:
1. HALLUCINATED FACTS: Claims about events or features NOT in any review
2. WRONG SENTIMENT: Report says positive when data shows negative (or vice versa)
3. TEMPORAL LEAKAGE: Mentions of products or events from other years
4. FABRICATED NUMBERS: Statistics that don't match actual review counts
5. UNSUPPORTED CLAIMS: Conclusions not backed by review data

{ground_truth}

<|eot_id|><|start_header_id|>user<|end_header_id|>

REPORT TO VERIFY:
---
{draft}
---

Output your verdict in EXACTLY this format:
STATUS: [PASS or FAIL]
ISSUES:
- [issue description, or "None found"]
REASONING: [1-2 sentences]

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def parse_verdict(self, raw_output: str) -> Dict:
        status = "UNKNOWN"
        status_match = re.search(r"STATUS:\s*(PASS|FAIL)", raw_output, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).upper()
        elif re.search(r"\bPASS\b", raw_output, re.IGNORECASE):
            status = "PASS"
        elif re.search(r"\bFAIL\b", raw_output, re.IGNORECASE):
            status = "FAIL"

        issues = []
        issues_match = re.search(r"ISSUES:\s*\n(.*?)(?:REASONING:|$)", raw_output, re.DOTALL | re.IGNORECASE)
        if issues_match:
            for line in issues_match.group(1).strip().split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line and line.lower() not in ("none", "none found", "n/a"):
                    issues.append(line)

        reasoning = ""
        reasoning_match = re.search(r"REASONING:\s*(.+?)$", raw_output, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()[:300]
        else:
            reasoning = raw_output[:200]

        return {"status": status, "issues_found": issues, "reasoning": reasoning, "raw_output": raw_output}
```

### Step 6.3: Rewrite `main.py`

```python
import gc
import sys
import torch

from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.llm import create_llm_backend
from src.agents.historian import HistorianAgent
from src.agents.critic import CriticAgent


def main():
    print("=" * 60)
    print("   TEMPORAL GRAPH AGENT — Interactive Brand Audit")
    print("=" * 60)

    print("\n[System] Loading graph...")
    graph = TemporalGraphEngine()
    if not graph.load_from_disk(settings.graph_path):
        print("[ERROR] Run 'python ingest.py' first.")
        sys.exit(1)

    print("[System] Loading LLM...")
    llm = create_llm_backend()

    historian = HistorianAgent(llm, graph)
    critic = CriticAgent(llm, graph)

    brands = graph.get_brands()
    print(f"\n[Ready] {len(brands)} brands. Examples: {brands[:10]}")

    while True:
        print("\n" + "=" * 60)
        target = input("\n>> Brand (or 'q'): ").strip()
        if target.lower() == "q":
            break

        matched = next((b for b in brands if target.lower() == b.lower()), None)
        if not matched:
            matches = [b for b in brands if target.lower() in b.lower()]
            print(f"  Not found. Similar: {matches[:5]}" if matches else f"  '{target}' not in graph.")
            continue

        try:
            y1 = int(input(">> Baseline year: ").strip())
            y2 = int(input(">> Comparison year: ").strip())
        except ValueError:
            print("  Invalid year.")
            continue

        for year in [y1, y2]:
            print(f"\n--- {matched} in {year} ---")
            report = historian.conduct_audit(matched, year)
            print(f"\n[Report]:\n{report}")
            verdict = critic.verify_audit(matched, report, year)
            print(f"\n[Verdict]: {verdict['status']} — {verdict['reasoning']}")
            if verdict["issues_found"]:
                for issue in verdict["issues_found"]:
                    print(f"  ! {issue}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
```

---

## PHASE 7-10: Metrics, API, HuggingFace, AWS

These phases remain the same as in the original plan. Create:
- `evaluation/performance_tracker.py` — timing decorator
- `evaluation/hallucination_benchmark.py` — synthetic injection + P/R/F1
- `evaluation/report_quality.py` — ROUGE + BERTScore
- `evaluation/test_cases.json` — curated test cases (UPDATE brands after ingestion!)
- `evaluation/run_eval.py` — CLI runner
- `api/main.py` — FastAPI with /audit, /brands, /health
- `hf_space/app.py` — Gradio demo
- `hf_space/requirements.txt` — Spaces dependencies
- `scripts/export_dataset.py` — graph to HF Dataset
- `infra/template.yaml` — SAM/CloudFormation
- `infra/handler.py` — Lambda function
- `infra/deploy.sh` — deployment script

For the complete code of these files, refer to the sections below:

(The evaluation, API, HF Space, and AWS code is identical to what was in the previous version of this plan — those components were already correctly designed. The key difference is they now work on top of a FIXED foundation.)

---

## Verification Checklist

```bash
# 1. Imports work
python -c "from config.settings import settings; print(settings.model_id)"

# 2. Ingestion works
python ingest.py --limit 100 --data-dir ./data/amazon_data --output test_graph.pkl

# 3. Graph queries work
python -c "
from src.graph.engine import TemporalGraphEngine
g = TemporalGraphEngine()
g.load_from_disk('test_graph.pkl')
print(g.stats())
brands = g.get_brands()
if brands:
    facts = g.get_snapshot_for_year(brands[0], 2023)
    print(f'{brands[0]}: {len(facts)} facts')
    if facts:
        print(f'  Sample: {facts[0].topic.value}, {facts[0].sentiment.value}, {facts[0].date}')
"

# 4. Full pipeline
python main.py

# 5. API
uvicorn api.main:app --port 8000
curl http://localhost:8000/health

# 6. Evaluation
python -m evaluation.run_eval --benchmark hallucination
```

---

## Critical Notes

1. **Phase order matters** — Phases 1-6 fix the broken core. Don't skip to deployment.
2. **Re-run ingestion** — the old `thesis_graph.pkl` was built with broken logic. Must rebuild.
3. **Test with `--limit 100` first** — verify correctness before full 5000/file run.
4. **Check brands after ingestion** — update `evaluation/test_cases.json` with real brands from your graph.
5. **Delete `src/llm/wrapper.py`** — it's replaced by the backend system.
6. **The old `__init__.py` renames (`src_init.py` etc.) must be deleted** — they serve no purpose.
7. **Amazon 2023 data uses millisecond timestamps** — the date parser handles `ts > 1e12` conversion.
8. **Commit after each phase** — one commit per phase for clean history.
