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
