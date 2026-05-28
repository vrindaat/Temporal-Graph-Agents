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
        self.stats = {
            "processed": 0, "added": 0, "skipped_no_brand": 0,
            "skipped_short": 0, "skipped_no_date": 0, "errors": 0,
        }

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
        # Check known brands first using word boundaries (no false matches like "pineapple" for "apple")
        for brand in KNOWN_BRANDS:
            if re.search(r'\b' + re.escape(brand) + r'\b', text_lower):
                return brand.title()

        # Only use NER as fallback, with stricter filtering
        # Disabled for now to avoid noise - only use KNOWN_BRANDS
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
                # Only assign topic if confidence > 0.4, else default to GENERAL
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
                # High rating but strongly negative text — mixed, call it neutral
                return Sentiment.NEUTRAL
            elif rating <= 2.0 and compound > 0.3:
                # Low rating but positive text (possible sarcasm) — trust the rating
                return Sentiment.NEGATIVE

        if compound >= 0.2:
            return Sentiment.POSITIVE
        elif compound <= -0.2:
            return Sentiment.NEGATIVE
        return Sentiment.NEUTRAL

    def parse_date(self, raw_date) -> Optional[datetime]:
        """Returns None on failure — never silently defaults to a fake date."""
        if raw_date is None or raw_date == "":
            return None
        # Try Unix timestamp (seconds or milliseconds)
        try:
            ts = float(raw_date)
            if ts > 1e12:  # milliseconds
                ts = ts / 1000
            if 946684800 < ts < 1893456000:  # 2000–2030 sanity check
                return datetime.fromtimestamp(ts)
        except (ValueError, TypeError, OSError):
            pass
        # Try common date string formats
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
            print(f"[Loader] No CSV/JSON files found in {data_dir}")
            return

        print(f"[Loader] Found {len(files)} files. Limit: {settings.ingest_limit} reviews/file")
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

            batch.append({
                "brand": brand,
                "text": full_text,
                "rating": rating,
                "date": date,
                "id": f"Rev_{filename}_{i}",
            })

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
            review_node = Node(
                item["id"], "Review",
                {"text": item["text"][:500], "rating": item.get("rating")},
            )
            edge = TemporalEdge(
                source=item["brand"],
                target=item["id"],
                relation="REVIEWED_IN",
                topic=topic,
                sentiment=sentiment,
                start_date=item["date"],
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
