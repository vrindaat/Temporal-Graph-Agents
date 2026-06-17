"""
Concrete implementations of entity extractors, sentiment classifiers, and topic classifiers.
"""
import re
from typing import Optional, List, Set
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline
from .base import EntityExtractor, SentimentClassifier, TopicClassifier


class KeywordEntityExtractor(EntityExtractor):
    """
    Simple keyword-based entity extraction.
    Fast and deterministic - good for known entity lists.

    Example usage:
        extractor = KeywordEntityExtractor(entities=["Apple", "Samsung", "Google"])
    """

    def __init__(self, entities: Set[str], case_sensitive: bool = False):
        """
        Args:
            entities: Set of entity names to look for
            case_sensitive: Whether matching should be case-sensitive
        """
        self.entities = entities
        self.case_sensitive = case_sensitive
        if not case_sensitive:
            self.entities_lower = {e.lower() for e in entities}

    def extract(self, text: str) -> Optional[str]:
        text_to_search = text if self.case_sensitive else text.lower()
        entity_set = self.entities if self.case_sensitive else self.entities_lower

        # Use word boundaries to avoid false matches (e.g., "pineapple" != "apple")
        for entity in entity_set:
            pattern = r'\b' + re.escape(entity) + r'\b'
            if re.search(pattern, text_to_search):
                # Return original casing from entity set
                return next(e for e in self.entities
                           if e.lower() == entity.lower())
        return None


class SpacyEntityExtractor(EntityExtractor):
    """
    NER-based entity extraction using spaCy.
    More flexible but slower than keyword matching.

    Example usage:
        extractor = SpacyEntityExtractor(
            entity_types=["ORG", "PRODUCT"],
            model="en_core_web_sm"
        )
    """

    def __init__(self, entity_types: List[str], model: str = "en_core_web_sm"):
        """
        Args:
            entity_types: List of spaCy entity types (ORG, PRODUCT, PERSON, etc.)
            model: spaCy model to use
        """
        import spacy
        try:
            self.nlp = spacy.load(model)
        except OSError:
            import os
            os.system(f"python -m spacy download {model}")
            self.nlp = spacy.load(model)

        self.entity_types = set(entity_types)

    def extract(self, text: str) -> Optional[str]:
        doc = self.nlp(text[:500])  # Limit to first 500 chars for speed
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                return ent.text
        return None


class VADERSentimentClassifier(SentimentClassifier):
    """
    Rule-based sentiment analysis using VADER.
    Fast, works well for social media and reviews.
    """

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        """
        Args:
            pos_threshold: Compound score >= this is POSITIVE
            neg_threshold: Compound score <= this is NEGATIVE
            Between thresholds is NEUTRAL
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def classify(self, text: str) -> str:
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']

        if compound >= self.pos_threshold:
            return "POSITIVE"
        elif compound <= self.neg_threshold:
            return "NEGATIVE"
        else:
            return "NEUTRAL"


class TransformerSentimentClassifier(SentimentClassifier):
    """
    Transformer-based sentiment analysis.
    More accurate but slower than VADER.

    Example usage:
        classifier = TransformerSentimentClassifier(
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    """

    def __init__(self, model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.pipe = hf_pipeline("sentiment-analysis", model=model, device=-1)

    def classify(self, text: str) -> str:
        result = self.pipe(text[:512])[0]  # Limit to 512 tokens
        label = result['label'].upper()

        # Map common labels to our format
        if label in ["POSITIVE", "POS"]:
            return "POSITIVE"
        elif label in ["NEGATIVE", "NEG"]:
            return "NEGATIVE"
        else:
            return "NEUTRAL"


class ZeroShotTopicClassifier(TopicClassifier):
    """
    Zero-shot topic classification using transformer models.
    No training required - just provide topic labels.

    Example usage:
        classifier = ZeroShotTopicClassifier(
            topics=["Product Quality", "Customer Service", "Price"]
        )
    """

    def __init__(self,
                 topics: List[str],
                 model: str = "facebook/bart-large-mnli",
                 confidence_threshold: float = 0.4):
        """
        Args:
            topics: List of topic labels to classify into
            model: Zero-shot classification model
            confidence_threshold: Minimum confidence to assign topic (else "General")
        """
        self.topics = topics
        self.confidence_threshold = confidence_threshold
        self.pipe = hf_pipeline("zero-shot-classification", model=model, device=-1)

    def classify(self, text: str) -> str:
        result = self.pipe(text[:512], candidate_labels=self.topics, multi_label=False)
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        if top_score >= self.confidence_threshold:
            return top_label
        else:
            return "General"

    def get_available_topics(self) -> List[str]:
        return self.topics


class KeywordTopicClassifier(TopicClassifier):
    """
    Simple keyword-based topic classification.
    Fast and interpretable.

    Example usage:
        classifier = KeywordTopicClassifier(
            keyword_map={
                "Quality": ["broken", "defective", "durable", "build quality"],
                "Price": ["expensive", "cheap", "value", "cost"],
                "Service": ["support", "customer service", "delivery"]
            }
        )
    """

    def __init__(self, keyword_map: dict[str, List[str]], default_topic: str = "General"):
        """
        Args:
            keyword_map: Dict mapping topic names to keyword lists
            default_topic: Topic to assign if no keywords match
        """
        self.keyword_map = keyword_map
        self.default_topic = default_topic

        # Pre-compile regex patterns for speed
        self.patterns = {}
        for topic, keywords in keyword_map.items():
            pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.patterns[topic] = re.compile(pattern, re.IGNORECASE)

    def classify(self, text: str) -> str:
        text_lower = text.lower()
        topic_scores = {}

        for topic, pattern in self.patterns.items():
            matches = pattern.findall(text_lower)
            topic_scores[topic] = len(matches)

        if not topic_scores or max(topic_scores.values()) == 0:
            return self.default_topic

        return max(topic_scores, key=topic_scores.get)

    def get_available_topics(self) -> List[str]:
        return list(self.keyword_map.keys()) + [self.default_topic]
