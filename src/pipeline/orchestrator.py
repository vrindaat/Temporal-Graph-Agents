"""
Pipeline orchestrator - coordinates the entire workflow.
"""
from pathlib import Path
from typing import Optional, Dict, Any

from src.config.pipeline_config import PipelineConfig
from src.graph.engine import TemporalGraphEngine
from src.llm import create_llm_backend
from src.agents.factory import create_historian, create_critic
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion import (
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


class PipelineOrchestrator:
    """
    Orchestrates the complete temporal graph agent pipeline.

    Handles:
    1. Configuration loading
    2. Component initialization
    3. Data ingestion (optional)
    4. Graph loading
    5. Agent creation
    6. Query execution
    """

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.graph: Optional[TemporalGraphEngine] = None
        self.llm = None
        self.historian = None
        self.critic = None

        # Validate config
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")

    def initialize(self, skip_ingestion: bool = False):
        """
        Initialize all pipeline components.

        Args:
            skip_ingestion: If True, skip data ingestion and load existing graph
        """
        print("=" * 80)
        print(f"  {self.config.project_name}")
        if self.config.description:
            print(f"  {self.config.description}")
        print("=" * 80)

        # Step 1: Initialize graph
        print("\n[1/5] Initializing knowledge graph...")
        self.graph = TemporalGraphEngine()

        # Step 2: Data ingestion or load existing graph
        if not skip_ingestion and self.config.ingestion:
            print("\n[2/5] Running data ingestion...")
            self._run_ingestion()
        else:
            print("\n[2/5] Loading existing graph...")
            if not self.graph.load_from_disk(self.config.graph_path):
                raise FileNotFoundError(
                    f"Graph file not found: {self.config.graph_path}. "
                    "Run with skip_ingestion=False to ingest data first."
                )

        stats = self.graph.stats()
        print(f"  Graph loaded: {stats['total_edges']} edges, {stats['brand_count']} entities")

        # Step 3: Initialize LLM
        print("\n[3/5] Initializing LLM backend...")
        self.llm = self._create_llm()
        print(f"  LLM: {self.config.llm.backend} / {self.config.llm.model_id}")

        # Step 4: Initialize Historian
        print("\n[4/5] Initializing Historian agent...")
        historian_config = {
            'max_tokens': self.config.historian.max_tokens,
            'temperature': self.config.historian.temperature,
            **self.config.historian.custom_params
        }
        self.historian = create_historian(
            self.llm,
            self.graph,
            historian_type=self.config.historian.type,
            config=historian_config
        )
        print(f"  Historian: {self.config.historian.type}")

        # Step 5: Initialize Critic
        print("\n[5/5] Initializing Critic agent...")
        critic_config = {
            'max_tokens': self.config.critic.max_tokens,
            'temperature': self.config.critic.temperature,
            **self.config.critic.custom_params
        }
        self.critic = create_critic(
            self.llm,
            self.graph,
            critic_type=self.config.critic.type,
            config=critic_config
        )
        print(f"  Critic: {self.config.critic.type}")

        print("\n" + "=" * 80)
        print("Pipeline initialized successfully!")
        print("=" * 80)

    def _create_llm(self):
        """Create LLM backend from config"""
        # Set environment variables for create_llm_backend
        import os
        os.environ['LLM_BACKEND'] = self.config.llm.backend
        os.environ['BEDROCK_MODEL_ID'] = self.config.llm.model_id

        if self.config.llm.region:
            os.environ['AWS_REGION'] = self.config.llm.region

        return create_llm_backend()

    def _run_ingestion(self):
        """Run data ingestion pipeline"""
        if not self.config.ingestion:
            raise ValueError("Ingestion config not provided")

        ing_config = self.config.ingestion

        # Create connector
        connector = self._create_connector(ing_config.connector)

        # Create extractors/classifiers
        entity_extractor = self._create_entity_extractor(ing_config.entity_extractor)
        sentiment_classifier = self._create_sentiment_classifier(ing_config.sentiment_classifier)
        topic_classifier = self._create_topic_classifier(ing_config.topic_classifier)

        # Create and run pipeline
        pipeline = IngestionPipeline(
            connector=connector,
            entity_extractor=entity_extractor,
            sentiment_classifier=sentiment_classifier,
            topic_classifier=topic_classifier
        )

        stats = pipeline.run(
            self.graph,
            limit=ing_config.limit,
            batch_size=ing_config.batch_size
        )

        # Save graph
        self.graph.save_to_disk(self.config.graph_path)
        print(f"  Saved graph to {self.config.graph_path}")

    def _create_connector(self, config: Dict):
        """Factory for connectors"""
        connector_type = config['type']
        connector_config = config['config']

        if connector_type == 'csv':
            return CSVConnector(connector_config)
        elif connector_type == 'jsonl':
            return JSONLinesConnector(connector_config)
        elif connector_type == 'directory':
            return DirectoryConnector(connector_config)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")

    def _create_entity_extractor(self, config: Optional[Dict]):
        """Factory for entity extractors"""
        if not config or config.get('type') == 'none':
            return None

        extractor_type = config['type']

        if extractor_type == 'keyword':
            return KeywordEntityExtractor(
                entities=set(config['entities']),
                case_sensitive=config.get('case_sensitive', False)
            )
        elif extractor_type == 'spacy':
            return SpacyEntityExtractor(
                entity_types=config['entity_types'],
                model=config.get('model', 'en_core_web_sm')
            )
        else:
            raise ValueError(f"Unknown entity extractor type: {extractor_type}")

    def _create_sentiment_classifier(self, config: Optional[Dict]):
        """Factory for sentiment classifiers"""
        if not config or config.get('type') == 'none':
            return None

        classifier_type = config['type']

        if classifier_type == 'vader':
            return VADERSentimentClassifier(
                pos_threshold=config.get('pos_threshold', 0.05),
                neg_threshold=config.get('neg_threshold', -0.05)
            )
        elif classifier_type == 'transformer':
            return TransformerSentimentClassifier(
                model=config.get('model', 'distilbert-base-uncased-finetuned-sst-2-english')
            )
        else:
            raise ValueError(f"Unknown sentiment classifier type: {classifier_type}")

    def _create_topic_classifier(self, config: Optional[Dict]):
        """Factory for topic classifiers"""
        if not config or config.get('type') == 'none':
            return None

        classifier_type = config['type']

        if classifier_type == 'zero-shot':
            return ZeroShotTopicClassifier(
                topics=config['topics'],
                model=config.get('model', 'facebook/bart-large-mnli'),
                confidence_threshold=config.get('confidence_threshold', 0.4)
            )
        elif classifier_type == 'keyword':
            return KeywordTopicClassifier(
                keyword_map=config['keyword_map'],
                default_topic=config.get('default_topic', 'General')
            )
        else:
            raise ValueError(f"Unknown topic classifier type: {classifier_type}")

    def run_audit(self, entity: str, year: int) -> Dict[str, Any]:
        """
        Run a complete audit for an entity and year.

        Args:
            entity: Entity to audit
            year: Year to analyze

        Returns:
            Dict with report, verification, and metadata
        """
        if not self.historian or not self.critic:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        print(f"\n--- {entity} in {year} ---")

        # Generate report
        print(f"  [Historian] Generating report...")
        report = self.historian.conduct_audit(entity, year)

        # Verify report
        print(f"  [Critic] Verifying report...")
        verification = self.critic.verify_audit(entity, report, year)

        return {
            'entity': entity,
            'year': year,
            'report': report,
            'verification': verification,
            'status': verification['status'],
        }

    def get_available_entities(self):
        """Get list of entities in the graph"""
        if not self.graph:
            raise RuntimeError("Pipeline not initialized")
        return self.graph.get_brands()
