"""
Configuration classes for the complete pipeline.
Defines structure for end-to-end workflow configs.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class LLMConfig:
    """LLM backend configuration"""
    backend: str  # 'bedrock' or 'local'
    model_id: str
    region: Optional[str] = None  # For Bedrock
    max_tokens: int = 1024
    temperature: float = 0.4


@dataclass
class IngestionConfig:
    """Data ingestion configuration"""
    connector: Dict[str, Any]
    entity_extractor: Optional[Dict[str, Any]] = None
    sentiment_classifier: Optional[Dict[str, Any]] = None
    topic_classifier: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    batch_size: int = 32


@dataclass
class HistorianConfig:
    """Historian agent configuration"""
    type: str = 'default'  # Which historian implementation to use
    max_tokens: int = 1024
    temperature: float = 0.4
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticConfig:
    """Critic agent configuration"""
    type: str = 'robust'  # Which critic implementation to use
    max_tokens: int = 512
    temperature: float = 0.1
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Complete end-to-end pipeline configuration"""
    # Project metadata
    project_name: str
    description: Optional[str] = None

    # Data paths
    graph_path: str = './graph.pkl'
    output_dir: str = './output'

    # Components
    llm: LLMConfig = None
    ingestion: Optional[IngestionConfig] = None
    historian: HistorianConfig = field(default_factory=HistorianConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)

    # Query settings (for main.py style usage)
    available_entities: Optional[List[str]] = None
    date_range: Optional[Dict[str, int]] = None  # {'min': 2014, 'max': 2024}

    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse nested configs
        llm_config = None
        if 'llm' in data:
            llm_config = LLMConfig(**data['llm'])

        ingestion_config = None
        if 'ingestion' in data and data['ingestion'] is not None:
            ingestion_config = IngestionConfig(**data['ingestion'])

        historian_config = HistorianConfig()
        if 'historian' in data:
            historian_config = HistorianConfig(**data['historian'])

        critic_config = CriticConfig()
        if 'critic' in data:
            critic_config = CriticConfig(**data['critic'])

        return cls(
            project_name=data['project_name'],
            description=data.get('description'),
            graph_path=data.get('graph_path', './graph.pkl'),
            output_dir=data.get('output_dir', './output'),
            llm=llm_config,
            ingestion=ingestion_config,
            historian=historian_config,
            critic=critic_config,
            available_entities=data.get('available_entities'),
            date_range=data.get('date_range')
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        data = {
            'project_name': self.project_name,
            'description': self.description,
            'graph_path': self.graph_path,
            'output_dir': self.output_dir,
        }

        if self.llm:
            data['llm'] = {
                'backend': self.llm.backend,
                'model_id': self.llm.model_id,
                'region': self.llm.region,
                'max_tokens': self.llm.max_tokens,
                'temperature': self.llm.temperature,
            }

        if self.ingestion:
            data['ingestion'] = {
                'connector': self.ingestion.connector,
                'entity_extractor': self.ingestion.entity_extractor,
                'sentiment_classifier': self.ingestion.sentiment_classifier,
                'topic_classifier': self.ingestion.topic_classifier,
                'limit': self.ingestion.limit,
                'batch_size': self.ingestion.batch_size,
            }

        data['historian'] = {
            'type': self.historian.type,
            'max_tokens': self.historian.max_tokens,
            'temperature': self.historian.temperature,
            **self.historian.custom_params
        }

        data['critic'] = {
            'type': self.critic.type,
            'max_tokens': self.critic.max_tokens,
            'temperature': self.critic.temperature,
            **self.critic.custom_params
        }

        if self.available_entities:
            data['available_entities'] = self.available_entities

        if self.date_range:
            data['date_range'] = self.date_range

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        if not self.project_name:
            errors.append("project_name is required")

        if self.llm is None:
            errors.append("llm configuration is required")
        elif self.llm.backend not in ['bedrock', 'local']:
            errors.append(f"Invalid llm.backend: {self.llm.backend} (must be 'bedrock' or 'local')")

        if self.ingestion:
            if 'type' not in self.ingestion.connector:
                errors.append("ingestion.connector.type is required")
            if 'config' not in self.ingestion.connector:
                errors.append("ingestion.connector.config is required")

        return errors
