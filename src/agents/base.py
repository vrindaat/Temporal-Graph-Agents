"""
Abstract base classes for agents.
Allows different implementations and configurations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.graph.engine import TemporalGraphEngine
from src.llm.base import LLMBackend


@dataclass
class AuditReport:
    """Standardized audit report format"""
    entity: str
    year: int
    text: str
    metadata: Dict[str, Any]


@dataclass
class VerificationResult:
    """Standardized verification result format"""
    status: str  # PASS, FAIL, ERROR
    issues_found: list
    reasoning: str
    raw_output: str = ""


class HistorianBase(ABC):
    """
    Abstract base class for Historian agents.
    Generates temporal reports from knowledge graph data.
    """

    def __init__(self, llm: LLMBackend, graph: TemporalGraphEngine, config: Optional[Dict] = None):
        """
        Args:
            llm: LLM backend to use
            graph: Temporal knowledge graph
            config: Optional configuration dict
        """
        self.llm = llm
        self.graph = graph
        self.config = config or {}

    @abstractmethod
    def conduct_audit(self, entity: str, year: int) -> str:
        """
        Generate an audit report for the given entity and year.

        Args:
            entity: Entity name (brand, product, etc.)
            year: Year to analyze

        Returns:
            Audit report text
        """
        pass

    def get_max_tokens(self) -> int:
        """Get max tokens setting from config or default"""
        return self.config.get('max_tokens', 1024)

    def get_temperature(self) -> float:
        """Get temperature setting from config or default"""
        return self.config.get('temperature', 0.4)


class CriticBase(ABC):
    """
    Abstract base class for Critic agents.
    Verifies reports against ground truth data.
    """

    def __init__(self, llm: LLMBackend, graph: TemporalGraphEngine, config: Optional[Dict] = None):
        """
        Args:
            llm: LLM backend to use
            graph: Temporal knowledge graph
            config: Optional configuration dict
        """
        self.llm = llm
        self.graph = graph
        self.config = config or {}

    @abstractmethod
    def verify_audit(self, entity: str, audit_draft: str, year: int) -> Dict:
        """
        Verify an audit report against ground truth.

        Args:
            entity: Entity name
            audit_draft: Report text to verify
            year: Year being analyzed

        Returns:
            Verification result dict with status, issues_found, reasoning
        """
        pass

    def get_max_tokens(self) -> int:
        """Get max tokens setting from config or default"""
        return self.config.get('max_tokens', 512)

    def get_temperature(self) -> float:
        """Get temperature setting from config or default"""
        return self.config.get('temperature', 0.1)
