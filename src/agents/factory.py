"""
Factory functions for creating agents with different configurations.
"""
from typing import Dict, Optional

from src.graph.engine import TemporalGraphEngine
from src.llm.base import LLMBackend
from .base import HistorianBase, CriticBase
from .historian import HistorianAgent
from .critic_v3 import RobustCriticAgent


# Registry of available agent implementations
HISTORIAN_REGISTRY = {
    'default': HistorianAgent,
    'standard': HistorianAgent,
}

CRITIC_REGISTRY = {
    'default': RobustCriticAgent,
    'robust': RobustCriticAgent,
    'v3': RobustCriticAgent,
}


def create_historian(
    llm: LLMBackend,
    graph: TemporalGraphEngine,
    historian_type: str = 'default',
    config: Optional[Dict] = None
) -> HistorianBase:
    """
    Factory function to create a Historian agent.

    Args:
        llm: LLM backend
        graph: Knowledge graph
        historian_type: Type of historian ('default', 'standard')
        config: Optional configuration dict with:
            - max_tokens: Maximum tokens for generation
            - temperature: Temperature for sampling

    Returns:
        Historian instance

    Example:
        historian = create_historian(
            llm, graph,
            historian_type='default',
            config={'max_tokens': 2048, 'temperature': 0.3}
        )
    """
    if historian_type not in HISTORIAN_REGISTRY:
        raise ValueError(f"Unknown historian type: {historian_type}. "
                        f"Available: {list(HISTORIAN_REGISTRY.keys())}")

    HistorianClass = HISTORIAN_REGISTRY[historian_type]
    return HistorianClass(llm, graph, config)


def create_critic(
    llm: LLMBackend,
    graph: TemporalGraphEngine,
    critic_type: str = 'default',
    config: Optional[Dict] = None
) -> CriticBase:
    """
    Factory function to create a Critic agent.

    Args:
        llm: LLM backend
        graph: Knowledge graph
        critic_type: Type of critic ('default', 'robust', 'v3')
        config: Optional configuration dict with:
            - max_tokens: Maximum tokens for generation
            - temperature: Temperature for sampling

    Returns:
        Critic instance

    Example:
        critic = create_critic(
            llm, graph,
            critic_type='robust',
            config={'max_tokens': 512, 'temperature': 0.05}
        )
    """
    if critic_type not in CRITIC_REGISTRY:
        raise ValueError(f"Unknown critic type: {critic_type}. "
                        f"Available: {list(CRITIC_REGISTRY.keys())}")

    CriticClass = CRITIC_REGISTRY[critic_type]
    return CriticClass(llm, graph, config)


def register_historian(name: str, historian_class: type):
    """
    Register a custom historian implementation.

    Args:
        name: Name to register under
        historian_class: Class that inherits from HistorianBase

    Example:
        class MyHistorian(HistorianBase):
            ...

        register_historian('my_custom', MyHistorian)
    """
    if not issubclass(historian_class, HistorianBase):
        raise TypeError(f"{historian_class} must inherit from HistorianBase")
    HISTORIAN_REGISTRY[name] = historian_class


def register_critic(name: str, critic_class: type):
    """
    Register a custom critic implementation.

    Args:
        name: Name to register under
        critic_class: Class that inherits from CriticBase

    Example:
        class MyStrict Critic(CriticBase):
            ...

        register_critic('strict', MyStrictCritic)
    """
    if not issubclass(critic_class, CriticBase):
        raise TypeError(f"{critic_class} must inherit from CriticBase")
    CRITIC_REGISTRY[name] = critic_class
