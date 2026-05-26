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
