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
