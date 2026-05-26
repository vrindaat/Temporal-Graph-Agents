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
