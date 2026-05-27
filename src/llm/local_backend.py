import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from config.settings import settings
from .base import LLMBackend, GenerationMetrics


class LocalHFBackend(LLMBackend):
    def __init__(self):
        super().__init__()
        print(f"[LLM] Loading {settings.model_id} (4-bit)...")

        # Use token from settings if available
        token = settings.hugging_face_hub_token if settings.hugging_face_hub_token else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id,
            cache_dir=settings.hf_home,
            token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=settings.hf_home,
            token=token
        )
        print("[LLM] Ready.")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        start = time.perf_counter()
        pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer,
            max_new_tokens=max_tokens, temperature=max(temperature, 0.01),
            do_sample=temperature > 0.05, return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        result = pipe(prompt)
        text = result[0]["generated_text"].strip()
        self.last_metrics = GenerationMetrics(
            prompt_tokens=self.token_count(prompt),
            response_tokens=self.token_count(text),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return text

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
