import os
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_backend: Literal["local", "bedrock"] = "local"
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_home: str = "./models_cache"
    historian_max_tokens: int = 1024
    historian_temperature: float = 0.4
    critic_max_tokens: int = 512
    critic_temperature: float = 0.1

    # Graph
    graph_path: str = "./thesis_graph.pkl"

    # Ingestion
    ingest_limit: int = 5000
    ingest_batch_size: int = 32
    data_dir: str = "./data/amazon_data"

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "meta.llama3-8b-instruct-v1:0"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Evaluation
    eval_output_dir: str = "./results"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
os.environ["HF_HOME"] = settings.hf_home
