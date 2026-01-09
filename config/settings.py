import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define where models live (Crucial for SCC quotas)
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Point Hugging Face to this directory
os.environ["HF_HOME"] = MODEL_CACHE_DIR

# The Model ID
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"