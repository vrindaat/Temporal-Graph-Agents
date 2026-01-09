import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the local path where the model must be saved
save_path = "/projectnb/cs599x1/students/akhilg/directed_study_v/brand_audit/models/bart-large-mnli"
model_name = "facebook/bart-large-mnli"

print(f"Downloading model to: {save_path}")
os.makedirs(save_path, exist_ok=True)

try:
    # 1. Load the Model (Specifying the exact classification class)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Save Locally
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("\n[SUCCESS] Model and tokenizer saved locally.")
    
except Exception as e:
    print(f"\n[CRITICAL ERROR] Download or save failed. Check your internet connection or Hugging Face authentication (if needed). Error: {e}")