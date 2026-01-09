import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import settings

class SCCLlama:
    def __init__(self):
        print(f"Loading {settings.MODEL_ID} to SCC GPU...")
        
        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.MODEL_ID, 
            cache_dir=settings.MODEL_CACHE_DIR
        )
        # Fix for Llama 3 padding issues
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 2. Load Model (4-bit for efficiency)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID,
            device_map="auto",
            load_in_4bit=True, # Requires bitsandbytes
            cache_dir=settings.MODEL_CACHE_DIR,
            torch_dtype=torch.float16,
        )
        
        # 3. Create Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            # CRITICAL: This prevents the model from repeating your long prompt in the output
            return_full_text=False 
        )

    def generate_raw(self, full_prompt: str) -> str:
        """
        Takes a fully formatted prompt (with special tokens) and returns ONLY the new text.
        Used by Agents (Historian/Critic) who need precise control over the prompt structure.
        """
        # The pipeline automatically handles the generation
        # return_full_text=False in __init__ ensures we only get the answer
        try:
            output = self.pipe(full_prompt)
            return output[0]['generated_text'].strip()
        except Exception as e:
            return f"Error generating text: {str(e)}"

    def analyze(self, context: str, query: str) -> str:
        """
        Helper for simple Q&A. 
        Constructs the Llama 3 prompt format automatically.
        """
        # We manually format the prompt to match Llama 3's template
        # This is safer than apply_chat_template for custom RAG contexts
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Data Historian. Use ONLY the provided Context Facts to answer.
If the answer is not in the facts, say "I don't know."

Context Facts:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return self.generate_raw(prompt)