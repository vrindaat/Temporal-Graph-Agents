from datetime import datetime
from ..graph.engine import TemporalGraphEngine
from ..llm.wrapper import SCCLlama

class CriticAgent:
    def __init__(self, graph: TemporalGraphEngine, llm: SCCLlama):
        self.graph = graph
        self.llm = llm

    def verify_audit(self, brand: str, audit_draft: str, target_date: datetime) -> str:
        # print(f"\n[DEBUG CRITIC] 1. Starting verification for {brand} in {target_date.year}...", flush=True)
        
        # 1. Check Inputs
        if not audit_draft or len(audit_draft) < 10:
            # print("[DEBUG CRITIC] Error: Input draft was empty!", flush=True)
            return "[Critic Error] I cannot verify an empty draft."

        # 2. Get Facts
        context_facts = self.graph.get_snapshot(target_date, target_brand=brand)
        # print(f"[DEBUG CRITIC] 2. Retrieved Context Facts (Length: {len(context_facts)} chars)", flush=True)
        
        # 3. Construct Prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Senior Editor. Check the DRAFT AUDIT against the GROUND TRUTH FACTS.

Ground Truth Facts:
{context_facts}

<|eot_id|><|start_header_id|>user<|end_header_id|>
Draft Audit:
"{audit_draft}"

Task:
1. Does the draft mention events NOT in the Ground Truth?
2. Is the sentiment accurate?

Output format:
## CRITIC'S VERDICT
Status: [PASS/FAIL]
Reasoning: [1 sentence]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        # print("[DEBUG CRITIC] 3. Sending Prompt to LLM... (Please Wait)", flush=True)

        # 4. Generate
        try:
            response = self.llm.generate_raw(prompt)
            
            # CRITICAL DEBUG: Print exactly what the LLM gave back, even if it's weird
            print(f"[DEBUG CRITIC] 4. LLM Raw Output: '{response}'", flush=True)
            
            if not response or not response.strip():
                return "[Critic Error] The LLM returned an empty string. Attempting fallback..."
                
            return response
            
        except Exception as e:
            print(f"[DEBUG CRITIC] CRASHED: {e}", flush=True)
            return f"[Critic Error] System Exception: {str(e)}"