from datetime import datetime
from ..graph.engine import TemporalGraphEngine
from ..llm.wrapper import SCCLlama

class HistorianAgent:
    def __init__(self, graph: TemporalGraphEngine, llm: SCCLlama):
        self.graph = graph
        self.llm = llm

    def conduct_audit(self, brand: str, target_date: datetime) -> str:
        print(f"   [Evaluator] Assessing brand health for '{brand}' in {target_date.year}...")
        
        # 1. RETRIEVE DATA
        context_facts = self.graph.get_snapshot(target_date, target_brand=brand)
        
        if "No recorded events" in context_facts:
            return f"Insufficient data to evaluate {brand} for {target_date.year}."

        # 2. THE NEW "EVALUATOR" PROMPT
        # We strip away the "Historian" role. 
        # We ask for a "Health Report" based on the VADER sentiment and Topics.
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Product Strategy Consultant. Your job is to analyze customer reviews and generate a Brand Health Report.

DATASET ({target_date.year}):
{context_facts}

INSTRUCTIONS:
1. Identify the top Product Flaws (Negative Sentiment + Quality/Performance).
2. Identify the top Value Drivers (Positive Sentiment + Price/Usability).
3. Do not hallucinate. Use only the reviews provided.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Audit the brand: {brand}

Output Format:
1. **Executive Summary:**
2. **Critical Product Flaws:** (Bullet points)
3. **Customer Praises:** (Bullet points)
4. **Strategic Verdict:**
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        # 3. GENERATE
        try:
            # We use generate_raw to get the direct output
            response = self.llm.generate_raw(prompt)
            return response
        except Exception as e:
            return f"[Error] Evaluation failed: {str(e)}"