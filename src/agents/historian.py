from typing import Optional, Dict
from src.graph.engine import TemporalGraphEngine
from src.llm.base import LLMBackend
from .base import HistorianBase


class HistorianAgent(HistorianBase):
    def __init__(self, llm: LLMBackend, graph: TemporalGraphEngine, config: Optional[Dict] = None):
        super().__init__(llm, graph, config)

    def conduct_audit(self, brand: str, year: int) -> str:
        print(f"  [Historian] Generating report for '{brand}' ({year})...")

        facts = self.graph.get_snapshot_for_year(brand, year)
        if not facts:
            return f"No data available for {brand} in {year}."
        if len(facts) < 3:
            return f"Insufficient data for {brand} in {year} (only {len(facts)} reviews)."

        # Pre-calculate sentiment counts to prevent LLM miscounting
        pos_count = sum(1 for f in facts if f.sentiment.value == "POSITIVE")
        neg_count = sum(1 for f in facts if f.sentiment.value == "NEGATIVE")
        neu_count = sum(1 for f in facts if f.sentiment.value == "NEUTRAL")

        context = self.graph.format_facts_for_prompt(facts, max_facts=80)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Brand Health Analyst generating a report from customer review data.

STRICT RULES:
- Use ONLY the data below. Do NOT add information from your training data.
- Every claim must be supported by at least one review from the data.
- If a topic has no data, say "No data available" — do not speculate.
- Use specific quotes or paraphrases from reviews as evidence.
- Use the EXACT sentiment counts provided below.

CUSTOMER REVIEW DATA FOR {brand} ({year}):
Total reviews: {len(facts)}
Sentiment distribution:
- POSITIVE: {pos_count} reviews
- NEGATIVE: {neg_count} reviews
- NEUTRAL: {neu_count} reviews

{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Generate a Brand Health Report for {brand} ({year}).

Format:
## Executive Summary
(2-3 sentences: sentiment breakdown, key finding)

## Critical Issues
- [topic]: [description with evidence from reviews]

## Strengths
- [topic]: [description with evidence from reviews]

## Sentiment Breakdown
- Positive: {pos_count} reviews ({pos_count * 100 // len(facts)}%)
- Negative: {neg_count} reviews ({neg_count * 100 // len(facts)}%)
- Neutral: {neu_count} reviews ({neu_count * 100 // len(facts)}%)

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return self.llm.generate(
            prompt,
            max_tokens=self.get_max_tokens(),
            temperature=self.get_temperature(),
        )
