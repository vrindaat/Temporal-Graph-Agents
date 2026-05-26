from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.llm.base import LLMBackend


class HistorianAgent:
    def __init__(self, llm: LLMBackend, graph: TemporalGraphEngine):
        self.llm = llm
        self.graph = graph

    def conduct_audit(self, brand: str, year: int) -> str:
        print(f"  [Historian] Generating report for '{brand}' ({year})...")

        facts = self.graph.get_snapshot_for_year(brand, year)
        if not facts:
            return f"No data available for {brand} in {year}."
        if len(facts) < 3:
            return f"Insufficient data for {brand} in {year} (only {len(facts)} reviews)."

        context = self.graph.format_facts_for_prompt(facts, max_facts=80)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Brand Health Analyst generating a report from customer review data.

STRICT RULES:
- Use ONLY the data below. Do NOT add information from your training data.
- Every claim must be supported by at least one review from the data.
- If a topic has no data, say "No data available" — do not speculate.
- Report sentiment counts as they appear (e.g., "7 of 10 reviews were negative").
- Use specific quotes or paraphrases from reviews as evidence.

CUSTOMER REVIEW DATA FOR {brand} ({year}):
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Generate a Brand Health Report for {brand} ({year}, {len(facts)} reviews).

Format:
## Executive Summary
(2-3 sentences: sentiment breakdown, key finding)

## Critical Issues
- [topic]: [description with evidence from reviews]

## Strengths
- [topic]: [description with evidence from reviews]

## Sentiment Breakdown
- Positive: X reviews (Y%)
- Negative: X reviews (Y%)
- Neutral: X reviews (Y%)

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return self.llm.generate(
            prompt,
            max_tokens=settings.historian_max_tokens,
            temperature=settings.historian_temperature,
        )
