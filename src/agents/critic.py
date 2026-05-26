import re
from collections import Counter
from typing import Dict, List

from config.settings import settings
from src.graph.engine import TemporalGraphEngine, SnapshotFact
from src.llm.base import LLMBackend


class CriticAgent:
    def __init__(self, llm: LLMBackend, graph: TemporalGraphEngine):
        self.llm = llm
        self.graph = graph

    def verify_audit(self, brand: str, audit_draft: str, year: int) -> Dict:
        print(f"  [Critic] Verifying report for '{brand}' ({year})...")

        if not audit_draft or len(audit_draft.strip()) < 20:
            return {"status": "ERROR", "reasoning": "Draft too short.", "issues_found": [], "raw_output": ""}

        facts = self.graph.get_snapshot_for_year(brand, year)
        if not facts:
            return {"status": "ERROR", "reasoning": f"No ground truth for {brand} in {year}.", "issues_found": [], "raw_output": ""}

        ground_truth = self._build_ground_truth(facts)
        prompt = self._build_prompt(brand, year, audit_draft, ground_truth)

        raw = self.llm.generate(
            prompt,
            max_tokens=settings.critic_max_tokens,
            temperature=settings.critic_temperature,
        )
        return self.parse_verdict(raw)

    def _build_ground_truth(self, facts: List[SnapshotFact]) -> str:
        pos = sum(1 for f in facts if f.sentiment.value == "POSITIVE")
        neg = sum(1 for f in facts if f.sentiment.value == "NEGATIVE")
        neu = sum(1 for f in facts if f.sentiment.value == "NEUTRAL")
        topics = Counter(f.topic.value for f in facts)

        lines = [
            f"GROUND TRUTH ({len(facts)} reviews):",
            f"Sentiment: {pos} positive, {neg} negative, {neu} neutral",
            "",
            "Topics present:",
        ]
        for topic, count in topics.most_common():
            lines.append(f"  - {topic}: {count} reviews")
        lines.append("")
        lines.append("Sample reviews:")
        for f in facts[:25]:
            lines.append(f"  [{f.sentiment.value}] [{f.topic.value}] {f.review_text[:200]}")
        return "\n".join(lines)

    def _build_prompt(self, brand: str, year: int, draft: str, ground_truth: str) -> str:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Fact-Checking Critic. Verify a brand report against ground truth data.

CHECK FOR:
1. HALLUCINATED FACTS: Claims about events or features NOT in any review
2. WRONG SENTIMENT: Report says positive when data shows negative (or vice versa)
3. TEMPORAL LEAKAGE: Mentions of products or events from other years
4. FABRICATED NUMBERS: Statistics that don't match actual review counts
5. UNSUPPORTED CLAIMS: Conclusions not backed by review data

{ground_truth}

<|eot_id|><|start_header_id|>user<|end_header_id|>

REPORT TO VERIFY:
---
{draft}
---

Output your verdict in EXACTLY this format:
STATUS: [PASS or FAIL]
ISSUES:
- [issue description, or "None found"]
REASONING: [1-2 sentences]

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def parse_verdict(self, raw_output: str) -> Dict:
        status = "UNKNOWN"
        status_match = re.search(r"STATUS:\s*(PASS|FAIL)", raw_output, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).upper()
        elif re.search(r"\bPASS\b", raw_output, re.IGNORECASE):
            status = "PASS"
        elif re.search(r"\bFAIL\b", raw_output, re.IGNORECASE):
            status = "FAIL"

        issues = []
        issues_match = re.search(r"ISSUES:\s*\n(.*?)(?:REASONING:|$)", raw_output, re.DOTALL | re.IGNORECASE)
        if issues_match:
            for line in issues_match.group(1).strip().split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line and line.lower() not in ("none", "none found", "n/a"):
                    issues.append(line)

        reasoning = ""
        reasoning_match = re.search(r"REASONING:\s*(.+?)$", raw_output, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()[:300]
        else:
            reasoning = raw_output[:200]

        return {"status": status, "issues_found": issues, "reasoning": reasoning, "raw_output": raw_output}
