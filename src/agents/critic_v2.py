"""Simplified Critic Agent - focuses on objective, verifiable checks only"""
import re
from collections import Counter
from typing import Dict, List

from config.settings import settings
from src.graph.engine import TemporalGraphEngine, SnapshotFact
from src.llm.base import LLMBackend


class SimplifiedCriticAgent:
    """A simpler critic that does objective fact-checking without complex reasoning"""

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

        # Step 1: Objective rule-based checks
        rule_issues = self._rule_based_checks(audit_draft, facts, year)

        # Step 2: LLM-based simple verification
        ground_truth = self._build_ground_truth(facts)
        prompt = self._build_simplified_prompt(brand, year, audit_draft, ground_truth)

        raw = self.llm.generate(
            prompt,
            max_tokens=settings.critic_max_tokens,
            temperature=settings.critic_temperature,
        )

        # Parse and combine results
        verdict = self.parse_verdict(raw)
        all_issues = rule_issues + verdict['issues_found']

        return {
            "status": "FAIL" if all_issues else "PASS",
            "issues_found": all_issues,
            "reasoning": verdict['reasoning'] if verdict['reasoning'] else "Passed all checks",
            "raw_output": raw
        }

    def _rule_based_checks(self, draft: str, facts: List[SnapshotFact], year: int) -> List[str]:
        """Objective rule-based checks that don't require LLM reasoning"""
        issues = []

        # Extract sentiment counts from draft
        pos_match = re.search(r'Positive:\s*(\d+)', draft, re.IGNORECASE)
        neg_match = re.search(r'Negative:\s*(\d+)', draft, re.IGNORECASE)
        neu_match = re.search(r'Neutral:\s*(\d+)', draft, re.IGNORECASE)

        if pos_match and neg_match and neu_match:
            reported_pos = int(pos_match.group(1))
            reported_neg = int(neg_match.group(1))
            reported_neu = int(neu_match.group(1))

            # Ground truth
            actual_pos = sum(1 for f in facts if f.sentiment.value == "POSITIVE")
            actual_neg = sum(1 for f in facts if f.sentiment.value == "NEGATIVE")
            actual_neu = sum(1 for f in facts if f.sentiment.value == "NEUTRAL")

            # Check for exact mismatches
            if reported_pos != actual_pos:
                issues.append(f"Sentiment mismatch: reported {reported_pos} positive but actual is {actual_pos}")
            if reported_neg != actual_neg:
                issues.append(f"Sentiment mismatch: reported {reported_neg} negative but actual is {actual_neg}")
            if reported_neu != actual_neu:
                issues.append(f"Sentiment mismatch: reported {reported_neu} neutral but actual is {actual_neu}")

        # Check if report claims NO data when data exists (check first 200 chars for overall "No data")
        if len(facts) > 0:
            # Only flag if "No data available" appears in the summary/start, not in sections
            draft_start = draft[:200].lower()
            if "no data available" in draft_start or "insufficient data" in draft_start:
                issues.append("Report says 'No data available' but reviews exist")

        return issues

    def _build_ground_truth(self, facts: List[SnapshotFact]) -> str:
        """Build a comprehensive ground truth with full review text"""
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
        lines.append("Full reviews with dates:")
        for i, f in enumerate(facts[:25], 1):
            date_str = f.date.strftime("%Y-%m-%d") if f.date else "Unknown"
            lines.append(f"\nReview {i}: [{f.sentiment.value}] [{f.topic.value}] ({date_str})")
            lines.append(f"  {f.review_text}")
        return "\n".join(lines)

    def _build_simplified_prompt(self, brand: str, year: int, draft: str, ground_truth: str) -> str:
        """Simplified prompt focused on detecting actual fabrications"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are verifying a brand report. Only flag MAJOR FABRICATIONS - do NOT fail for paraphrasing.

{ground_truth}

<|eot_id|><|start_header_id|>user<|end_header_id|>

REPORT TO VERIFY:
{draft}

WHAT COUNTS AS A MAJOR FABRICATION:
✗ Inventing specific products, features, or events NOT in ANY review
✗ Claiming sentiment opposite of reality (e.g., "all negative" when 80% positive)
✗ Fabricating specific quotes, dates, or statistics absent from reviews

WHAT IS NOT A FABRICATION (these are VALID):
✓ Paraphrasing reviews (e.g., "charges well" → "good charging performance")
✓ Summarizing trends (e.g., "majority praised X" when 5/6 reviews praise X)
✓ Using synonyms (e.g., "disappointed" → "negative experience")
✓ Acknowledging exceptions (e.g., "most liked it" doesn't mean "all" - having 1 negative is fine)

CRITICAL: "Majority" means >50%, not 100%. A report saying "majority praised X" is VALID if 5/6 reviews praise X, even if 1 doesn't.

Does the report contain MAJOR FABRICATIONS (invented claims absent from ALL reviews)?

ANSWER: YES or NO
REASONING: [One sentence - cite specific fabrication or say "accurately reflects reviews"]

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def parse_verdict(self, raw_output: str) -> Dict:
        """Parse the simplified verdict format"""
        issues = []

        # Check for ANSWER = YES (found major fabrications)
        answer_match = re.search(r'ANSWER:\s*(YES|NO)', raw_output, re.IGNORECASE)
        if answer_match and answer_match.group(1).upper() == "YES":
            issues.append("Report contains major fabrications not found in reviews")

        # Extract reasoning
        reasoning = ""
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', raw_output, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()[:300]

        return {
            "status": "FAIL" if issues else "PASS",
            "issues_found": issues,
            "reasoning": reasoning,
            "raw_output": raw_output
        }
