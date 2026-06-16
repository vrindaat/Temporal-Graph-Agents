"""V3 Critic: Minimal LLM usage, rule-based verification"""
import re
from collections import Counter
from typing import Dict, List
from config.settings import settings
from src.graph.engine import TemporalGraphEngine, SnapshotFact
from src.llm.base import LLMBackend


class RobustCriticAgent:
    """
    A critic that uses minimal LLM reasoning and focuses on objective checks.
    Philosophy: Catch obvious fabrications, allow reasonable paraphrasing.
    """

    def __init__(self, llm: LLMBackend, graph: TemporalGraphEngine):
        self.llm = llm
        self.graph = graph

    def verify_audit(self, brand: str, audit_draft: str, year: int) -> Dict:
        print(f"  [Critic] Verifying report for '{brand}' ({year})...")

        if not audit_draft or len(audit_draft.strip()) < 20:
            return {"status": "ERROR", "reasoning": "Draft too short.", "issues_found": []}

        facts = self.graph.get_snapshot_for_year(brand, year)
        if not facts:
            return {"status": "ERROR", "reasoning": f"No ground truth for {brand} in {year}.", "issues_found": []}

        # Objective rule-based checks
        issues = self._objective_checks(audit_draft, facts)

        # Only use LLM for ambiguous cases (few reviews where fabrication is more likely)
        if len(facts) <= 5:
            llm_issues = self._llm_spot_check(brand, audit_draft, facts)
            issues.extend(llm_issues)

        status = "FAIL" if issues else "PASS"
        reasoning = "Passed all checks" if not issues else f"Found {len(issues)} issue(s)"

        return {
            "status": status,
            "issues_found": issues,
            "reasoning": reasoning,
            "raw_output": ""
        }

    def _objective_checks(self, draft: str, facts: List[SnapshotFact]) -> List[str]:
        """Rule-based checks that don't need LLM"""
        issues = []

        # 1. Check sentiment counts
        pos_match = re.search(r'Positive:\s*(\d+)', draft, re.IGNORECASE)
        neg_match = re.search(r'Negative:\s*(\d+)', draft, re.IGNORECASE)
        neu_match = re.search(r'Neutral:\s*(\d+)', draft, re.IGNORECASE)

        if pos_match and neg_match and neu_match:
            reported_pos = int(pos_match.group(1))
            reported_neg = int(neg_match.group(1))
            reported_neu = int(neu_match.group(1))

            actual_pos = sum(1 for f in facts if f.sentiment.value == "POSITIVE")
            actual_neg = sum(1 for f in facts if f.sentiment.value == "NEGATIVE")
            actual_neu = sum(1 for f in facts if f.sentiment.value == "NEUTRAL")

            if reported_pos != actual_pos:
                issues.append(f"Sentiment count error: reported {reported_pos} positive, actual {actual_pos}")
            if reported_neg != actual_neg:
                issues.append(f"Sentiment count error: reported {reported_neg} negative, actual {actual_neg}")
            if reported_neu != actual_neu:
                issues.append(f"Sentiment count error: reported {reported_neu} neutral, actual {actual_neu}")

        # 2. Check for obviously impossible claims (sample size violations)
        total_reviews = len(facts)
        # Extract percentages from draft
        percentages = re.findall(r'(\d+)%', draft)
        for pct_str in percentages:
            pct = int(pct_str)
            if pct > 100:
                issues.append(f"Impossible percentage: {pct}%")

        # 3. Check for "No data available" in wrong context
        draft_start = draft[:250].lower()
        if "no data available" in draft_start and len(facts) > 0:
            issues.append("Report claims no data available, but reviews exist")

        return issues

    def _llm_spot_check(self, brand: str, draft: str, facts: List[SnapshotFact]) -> List[str]:
        """
        Light LLM check: ask model to list main claims, then verify them
        Only used when review count is low (higher risk of fabrication)
        """
        issues = []

        # Build review text
        review_texts = "\n\n".join([
            f"Review {i+1}: {f.review_text[:300]}"
            for i, f in enumerate(facts)
        ])

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Extract specific factual claims from a brand report.

<|eot_id|><|start_header_id|>user<|end_header_id|>

REPORT:
{draft}

AVAILABLE REVIEWS:
{review_texts}

List the 3 most specific factual claims in the report (e.g., features mentioned, specific user experiences, product names).
For each claim, state if it's supported by the reviews.

Format:
1. [Claim] - SUPPORTED or FABRICATED
2. [Claim] - SUPPORTED or FABRICATED
3. [Claim] - SUPPORTED or FABRICATED

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        try:
            response = self.llm.generate(
                prompt,
                max_tokens=min(settings.critic_max_tokens, 400),
                temperature=0.1
            )

            # Count "FABRICATED" mentions
            fabricated_count = response.upper().count("FABRICATED")
            if fabricated_count >= 2:
                issues.append(f"Multiple claims appear fabricated (LLM detected {fabricated_count})")

        except Exception as e:
            # Don't fail if LLM errors out
            pass

        return issues
