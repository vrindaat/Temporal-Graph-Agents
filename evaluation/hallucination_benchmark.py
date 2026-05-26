import json
import random
from dataclasses import dataclass, field
from typing import List

from src.graph.engine import TemporalGraphEngine
from src.agents.historian import HistorianAgent
from src.agents.critic import CriticAgent


@dataclass
class TestCase:
    brand: str
    year: int
    hallucination_type: str  # "temporal_leakage", "fabricated_claim", "sentiment_reversal", "clean"
    injected_claim: str
    expected_verdict: str  # "PASS" or "FAIL"
    description: str


@dataclass
class BenchmarkResult:
    total: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    details: List[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


class HallucinationBenchmark:
    def __init__(self, graph: TemporalGraphEngine, historian: HistorianAgent, critic: CriticAgent):
        self.graph = graph
        self.historian = historian
        self.critic = critic

    def load_test_cases(self, path: str = "evaluation/test_cases.json") -> List[TestCase]:
        with open(path) as f:
            data = json.load(f)
        return [TestCase(**case) for case in data]

    def inject_temporal_leakage(self, report: str, claim: str) -> str:
        lines = report.split("\n")
        idx = max(1, len(lines) // 2)
        lines.insert(idx, f"- {claim}")
        return "\n".join(lines)

    def inject_fabricated_claim(self, report: str, claim: str) -> str:
        lines = report.split("\n")
        idx = random.randint(1, max(1, len(lines) - 1))
        lines.insert(idx, f"- {claim}")
        return "\n".join(lines)

    def inject_sentiment_reversal(self, report: str) -> str:
        swaps = {
            "positive": "negative", "negative": "positive",
            "praised": "criticized", "loved": "hated",
            "excellent": "terrible", "strong": "weak",
            "improved": "declined", "growth": "decline",
        }
        result = report
        for old, new in swaps.items():
            result = result.replace(old, new).replace(old.capitalize(), new.capitalize())
        return result

    def run_single(self, case: TestCase) -> dict:
        clean_report = self.historian.conduct_audit(case.brand, case.year)

        if case.hallucination_type == "clean":
            test_report = clean_report
        elif case.hallucination_type == "temporal_leakage":
            test_report = self.inject_temporal_leakage(clean_report, case.injected_claim)
        elif case.hallucination_type == "fabricated_claim":
            test_report = self.inject_fabricated_claim(clean_report, case.injected_claim)
        elif case.hallucination_type == "sentiment_reversal":
            test_report = self.inject_sentiment_reversal(clean_report)
        else:
            test_report = clean_report

        verdict = self.critic.verify_audit(case.brand, test_report, case.year)
        actual = verdict["status"]

        return {
            "case": case.description,
            "type": case.hallucination_type,
            "expected": case.expected_verdict,
            "actual": actual,
            "correct": actual == case.expected_verdict,
            "reasoning": verdict["reasoning"],
            "issues": verdict.get("issues_found", []),
        }

    def run_benchmark(self, test_cases: List[TestCase] = None) -> BenchmarkResult:
        if test_cases is None:
            test_cases = self.load_test_cases()

        tp = fp = tn = fn = 0
        details = []

        for case in test_cases:
            result = self.run_single(case)
            details.append(result)

            expected = result["expected"]
            actual = result["actual"]

            if expected == "FAIL" and actual == "FAIL":
                tp += 1
            elif expected == "PASS" and actual == "FAIL":
                fp += 1
            elif expected == "PASS" and actual == "PASS":
                tn += 1
            elif expected == "FAIL" and actual == "PASS":
                fn += 1

        return BenchmarkResult(
            total=len(test_cases),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            details=details,
        )
