from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from typing import List


class ReportQualityEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_rouge(self, report: str, reference_facts: List[str]) -> dict:
        reference = " ".join(reference_facts)
        scores = self.rouge.score(reference, report)
        return {
            "rouge1_f": round(scores["rouge1"].fmeasure, 4),
            "rouge2_f": round(scores["rouge2"].fmeasure, 4),
            "rougeL_f": round(scores["rougeL"].fmeasure, 4),
        }

    def compute_bertscore(self, report: str, reference_facts: List[str]) -> dict:
        reference = " ".join(reference_facts)
        P, R, F1 = bert_score_fn([report], [reference], lang="en", verbose=False)
        return {
            "bertscore_precision": round(P.item(), 4),
            "bertscore_recall": round(R.item(), 4),
            "bertscore_f1": round(F1.item(), 4),
        }

    def evaluate(self, report: str, reference_facts: List[str]) -> dict:
        results = {}
        results.update(self.compute_rouge(report, reference_facts))
        results.update(self.compute_bertscore(report, reference_facts))
        return results
