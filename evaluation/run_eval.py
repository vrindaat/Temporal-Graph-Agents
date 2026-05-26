"""
TGA Evaluation Suite.
Usage: python -m evaluation.run_eval [--benchmark all|hallucination|quality] [--output results/]
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.llm import create_llm_backend
from src.agents.historian import HistorianAgent
from src.agents.critic import CriticAgent
from .hallucination_benchmark import HallucinationBenchmark
from .report_quality import ReportQualityEvaluator
from .performance_tracker import tracker


def run_hallucination_eval(historian, critic, graph) -> dict:
    print("\n--- Hallucination Detection Benchmark ---")
    benchmark = HallucinationBenchmark(graph, historian, critic)
    result = benchmark.run_benchmark()
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  F1:        {result.f1:.4f}")
    for d in result.details:
        icon = "Y" if d["correct"] else "X"
        print(f"    [{icon}] {d['type']:20s} expected={d['expected']} actual={d['actual']} — {d['case']}")
    return result.to_dict()


def run_quality_eval(historian, graph) -> dict:
    print("\n--- Report Quality Evaluation ---")
    evaluator = ReportQualityEvaluator()
    brands = graph.get_brands()[:3]

    results_list = []
    for brand in brands:
        print(f"  Evaluating {brand}...")
        report = historian.conduct_audit(brand, 2023)
        facts = graph.get_snapshot_for_year(brand, 2023)
        if not facts:
            continue
        reference = [f.review_text for f in facts if f.review_text]
        result = evaluator.evaluate(report, reference)
        result["brand"] = brand
        results_list.append(result)
        print(f"    ROUGE-L={result['rougeL_f']}, BERTScore={result['bertscore_f1']}")

    if results_list:
        avg = {}
        for key in results_list[0]:
            if key != "brand" and isinstance(results_list[0][key], float):
                avg[f"avg_{key}"] = round(sum(r[key] for r in results_list) / len(results_list), 4)
        return {"per_brand": results_list, "averages": avg}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="all", choices=["all", "hallucination", "quality"])
    parser.add_argument("--output", default=settings.eval_output_dir)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading graph and LLM...")
    graph = TemporalGraphEngine()
    graph.load_from_disk(settings.graph_path)
    llm = create_llm_backend()
    historian = HistorianAgent(llm, graph)
    critic = CriticAgent(llm, graph)

    results = {"timestamp": datetime.now().isoformat(), "backend": settings.llm_backend}

    if args.benchmark in ("hallucination", "all"):
        results["hallucination"] = run_hallucination_eval(historian, critic, graph)

    if args.benchmark in ("quality", "all"):
        results["quality"] = run_quality_eval(historian, graph)

    results["performance"] = tracker.summary()

    out_path = Path(args.output) / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
