import gc
import sys
import torch

from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.llm import create_llm_backend
from src.agents.historian import HistorianAgent
from src.agents.critic import CriticAgent


def main():
    print("=" * 60)
    print("   TEMPORAL GRAPH AGENT — Interactive Brand Audit")
    print("=" * 60)

    print("\n[System] Loading graph...")
    graph = TemporalGraphEngine()
    if not graph.load_from_disk(settings.graph_path):
        print("[ERROR] Run 'python ingest.py' first.")
        sys.exit(1)

    print("[System] Loading LLM...")
    llm = create_llm_backend()

    historian = HistorianAgent(llm, graph)
    critic = CriticAgent(llm, graph)

    brands = graph.get_brands()
    print(f"\n[Ready] {len(brands)} brands. Examples: {brands[:10]}")

    while True:
        print("\n" + "=" * 60)
        target = input("\n>> Brand (or 'q'): ").strip()
        if target.lower() == "q":
            break

        matched = next((b for b in brands if target.lower() == b.lower()), None)
        if not matched:
            matches = [b for b in brands if target.lower() in b.lower()]
            print(f"  Not found. Similar: {matches[:5]}" if matches else f"  '{target}' not in graph.")
            continue

        try:
            y1 = int(input(">> Baseline year: ").strip())
            y2 = int(input(">> Comparison year: ").strip())
        except ValueError:
            print("  Invalid year.")
            continue

        for year in [y1, y2]:
            print(f"\n--- {matched} in {year} ---")
            report = historian.conduct_audit(matched, year)
            print(f"\n[Report]:\n{report}")
            verdict = critic.verify_audit(matched, report, year)
            print(f"\n[Verdict]: {verdict['status']} — {verdict['reasoning']}")
            if verdict["issues_found"]:
                for issue in verdict["issues_found"]:
                    print(f"  ! {issue}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
