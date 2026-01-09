import sys
import os
import torch
import gc
from datetime import datetime

from src.graph.engine import TemporalGraphEngine
from src.llm.wrapper import SCCLlama
from src.agents.historian import HistorianAgent
from src.agents.critic import CriticAgent

def main():
    print("==================================================")
    print("   PHASE 2: NEURO-SYMBOLIC AUDIT (INTERACTIVE)    ")
    print("==================================================")

    # 1. Load the Brain (Llama 3)
    print("\n[System] Loading Llama-3 Model...")
    llm = SCCLlama()

    # 2. Load the Memory (The Graph)
    print("\n[System] Loading Knowledge Graph from Disk...")
    graph = TemporalGraphEngine()
    success = graph.load_from_disk("thesis_graph.pkl")

    if not success:
        print("[Critical] Could not load graph. Did you run 'ingest.py' first?")
        sys.exit()

    # 3. Agents
    available_brands = [n for n, d in graph.graph.nodes(data=True) if d.get('type') == 'Brand']
    print(f"\n[System] Ready! Available Brands: {available_brands[:10]}...")

    historian = HistorianAgent(graph, llm)
    critic = CriticAgent(graph, llm)

    # 4. Interactive Loop
    while True:
        print("\n" + "="*60)
        target_brand = input("\n>> Enter Brand Name (or 'q'): ").strip()
        if target_brand.lower() == 'q': break
        
        # Fuzzy Match
        matched_brand = next((b for b in available_brands if target_brand.lower() == b.lower()), None)
        if not matched_brand:
            print("[Error] Brand not found in graph.")
            continue
            
        print(f"\n[Selected] {matched_brand}")
        
        try:
            y1 = input(">> Baseline Year (e.g., 2016): ").strip()
            y2 = input(">> Comparison Year (e.g., 2018): ").strip()
            d1 = datetime(int(y1), 1, 1)
            d2 = datetime(int(y2), 1, 1)
        except:
            print("[Error] Invalid year.")
            continue

        # 1. Historian generates the draft report
        draft1 = historian.conduct_audit(matched_brand, d1)
        print(f"\n[Historian's Draft {d1.year}]:")
        print(draft1)
        
        # 2. CRITIC VERIFIES THE REPORT <-- ADDED
        verify1 = critic.verify_audit(matched_brand, draft1, d1)
        print(f"\n[Critic's Review {d1.year}]:")
        print(verify1)

        # GPU Cleanup (After the first full audit: Historian + Critic)
        gc.collect()
        torch.cuda.empty_cache()

        # --- AUDIT 2: COMPARISON ANALYSIS (d2) ---
        print(f"\n--- [2/2] Analyzing Comparison ({d2.year}) ---")
        
        # 3. Historian generates the comparison draft
        draft2 = historian.conduct_audit(matched_brand, d2)
        print(f"\n[Historian's Draft {d2.year}]:")
        print(draft2)

        # 4. CRITIC VERIFIES THE REPORT <-- ADDED
        verify2 = critic.verify_audit(matched_brand, draft2, d2)
        print(f"\n[Critic's Review {d2.year}]:")
        print(verify2)

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()