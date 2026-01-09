# ingest.py
import sys
import os
import pickle # <-- CRITICAL: Ensure pickle is imported here
from src.graph.engine import TemporalGraphEngine
# Explicitly import all schema components the graph relies on:
from src.graph.schema import Node, TemporalEdge, MarketingTopic, Sentiment 
from src.utils.loader import UnsupervisedLoader
from src.llm.wrapper import SCCLlama # We don't use it, but keeping the import path clean helps

def run_ingestion():
    print("==================================================")
    print("   PHASE 1: DATA INGESTION & GRAPH BUILDING       ")
    print("==================================================")

    # 1. Initialize ONLY the Graph (No Llama needed here)
    graph = TemporalGraphEngine()
    
    # Pass 'None' for LLM because we aren't generating text yet
    # We must pass None, not a dummy object, as the loader expects SCCLlama() or None
    loader = UnsupervisedLoader(graph, llm_engine=None)

    # 2. Load Data
    data_folder = "/projectnb/cs599x1/students/akhilg/directed_study_v/brand_audit/data/amazon_data" # Use your path
    loader.load_directory(data_folder)

    # 3. Check & Save
    if graph.graph.number_of_edges() == 0:
        print("[Error] Graph is empty. Ingestion failed.")
        sys.exit()

    print(f"\n[System] Ingestion Success! Total Facts: {graph.graph.number_of_edges()}")
    
    # Save to disk
    graph.save_to_disk("thesis_graph.pkl")
    print("[System] Graph saved to 'thesis_graph.pkl'. You can now run main.py.")

if __name__ == "__main__":
    run_ingestion()