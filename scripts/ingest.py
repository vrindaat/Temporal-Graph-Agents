"""
Phase 1: Data Ingestion & Graph Building.
Usage: python ingest.py [--data-dir ./data/amazon_data] [--output thesis_graph.pkl] [--limit 5000]
"""
import argparse
import sys

from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.utils.loader import ReviewLoader


def run_ingestion(data_dir=None, output=None, limit=None):
    print("=" * 60)
    print("   PHASE 1: DATA INGESTION & GRAPH BUILDING")
    print("=" * 60)

    if limit:
        settings.ingest_limit = limit
    if data_dir:
        settings.data_dir = data_dir

    graph = TemporalGraphEngine()
    loader = ReviewLoader(graph)
    loader.load_directory(settings.data_dir)

    if graph.graph.number_of_edges() == 0:
        print("[ERROR] Graph is empty. Check your data directory.")
        sys.exit(1)

    stats = graph.stats()
    print(f"\n[Success] {stats['total_edges']} edges, {stats['brand_count']} brands")
    print(f"  Brands: {stats['brands'][:10]}")

    output_path = output or settings.graph_path
    graph.save_to_disk(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_ingestion(args.data_dir, args.output, args.limit)
