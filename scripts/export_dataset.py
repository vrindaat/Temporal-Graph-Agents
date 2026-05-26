"""Export the temporal knowledge graph to HuggingFace Dataset format.

Usage:
    python scripts/export_dataset.py --input thesis_graph.pkl --push-to-hub username/tga-dataset
"""
import argparse
import pickle
import pandas as pd
from datasets import Dataset, DatasetDict


def export(pkl_path, push_to_hub=None, output_dir="./data/hf_dataset"):
    with open(pkl_path, "rb") as f:
        graph = pickle.load(f)

    nodes = [
        {"id": n, "type": d.get("type", ""), "text": d.get("text", "")}
        for n, d in graph.nodes(data=True)
    ]
    edges = [
        {
            "source": u, "target": v,
            "topic": d.get("topic", ""),
            "sentiment": d.get("sentiment", ""),
            "date": str(d.get("start_date", "")),
        }
        for u, v, d in graph.edges(data=True)
    ]

    ds = DatasetDict({
        "nodes": Dataset.from_pandas(pd.DataFrame(nodes)),
        "edges": Dataset.from_pandas(pd.DataFrame(edges)),
    })

    ds.save_to_disk(output_dir)
    print(f"Saved: {len(nodes)} nodes, {len(edges)} edges → {output_dir}")

    if push_to_hub:
        ds.push_to_hub(push_to_hub)
        print(f"Pushed to HuggingFace Hub: {push_to_hub}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="thesis_graph.pkl")
    p.add_argument("--push-to-hub", default=None, help="HF Hub repo (e.g., username/tga-dataset)")
    p.add_argument("--output-dir", default="./data/hf_dataset")
    args = p.parse_args()
    export(args.input, args.push_to_hub, args.output_dir)
