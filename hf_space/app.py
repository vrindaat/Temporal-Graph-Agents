import os
import gc
import sys
import torch
import gradio as gr

os.environ["HF_HOME"] = "./models_cache"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.engine import TemporalGraphEngine
from src.llm.local_backend import LocalHFBackend
from src.agents.historian import HistorianAgent
from src.agents.critic import CriticAgent

graph_engine = None
historian = None
critic = None


def load_system():
    global graph_engine, historian, critic
    if graph_engine is not None:
        return
    graph_engine = TemporalGraphEngine()
    graph_engine.load_from_disk("thesis_graph.pkl")
    llm = LocalHFBackend()
    historian = HistorianAgent(llm, graph_engine)
    critic = CriticAgent(llm, graph_engine)


def get_brands():
    load_system()
    return graph_engine.get_brands()


def run_audit(brand, year1, year2, progress=gr.Progress()):
    progress(0.1, desc="Loading system...")
    load_system()

    progress(0.25, desc=f"Generating {int(year1)} report...")
    r1 = historian.conduct_audit(brand, int(year1))

    progress(0.4, desc=f"Verifying {int(year1)} report...")
    v1 = critic.verify_audit(brand, r1, int(year1))

    progress(0.6, desc=f"Generating {int(year2)} report...")
    r2 = historian.conduct_audit(brand, int(year2))

    progress(0.8, desc=f"Verifying {int(year2)} report...")
    v2 = critic.verify_audit(brand, r2, int(year2))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    verdict1 = f"**{v1['status']}** — {v1['reasoning']}"
    verdict2 = f"**{v2['status']}** — {v2['reasoning']}"

    return r1, verdict1, r2, verdict2


with gr.Blocks(title="Temporal Graph Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Temporal Graph Agent — Brand Auditor
    A neuro-symbolic system that eliminates temporal hallucinations in LLM-based brand audits.
    Select a brand and two years to compare how public sentiment evolved over time.
    """)

    with gr.Row():
        brand_input = gr.Dropdown(choices=get_brands(), label="Brand")
        y1_input = gr.Slider(2013, 2023, value=2020, step=1, label="Baseline Year")
        y2_input = gr.Slider(2013, 2023, value=2023, step=1, label="Comparison Year")

    run_btn = gr.Button("Run Audit", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline Report")
            out_r1 = gr.Markdown()
            gr.Markdown("### Verdict")
            out_v1 = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Comparison Report")
            out_r2 = gr.Markdown()
            gr.Markdown("### Verdict")
            out_v2 = gr.Markdown()

    run_btn.click(run_audit, [brand_input, y1_input, y2_input], [out_r1, out_v1, out_r2, out_v2])

if __name__ == "__main__":
    demo.launch()
