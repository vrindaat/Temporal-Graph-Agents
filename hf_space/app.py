import os
import gc
import sys
import torch
import gradio as gr

os.environ["HF_HOME"] = "./models_cache"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.pipeline_config import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator

pipeline = None


def load_system():
    global pipeline
    if pipeline is not None:
        return

    # Load configuration (use local config for HF Space)
    config_path = os.environ.get('TGA_CONFIG', 'configs/pipeline_query_only.yaml')
    config = PipelineConfig.from_yaml(config_path)

    # Initialize pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.initialize(skip_ingestion=True)


def get_brands():
    load_system()
    return pipeline.get_available_entities()


def run_audit(brand, year1, year2, progress=gr.Progress()):
    progress(0.1, desc="Loading system...")
    load_system()

    progress(0.25, desc=f"Generating {int(year1)} report...")
    result1 = pipeline.run_audit(brand, int(year1))

    progress(0.6, desc=f"Generating {int(year2)} report...")
    result2 = pipeline.run_audit(brand, int(year2))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    verdict1 = f"**{result1['status']}** — {result1['verification']['reasoning']}"
    verdict2 = f"**{result2['status']}** — {result2['verification']['reasoning']}"

    return result1['report'], verdict1, result2['report'], verdict2


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
