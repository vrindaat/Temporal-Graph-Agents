---
title: Temporal Graph Agent
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44"
app_file: app.py
pinned: false
license: mit
hardware: t4-small
---

# Temporal Graph Agent (TGA)

Neuro-symbolic brand audit system that eliminates temporal hallucinations in LLM-based analysis.

## How It Works

1. **Temporal Knowledge Graph**: Customer reviews indexed by brand, topic, sentiment, and exact date
2. **Historian Agent**: Generates brand health reports from time-filtered graph snapshots
3. **Critic Agent**: Verifies every claim against ground truth — catches hallucinations, wrong sentiment, fabricated stats
4. **Temporal Firewall**: The LLM never sees data outside the queried year window

## Architecture

```
Raw Reviews → [Known-Brand NER + BART-MNLI topics + VADER sentiment] → Temporal Knowledge Graph
                                                                                    ↓
                                                         [Year-range snapshot for brand + year]
                                                                                    ↓
                                                               [Historian Agent → Structured Report]
                                                                                    ↓
                                                               [Critic Agent → PASS/FAIL + Issues]
```

## Citation

Vrinda Anil Thaker, Boston University, 2025.
