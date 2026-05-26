import gc
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from config.settings import settings
from src.graph.engine import TemporalGraphEngine
from src.llm import create_llm_backend
from src.agents.historian import HistorianAgent
from src.agents.critic import CriticAgent

app = FastAPI(title="Temporal Graph Agent", version="1.0.0")

graph_engine: Optional[TemporalGraphEngine] = None
historian: Optional[HistorianAgent] = None
critic: Optional[CriticAgent] = None


@app.on_event("startup")
async def startup():
    global graph_engine, historian, critic
    graph_engine = TemporalGraphEngine()
    graph_engine.load_from_disk(settings.graph_path)
    llm = create_llm_backend()
    historian = HistorianAgent(llm, graph_engine)
    critic = CriticAgent(llm, graph_engine)


class AuditRequest(BaseModel):
    brand: str
    baseline_year: int
    comparison_year: int


class VerdictResponse(BaseModel):
    status: str
    reasoning: str
    issues_found: List[str]


class AuditResponse(BaseModel):
    brand: str
    baseline_report: str
    baseline_verdict: VerdictResponse
    comparison_report: str
    comparison_verdict: VerdictResponse


@app.post("/audit", response_model=AuditResponse)
async def run_audit(req: AuditRequest):
    try:
        r1 = historian.conduct_audit(req.brand, req.baseline_year)
        v1 = critic.verify_audit(req.brand, r1, req.baseline_year)

        r2 = historian.conduct_audit(req.brand, req.comparison_year)
        v2 = critic.verify_audit(req.brand, r2, req.comparison_year)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return AuditResponse(
            brand=req.brand,
            baseline_report=r1,
            baseline_verdict=VerdictResponse(
                status=v1["status"], reasoning=v1["reasoning"], issues_found=v1["issues_found"]
            ),
            comparison_report=r2,
            comparison_verdict=VerdictResponse(
                status=v2["status"], reasoning=v2["reasoning"], issues_found=v2["issues_found"]
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brands")
async def list_brands():
    return {"brands": graph_engine.get_brands()}


@app.get("/health")
async def health():
    stats = graph_engine.stats()
    return {"status": "healthy", **stats}
