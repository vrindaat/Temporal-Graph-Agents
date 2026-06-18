"""
FastAPI service for Temporal Graph Agent.
Uses the new pipeline orchestration system with YAML configuration.
"""
import gc
import os
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from src.config.pipeline_config import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator


# Load configuration
CONFIG_PATH = os.environ.get('TGA_CONFIG', 'configs/pipeline_query_only.yaml')
API_KEY = os.environ.get('TGA_API_KEY', None)  # Optional API key for basic auth

app = FastAPI(
    title="Temporal Graph Agent API",
    version="2.0.0",
    description="Temporal knowledge graph analysis with fact-checking"
)

# Global pipeline instance
pipeline: Optional[PipelineOrchestrator] = None


@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup"""
    global pipeline

    print(f"Loading configuration from {CONFIG_PATH}...")

    if not Path(CONFIG_PATH).exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    config = PipelineConfig.from_yaml(CONFIG_PATH)
    pipeline = PipelineOrchestrator(config)
    pipeline.initialize(skip_ingestion=True)  # API assumes graph already exists

    print("API ready!")


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Simple API key verification (if configured)"""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# Request/Response models
class AuditRequest(BaseModel):
    entity: str = Field(..., description="Entity to analyze (brand, product, etc.)")
    year: int = Field(..., ge=1990, le=2030, description="Year to analyze")

    class Config:
        schema_extra = {
            "example": {
                "entity": "Apple",
                "year": 2021
            }
        }


class VerdictResponse(BaseModel):
    status: str = Field(..., description="PASS, FAIL, or ERROR")
    reasoning: str = Field(..., description="Explanation of verdict")
    issues_found: List[str] = Field(default_factory=list, description="List of issues if status is FAIL")


class AuditResponse(BaseModel):
    entity: str
    year: int
    report: str = Field(..., description="Generated temporal report")
    verification: VerdictResponse

    class Config:
        schema_extra = {
            "example": {
                "entity": "Apple",
                "year": 2021,
                "report": "## Executive Summary\nApple's brand health in 2021...",
                "verification": {
                    "status": "PASS",
                    "reasoning": "Passed all checks",
                    "issues_found": []
                }
            }
        }


class ComparisonRequest(BaseModel):
    entity: str = Field(..., description="Entity to analyze")
    baseline_year: int = Field(..., ge=1990, le=2030)
    comparison_year: int = Field(..., ge=1990, le=2030)

    class Config:
        schema_extra = {
            "example": {
                "entity": "Samsung",
                "baseline_year": 2020,
                "comparison_year": 2023
            }
        }


class ComparisonResponse(BaseModel):
    entity: str
    baseline: AuditResponse
    comparison: AuditResponse


# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    if not pipeline or not pipeline.graph:
        return {"status": "initializing"}

    stats = pipeline.graph.stats()
    return {
        "status": "healthy",
        "entities": stats["brand_count"],
        "edges": stats["total_edges"],
        "config": pipeline.config.project_name
    }


@app.get("/entities")
async def list_entities(api_key: str = Depends(verify_api_key)):
    """Get list of available entities in the knowledge graph"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    entities = pipeline.get_available_entities()
    return {
        "count": len(entities),
        "entities": entities
    }


@app.post("/audit", response_model=AuditResponse)
async def run_audit(req: AuditRequest, api_key: str = Depends(verify_api_key)):
    """
    Generate a temporal audit report for a specific entity and year.

    The report is fact-checked by the Critic agent before being returned.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate entity exists
    entities = pipeline.get_available_entities()
    matched_entity = next((e for e in entities if e.lower() == req.entity.lower()), None)

    if not matched_entity:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{req.entity}' not found. Available: {entities[:10]}"
        )

    try:
        result = pipeline.run_audit(matched_entity, req.year)

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return AuditResponse(
            entity=result['entity'],
            year=result['year'],
            report=result['report'],
            verification=VerdictResponse(
                status=result['verification']['status'],
                reasoning=result['verification']['reasoning'],
                issues_found=result['verification']['issues_found']
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


@app.post("/compare", response_model=ComparisonResponse)
async def compare_years(req: ComparisonRequest, api_key: str = Depends(verify_api_key)):
    """
    Compare an entity across two different years.

    Generates two reports (baseline and comparison) with fact-checking.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate entity
    entities = pipeline.get_available_entities()
    matched_entity = next((e for e in entities if e.lower() == req.entity.lower()), None)

    if not matched_entity:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{req.entity}' not found"
        )

    try:
        # Run both audits
        baseline_result = pipeline.run_audit(matched_entity, req.baseline_year)
        comparison_result = pipeline.run_audit(matched_entity, req.comparison_year)

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ComparisonResponse(
            entity=matched_entity,
            baseline=AuditResponse(
                entity=baseline_result['entity'],
                year=baseline_result['year'],
                report=baseline_result['report'],
                verification=VerdictResponse(**baseline_result['verification'])
            ),
            comparison=AuditResponse(
                entity=comparison_result['entity'],
                year=comparison_result['year'],
                report=comparison_result['report'],
                verification=VerdictResponse(**comparison_result['verification'])
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/config")
async def get_config(api_key: str = Depends(verify_api_key)):
    """Get current pipeline configuration (non-sensitive parts)"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return {
        "project_name": pipeline.config.project_name,
        "description": pipeline.config.description,
        "historian_type": pipeline.config.historian.type,
        "critic_type": pipeline.config.critic.type,
        "date_range": pipeline.config.date_range,
    }


# Documentation
@app.get("/")
async def root():
    """API documentation and usage information"""
    return {
        "name": "Temporal Graph Agent API",
        "version": "2.0.0",
        "description": "Temporal knowledge graph analysis with fact-checking",
        "endpoints": {
            "GET /health": "Health check and stats",
            "GET /entities": "List available entities",
            "POST /audit": "Generate temporal audit report",
            "POST /compare": "Compare entity across two years",
            "GET /config": "View pipeline configuration"
        },
        "authentication": "Set TGA_API_KEY environment variable to enable API key auth",
        "configuration": f"Using config: {CONFIG_PATH}",
        "docs": "/docs (Swagger UI)"
    }
