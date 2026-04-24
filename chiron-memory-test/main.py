"""Chiron Memory Smart Chunking Schema — Test Backend.

Two endpoints:
  POST /test  — run baseline vs. schema comparison
  GET  /health — model load status check

No auth, no database, no Redis. All state is in-memory per request.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from model_loader import load_model, get_model, is_model_loaded
from memory.schema import run_schema, _score_to_risk
from inference.prompts import build_baseline_prompt
from inference.runner import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / shutdown ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; no special shutdown needed."""
    logger.info("Starting up — loading model...")
    try:
        await load_model()
        logger.info("Model ready. Server accepting requests.")
    except RuntimeError as exc:
        logger.error("Model failed to load: %s", exc)
        # Allow the server to start anyway so /health can report the failure
    yield


app = FastAPI(
    title="Chiron Memory Schema Test",
    description="Tests the Memory Smart Chunking Schema against a direct baseline call.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / response models ───────────────────────────────────────────────

class Chunk(BaseModel):
    """A single text chunk with its source label."""

    text: str = Field(..., min_length=1, description="The chunk content.")
    source: str = Field(..., min_length=1, description="A label identifying where this chunk came from.")


class TestRequest(BaseModel):
    """Payload for POST /test."""

    question: str = Field(..., min_length=3, description="The question to answer.")
    chunks: List[Chunk] = Field(
        ...,
        min_length=1,
        description="Text chunks to process through the schema.",
    )

    @field_validator("chunks")
    @classmethod
    def at_least_one_chunk(cls, v: List[Chunk]) -> List[Chunk]:
        """Ensure at least one chunk is provided."""
        if not v:
            raise ValueError("At least one chunk is required.")
        return v


class BaselineResult(BaseModel):
    answer: str
    confidence: float
    sources_cited: List[str]
    hallucination_risk: str


class SchemaResult(BaseModel):
    answer: str
    confidence: float
    sources_cited: List[str]
    hallucination_risk: str
    cycles_used: int
    gate_passed: bool
    reasoning_trace: List[Dict[str, Any]]


class TestResponse(BaseModel):
    session_id: str
    question: str
    baseline: BaselineResult
    schema_result: SchemaResult
    chunk_records: List[Dict[str, Any]]
    working_memory_final: Dict[str, Any]
    episodic_memory: List[Dict[str, Any]]
    verdict: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Return server and model load status.

    Use this endpoint to confirm the model is loaded before sending /test queries.
    """
    model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
    if is_model_loaded():
        try:
            client = await get_model()
            model_name = client.model_name
        except RuntimeError:
            pass

    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        model_name=model_name,
    )


@app.post("/test", response_model=TestResponse, tags=["test"])
async def test_schema(body: TestRequest) -> TestResponse:
    """Run a side-by-side comparison: baseline vs. Memory Smart Chunking Schema.

    The baseline feeds the question directly to gpt-oss-20b with no chunks.
    The schema run processes all provided chunks through all five schema layers.

    Returns both results plus the full reasoning trace and verdict.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Check /health and retry.",
        )

    client = await get_model()

    async def inference_fn(prompt: str, is_baseline: bool = False) -> Dict[str, Any]:
        return await run_inference(prompt, client, is_baseline=is_baseline)

    # ── Call A: Baseline (no schema, no chunks) ──────────────────────────────
    logger.info("Running baseline call for question: %r", body.question[:80])
    baseline_prompt = build_baseline_prompt(body.question)
    baseline_output = await run_inference(baseline_prompt, client, is_baseline=True)

    baseline_result = BaselineResult(
        answer=baseline_output.get("answer", ""),
        confidence=float(baseline_output.get("confidence", 0.0)),
        sources_cited=_ensure_list(baseline_output.get("sources_cited")),
        hallucination_risk=_score_to_risk(float(baseline_output.get("confidence", 0.0))),
    )

    # ── Call B: Schema run ───────────────────────────────────────────────────
    logger.info("Running schema pipeline for question: %r", body.question[:80])
    chunks_dicts = [{"text": c.text, "source": c.source} for c in body.chunks]
    schema_output = await run_schema(body.question, chunks_dicts, inference_fn)

    schema_result = SchemaResult(
        answer=schema_output.get("answer", ""),
        confidence=float(schema_output.get("confidence", 0.0)),
        sources_cited=_ensure_list(schema_output.get("sources_cited")),
        hallucination_risk=schema_output.get("hallucination_risk", "high"),
        cycles_used=int(schema_output.get("cycles_used", 0)),
        gate_passed=bool(schema_output.get("gate_passed", False)),
        reasoning_trace=schema_output.get("reasoning_trace", []),
    )

    # ── Verdict ──────────────────────────────────────────────────────────────
    verdict = _calculate_verdict(baseline_result, schema_result)

    return TestResponse(
        session_id=schema_output.get("session_id", ""),
        question=body.question,
        baseline=baseline_result,
        schema_result=schema_result,
        chunk_records=schema_output.get("chunk_records", []),
        working_memory_final=schema_output.get("working_memory_final", {}),
        episodic_memory=schema_output.get("episodic_memory", []),
        verdict=verdict,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _calculate_verdict(baseline: BaselineResult, schema: SchemaResult) -> str:
    """Determine which approach produced the better result.

    Rules (from spec):
    - schema_better: schema confidence > baseline confidence by > 0.1 AND
                     schema has more sources cited.
    - baseline_better: baseline confidence > schema confidence by > 0.1 AND
                       baseline has more sources cited.
    - equal: neither condition met.

    Args:
        baseline: The baseline inference result.
        schema: The schema pipeline result.

    Returns:
        One of 'schema_better', 'baseline_better', or 'equal'.
    """
    conf_diff = schema.confidence - baseline.confidence
    schema_sources = len(schema.sources_cited)
    baseline_sources = len(baseline.sources_cited)

    if conf_diff > 0.1 and schema_sources > baseline_sources:
        return "schema_better"
    if conf_diff < -0.1 and baseline_sources > schema_sources:
        return "baseline_better"
    return "equal"


def _ensure_list(value: Any) -> List[str]:
    """Coerce a value to a list of strings.

    Args:
        value: May be a list, a string, or None.

    Returns:
        A clean list of strings.
    """
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value:
        return [value]
    return []
