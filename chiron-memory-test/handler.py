"""RunPod Serverless entry point.

Wraps the FastAPI app so it can be invoked as a RunPod serverless job.
The model is loaded once at module-level (container startup) and reused
for every subsequent request, giving near-zero cold-start overhead after
the first invocation.

RunPod calls handler(event) for each job. The event shape matches the
rp_schema.json definition.
"""

import asyncio
import json
import logging
import sys

import runpod

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)

# ── Model warm-up at container startup ──────────────────────────────────────
# This runs once when RunPod starts the container. All subsequent invocations
# reuse the already-loaded model, so only the first request pays the load time.

logger.info("Container starting — loading model...")
try:
    from model_loader import load_model, is_model_loaded
    asyncio.get_event_loop().run_until_complete(load_model())
    logger.info("Model loaded and ready for inference.")
except Exception as exc:
    logger.error("Model load failed at startup: %s", exc)
    # Let RunPod know via the health check; don't crash the container


# ── Handler ──────────────────────────────────────────────────────────────────

def handler(event: dict) -> dict:
    """RunPod job handler.

    Dispatches to the correct endpoint based on the 'endpoint' field in the
    event input, or defaults to '/test' for backward compatibility.

    Args:
        event: RunPod event dict. Expected shape:
               {
                 "input": {
                   "endpoint": "/test" | "/health",   # optional, defaults to /test
                   "question": "...",                  # required for /test
                   "chunks": [{"text": "...", "source": "..."}]  # required for /test
                 }
               }

    Returns:
        Dict matching the response schema defined in rp_schema.json.
    """
    job_input = event.get("input", {})
    endpoint = job_input.get("endpoint", "/test")

    if endpoint == "/health":
        return _handle_health()
    if endpoint == "/test":
        return _handle_test(job_input)

    return {
        "error": f"Unknown endpoint '{endpoint}'. Supported: /test, /health"
    }


def _handle_health() -> dict:
    """Handle a /health check job.

    Returns:
        Health status dict.
    """
    from model_loader import is_model_loaded
    import os
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "model_name": os.getenv("MODEL_NAME", "openai/gpt-oss-20b"),
    }


def _handle_test(job_input: dict) -> dict:
    """Handle a /test job by running baseline + schema comparison.

    Args:
        job_input: The 'input' section of the RunPod event.

    Returns:
        The full comparison result dict or an error dict.
    """
    question = job_input.get("question", "").strip()
    chunks_raw = job_input.get("chunks", [])

    if not question:
        return {"error": "Missing required field: 'question'"}
    if not chunks_raw or not isinstance(chunks_raw, list):
        return {"error": "Missing required field: 'chunks' (must be a non-empty list)"}

    chunks = []
    for i, c in enumerate(chunks_raw):
        if not isinstance(c, dict):
            return {"error": f"chunks[{i}] must be a dict with 'text' and 'source' keys"}
        text = c.get("text", "").strip()
        source = c.get("source", f"chunk_{i+1}").strip()
        if not text:
            return {"error": f"chunks[{i}].text is empty"}
        chunks.append({"text": text, "source": source})

    try:
        result = asyncio.get_event_loop().run_until_complete(
            _run_test_async(question, chunks)
        )
        return result
    except Exception as exc:
        logger.exception("Error during test run: %s", exc)
        return {"error": str(exc)}


async def _run_test_async(question: str, chunks: list) -> dict:
    """Async core of the test handler — mirrors main.py POST /test logic.

    Args:
        question: The question to answer.
        chunks: List of dicts with 'text' and 'source' keys.

    Returns:
        Full comparison result dict.
    """
    from model_loader import get_model
    from inference.runner import run_inference
    from inference.prompts import build_baseline_prompt
    from memory.schema import run_schema, _score_to_risk

    client = await get_model()

    async def inference_fn(prompt: str, is_baseline: bool = False) -> dict:
        return await run_inference(prompt, client, is_baseline=is_baseline)

    # Baseline call
    baseline_prompt = build_baseline_prompt(question)
    baseline_output = await run_inference(baseline_prompt, client, is_baseline=True)

    baseline = {
        "answer": baseline_output.get("answer", ""),
        "confidence": float(baseline_output.get("confidence", 0.0)),
        "sources_cited": _ensure_list(baseline_output.get("sources_cited")),
        "hallucination_risk": _score_to_risk(float(baseline_output.get("confidence", 0.0))),
    }

    # Schema call
    schema_output = await run_schema(question, chunks, inference_fn)

    schema = {
        "answer": schema_output.get("answer", ""),
        "confidence": float(schema_output.get("confidence", 0.0)),
        "sources_cited": _ensure_list(schema_output.get("sources_cited")),
        "hallucination_risk": schema_output.get("hallucination_risk", "high"),
        "cycles_used": int(schema_output.get("cycles_used", 0)),
        "gate_passed": bool(schema_output.get("gate_passed", False)),
        "reasoning_trace": schema_output.get("reasoning_trace", []),
    }

    # Verdict
    conf_diff = schema["confidence"] - baseline["confidence"]
    if conf_diff > 0.1 and len(schema["sources_cited"]) > len(baseline["sources_cited"]):
        verdict = "schema_better"
    elif conf_diff < -0.1 and len(baseline["sources_cited"]) > len(schema["sources_cited"]):
        verdict = "baseline_better"
    else:
        verdict = "equal"

    return {
        "session_id": schema_output.get("session_id", ""),
        "question": question,
        "baseline": baseline,
        "schema_result": schema,
        "chunk_records": schema_output.get("chunk_records", []),
        "working_memory_final": schema_output.get("working_memory_final", {}),
        "episodic_memory": schema_output.get("episodic_memory", []),
        "verdict": verdict,
    }


def _ensure_list(value) -> list:
    """Coerce a value to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value:
        return [value]
    return []


# ── Entry point ───────────────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
