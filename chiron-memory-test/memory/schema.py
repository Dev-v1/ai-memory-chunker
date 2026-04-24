"""Memory Smart Chunking Schema orchestrator.

Coordinates all five layers across the full processing cycle:
  Layer 1: Mission Anchor (created once, never modified)
  Layer 2: Working Memory (updated each cycle)
  Layer 3: Chunk Processing Record (audit trail)
  Layer 4: Episodic Memory (high-relevance recall)
  Layer 5: Synthesis Gate (four-condition pass check)
"""

import logging
from typing import List, Dict, Any, Optional

from memory.mission_anchor import MissionAnchor, create_mission_anchor
from memory.working_memory import WorkingMemory, create_working_memory, update_working_memory
from memory.chunk_record import ChunkRecord, create_chunk_record
from memory.episodic_memory import (
    EpisodicEntry,
    maybe_store_episodic,
    recall_relevant_entries,
    format_recalled_entries,
)
from memory.synthesis_gate import GateResult, check_synthesis_gate

logger = logging.getLogger(__name__)

_MAX_CYCLES = 7
_SYNTHESIS_RETRY_ALLOWED = True


class SchemaSession:
    """Holds all mutable state for a single schema processing session."""

    def __init__(self, query: str) -> None:
        self.anchor: MissionAnchor = create_mission_anchor(query)
        self.working_memory: WorkingMemory = create_working_memory()
        self.chunk_records: List[ChunkRecord] = []
        self.episodic_store: List[EpisodicEntry] = []
        self.gate_result: Optional[GateResult] = None
        self.final_answer: str = ""
        self.reasoning_trace: List[Dict[str, Any]] = []


async def run_schema(
    query: str,
    chunks: List[Dict[str, str]],
    run_inference,
) -> Dict[str, Any]:
    """Execute the full Memory Smart Chunking Schema pipeline.

    Args:
        query: The original question.
        chunks: List of dicts with 'text' and 'source' keys.
        run_inference: Async callable (prompt: str) -> Dict[str, Any]
                       that calls the model and returns parsed JSON output.

    Returns:
        A dict containing the final answer, reasoning trace, working memory,
        episodic memory, chunk records, and gate result.
    """
    session = SchemaSession(query)
    chunks_to_process = list(chunks)
    chunk_index = 0
    retry_used = False

    while chunk_index < len(chunks_to_process) and chunk_index < _MAX_CYCLES:
        chunk = chunks_to_process[chunk_index]
        cycle_number = chunk_index + 1
        chunk_text: str = chunk.get("text", "")
        chunk_source: str = chunk.get("source", f"chunk_{cycle_number}")

        logger.info("Processing cycle %d — source: %s", cycle_number, chunk_source)

        # Layer 4: recall any episodic entries triggered by this chunk
        recalled = recall_relevant_entries(session.episodic_store, chunk_text)
        recalled_block = format_recalled_entries(recalled)

        # Build the full prompt for this cycle
        from inference.prompts import build_chunk_prompt
        prompt = build_chunk_prompt(
            anchor=session.anchor,
            working_memory=session.working_memory,
            chunk_text=chunk_text,
            chunk_source=chunk_source,
            recalled_block=recalled_block,
        )

        # Run inference (includes JSON enforcement + retry logic internally)
        model_output = await run_inference(prompt)

        # Layer 3: write chunk record
        record = create_chunk_record(cycle_number, chunk_source, chunk_text, model_output)
        session.chunk_records.append(record)

        # Layer 2: update working memory
        update_working_memory(session.working_memory, model_output, chunk_source, cycle_number)

        # Layer 4: maybe store this chunk in episodic memory
        maybe_store_episodic(
            session.episodic_store, chunk_text, chunk_source, model_output
        )

        # Reasoning trace entry
        session.reasoning_trace.append({
            "cycle": cycle_number,
            "insight": model_output.get("summary", ""),
            "source": chunk_source,
            "confidence": float(model_output.get("confidence", 0.0)),
        })

        # Layer 5: check synthesis gate
        gate = check_synthesis_gate(session.working_memory, query)
        session.gate_result = gate

        if gate.passed:
            logger.info("Synthesis gate PASSED at cycle %d", cycle_number)
            break

        chunk_index += 1

        # If all chunks exhausted and gate hasn't passed, attempt one refined retry
        if chunk_index >= len(chunks_to_process) and not retry_used and not gate.passed:
            logger.info("Gate not passed after all chunks. Attempting one refinement retry.")
            retry_used = True
            # Re-queue unresolved chunks from episodic memory as a second pass
            if session.episodic_store:
                retry_chunks = [
                    {"text": e.summary, "source": f"{e.source_chunk}__retry"}
                    for e in session.episodic_store
                ]
                chunks_to_process.extend(retry_chunks)

    # Generate final synthesised answer
    session.final_answer = await _generate_final_answer(session, run_inference)

    return _build_result(session)


async def _generate_final_answer(session: SchemaSession, run_inference) -> str:
    """Synthesise the final answer from working memory.

    Args:
        session: The current SchemaSession with populated working memory.
        run_inference: Async callable for model inference.

    Returns:
        The final answer string.
    """
    from inference.prompts import build_synthesis_prompt
    prompt = build_synthesis_prompt(
        anchor=session.anchor,
        working_memory=session.working_memory,
        gate_passed=session.gate_result.passed if session.gate_result else False,
    )
    output = await run_inference(prompt)
    answer = output.get("answer", "").strip()

    if not answer:
        # Fallback: use the current hypothesis directly
        answer = session.working_memory.hypothesis or "(No answer could be generated.)"

    return answer


def _build_result(session: SchemaSession) -> Dict[str, Any]:
    """Package all session data into the API response payload.

    Args:
        session: The fully processed SchemaSession.

    Returns:
        Dict ready for JSON serialisation.
    """
    gate_passed = session.gate_result.passed if session.gate_result else False

    return {
        "answer": session.final_answer,
        "confidence": session.working_memory.confidence_score,
        "sources_cited": list({r.chunk_source for r in session.chunk_records if r.status != "invalid"}),
        "hallucination_risk": _score_to_risk(session.working_memory.confidence_score),
        "cycles_used": session.working_memory.cycles_completed,
        "gate_passed": gate_passed,
        "reasoning_trace": session.reasoning_trace,
        "chunk_records": [r.to_dict() for r in session.chunk_records],
        "working_memory_final": session.working_memory.to_dict(),
        "episodic_memory": [e.to_dict() for e in session.episodic_store],
        "session_id": session.anchor.session_id,
    }


def _score_to_risk(score: float) -> str:
    """Map a confidence score to a human-readable hallucination risk label.

    Args:
        score: Float confidence in [0, 1].

    Returns:
        'low', 'medium', or 'high'.
    """
    if score >= 0.75:
        return "low"
    if score >= 0.50:
        return "medium"
    return "high"
