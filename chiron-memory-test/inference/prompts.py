"""All prompt templates for the Memory Smart Chunking Schema test backend.

Every template produces a string that instructs gpt-oss-20b to return
structured JSON with the required fields. Strict vs. lenient variants are
used during JSON-enforcement retries.
"""

from memory.mission_anchor import MissionAnchor
from memory.working_memory import WorkingMemory

_JSON_FIELD_LIST = """
{
  "answer": "string — your answer based only on this chunk",
  "key_insights": ["string", "..."],
  "relevance_score": 0.0,
  "hypothesis_changed": false,
  "hypothesis_update": "string — new hypothesis if changed, else empty string",
  "summary": "string — one-sentence summary of this chunk",
  "confidence": 0.0,
  "sources_cited": ["string", "..."],
  "open_questions": ["string", "..."],
  "potential_hallucinations": ["string", "..."],
  "episodic_flag": false,
  "episodic_reason": "string — why to store in episodic memory, else empty string"
}
""".strip()

_STRICT_JSON_REMINDER = (
    "CRITICAL: Your entire response must be ONLY the JSON object above. "
    "No explanation, no markdown fences, no text before or after the JSON. "
    "Start your response with '{' and end with '}'. "
    "If you cannot determine a value use null for strings and 0.0 for floats."
)

_SYNTHESIS_JSON_FIELDS = """
{
  "answer": "string — complete, well-sourced final answer to the original query",
  "key_insights": ["string", "..."],
  "confidence": 0.0,
  "sources_cited": ["string", "..."],
  "potential_hallucinations": ["string", "..."]
}
""".strip()

_BASELINE_JSON_FIELDS = """
{
  "answer": "string — your best answer to the question",
  "confidence": 0.0,
  "sources_cited": ["string", "..."],
  "potential_hallucinations": ["string", "..."]
}
""".strip()


def build_chunk_prompt(
    anchor: MissionAnchor,
    working_memory: WorkingMemory,
    chunk_text: str,
    chunk_source: str,
    recalled_block: str = "",
    strict: bool = False,
) -> str:
    """Build the prompt for processing a single chunk during the schema cycle.

    Args:
        anchor: The immutable MissionAnchor for this session.
        working_memory: The current WorkingMemory state.
        chunk_text: The raw text of the chunk to process.
        chunk_source: Source label for the chunk.
        recalled_block: Optional block of recalled episodic entries.
        strict: If True, add a stricter JSON-only reminder (used on retry).

    Returns:
        Fully assembled prompt string.
    """
    parts = [
        anchor.to_prompt_block(),
        "",
        working_memory.to_prompt_block(),
        "",
    ]

    if recalled_block:
        parts += [recalled_block, ""]

    parts += [
        f"=== CURRENT CHUNK (source: {chunk_source}) ===",
        chunk_text.strip(),
        "=" * 48,
        "",
        "INSTRUCTIONS:",
        "1. Analyse the chunk above in the context of the Mission Anchor and Working Memory.",
        "2. Extract key insights that are DIRECTLY stated in the chunk — do not infer beyond it.",
        "3. Update the confidence score to reflect how well this chunk addresses the query.",
        "4. Flag any claims you are uncertain about in potential_hallucinations.",
        "5. If this chunk significantly changes your best hypothesis, set hypothesis_changed=true.",
        "6. If this chunk is likely to be relevant to a future query, set episodic_flag=true.",
        "",
        "Respond ONLY with this exact JSON structure:",
        _JSON_FIELD_LIST,
    ]

    if strict:
        parts += ["", _STRICT_JSON_REMINDER]

    return "\n".join(parts)


def build_synthesis_prompt(
    anchor: MissionAnchor,
    working_memory: WorkingMemory,
    gate_passed: bool,
    strict: bool = False,
) -> str:
    """Build the final synthesis prompt after the processing cycle ends.

    Args:
        anchor: The immutable MissionAnchor for this session.
        working_memory: The fully updated WorkingMemory.
        gate_passed: Whether the Synthesis Gate passed.
        strict: If True, add stricter JSON enforcement reminder.

    Returns:
        Fully assembled synthesis prompt string.
    """
    gate_note = (
        "The Synthesis Gate has PASSED — you have sufficient evidence to answer."
        if gate_passed
        else (
            "WARNING: The Synthesis Gate did NOT fully pass. "
            "Provide your best available answer but be explicit about uncertainties."
        )
    )

    facts_text = "\n".join(
        f"  [{i+1}] {f.fact} (source: {f.source}, confidence: {f.confidence:.2f})"
        for i, f in enumerate(working_memory.confirmed_facts)
    ) or "  (no confirmed facts)"

    parts = [
        anchor.to_prompt_block(),
        "",
        "=== SYNTHESIS INSTRUCTION ===",
        gate_note,
        "",
        f"Original Query: {anchor.original_query}",
        f"Current Hypothesis: {working_memory.hypothesis or '(none)'}",
        f"Overall Confidence: {working_memory.confidence_score:.2f}",
        "",
        "Confirmed Facts:",
        facts_text,
        "",
        "Using ONLY the confirmed facts above, write a complete answer to the original query.",
        "Do not add information beyond what is in the confirmed facts.",
        "Cite your sources inline using (source: <name>) notation.",
        "",
        "Respond ONLY with this exact JSON structure:",
        _SYNTHESIS_JSON_FIELDS,
    ]

    if strict:
        parts += ["", _STRICT_JSON_REMINDER]

    return "\n".join(parts)


def build_baseline_prompt(question: str, strict: bool = False) -> str:
    """Build the prompt for the baseline (no-schema) inference call.

    Args:
        question: The raw question to answer.
        strict: If True, add stricter JSON enforcement reminder.

    Returns:
        Fully assembled baseline prompt string.
    """
    parts = [
        "You are a helpful assistant. Answer the following question to the best of your ability.",
        "Be honest about uncertainty. Cite any sources you draw on.",
        "",
        f"Question: {question}",
        "",
        "Respond ONLY with this exact JSON structure:",
        _BASELINE_JSON_FIELDS,
    ]

    if strict:
        parts += ["", _STRICT_JSON_REMINDER]

    return "\n".join(parts)
