"""Layer 1 of the Memory Smart Chunking Schema.

Created once per session at the start of a test run. Never modified after creation.
Pinned into every model call so the system never loses sight of the original goal.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


_DEFAULT_CONSTRAINTS: List[str] = [
    "Be factual: only state information present in the provided context chunks.",
    "Cite sources for every factual claim using the chunk source label.",
    "Flag potential hallucinations explicitly if you detect uncertainty.",
    "Never invent facts that are not present in the provided context.",
    "Acknowledge when the context is insufficient to fully answer the query.",
]


@dataclass(frozen=True)
class MissionAnchor:
    """Immutable mission anchor created once per session."""

    session_id: str
    original_query: str
    success_criteria: str
    constraints: tuple
    created_at: str

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary for JSON responses."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "success_criteria": self.success_criteria,
            "constraints": list(self.constraints),
            "created_at": self.created_at,
        }

    def to_prompt_block(self) -> str:
        """Render as a formatted block for inclusion in model prompts."""
        constraints_text = "\n".join(f"  - {c}" for c in self.constraints)
        return (
            "=== MISSION ANCHOR (read-only — do not modify) ===\n"
            f"Session ID  : {self.session_id}\n"
            f"Query       : {self.original_query}\n"
            f"Success     : {self.success_criteria}\n"
            f"Constraints :\n{constraints_text}\n"
            f"Created At  : {self.created_at}\n"
            "===================================================="
        )


def create_mission_anchor(query: str) -> MissionAnchor:
    """Create a new MissionAnchor for the given query.

    Args:
        query: The original question from the test run.

    Returns:
        A fully initialised, frozen MissionAnchor.
    """
    return MissionAnchor(
        session_id=str(uuid.uuid4()),
        original_query=query,
        success_criteria=_derive_success_criteria(query),
        constraints=tuple(_DEFAULT_CONSTRAINTS),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _derive_success_criteria(query: str) -> str:
    """Auto-derive success criteria from the query wording.

    Args:
        query: The raw question string.

    Returns:
        A one-sentence success criterion tailored to the query type.
    """
    q = query.lower()

    if any(w in q for w in ("why", "cause", "reason", "lead to", "resulted")):
        return f"Identify and explain the primary causes or reasons that answer: {query}"
    if any(w in q for w in ("how", "process", "mechanism", "work", "operate")):
        return f"Explain the process or mechanism that fully answers: {query}"
    if any(w in q for w in ("what", "define", "describe", "explain")):
        return f"Provide a complete, sourced description that answers: {query}"
    if any(w in q for w in ("when", "date", "year", "timeline", "sequence")):
        return f"Establish a clear timeline with specific dates that answers: {query}"
    if any(w in q for w in ("who", "person", "people", "entity", "company")):
        return f"Identify the relevant people or organisations that answer: {query}"
    if any(w in q for w in ("compare", "difference", "versus", "vs")):
        return f"Provide a structured comparison that fully answers: {query}"

    return f"Provide a complete, accurate, and well-sourced answer to: {query}"
