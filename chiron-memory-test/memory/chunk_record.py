"""Layer 3 of the Memory Smart Chunking Schema.

Every processed chunk creates a permanent audit receipt. Enterprise customers can inspect
exactly what information was considered, what was found useful, and what was discarded.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any

_CHUNK_PREVIEW_LEN = 500


@dataclass
class ChunkRecord:
    """Immutable audit record for a single processed chunk."""

    cycle_number: int
    chunk_source: str
    chunk_text_preview: str          # Truncated to _CHUNK_PREVIEW_LEN
    key_insights: List[str]
    relevance_score: float
    hypothesis_changed: bool
    status: str                      # "valid" | "invalid" | "flagged"
    processed_at: str

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "chunk_source": self.chunk_source,
            "chunk_text_preview": self.chunk_text_preview,
            "key_insights": self.key_insights,
            "relevance_score": round(self.relevance_score, 4),
            "hypothesis_changed": self.hypothesis_changed,
            "status": self.status,
            "processed_at": self.processed_at,
        }


def create_chunk_record(
    cycle_number: int,
    chunk_source: str,
    chunk_text: str,
    model_output: Dict[str, Any],
) -> ChunkRecord:
    """Build a ChunkRecord from raw chunk data and the model's structured output.

    Args:
        cycle_number: 1-based index of the current processing cycle.
        chunk_source: Source label attached to this chunk.
        chunk_text: The full raw text of the chunk.
        model_output: Parsed JSON dict from the model's structured response.

    Returns:
        A populated ChunkRecord ready to append to the session audit trail.
    """
    relevance: float = float(model_output.get("relevance_score", 0.0))
    hallucinations: List[str] = model_output.get("potential_hallucinations", [])

    if relevance < 0.2:
        status = "invalid"
    elif hallucinations:
        status = "flagged"
    else:
        status = "valid"

    return ChunkRecord(
        cycle_number=cycle_number,
        chunk_source=chunk_source,
        chunk_text_preview=chunk_text[:_CHUNK_PREVIEW_LEN],
        key_insights=_clean_list(model_output.get("key_insights", [])),
        relevance_score=relevance,
        hypothesis_changed=bool(model_output.get("hypothesis_changed", False)),
        status=status,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


def _clean_list(raw: Any) -> List[str]:
    """Normalise a value that should be a list of strings.

    Args:
        raw: The value from model output — may be a list, a string, or None.

    Returns:
        A clean list of non-empty strings.
    """
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if item and str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []
