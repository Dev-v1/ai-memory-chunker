"""Layer 4 of the Memory Smart Chunking Schema.

Chunks with high relevance are flagged for potential later recall.
When a subsequent chunk semantically matches a stored retrieval trigger, the
flagged content is re-injected into context automatically.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

_EPISODIC_RELEVANCE_THRESHOLD = 0.8


@dataclass
class EpisodicEntry:
    """A single stored episodic memory entry."""

    source_chunk: str
    summary: str
    reason_flagged: str
    retrieval_trigger: List[str]    # Keywords that will cause recall

    def to_dict(self) -> dict:
        return {
            "source_chunk": self.source_chunk,
            "summary": self.summary,
            "reason_flagged": self.reason_flagged,
            "retrieval_trigger": self.retrieval_trigger,
        }


def maybe_store_episodic(
    episodic_store: List[EpisodicEntry],
    chunk_text: str,
    chunk_source: str,
    model_output: Dict[str, Any],
) -> None:
    """Conditionally store a chunk in episodic memory based on the model's flag.

    Stores when EITHER the model explicitly requested episodic storage OR the
    relevance score exceeds the threshold.

    Args:
        episodic_store: The mutable list of EpisodicEntry objects for this session.
        chunk_text: The raw text of the chunk.
        chunk_source: Source label for the chunk.
        model_output: Parsed JSON dict from the model's structured response.
    """
    relevance: float = float(model_output.get("relevance_score", 0.0))
    should_flag: bool = bool(model_output.get("episodic_flag", False))

    if not should_flag and relevance < _EPISODIC_RELEVANCE_THRESHOLD:
        return

    summary: str = model_output.get("summary", "").strip() or chunk_text[:200]
    reason: str = model_output.get("episodic_reason", "").strip() or "High relevance score."

    # Build retrieval triggers from the key insights and the summary
    trigger_words: List[str] = _extract_trigger_keywords(
        summary, model_output.get("key_insights", [])
    )

    if not trigger_words:
        return

    entry = EpisodicEntry(
        source_chunk=chunk_source,
        summary=summary,
        reason_flagged=reason,
        retrieval_trigger=trigger_words,
    )
    episodic_store.append(entry)


def recall_relevant_entries(
    episodic_store: List[EpisodicEntry],
    current_chunk_text: str,
) -> List[EpisodicEntry]:
    """Find episodic entries whose retrieval triggers match the current chunk.

    Args:
        episodic_store: The mutable list of all stored EpisodicEntry objects.
        current_chunk_text: The raw text of the chunk currently being processed.

    Returns:
        List of EpisodicEntry objects whose triggers fired on the current chunk.
    """
    chunk_tokens = set(_tokenise(current_chunk_text))
    matched: List[EpisodicEntry] = []

    for entry in episodic_store:
        trigger_tokens = set(kw.lower() for kw in entry.retrieval_trigger)
        if trigger_tokens & chunk_tokens:
            matched.append(entry)

    return matched


def format_recalled_entries(recalled: List[EpisodicEntry]) -> str:
    """Render recalled episodic entries as a prompt block.

    Args:
        recalled: List of recalled EpisodicEntry objects.

    Returns:
        Formatted string for injection into the model prompt.
    """
    if not recalled:
        return ""

    lines = ["=== RECALLED EPISODIC MEMORY ==="]
    for i, entry in enumerate(recalled, start=1):
        lines.append(
            f"[{i}] Source: {entry.source_chunk}\n"
            f"    Summary: {entry.summary}\n"
            f"    Why stored: {entry.reason_flagged}"
        )
    lines.append("================================")
    return "\n".join(lines)


def _extract_trigger_keywords(summary: str, insights: Any) -> List[str]:
    """Extract meaningful keywords to use as retrieval triggers.

    Args:
        summary: One-sentence summary of the chunk.
        insights: List of key insight strings from the model output.

    Returns:
        Deduplicated list of lowercase keyword strings.
    """
    combined = summary
    if isinstance(insights, list):
        combined += " " + " ".join(str(i) for i in insights if i)

    tokens = _tokenise(combined)
    stopwords = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "but", "is", "was", "are", "were", "be", "been", "has", "had", "have",
        "it", "its", "this", "that", "with", "from", "by", "as", "not", "no",
        "can", "will", "may", "would", "could", "should", "do", "did", "does",
    }
    keywords = [t for t in tokens if t not in stopwords and len(t) > 3]

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return unique[:15]  # Cap at 15 triggers to keep matching focused


def _tokenise(text: str) -> List[str]:
    """Normalise and split text into word tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())
