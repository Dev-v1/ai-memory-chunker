"""Layer 2 of the Memory Smart Chunking Schema.

The evolving brain-state of the system. Updated after every chunk is processed.
Deduplicated by semantic similarity so the fact list stays clean across cycles.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

_DEDUP_THRESHOLD = 0.85


@dataclass
class ConfirmedFact:
    """A single grounded fact extracted from a processed chunk."""

    fact: str
    source: str
    confidence: float
    cycle_number: int

    def to_dict(self) -> dict:
        return {
            "fact": self.fact,
            "source": self.source,
            "confidence": self.confidence,
            "cycle_number": self.cycle_number,
        }


@dataclass
class ContradictionFlag:
    """A detected conflict between two claims."""

    claim_1: str
    claim_2: str
    severity: str          # "low" | "medium" | "high"
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "claim_1": self.claim_1,
            "claim_2": self.claim_2,
            "severity": self.severity,
            "resolved": self.resolved,
        }


@dataclass
class WorkingMemory:
    """Mutable working memory updated after every chunk cycle."""

    confirmed_facts: List[ConfirmedFact] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    hypothesis: str = ""
    confidence_score: float = 0.0
    contradiction_flags: List[ContradictionFlag] = field(default_factory=list)
    cycles_completed: int = 0

    def to_dict(self) -> dict:
        return {
            "confirmed_facts": [f.to_dict() for f in self.confirmed_facts],
            "open_questions": self.open_questions,
            "hypothesis": self.hypothesis,
            "confidence_score": round(self.confidence_score, 4),
            "contradiction_flags": [c.to_dict() for c in self.contradiction_flags],
            "cycles_completed": self.cycles_completed,
        }

    def to_prompt_block(self) -> str:
        """Render current working memory as a compact block for model prompts."""
        facts_text = (
            "\n".join(
                f"  [{i+1}] {f.fact} (source: {f.source}, confidence: {f.confidence:.2f})"
                for i, f in enumerate(self.confirmed_facts)
            )
            or "  (none yet)"
        )
        questions_text = (
            "\n".join(f"  - {q}" for q in self.open_questions) or "  (none)"
        )
        contradictions_text = (
            "\n".join(
                f"  ! [{c.severity.upper()}] '{c.claim_1}' vs '{c.claim_2}' resolved={c.resolved}"
                for c in self.contradiction_flags
            )
            or "  (none)"
        )
        return (
            "=== WORKING MEMORY (current state) ===\n"
            f"Hypothesis       : {self.hypothesis or '(not yet formed)'}\n"
            f"Confidence Score : {self.confidence_score:.2f}\n"
            f"Cycles Completed : {self.cycles_completed}\n"
            f"Confirmed Facts  :\n{facts_text}\n"
            f"Open Questions   :\n{questions_text}\n"
            f"Contradictions   :\n{contradictions_text}\n"
            "======================================="
        )


def create_working_memory() -> WorkingMemory:
    """Initialise a blank WorkingMemory for a new session."""
    return WorkingMemory()


def update_working_memory(
    wm: WorkingMemory,
    model_output: Dict[str, Any],
    chunk_source: str,
    cycle_number: int,
) -> WorkingMemory:
    """Apply a single model output JSON to the working memory.

    Args:
        wm: The current WorkingMemory instance (mutated in place).
        model_output: Parsed JSON dict from the model's structured response.
        chunk_source: Label of the chunk that produced this output.
        cycle_number: Current processing cycle index (1-based).

    Returns:
        The mutated WorkingMemory for convenience chaining.
    """
    chunk_confidence: float = float(model_output.get("confidence", 0.0))

    # --- Confirmed facts ---
    for insight in model_output.get("key_insights", []):
        if not insight or not isinstance(insight, str):
            continue
        new_fact = ConfirmedFact(
            fact=insight.strip(),
            source=chunk_source,
            confidence=chunk_confidence,
            cycle_number=cycle_number,
        )
        if not _is_duplicate_fact(new_fact, wm.confirmed_facts):
            wm.confirmed_facts.append(new_fact)

    # --- Open questions ---
    for question in model_output.get("open_questions", []):
        if question and isinstance(question, str) and question.strip():
            q = question.strip()
            if q not in wm.open_questions:
                wm.open_questions.append(q)

    # --- Hypothesis update ---
    if model_output.get("hypothesis_changed", False):
        new_hyp = model_output.get("hypothesis_update", "").strip()
        if new_hyp:
            wm.hypothesis = new_hyp

    # --- Confidence: weighted average biased toward most-recent ---
    if wm.confidence_score == 0.0:
        wm.confidence_score = chunk_confidence
    else:
        wm.confidence_score = _weighted_average_confidence(
            wm.confidence_score, chunk_confidence, wm.cycles_completed
        )

    # --- Contradiction detection ---
    sources_cited = model_output.get("sources_cited", [])
    hallucinations = model_output.get("potential_hallucinations", [])
    if hallucinations:
        for h in hallucinations:
            if isinstance(h, str) and h.strip():
                flag = ContradictionFlag(
                    claim_1=h.strip(),
                    claim_2="(flagged as potential hallucination by model)",
                    severity="medium",
                    resolved=False,
                )
                wm.contradiction_flags.append(flag)

    wm.cycles_completed += 1
    return wm


def _is_duplicate_fact(
    candidate: ConfirmedFact, existing: List[ConfirmedFact]
) -> bool:
    """Return True if candidate is semantically similar (>= threshold) to any existing fact.

    Uses bag-of-words cosine similarity — fast enough for in-memory test workloads.

    Args:
        candidate: The new fact being considered.
        existing: The current list of confirmed facts.

    Returns:
        True if the fact should be skipped as a near-duplicate.
    """
    for fact in existing:
        if _cosine_similarity(candidate.fact, fact.fact) >= _DEDUP_THRESHOLD:
            return True
    return False


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute word-level cosine similarity between two strings.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Float in [0, 1] representing similarity.
    """
    words_a = _tokenise(text_a)
    words_b = _tokenise(text_b)

    if not words_a or not words_b:
        return 0.0

    vocab = set(words_a) | set(words_b)
    vec_a = {w: words_a.count(w) for w in vocab}
    vec_b = {w: words_b.count(w) for w in vocab}

    dot = sum(vec_a[w] * vec_b[w] for w in vocab)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def _tokenise(text: str) -> List[str]:
    """Lower-case and split text into word tokens, stripping punctuation.

    Args:
        text: Raw input string.

    Returns:
        List of normalised word tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _weighted_average_confidence(
    current: float, new_value: float, completed_cycles: int
) -> float:
    """Blend current confidence with the new chunk's confidence.

    More recent cycles have higher weight so the score tracks improving evidence.

    Args:
        current: The confidence score accumulated so far.
        new_value: The confidence score from the latest chunk.
        completed_cycles: How many cycles have been completed (before this one).

    Returns:
        Updated confidence score in [0, 1].
    """
    # Weight new evidence more heavily as cycle count grows (evidence converges)
    new_weight = min(0.6, 0.3 + 0.05 * completed_cycles)
    old_weight = 1.0 - new_weight
    blended = old_weight * current + new_weight * new_value
    return round(min(1.0, max(0.0, blended)), 4)
