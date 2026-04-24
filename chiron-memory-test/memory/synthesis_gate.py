"""Layer 5 of the Memory Smart Chunking Schema.

Four conditions must all pass before the final answer is generated. If any fail,
the system retrieves the next chunk and retries (one retry allowed), then returns
the best available answer with a gate_passed=False flag.
"""

import re
from dataclasses import dataclass
from typing import List

from memory.working_memory import WorkingMemory

_CONFIDENCE_THRESHOLD = 0.70


@dataclass
class GateResult:
    """Result of running the Synthesis Gate check."""

    passed: bool
    confidence_ok: bool
    questions_resolved: bool
    contradictions_resolved: bool
    hypothesis_aligned: bool
    details: str

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "confidence_ok": self.confidence_ok,
            "questions_resolved": self.questions_resolved,
            "contradictions_resolved": self.contradictions_resolved,
            "hypothesis_aligned": self.hypothesis_aligned,
            "details": self.details,
        }


def check_synthesis_gate(
    wm: WorkingMemory,
    original_query: str,
) -> GateResult:
    """Evaluate all four gate conditions against the current working memory.

    Args:
        wm: The current WorkingMemory state.
        original_query: The original question from the MissionAnchor.

    Returns:
        A GateResult describing which conditions passed or failed.
    """
    confidence_ok = _check_confidence(wm.confidence_score)
    questions_resolved = _check_open_questions(wm.open_questions)
    contradictions_resolved = _check_contradictions(wm.contradiction_flags)
    hypothesis_aligned = _check_hypothesis_alignment(wm.hypothesis, original_query)

    passed = (
        confidence_ok
        and questions_resolved
        and contradictions_resolved
        and hypothesis_aligned
    )

    details = _build_details(
        confidence_ok, questions_resolved, contradictions_resolved, hypothesis_aligned, wm
    )

    return GateResult(
        passed=passed,
        confidence_ok=confidence_ok,
        questions_resolved=questions_resolved,
        contradictions_resolved=contradictions_resolved,
        hypothesis_aligned=hypothesis_aligned,
        details=details,
    )


def _check_confidence(score: float) -> bool:
    """Condition 1: confidence score must meet the threshold.

    Args:
        score: Current confidence score from WorkingMemory.

    Returns:
        True if the score meets or exceeds the threshold.
    """
    return score >= _CONFIDENCE_THRESHOLD


def _check_open_questions(questions: List[str]) -> bool:
    """Condition 2: all open questions must be resolved or explicitly marked unresolvable.

    A question is considered resolved if it contains the word 'unresolvable'.

    Args:
        questions: List of open question strings from WorkingMemory.

    Returns:
        True if the list is empty or every item is marked unresolvable.
    """
    if not questions:
        return True
    return all("unresolvable" in q.lower() for q in questions)


def _check_contradictions(flags: list) -> bool:
    """Condition 3: no unresolved contradictions with severity 'high'.

    Args:
        flags: List of ContradictionFlag objects from WorkingMemory.

    Returns:
        True if there are no unresolved high-severity contradictions.
    """
    for flag in flags:
        if flag.severity == "high" and not flag.resolved:
            return False
    return True


def _check_hypothesis_alignment(hypothesis: str, query: str) -> bool:
    """Condition 4: hypothesis must semantically relate to the original query.

    Uses keyword overlap — acceptable for a test build where we're validating
    the schema logic rather than production-quality NLU.

    Args:
        hypothesis: Current hypothesis string from WorkingMemory.
        query: The original query from MissionAnchor.

    Returns:
        True if there is meaningful keyword overlap between hypothesis and query.
    """
    if not hypothesis:
        return False

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "has",
        "have", "had", "in", "on", "at", "to", "for", "of", "and", "or",
        "but", "with", "from", "by", "as", "not", "what", "how", "why",
        "when", "who", "did", "does", "do", "will", "can", "it", "this",
        "that", "these", "those",
    }

    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower())) - stopwords
    hyp_tokens = set(re.findall(r"[a-z0-9]+", hypothesis.lower()))

    if not query_tokens:
        return bool(hyp_tokens)

    overlap = query_tokens & hyp_tokens
    overlap_ratio = len(overlap) / len(query_tokens)
    return overlap_ratio >= 0.25


def _build_details(
    confidence_ok: bool,
    questions_resolved: bool,
    contradictions_resolved: bool,
    hypothesis_aligned: bool,
    wm: WorkingMemory,
) -> str:
    """Build a human-readable summary of the gate evaluation.

    Args:
        confidence_ok: Result of confidence check.
        questions_resolved: Result of open questions check.
        contradictions_resolved: Result of contradiction check.
        hypothesis_aligned: Result of hypothesis alignment check.
        wm: WorkingMemory for additional context in the message.

    Returns:
        A multi-line string describing which checks passed or failed.
    """
    lines = []
    lines.append(
        f"[{'PASS' if confidence_ok else 'FAIL'}] "
        f"Confidence {wm.confidence_score:.2f} >= {_CONFIDENCE_THRESHOLD}"
    )

    unresolved_q = [q for q in wm.open_questions if "unresolvable" not in q.lower()]
    lines.append(
        f"[{'PASS' if questions_resolved else 'FAIL'}] "
        f"Open questions: {len(unresolved_q)} unresolved"
    )

    high_conflicts = [
        f for f in wm.contradiction_flags if f.severity == "high" and not f.resolved
    ]
    lines.append(
        f"[{'PASS' if contradictions_resolved else 'FAIL'}] "
        f"High-severity contradictions: {len(high_conflicts)} unresolved"
    )

    lines.append(
        f"[{'PASS' if hypothesis_aligned else 'FAIL'}] "
        f"Hypothesis alignment with query"
    )

    return "\n".join(lines)
