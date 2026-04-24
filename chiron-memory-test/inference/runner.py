"""Model inference runner with JSON enforcement.

Calls gpt-oss-20b and guarantees structured JSON output through:
  1. Prompt instruction (primary)
  2. Retry up to 3 times with a stricter prompt
  3. Regex extraction fallback from partial output
  4. Degraded-data log and continue

All calls are async to match the FastAPI async event loop.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MAX_JSON_RETRIES = 3
_REQUIRED_FIELDS = {
    "answer", "key_insights", "relevance_score", "hypothesis_changed",
    "hypothesis_update", "summary", "confidence", "sources_cited",
    "open_questions", "potential_hallucinations", "episodic_flag", "episodic_reason",
}
_BASELINE_REQUIRED_FIELDS = {"answer", "confidence", "sources_cited", "potential_hallucinations"}


async def run_inference(
    prompt: str,
    model_client,
    is_baseline: bool = False,
) -> Dict[str, Any]:
    """Call the model and return a validated JSON dict.

    Args:
        prompt: The fully assembled prompt string.
        model_client: The loaded ModelClient instance from model_loader.py.
        is_baseline: If True, use the smaller set of required fields.

    Returns:
        A dict with at minimum the required fields populated.
        Missing or unparseable fields are filled with safe defaults.
    """
    required = _BASELINE_REQUIRED_FIELDS if is_baseline else _REQUIRED_FIELDS

    for attempt in range(1, _MAX_JSON_RETRIES + 1):
        # On retries, append a stricter instruction to the prompt
        final_prompt = prompt if attempt == 1 else _append_strict_reminder(prompt, attempt)

        raw_output = await model_client.generate(final_prompt)

        parsed, error = _try_parse_json(raw_output)
        if parsed is not None:
            return _fill_defaults(parsed, required)

        logger.warning(
            "JSON parse attempt %d/%d failed: %s. Raw output (first 300): %r",
            attempt, _MAX_JSON_RETRIES, error, raw_output[:300],
        )

    # Regex fallback: try to extract a JSON object from anywhere in the output
    raw_output_last = await model_client.generate(prompt)
    extracted = _regex_extract_json(raw_output_last)
    if extracted is not None:
        logger.warning("Used regex extraction fallback for JSON parsing.")
        return _fill_defaults(extracted, required)

    # Complete fallback: return degraded data with the raw text as the answer
    logger.error("All JSON extraction methods failed. Returning degraded data.")
    return _degraded_defaults(raw_output_last, required)


def _try_parse_json(text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Attempt to parse JSON from the model output.

    Strips markdown fences and leading/trailing whitespace before parsing.

    Args:
        text: Raw string output from the model.

    Returns:
        Tuple of (parsed_dict, None) on success or (None, error_message) on failure.
    """
    cleaned = _strip_markdown_fences(text.strip())
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result, None
        return None, f"JSON parsed but not a dict: {type(result)}"
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _regex_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract a JSON object from anywhere in the text using regex.

    Args:
        text: Raw string that may contain JSON embedded in prose.

    Returns:
        Parsed dict if a valid JSON object is found, else None.
    """
    # Find the outermost { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group())
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def _append_strict_reminder(prompt: str, attempt: int) -> str:
    """Append a progressively more insistent JSON-only reminder to the prompt.

    Args:
        prompt: The original prompt string.
        attempt: Current attempt number (2 or 3).

    Returns:
        The prompt with an appended JSON reminder.
    """
    reminder = (
        f"\n\n[RETRY {attempt}] Your previous response was not valid JSON. "
        "You MUST respond with ONLY the JSON object. No prose, no markdown, no explanation. "
        "Start immediately with '{' and end with '}'."
    )
    return prompt + reminder


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from model output.

    Args:
        text: Possibly fence-wrapped JSON string.

    Returns:
        The unwrapped content.
    """
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fill_defaults(data: Dict[str, Any], required: set) -> Dict[str, Any]:
    """Ensure all required fields are present, filling safe defaults where missing.

    Args:
        data: Parsed JSON dict from the model.
        required: Set of required field names.

    Returns:
        The dict with any missing required fields filled.
    """
    defaults: Dict[str, Any] = {
        "answer": "",
        "key_insights": [],
        "relevance_score": 0.0,
        "hypothesis_changed": False,
        "hypothesis_update": "",
        "summary": "",
        "confidence": 0.0,
        "sources_cited": [],
        "open_questions": [],
        "potential_hallucinations": [],
        "episodic_flag": False,
        "episodic_reason": "",
    }
    for field in required:
        if field not in data or data[field] is None:
            data[field] = defaults.get(field, "")
    return data


def _degraded_defaults(raw_text: str, required: set) -> Dict[str, Any]:
    """Return a safe degraded response when all parsing attempts fail.

    Args:
        raw_text: The last raw output from the model.
        required: Set of required field names.

    Returns:
        A dict with safe defaults and the raw text stored in the answer field.
    """
    result = _fill_defaults({}, required)
    result["answer"] = raw_text[:1000] if raw_text else "(model produced no output)"
    result["potential_hallucinations"] = ["JSON parsing failed — treat entire response with caution"]
    return result
