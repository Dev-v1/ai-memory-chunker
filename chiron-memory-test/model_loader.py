"""gpt-oss-20b model loader — vLLM on A100 80GB.

Weights are loaded directly into VRAM on startup and stay there for the
lifetime of the container. No disk caching between requests.

gpu_memory_utilization=0.95 gives the model ~76 GB of the 80 GB A100.
The remaining ~4 GB covers CUDA overhead and the KV cache for concurrent
requests. max_model_len=8192 keeps the KV cache footprint small — more than
enough for the schema prompts used in this test backend.
"""

import asyncio
import logging
import os
from typing import Optional

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

_MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
_HF_TOKEN = os.getenv("HF_TOKEN", "")

_model_client: Optional["ModelClient"] = None


class ModelClient:
    """Thin wrapper around the vLLM LLM instance."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        return _MODEL_NAME

    async def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Generate text from a prompt.

        Runs vLLM's synchronous generate() in a thread pool so the FastAPI
        event loop stays unblocked while the GPU works.

        Args:
            prompt: The full input prompt string.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text only (prompt not included).
        """
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=max_new_tokens,
        )
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._llm.generate([prompt], sampling_params),
        )
        return outputs[0].outputs[0].text.strip()


async def load_model() -> "ModelClient":
    """Load gpt-oss-20b into VRAM via vLLM.

    Called once at container startup. Raises RuntimeError immediately if the
    load fails — no silent fallback.

    Returns:
        A ready-to-use ModelClient with the model resident in VRAM.

    Raises:
        RuntimeError: If vLLM cannot load the model.
    """
    global _model_client

    if _model_client is not None:
        return _model_client

    logger.info("Loading %s into VRAM...", _MODEL_NAME)

    env = {"HUGGING_FACE_HUB_TOKEN": _HF_TOKEN} if _HF_TOKEN else {}
    for key, val in env.items():
        os.environ.setdefault(key, val)

    try:
        loop = asyncio.get_event_loop()
        llm = await loop.run_in_executor(
            None,
            lambda: LLM(
                model=_MODEL_NAME,
                tokenizer=_MODEL_NAME,
                dtype="float16",
                gpu_memory_utilization=0.95,
                max_model_len=8192,
                trust_remote_code=False,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {_MODEL_NAME} into VRAM: {exc}. "
            "Verify HF_TOKEN is set and the A100 has sufficient free memory."
        ) from exc

    _model_client = ModelClient(llm)
    logger.info("%s loaded into VRAM and ready.", _MODEL_NAME)
    return _model_client


async def get_model() -> "ModelClient":
    """Return the already-loaded ModelClient.

    Raises:
        RuntimeError: If load_model() has not been called yet.
    """
    if _model_client is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")
    return _model_client


def is_model_loaded() -> bool:
    """Return True if the model is resident in VRAM and ready."""
    return _model_client is not None
