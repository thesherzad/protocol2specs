"""
Wrapper for Ollama local model chat (gemma3:1b).
Provides a simple chat() function that accepts a list of messages (role/content).
"""

import logging
from typing import List, Dict, Optional
import json
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import ollama python package, otherwise fail with clear error
try:
    import ollama
except Exception as e:
    ollama = None
    logger.debug("ollama python package not available; ensure 'pip install ollama' if you want Python API usage.")


def chat(
    messages: List[Dict[str, str]],
    model: str = "gemma3:1b",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout_sec: int = 300,
) -> str:
    """
    Send messages to the local Ollama model and return its text output.

    `messages` should be a list of dicts: [{"role": "system", "content": "..."},
                                           {"role": "user", "content": "..."}]

    Returns the final assistant string (not the raw object).
    """
    if not ollama:
        raise RuntimeError(
            "Ollama python client not installed (pip install ollama). "
            "Alternatively ensure the 'ollama' binary is available and change this code to use subprocess."
        )

    # Ollama's python API accepts messages like OpenAI-style
    try:
        # use the chat endpoint
        response = ollama.chat(model=model, messages=messages,
            options={
                'temperature': temperature
            }
        )
        # response format may vary; attempt to extract text robustly
        if isinstance(response, dict):
            # some versions return {'message': {'role':..., 'content': '...'}}
            if "message" in response and isinstance(response["message"], dict):
                return response["message"].get("content", "")
            # or {'content': '...'}
            return response.get("content", "") or str(response)
        # fallback: str conversion
        return str(response)
    except Exception as exc:
        logger.exception("Ollama chat call failed; error: %s", exc)
        raise
