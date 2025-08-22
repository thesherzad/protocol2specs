# clinical_agent/tools/response_validator.py
"""
Validate LLM JSON output against a jsonschema. If invalid, ask the model to fix it once.
"""

import json
import logging
from typing import Tuple, Any, Dict
from jsonschema import validate, ValidationError
from tools.ollama_client import chat as ollama_chat
from config import OLLAMA_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _make_repair_prompt(invalid_text: str, schema: Dict) -> str:
    return (
        "The previous response is intended to be valid JSON matching this schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "However, it failed validation. Please return only valid JSON matching the schema exactly "
        "and do not include any commentary. Here is the original (possibly invalid) JSON:\n\n"
        f"{invalid_text}\n\n"
        "Please fix it now."
    )


def validate_and_repair(
    output_text: str,
    schema: Dict,
    model: str = OLLAMA_MODEL,
    timeout_sec: int = 30
) -> Tuple[bool, Any]:
    """
    Validate `output_text` as JSON against the given jsonschema `schema`.
    If valid -> return (True, parsed_json)
    If invalid -> ask the model to repair once (via Ollama) and re-validate.
    Returns (bool_valid, parsed_json_or_error_message)
    """
    try:
        parsed = json.loads(output_text)
    except Exception as e:
        logger.warning("Initial JSON parse failed: %s", e)
        parsed = None

    # Try validate if parsed JSON
    if parsed is not None:
        try:
            validate(instance=parsed, schema=schema)
            return True, parsed
        except ValidationError as ve:
            logger.warning("JSON failed validation: %s", ve)

    # If no parsed JSON or invalid, request repair
    logger.info("Requesting model to repair JSON (one attempt).")
    repair_prompt = _make_repair_prompt(output_text, schema)
    messages = [
        {"role": "system", "content": "You are a JSON fixer. Return only valid JSON."},
        {"role": "user", "content": repair_prompt}
    ]
    try:
        repaired = ollama_chat(messages=messages, model=model)
        # ensure we only capture the JSON text
        repaired = repaired.strip()
        parsed2 = json.loads(repaired)
        validate(instance=parsed2, schema=schema)
        return True, parsed2
    except Exception as e:
        logger.exception("Repair attempt failed: %s", e)
        return False, {"error": "validation_and_repair_failed", "detail": str(e)}
