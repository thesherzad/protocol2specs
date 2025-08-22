import requests
import json
import re

def call_gemma(sys: str, user: str, schema: dict = None) -> dict:
    """Call Ollama (gemma3:1b) and return OpenAI-like response."""
    url = "http://localhost:11434/api/chat"
    data = {
        "model": "gemma3:1b",
        "messages": [{"role": "user", "content": user},
        {"role": "system", "content": sys},
        {"role": "user", "content": "Schema: " + json.dumps(schema)}],
        "stream": False,
    }

    resp = requests.post(url, json=data)
    resp.raise_for_status()
    ollama_json = resp.json()
    content = ollama_json["message"]["content"]
    clean_content = re.sub(r"^```[a-z]*\n|```$", "", content, flags=re.MULTILINE).strip()
    # Wrap into OpenAI-like structure
    return {
        "choices": [
            {
                "message": {
                    "role": ollama_json["message"]["role"],
                    "content": clean_content
                }
            }
        ]
    }
