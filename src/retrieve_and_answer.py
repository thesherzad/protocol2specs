from openai import OpenAI
from rapidfuzz import fuzz
import json
import time
from tools.ollama_client import chat as ollama_chat
from tools.response_validator import validate_and_repair
from config import CHAT_BACKEND, OLLAMA_MODEL
from tools.llm_client import call_gemma
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

client = OpenAI()

# Purpose: Makes the query richer for embeddings + LLM
def expand_query(variable, lexicon):
    syns = lexicon.get(variable, []) + lexicon.get(variable.upper(), []) + lexicon.get(variable.lower(), [])
    uniq = list(dict.fromkeys([variable] + syns))
    hints = " | ".join(uniq + ["definition", "units", "controlled terminology"])
    return f"Variable: {variable}\nSynonyms: {', '.join(uniq)}\nTask: Provide the definition and units."

# Purpose: Fix embedding search errors by preferring chunks that explicitly mention the variable
def rerank_keyword_boost(results, variable):
    # simple boost if text contains the variable string strongly
    def score(r):
        base = r["score"]
        kw = max(fuzz.partial_ratio(variable.lower(), r["text"].lower())/100.0, 0)
        return base + 0.1*kw
    return sorted(results, key=score, reverse=True)

# Load ColBERT once at startup
# Load tokenizer + model once
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model = AutoModel.from_pretrained("colbert-ir/colbertv2.0").to(device)


def texts_embedding(texts):
    """Get embeddings from ColBERT (using [CLS] token)."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token (first token) as sentence embedding
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
    return embeddings


def rerank_colbert(results, variable, top_k):
    """
    Rerank results with ColBERT embeddings.
    """
    if not results:
        return results

    texts = [r["text"] for r in results]

    # Encode query and documents
    query_emb = texts_embedding([variable])
    doc_embs = texts_embedding(texts)

    # Compute cosine similarity
    scores = (query_emb @ doc_embs.T).squeeze(0).cpu().tolist()

    # Update scores in results
    for r, score in zip(results, scores):
        r["score"] = float(score)

    # Sort and return top_k
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

def _extract_first_json_object(text: str):
    """
    Find the first complete JSON object in `text` by scanning for balanced braces.
    Handles braces inside JSON strings by tracking string/escape state.
    Returns the JSON substring or None if no complete object found.
    """
    start = text.find("{")
    if start == -1:
        return None

    in_string = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def ask_llm(model, variable, snippets):
    # Minimal, strict instructions
    # TODO make up some example about good answers.
    sys = (
        "You are an expert at creating data specification for clinical dataset development from clinical trials protocols."
        "You may get multiple similar text chunks, go through each one very carefully and craft your final answer. Usually the section header is hintful about the content, if it's provided, you can use that to find the best answer."
        "Your final answer will be used to derive a variable of an analysis dataset."
        "You extract concise, literal definitions of clinical variables from provided protocol snippets. It's import to include the exact numbers in your final response."
        "Use ONLY the snippets. Prefer direct quotes. If not found, indicate not_found. "
        "Always include page numbers in citations."
        # below is only needed for gemma3:1b model
        # "Return a JSON object that conforms exactly to the schema I'll provide."
    )

    # Build the context for the model
    context = "\n\n---\n\n".join([f"[p.{s['page']}] {s['text'][:1200]}" for s in snippets])
    user = (
        f"SNIPPETS:\n{context}\n\n"
        f"QUESTION:\nVariable: {variable}\nReturn a 1–2 sentence definition with units and citations to page numbers."
    )

    # JSON Schema for structured output (strict)
    schema = {
        "name": "variable_definition",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "variable": {"type": "string"},
                "definition": {"type": "string"},
                "units": {"type": "string"},
                "controlled_terms": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer"},
                            "section": {"type": "string"},
                            "quote": {"type": "string"}
                        },
                        "required": ["page", "section", "quote"],
                        "additionalProperties": False
                    }
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
            },
            "required": ["variable", "definition", "citations", "confidence", "units", "controlled_terms"],
            "additionalProperties": False
        }
    }

    # Recommended: increase this if your snippets are long (prevents truncation)
    MAX_TOKENS = 800

    def _call_model():
        if CHAT_BACKEND == "ollama":
            # raw_resp = ollama_chat(
            #     model=OLLAMA_MODEL,
            #     messages=[
            #         {"role": "system", "content": sys},
            #         {"role": "user", "content": user},
            #         {"role": "user", "content": "Schema: " + json.dumps(schema)}
            #     ]
            # )
            content = call_gemma(
                sys,
                user,
                schema
            )
            # print(content)

            # Validate & repair the JSON to match schema (one repair attempt)
            # valid, parsed = validate_and_repair(content, schema)
            # if valid:
            #     return parsed
            # else:
            #     return {"error": "model_failed_to_output_valid_schema", "detail": parsed}
            return content

        else:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": schema
                },
                max_completion_tokens=MAX_TOKENS,
                temperature=0
            )
        
    # 1) initial call
    resp = _call_model()
    
    # defensive access
    content = resp.choices[0].message.content

    # If API already returned a parsed object/dict, return it directly
    if isinstance(content, (dict, list)):
        return content

    # content should be a string — try robust parsing:
    if isinstance(content, str):
        # 1) Try direct JSON parse
        try:
            return json.loads(content)
        except Exception:
            pass

        # 2) Try to extract the first complete JSON object using balanced-brace parser
        js_str = _extract_first_json_object(content)
        if js_str:
            try:
                return json.loads(js_str)
            except Exception:
                # fallthrough to retry recovery
                pass

        # 3) If still not JSON, ask the model to re-output only the JSON (repair step)
        # We'll attempt a couple of short retries
        repair_prompt = (
            "The previous response was not valid JSON or was truncated. "
            "Here is the exact output I received:\n\n"
            f"{content}\n\n"
            "Please OUTPUT ONLY the JSON object that matches the specified schema (no surrounding text, "
            "no markdown, no backticks). If fields are unknown, use empty strings or empty arrays where appropriate. "
            "Return strictly valid JSON only."
        )

        for attempt in range(2):
            try:
                resp2 = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": repair_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": schema
                    },
                    max_completion_tokens=MAX_TOKENS,
                    temperature=0
                )
                content2 = resp2.choices[0].message.content
                if isinstance(content2, (dict, list)):
                    return content2
                # try direct parse
                try:
                    return json.loads(content2)
                except Exception:
                    # try extract first json substring
                    js2 = _extract_first_json_object(content2 if isinstance(content2, str) else "")
                    if js2:
                        try:
                            return json.loads(js2)
                        except Exception:
                            # continue retrying
                            content = content2
                            continue
                    content = content2
                    continue
            except Exception as exc:
                # short backoff and retry
                time.sleep(0.5)
                continue

    # If all attempts fail raise an informative error (include the raw text for debugging)
    raise ValueError(f"Model returned non-JSON content and recovery retries failed. Last raw output:\n{content}")