import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models
EMBED_MODEL = "text-embedding-3-small"   # or -small for cheaper runs
# TRY using gpt-oss from HuggingFace.
GEN_MODEL   = "gpt-4o-mini"

CHAT_BACKEND = os.getenv("CHAT_BACKEND", "ollamaxxx")   # default to ollama for privacy - not using, its output doesn't fit the required schema.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")  # if using HTTP server (optional)

# Retrieval params
TOP_K = 20
FINAL_K = 3
