"""Central configuration. Everything is overridable via environment variables or .env."""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# --- Storage ---
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
BASE_COLLECTION = os.getenv("COLLECTION_NAME", "knowledge_base")

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
_DEFAULT_LLM = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_LLM.get(LLM_PROVIDER, "claude-sonnet-5-5"))

# --- Embeddings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
_DEFAULT_EMB = {"huggingface": "BAAI/bge-small-en-v1.5", "openai": "text-embedding-3-small"}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMB.get(EMBEDDING_PROVIDER, "BAAI/bge-small-en-v1.5"))

# The collection name includes the embedding model, so switching models never
# mixes vectors of different dimensions: you simply get a fresh index.
_slug = re.sub(r"[^a-zA-Z0-9]+", "-", EMBEDDING_MODEL).strip("-")[:40]
COLLECTION_NAME = f"{BASE_COLLECTION}_{_slug}"

# --- Chunking & retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.0"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
