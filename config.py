from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

# ── Chunking settings ──────────────────────────────────────────
CHUNK_SIZE = 500 # Characters per chunk
CHUNK_OVERLAP = 100

# ── Vector store ───────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# ── Retrieval ──────────────────────────────────────────────────
TOP_K = 3 # number of chunks to retrieve

