import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import (
    OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_PATH, COLLECTION_NAME
)

# ── Step 1: Load documents ─────────────────────────────────────

def load_documents(docs_path: str = "./docs") -> list:
    """Load all .txt files from the docs folder."""
    loader = DirectoryLoader(
        docs_path,
        glob = "**/*.txt", # match all .txt files
        loader_cls = TextLoader # use TextLoader for each file
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {docs_path}")

    return documents

# ── Step 2: Chunk documents ────────────────────────────────────

def chunk_documents(documents: list) -> list:
    """Split documents into smaller chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len,
        separators = ["\n\n", "\n", ". ", " ", ""] # tries to split at paragraph → newline → sentence → word → character
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── Step 3: Embed and store ────────────────────────────────────

def create_vector_store(chunks: list) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(api_key = OPENAI_API_KEY)

    # Delete old store if it exists — fresh index each time
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        print("Deleted old vector store")

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        collection_name = COLLECTION_NAME,
        persist_directory = CHROMA_PATH
    )
    print(f"Created vector store at {CHROMA_PATH} with {len(chunks)} chunks")
    return vector_store


# ── Run full indexing pipeline ─────────────────────────────────

def index_documents(docs_path: str = "./docs") -> Chroma:
    """Full pipeline: load → chunk → embed → store."""
    print("\n--- Indexing Pipeline ---")
    documents    = load_documents(docs_path)
    chunks       = chunk_documents(documents)

    # Inspect a few chunks
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n Chunk {i+1} ({len(chunk.page_content)} chars):")
        print(f" '{chunk.page_content[:100]}...'")

    vector_store = create_vector_store(chunks)
    print("\nIndexing complete ✅")
    return vector_store

if __name__ == "__main__":
    index_documents()