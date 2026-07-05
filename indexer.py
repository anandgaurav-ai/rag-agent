import os
import re
import shutil
import pdfplumber
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import (
    OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_PATH, COLLECTION_NAME
)

# ── PDF loader ─────────────────────────────────────────────────

def table_to_natural_language(table: list) -> str:
    """Convert pdfplumber table to natural language sentences."""
    if not table or len(table) < 2:
        return ""
    
    header = table[0]
    rows = table[1:]
    sentences = []
    for row in rows:
        pairs = [f"{h}: {v}" for h, v in zip(header, row)]
        sentences.append(", ".join(pairs))
    return "\n".join(sentences)


def table_to_markdown(table: list) -> str:
    """Convert pdfplumber table to markdown format."""
    if not table:
        return ""
    header = table[0]
    rows   = table[1:]
    md  = "| " + " | ".join(str(h) for h in header) + " |\n"
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
    return md


def load_pdf(filepath: str) -> str:
    """
    Load a PDF file and extract content as Documents.
    Handles text and tables separately.
    Returns a list of Document objects.
    """
    documents = []
    filename = os.path.basename(filepath)


    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # ── Extract tables ─────────────────────────────────
            tables = page.extract_tables()
            table_regions = []  # track bounding boxes of tables

            for table in tables:
                if not table:
                    continue

                # Store in natural language for retrieval
                nl_text = table_to_natural_language(table)

                # Store markdown in metadata for LLM generation
                md_text = table_to_markdown(table)

                if nl_text:
                    documents.append(Document(
                        page_content = nl_text,
                        metadata = {
                            "source":      filename,
                            "page":        page_num + 1,
                            "type":        "table",
                            "markdown":    md_text  # kept for generation
                        }
                    ))

            # ── Extract regular text ───────────────────────────
            text = page.extract_text()
            if text:
                # Remove table content from text to avoid duplication
                # (pdfplumber includes table text in extract_text() too)
                for table in tables:
                    for row in table:
                        for cell in row:
                            if cell:
                                text = text.replace(cell, "")

                # Clean up extra whitespace
                text = re.sub(r'\n+', '\n', text).strip()

                if text:
                    documents.append(Document(
                        page_content = text,
                        metadata = {
                            "source": filename,
                            "page":   page_num + 1,
                            "type":   "text"
                        }
                    ))

    print(f"  Loaded PDF: {filename} — {len(documents)} sections")
    return documents


# ── Text loader ─────────────────────────────────────────────────

def load_text_file(filepath: str) -> list:
    """Load a plain text file."""
    loader = TextLoader(filepath)
    docs = loader.load()

    for doc in docs:
        doc.metadata["type"] = "text"

    print(f"  Loaded TXT: {os.path.basename(filepath)}")
    return docs

# ── Universal document loader ─────────────────────────────────────

def load_documents(docs_path: str = "./docs") -> list:
    """Load all supported files from the docs folder."""
    all_docs = []
    
    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)

        if filename.endswith(".pdf"):
            docs = load_pdf(filepath)
            all_docs.extend(docs)

        elif filename.endswith(".txt"):
            docs = load_text_file(filepath)
            all_docs.extend(docs)

        elif filename.startswith("."):
            continue # skip hidden files

    print(f"\nLoaded {len(all_docs)} sections from {docs_path}")
    return all_docs

# ── Chunking─────────────────

def chunk_documents_by_section(documents: list) -> list:
    """Split documents by logical sections — headings, numbered items, blank lines."""
    all_chunks = []

    for doc in documents:
        # Don't chunk table documents — they're already structured
        if doc.metadata.get("type") == "table":
            all_chunks.append(doc)
            continue

        text = doc.page_content
        source = doc.metadata.get("source", "unknown")

        # Split by numbered items or double newlines
        sections = re.split(r'\n(?=\d+\.)', text)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If section is still too large, split further
            if len(section) > CHUNK_SIZE:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size = CHUNK_SIZE,
                    chunk_overlap = CHUNK_OVERLAP
                )
                sub_chunks = splitter.split_text(section)
                for sc in sub_chunks:
                    all_chunks.append(Document(
                        page_content = sc,
                        metadata = {"Source": source}
                    ))
            else:
                all_chunks.append(Document(
                    page_content = section,
                    metadata = {"Source": source}
                ))
    print(f"[Section] Split into {len(all_chunks)} chunks")
    return all_chunks


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

def index_documents(docs_path: str = "./docs", method: str = "section") -> Chroma:
    """Full pipeline: load → chunk → embed → store."""
    print("\n--- Indexing Pipeline ---")
    documents    = load_documents(docs_path)

    chunks = chunk_documents_by_section(documents)
    
    # Inspect chunks
    print("\nAll chunks:")
    for i, chunk in enumerate(chunks):
        chunk_type = chunk.metadata.get("type", "text")
        preview = chunk.page_content[:80].replace('\n', ' ')
        print(f"\n Chunk {i+1} [{chunk_type}] ({len(chunk.page_content)} chars): '{preview}...'")
        

    vector_store = create_vector_store(chunks)
    print("\nIndexing complete ✅")
    return vector_store

if __name__ == "__main__":
    index_documents()