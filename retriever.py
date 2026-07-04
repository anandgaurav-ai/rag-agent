import  re
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from config import OPENAI_API_KEY, CHROMA_PATH, COLLECTION_NAME, TOP_K

def get_vector_store() -> Chroma:
    """Connect to existing ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(api_key = OPENAI_API_KEY)
    vector_store = Chroma(
        collection_name = COLLECTION_NAME,
        persist_directory = CHROMA_PATH,
        embedding_function = embeddings
    )
    return vector_store

# ── Semantic search (original) ──────────────────────────────────

def retrieve_semantic(query: str, top_k: int = TOP_K) -> list:
    """Search by meaning using embeddings."""
    vector_store = get_vector_store()

    # similarity_search_with_score returns (document, score) pairs
    results = vector_store.similarity_search_with_score(query, k = top_k)

    return results

# ── Keyword search (BM25) ─────────────────────────────────

def tokenize(text: str) -> list:
        """Lowercase, remove punctuation, split into tokens."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # remove all punctuations
        return text.split()

def retrieve_keyword(query: str, top_k: int = TOP_K) -> list:
    """Search by keyword relevance using BM25."""

    vector_store = get_vector_store()
    all_docs = vector_store.get()

    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    # Tokenize all chunks for BM25
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    # Score the query against all chunks
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Pair documents with scores and sort
    scored = []
    for i, score in enumerate(scores):
        if score > 0:
            doc = Document(
                page_content = documents[i],
                metadata = metadatas[i] if metadatas else {}
            )
            scored.append((doc, score))

    scored.sort(key = lambda x: x[1], reverse = True)
    return scored[:top_k]

# ── Hybrid search────────────────────────────────

def retrieve_hybrid(query: str, top_k: int = TOP_K) -> list:
    """Combine BM25 and semantic search using Reciprocal Rank Fusion."""
    semantic_results = retrieve_semantic(query, top_k=10)
    keyword_results  = retrieve_keyword(query, top_k=10)

    # Build rank maps (content → rank position)
    semantic_ranks = {}
    for rank, (doc, score) in enumerate(semantic_results):
        key = doc.page_content[:80]
        semantic_ranks[key] = {"doc": doc, "rank": rank}

    keyword_ranks = {}
    for rank, (doc, score) in enumerate(keyword_results):
        key = doc.page_content[:80]
        keyword_ranks[key] = {"doc": doc, "rank": rank}

    # Collect all unique chunks
    all_keys = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

    # Calculate RRF score for each chunk
    k = 60  # standard RRF constant
    rrf_scores = []

    for key in all_keys:
        # If chunk wasn't found by a method, use a large rank (low contribution)
        sem_rank = semantic_ranks[key]["rank"] if key in semantic_ranks else 100
        kw_rank  = keyword_ranks[key]["rank"]  if key in keyword_ranks  else 100

        rrf_score = (1 / (k + sem_rank)) + (1 / (k + kw_rank))

        # Get the doc object from whichever method found it
        doc = semantic_ranks[key]["doc"] if key in semantic_ranks else keyword_ranks[key]["doc"]

        rrf_scores.append((doc, rrf_score))

    # Sort by RRF score descending (higher = more relevant)
    rrf_scores.sort(key=lambda x: x[1], reverse=True)

    # Convert to distance format (lower = better) for consistency
    max_rrf = rrf_scores[0][1] if rrf_scores else 1
    results = [
        (doc, 1 - (score / max_rrf))
        for doc, score in rrf_scores[:top_k]
    ]
    
    return results



# ── Main retrieve function ─────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K, method: str = "hybrid") -> list:
    """Search vector store using specified method."""
    if method == "keyword":
        results = retrieve_keyword(query, top_k)
    elif method == "semantic":
        results = retrieve_semantic(query, top_k)
    else:
        results = retrieve_hybrid(query, top_k)

    print(f"\nQuery: '{query}' (method={method})")
    print(f"Retrieved {len(results)} chunks:")

    for i, (doc, score) in enumerate(results):
        print(f"\n  Chunk {i+1} (score: {score:.4f}):")
        print(f"  '{doc.page_content[:100]}...'")

    return results

if __name__ == "__main__":
    retrieve("What is checkpointing in LangGraph?")
    retrieve("What are the common use cases of LangGraph?")
