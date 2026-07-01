from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
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

def retrieve(query: str, top_k: int = TOP_K) -> list:
    """Search vector store for chunks most relevant to the query."""
    vector_store = get_vector_store()

    # similarity_search_with_score returns (document, score) pairs
    results = vector_store.similarity_search_with_score(query, k = top_k)

    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(results)} chunks:")

    for i, (doc, score) in enumerate(results):
        print(f"\n Chunk {i+1} (score: {score: .4f}):")
        print(f" '{doc.page_content[:100]}...'")

    return results

if __name__ == "__main__":
    # Test with different queries
    print("=" * 50)
    retrieve("What is checkpointing in LangGraph?")

    print("\n" + "=" * 50)
    retrieve("How much does LangGraph cost?")

    print("\n" + "=" * 50)
    retrieve("How does human-in-the-loop work?")