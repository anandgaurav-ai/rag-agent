from langchain_openai import ChatOpenAI
from config import MODEL_NAME, OPENAI_API_KEY
from retriever import retrieve

llm = ChatOpenAI(model = MODEL_NAME, api_key = OPENAI_API_KEY)

RAG_PROMPT = """You are a helpful assistant that answers questions 
based ONLY on the provided context.

RULES:
- Answer ONLY from the context below — do not use your own knowledge
- If the context doesn't contain enough information, say
  "I don't have enough information to answer this."
- Be concise and specific
- Quote relevant parts of the context when possible

CONTEXT:
{context}

QUESTION:
{question}"""

def format_chunks(results: list) -> str:
    """Convert retrieved chunks into a single context string."""
    context_parts = []

    for i, (doc, score) in enumerate(results):
        context_parts.append(f"[Chunk {i+1}]\n{doc.page_content}")
    return "\n\n".join(context_parts)

def ask(question: str) -> str:
    """Full RAG pipeline: retrieve → format → generate."""
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")

    # Step 1 — Retrieve relevant chunks
    results = retrieve(question)

    # Step 2 — Format chunks into context string
    context = format_chunks(results)

    # Step 3 — Build prompt with context + question
    prompt = RAG_PROMPT.format(
        context = context,
        question = question
    )

    # Step 4 — Generate answer
    response = llm.invoke(prompt)
    answer = response.content

    print(f"\nAnswer: {answer}")
    return answer

if __name__ == "__main__":
    # Test with 3 different questions
    ask("What is checkpointing in LangGraph?")
    ask("How much does LangGraph cost?")
    ask("How does human-in-the-loop work in LangGraph?")