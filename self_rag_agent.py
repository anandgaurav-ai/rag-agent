from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from config import MODEL_NAME, OPENAI_API_KEY
from retriever import retrieve
from tools import grade_relevance

# ── State ──────────────────────────────────────────────────────

class SelfRAGState(TypedDict):
    question: str
    query: str # current search query (can be reformulated)
    chunks_text: str
    verdict: str
    reason: str
    retry_count: int
    answer: str

MAX_RETRIES = 2

# ── LLM ────────────────────────────────────────────────────────

llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)

REFORMULATE_PROMPT = """The following search query did not return relevant results.

Original question: {question}
Failed query: {query}
Reason it failed: {reason}

Generate a NEW, DIFFERENT search query that might find better results.
Try different keywords, synonyms, or a more specific/general phrasing.
Respond with ONLY the new query text, nothing else."""

ANSWER_PROMPT = """Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so honestly.

CONTEXT:
{context}

QUESTION:
{question}"""

# ── Nodes ──────────────────────────────────────────────────────

def retrieve_node(state: SelfRAGState) -> dict:
    """Retrieve chunks for the current query."""
    query = state.get("query") or state["question"]
    print(f"\n  [retrieve] query: '{query}'")

    results = retrieve(query, top_k = 5, method = "hybrid")

    if not results:
        chunks_text = "No results found."
    else:
        chunks_text = "\n\n".join(
            f"[Source {i+1}]\n{doc.page_content}"
            for i, (doc, score) in enumerate(results)
        )

    return {"query": query, "chunks_text": chunks_text}

def grade_node(state: SelfRAGState) -> dict:
    """Grade whether the retrieved chunks are relevant."""
    grade = grade_relevance(state["question"], state["chunks_text"])
    print(f"  [grade] verdict: {grade['verdict']} | reason: {grade['reason']}")
    return {"verdict": grade["verdict"], "reason": grade["reason"]}

def reformulate_node(state: SelfRAGState) -> dict:
    """Generate a new search query when retrieval failed."""
    prompt = REFORMULATE_PROMPT.format(
        question=state["question"],
        query=state["query"],
        reason=state["reason"]
    )
    response  = llm.invoke(prompt)
    new_query = response.content.strip()

    print(f"  [reformulate] new query: '{new_query}'")

    return {
        "query":       new_query,
        "retry_count": state.get("retry_count", 0) + 1
    }

def generate_node(state: SelfRAGState) -> dict:
    """Generate the final answer from retrieved chunks."""
    prompt = ANSWER_PROMPT.format(
        context=state["chunks_text"],
        question=state["question"]
    )
    response = llm.invoke(prompt)
    print(f"\nAnswer: {response.content}")
    return {"answer": response.content}

# ── Routing ────────────────────────────────────────────────────

def should_retry(state: SelfRAGState) -> str:
    """Decide whether to generate, retry retrieval, or give up."""
    if state["verdict"] == "relevant":
        return "generate"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(f"  [routing] max retries reached — generating with best available context")
        return "generate"

    return "reformulate"

# ── Graph ──────────────────────────────────────────────────────

def build_self_rag_graph():
    builder = StateGraph(SelfRAGState)

    builder.add_node("retrieve",    retrieve_node)
    builder.add_node("grade",       grade_node)
    builder.add_node("reformulate", reformulate_node)
    builder.add_node("generate",    generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        should_retry,
        {"generate": "generate", "reformulate": "reformulate"}
    )
    builder.add_edge("reformulate", "retrieve")  # loop back
    builder.add_edge("generate", END)

    return builder.compile()

self_rag_graph = build_self_rag_graph()

# ── Run function ──────────────────────────────────────────────

def ask_self_rag(question: str) -> str:
    """Run one question through the Self-RAG pipeline."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    result = self_rag_graph.invoke({
        "question":    question,
        "retry_count": 0
    })

    return result["answer"]