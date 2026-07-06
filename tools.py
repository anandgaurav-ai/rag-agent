from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from retriever import retrieve
from config import MODEL_NAME, OPENAI_API_KEY

grader_llm = ChatOpenAI(model = MODEL_NAME, api_key = OPENAI_API_KEY, temperature = 0)

GRADER_PROMPT = """You are a strict relevance grader.

Given a user question and retrieved document chunks, determine if
the chunks contain information that can answer the question.

QUESTION: {question}

RETRIEVED CHUNKS:
{chunks}

Respond in EXACTLY this format:
VERDICT: [relevant/not_relevant]
REASON: [one sentence explanation]"""

def grade_relevance(question: str, chunks_text: str) -> dict:
    """Grade whether retrieved chunks can answer the question."""
    prompt = GRADER_PROMPT.format(question = question, chunks = chunks_text)
    response = grader_llm.invoke(prompt)
    raw = response.content
    
    verdict = "not_relevant" # safe default
    reason = raw

    try:
        if "VERDICT:" in raw:
            verdict = raw.split("VERDICT:")[1].split("REASON:")[0].strip().lower()
            reason = raw.split("REASON:")[1].strip()
    except Exception:
        pass

    return {"verdict": verdict, "reason": reason}


@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for information about LangGraph,
    its features, pricing, or use cases. Use this when the user asks
    a factual question that might be answered by indexed documents.
    Do NOT use this for greetings, small talk, or general knowledge
    questions unrelated to the documents."""

    results = retrieve(query, top_k = 5, method = "hybrid")

    if not results:
        return "NO relevant document found for this query"
    
    context_parts = []
    for i, (doc, score) in enumerate(results):
        context_parts.append(f"[Source {i+1}]\n{doc.page_content}")

    return "\n\n".join(context_parts)

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use for any arithmetic calculation.
    E.g. '20 * 0.15' or '99 / 12'."""
    try:
        import math
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator Error: {e}"
    
tools = [search_documents, calculator]
TOOL_MAP = {t.name: t for t in tools}