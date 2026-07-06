from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from config import MODEL_NAME, OPENAI_API_KEY
from tools import tools, TOOL_MAP

# ── State ──────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: list


# ── LLM ────────────────────────────────────────────────────────

llm = ChatOpenAI(model = MODEL_NAME, api_key = OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions
about LangGraph using a knowledge base.

RULES:
- For factual questions about LangGraph (features, pricing, use cases),
  use the search_documents tool to find relevant information
- For math questions, use the calculator tool
- For greetings, small talk, or questions unrelated to LangGraph,
  answer directly WITHOUT using any tool
- Only answer from retrieved context — if the documents don't
  contain the answer, say so honestly
- Be concise and cite what you found
"""

# ── Nodes ──────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """LLM decides whether to retrieve, calculate, or answer directly."""
    
    messages = [SystemMessage(content = SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}

def tools_node(state: AgentState) -> dict:
    """Execute whatever tool the LLM decided to call."""
    last_message = state["messages"][-1]
    
    new_messages = []

    for tool_call in last_message.tool_calls:
        
        name = tool_call["name"]
        args = tool_call["args"]

        print(f" -> {name} ({args})")

        if name not in TOOL_MAP:
            result = f"Error: tool '{name}' not found"
        else:
            result = TOOL_MAP[name].invoke(args)
            result = result if result is not None else "No output."

        tm = ToolMessage(
            content = str(result),
            tool_call_id = tool_call["id"]
        )
        
        new_messages.append(tm)

    
    return {"messages": state["messages"] + new_messages}


# ── Routing ────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route to tools if LLM called one, else finish."""
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END
    return "tools"

# ── Graph ──────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")
    
    return builder.compile()

graph = build_graph()


# ── Run function ──────────────────────────────────────────────

def ask_agent(question: str) -> str:
    """Run one question through the RAG agent."""
    print(f"\n{'='*60}")
    print(f"User: {question}")
    print(f"{'='*60}")

    result = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })

    answer = result["messages"][-1].content
    print(f"\nAssistant: {answer}")
    return answer