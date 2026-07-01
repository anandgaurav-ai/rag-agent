import os
import sys

os.system("chcp 65001 > nul")
sys.stdout.reconfigure(encoding="utf-8")

from indexer import index_documents
from rag_chain import ask

def main():
    print("="*50)
    print("RAG Pipeline — Document Q&A")
    print("="*50)

    # Step 1 — Index documents (only needed once, or when docs change)
    index_documents()

    # Step 2 — Ask questions
    questions = [
        "What is checkpointing in LangGraph?",
        "How much does LangGraph cost?",
        "How does human-in-the-loop work in LangGraph?",
        "What are the common use cases of LangGraph?",
        "How does LangGraph handle multi-agent workflows?",
    ]

    for q in questions:
        ask(q)

if __name__ == "__main__":
    main()