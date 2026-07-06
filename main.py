import os
import sys

os.system("chcp 65001 > nul")
sys.stdout.reconfigure(encoding="utf-8")

from indexer import index_documents
from agent import ask_agent

def main():
    print("="*60)
    print("RAG Agent — Document Q&A with Agentic Retrieval")
    print("="*60)

    # Index documents (run once, or when docs change)
    index_documents()

    # Test with a MIX of questions — some need retrieval, some don't
    questions = [
        "Hi there!",                                          # NO retrieval expected
        "Thanks, that's helpful!",                             # NO retrieval expected
        "What is checkpointing in LangGraph?",                 # retrieval expected
        "How much does the Developer plan cost?",              # retrieval expected
        "What is 20% of 99?",                                  # calculator expected
        "If the Team plan costs $99/month, what's the annual cost?",  # retrieval + calculator
    ]

    for q in questions:
        ask_agent(q)

if __name__ == "__main__":
    main()