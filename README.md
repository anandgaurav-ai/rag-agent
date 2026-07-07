# RAG Agent — Advanced Retrieval-Augmented Generation

A production-style RAG system progressing from a plain pipeline to an agentic, self-checking retriever. Handles text and PDF documents including tables.

## Features
- Hybrid search combining semantic (embeddings) and keyword (BM25) retrieval
- Reciprocal Rank Fusion (RRF) for merging ranked results
- Section-aware chunking that respects document structure
- PDF handling with table extraction and structure preservation
- Agentic retrieval — the agent decides when to search vs answer directly
- Self-RAG — grades retrieval quality and reformulates queries on failure
- LangSmith tracing for full observability

## Architecture

### Plain RAG pipeline
Documents are loaded, chunked by section, embedded, and stored in ChromaDB. Queries retrieve relevant chunks via hybrid search, then an LLM generates a grounded answer.

### Agentic RAG
Retrieval becomes a tool the agent can choose to call. Greetings and questions with provided context skip retrieval entirely; factual questions trigger a document search; math triggers a calculator.

### Self-RAG
After retrieval, a separate grading LLM evaluates whether the chunks answer the question. If not, the query is reformulated and retried up to a limit, then fails gracefully with an honest "not found" instead of hallucinating.

## Stack
LangGraph · LangChain · OpenAI GPT-4o · ChromaDB · rank-bm25 · pdfplumber · LangSmith

## Setup
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python indexer.py     # build the vector store
python main.py        # run the agent

## Project Structure
- config.py            — settings, chunk size, top-k
- indexer.py           — load, chunk, embed documents (txt + pdf)
- retriever.py         — semantic, keyword (BM25), and hybrid (RRF) search
- rag_chain.py         — plain RAG pipeline
- tools.py             — retrieval + calculator tools, relevance grader
- agent.py             — agentic RAG with tool-based retrieval
- self_rag_agent.py    — Self-RAG with grading and query reformulation
- main.py              — entry point
- docs/                — sample documents (txt + pdf)

## Key Engineering Decisions
- **Hybrid over pure semantic** — semantic search alone missed exact-match queries (e.g. "use cases"); BM25 catches keyword matches embeddings miss.
- **RRF over weighted scoring** — combining rank positions is more robust than tuning score weights, which fixed one query while breaking another.
- **Manual message state** — bypassed the add_messages reducer after a version incompatibility silently converted ToolMessages to HumanMessages.
- **Natural-language table conversion** — PDF tables embed better as "Plan: Developer, Price: $20/month" than as markdown pipes.