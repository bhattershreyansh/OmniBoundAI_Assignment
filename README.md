# OmniBoundAI Assignment

Backend take-home project for a multi-tenant Brand Q&A RAG agent.

## What It Does

- Ingests brand-related documents from raw text or URLs.
- Chunks and embeds documents into Pinecone under tenant-specific namespaces.
- Answers questions with a LangGraph RAG workflow.
- Grades retrieved chunks before answering.
- Returns citations and a trace for debugging.
- Enforces tenant isolation between customers.

## Run

```bash
make run
```

## Ingest Demo Corpus

```bash
make ingest
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

