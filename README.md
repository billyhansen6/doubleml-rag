# doubleml-rag

A RAG (Retrieval-Augmented Generation) demo over DoubleML documentation, built with the Anthropic Claude API and Voyage AI embeddings. No LangChain or LlamaIndex.

---

## Overview

This project implements a from-scratch RAG pipeline for answering questions about DoubleML (Double Machine Learning). It ingests documentation, papers, and book chapters, embeds and indexes them with ChromaDB, and retrieves relevant context for Claude to generate cited answers.

The goal was to simulate the kind of RAG setup a business might use over its internal documents — pointing an LLM at a specific corpus to answer context-specific questions. DoubleML isn't a perfect analogue, since Claude has already seen the public docs and papers in training, but the architecture and the failure modes should be similar. The eval is built to surface those failure modes rather than just produce a working demo.

---

## Eval results

The full evaluation is documented in [`notebooks/eval_results.ipynb`](notebooks/eval_results.ipynb). It covers retrieval metrics (Recall@k, MRR), generation quality, faithfulness, and abstention behavior across 50 labeled questions — including a RAG vs no-RAG ablation.

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/billyhansen6/doubleml-rag.git
cd doubleml-rag

# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY
```

---

## Usage (CLI)

```bash
# Ingest documents into the vector store
uv run python -m doubleml_rag.ingestion

# Query the RAG pipeline
uv run python -m doubleml_rag.retrieval "What is the Neyman orthogonality condition?"
```

---

## Eval

Golden questions are defined in `eval/golden.yaml`. Run evaluation with:

```bash
uv run pytest eval/
```

Results are written to `data/processed/eval_results.csv`.

---

## Architecture

```
src/doubleml_rag/
├── config.py        # Settings and path constants
├── ingestion/       # PDF/HTML/Markdown parsing and chunking
├── retrieval/       # ChromaDB vector search
├── generation/      # Claude API calls, prompt construction
└── eval/            # Scoring and metrics
```
