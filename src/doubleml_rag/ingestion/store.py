"""
store.py — Persist chunks + embeddings in ChromaDB.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from doubleml_rag.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR

_DEFAULT_PERSIST = CHROMA_PERSIST_DIR
_DEFAULT_COLLECTION = CHROMA_COLLECTION_NAME

# Metadata fields (everything in chunk dict except text and embedding)
_META_FIELDS = {
    "chunk_id", "source_type", "source_name", "section_path",
    "original_path", "chunk_index", "total_chunks_in_section", "token_count",
}


def get_chroma_collection(
    persist_dir: Path = _DEFAULT_PERSIST,
    collection_name: str = _DEFAULT_COLLECTION,
) -> Collection:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[dict], collection: Collection) -> None:
    """Upsert all chunks (with embeddings) into the collection."""
    if not chunks:
        return

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [
        {k: v for k, v in c.items() if k in _META_FIELDS}
        for c in chunks
    ]

    # Upsert in batches of 500 to stay well under ChromaDB limits
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
