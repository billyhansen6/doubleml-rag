"""
retriever.py — Semantic retrieval over the doubleml-rag ChromaDB collection.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
import voyageai

from doubleml_rag.config import settings


class Retriever:
    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: Path | None = None,
        embed_model: str = "voyage-3",
    ) -> None:
        self.embed_model = embed_model
        resolved_dir = persist_dir or settings.chroma_persist_dir
        resolved_name = collection_name or settings.chroma_collection_name

        self._client = chromadb.PersistentClient(path=str(resolved_dir))
        self._collection = self._client.get_collection(resolved_name)
        self._vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", ""))

    def retrieve(
        self,
        query: str,
        k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Embed the query and return top-k chunks ranked by cosine similarity.

        Parameters
        ----------
        query         : natural-language question
        k             : number of results to return
        source_filter : optional list of source_type values to restrict results
                        e.g. ["paper", "book"]

        Returns
        -------
        list of dicts with keys:
            chunk_id, text, score, source_type, source_name,
            section_path, original_path
        """
        # Embed with input_type="query" — Voyage uses an asymmetric embedding
        # space: documents are indexed with "document", queries use "query".
        result = self._vo.embed([query], model=self.embed_model, input_type="query")
        query_embedding = result.embeddings[0]

        where = None
        if source_filter:
            if len(source_filter) == 1:
                where = {"source_type": source_filter[0]}
            else:
                where = {"source_type": {"$in": source_filter}}

        query_kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if where is not None:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results = []
        for cid, text, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB returns cosine *distance* in [0, 2]; similarity = 1 - distance
            score = max(0.0, min(1.0, 1.0 - dist))
            results.append(
                {
                    "chunk_id": cid,
                    "text": text,
                    "score": score,
                    "source_type": meta.get("source_type", ""),
                    "source_name": meta.get("source_name", ""),
                    "section_path": meta.get("section_path", ""),
                    "original_path": meta.get("original_path", ""),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
