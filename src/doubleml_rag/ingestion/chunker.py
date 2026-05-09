"""
chunker.py — Split section-level dicts into token-bounded chunks.

Strategy:
- Sections under max_tokens → single chunk.
- Sections over max_tokens → split at sentence boundaries with overlap.
- Code blocks are atomic: never split mid-block.

Output dict shape:
    text, chunk_id, source_type, source_name, section_path,
    original_path, chunk_index, total_chunks_in_section, token_count
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Sequence

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

# Regex that matches a code block fence: lines starting with 4+ spaces or
# a stretch of text that looks like a code fence (``` ... ```)
_CODE_FENCE = re.compile(r"```[\s\S]*?```", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _count(text: str) -> int:
    return len(_ENC.encode(text))


def _slugify(text: str) -> str:
    """Convert a heading to a URL-safe slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "_", text)[:60]


def _split_preserving_code(text: str, target: int, max_tokens: int, overlap: int) -> list[str]:
    """
    Split text into chunks of ~target tokens, keeping code blocks atomic.
    Overlaps are added at sentence boundaries.
    """
    # Identify code-block spans so we never split inside them
    protected: list[tuple[int, int]] = []
    for m in _CODE_FENCE.finditer(text):
        protected.append((m.start(), m.end()))

    def in_protected(pos: int) -> bool:
        return any(s <= pos < e for s, e in protected)

    # Split into sentence-level atoms first
    # We'll build atoms that are either code blocks or sentence runs
    atoms: list[str] = []
    last = 0
    for s, e in protected:
        # prose before this code block
        prose = text[last:s]
        if prose.strip():
            atoms.extend(p.strip() for p in _SENTENCE_END.split(prose) if p.strip())
        # the code block itself as one atom
        atoms.append(text[s:e])
        last = e
    # trailing prose
    prose = text[last:]
    if prose.strip():
        atoms.extend(p.strip() for p in _SENTENCE_END.split(prose) if p.strip())

    if not atoms:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for atom in atoms:
        atom_tokens = _count(atom)

        # If a single atom exceeds max_tokens (e.g. huge code block), emit it alone
        if atom_tokens > max_tokens:
            if buf:
                chunks.append(" ".join(buf))
                buf = []
                buf_tokens = 0
            chunks.append(atom)
            continue

        if buf_tokens + atom_tokens > max_tokens:
            chunks.append(" ".join(buf))
            # Overlap: carry last N tokens worth of sentences into next chunk
            overlap_buf: list[str] = []
            overlap_count = 0
            for sentence in reversed(buf):
                st = _count(sentence)
                if overlap_count + st <= overlap:
                    overlap_buf.insert(0, sentence)
                    overlap_count += st
                else:
                    break
            buf = overlap_buf + [atom]
            buf_tokens = overlap_count + atom_tokens
        else:
            buf.append(atom)
            buf_tokens += atom_tokens

    if buf:
        chunks.append(" ".join(buf))

    return [c for c in chunks if c.strip()]


def chunk_documents(
    docs: Sequence[dict],
    target_tokens: int = 800,
    max_tokens: int = 1500,
    overlap_tokens: int = 100,
) -> list[dict]:
    """Chunk a list of section dicts into token-bounded chunk dicts."""
    chunks: list[dict] = []
    # Count occurrences of each (original_path, section_path) pair so that
    # genuinely repeated section headings within the same file get unique IDs.
    key_counter: dict[str, int] = {}

    for doc in docs:
        text = doc["text"]
        source_name = doc["source_name"]
        section_hierarchy = doc.get("section_hierarchy", [])
        section_path = " > ".join(section_hierarchy) if section_hierarchy else source_name

        # Build chunk_id: readable slug + 8-char hash.
        # Hash includes original_path + occurrence index so even repeated
        # headings within the same file get unique, stable IDs.
        last_heading = _slugify(section_hierarchy[-1]) if section_hierarchy else _slugify(source_name)
        unique_key = f"{doc.get('original_path', '')}::{section_path}"
        occurrence = key_counter.get(unique_key, 0)
        key_counter[unique_key] = occurrence + 1
        path_hash = hashlib.md5(f"{unique_key}::{occurrence}".encode()).hexdigest()[:8]

        token_count = _count(text)

        if token_count <= max_tokens:
            raw_chunks = [text]
        else:
            raw_chunks = _split_preserving_code(text, target_tokens, max_tokens, overlap_tokens)
            if not raw_chunks:
                raw_chunks = [text]

        total = len(raw_chunks)
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_id": f"{source_name}::{last_heading}::{path_hash}::{i}",
                    "source_type": doc["source_type"],
                    "source_name": source_name,
                    "section_path": section_path,
                    "original_path": doc.get("original_path", ""),
                    "chunk_index": i,
                    "total_chunks_in_section": total,
                    "token_count": _count(chunk_text),
                }
            )

    # Sanity check: all IDs must be unique before returning
    ids = [c["chunk_id"] for c in chunks]
    seen: set[str] = set()
    dupes = [cid for cid in ids if cid in seen or seen.add(cid)]  # type: ignore[func-returns-value]
    if dupes:
        raise ValueError(
            f"chunk_documents produced {len(dupes)} duplicate chunk_ids: {dupes[:5]}"
        )

    return chunks
