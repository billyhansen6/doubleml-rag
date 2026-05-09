"""
ingest.py — Run the full ingestion pipeline.

    uv run python scripts/ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doubleml_rag.ingestion.chunker import chunk_documents
from doubleml_rag.ingestion.embedder import embed_chunks
from doubleml_rag.ingestion.loaders import load_book_html, load_pdf_papers, load_rst_files
from doubleml_rag.ingestion.store import get_chroma_collection, index_chunks

console = Console(highlight=False)

DATA_RAW = ROOT / "data" / "raw"
DOCS_DIR = DATA_RAW / "doubleml_docs" / "repo" / "doc"
PAPERS_DIR = DATA_RAW / "papers"
BOOK_DIR = DATA_RAW / "book"


def _rule(title: str = "") -> None:
    console.rule(title, characters="-")


def main() -> None:
    _rule("[bold blue]doubleml-rag ingestion pipeline[/bold blue]")

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    _rule("[dim]1 / 4  Loading raw sources[/dim]")

    console.print("[cyan]loaders:[/cyan] loading RST docs...")
    rst_docs = load_rst_files(DOCS_DIR)
    console.print(f"[green]loaders:[/green] {len(rst_docs)} RST sections loaded")

    console.print("[cyan]loaders:[/cyan] loading PDF papers...")
    pdf_docs = load_pdf_papers(PAPERS_DIR)
    console.print(f"[green]loaders:[/green] {len(pdf_docs)} PDF sections loaded")

    console.print("[cyan]loaders:[/cyan] loading book HTML...")
    book_docs = load_book_html(BOOK_DIR)
    console.print(f"[green]loaders:[/green] {len(book_docs)} book sections loaded")

    all_docs = rst_docs + pdf_docs + book_docs
    console.print(f"[green]loaders:[/green] total sections: {len(all_docs)}")

    # ------------------------------------------------------------------
    # 2. Chunk
    # ------------------------------------------------------------------
    _rule("[dim]2 / 4  Chunking[/dim]")

    console.print("[cyan]chunker:[/cyan] splitting into token-bounded chunks...")
    chunks = chunk_documents(all_docs)
    console.print(f"[green]chunker:[/green] {len(chunks)} chunks produced")

    # ------------------------------------------------------------------
    # 3. Embed  (skip chunks already in ChromaDB — idempotent resume)
    # ------------------------------------------------------------------
    _rule("[dim]3 / 4  Embedding via Voyage AI[/dim]")

    collection = get_chroma_collection()
    existing_ids: set[str] = set()
    existing_count = collection.count()
    if existing_count > 0:
        existing_ids = set(collection.get(include=[])["ids"])
        console.print(
            f"[yellow]embedder:[/yellow] {existing_count} chunks already in ChromaDB "
            f"— skipping those"
        )

    to_embed = [c for c in chunks if c["chunk_id"] not in existing_ids]
    console.print(
        f"[cyan]embedder:[/cyan] embedding {len(to_embed)} new chunks "
        f"({len(chunks) - len(to_embed)} already present)..."
    )

    if to_embed:
        to_embed = embed_chunks(to_embed)
        console.print("[green]embedder:[/green] embedding complete")
    else:
        console.print("[green]embedder:[/green] nothing new to embed")

    # ------------------------------------------------------------------
    # 4. Store
    # ------------------------------------------------------------------
    _rule("[dim]4 / 4  Indexing into ChromaDB[/dim]")

    console.print("[cyan]store:[/cyan] upserting into ChromaDB...")
    index_chunks(to_embed, collection)
    chroma_count = collection.count()
    console.print(f"[green]store:[/green] ChromaDB collection size: {chroma_count}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _rule()

    # Per-source_type breakdown (all chunks, not just newly embedded)
    from collections import defaultdict
    type_chunks: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:  # full list from chunker
        type_chunks[c["source_type"]].append(c)

    table = Table(title="Ingestion summary", show_lines=True)
    table.add_column("Source type", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column("Avg tokens/chunk", justify="right")
    table.add_column("Total tokens", justify="right")

    total_chunks = 0
    total_tokens = 0
    for src_type, src_chunks in sorted(type_chunks.items()):
        n = len(src_chunks)
        tok = sum(c["token_count"] for c in src_chunks)
        avg = tok // n if n else 0
        table.add_row(src_type, str(n), str(avg), str(tok))
        total_chunks += n
        total_tokens += tok

    table.add_section()
    avg_all = total_tokens // total_chunks if total_chunks else 0
    table.add_row("[bold]TOTAL[/bold]", str(total_chunks), str(avg_all), str(total_tokens))
    console.print(table)

    console.print(f"\nChromaDB collection size: [bold]{chroma_count}[/bold]")

    # ------------------------------------------------------------------
    # Sample 3 chunks for quality inspection
    # ------------------------------------------------------------------
    _rule("[dim]Sample chunks[/dim]")
    sample_indices = [0, len(chunks) // 2, len(chunks) - 1]
    for idx in sample_indices:
        c = chunks[idx]
        console.print(f"\n[bold yellow]-- Chunk {idx} --[/bold yellow]")
        console.print(f"  chunk_id     : {c['chunk_id']}")
        console.print(f"  source_type  : {c['source_type']}")
        console.print(f"  source_name  : {c['source_name']}")
        console.print(f"  section_path : {c['section_path']}")
        console.print(f"  original_path: {c['original_path']}")
        console.print(f"  token_count  : {c['token_count']}")
        console.print(f"  chunk_index  : {c['chunk_index']} / {c['total_chunks_in_section']}")
        preview = c["text"][:300].replace("\n", " ")
        console.print(f"  text preview : {preview}...")


if __name__ == "__main__":
    main()
