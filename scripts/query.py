"""
query.py — CLI for querying the doubleml-rag retrieval system.

Usage:
    uv run python scripts/query.py --query "what is double machine learning"
    uv run python scripts/query.py --query "cross-fitting" --k 8
    uv run python scripts/query.py --query "propensity score" --k 5 --source book paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from doubleml_rag.generation.answerer import Answerer
from doubleml_rag.retrieval.retriever import Retriever

console = Console(highlight=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query the doubleml-rag retrieval system")
    p.add_argument("--query", required=True, help="Question or search string")
    p.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    p.add_argument(
        "--source",
        nargs="+",
        metavar="TYPE",
        help="Filter by source_type: paper, book, docs (can list multiple)",
    )
    p.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation and show retrieval results only",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    retriever = Retriever()
    results = retriever.retrieve(args.query, k=args.k, source_filter=args.source)

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    filter_note = f" [source: {', '.join(args.source)}]" if args.source else ""
    console.print(Rule(f'[bold blue]Query[/bold blue]: "{args.query}"{filter_note}', characters="-"))
    console.print()

    table = Table(show_lines=True, expand=False)
    table.add_column("Rank", justify="right", style="bold", width=5)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Type", width=7)
    table.add_column("Source", width=28)
    table.add_column("Section path", no_wrap=False)

    for i, r in enumerate(results, start=1):
        score_color = "green" if r["score"] >= 0.80 else "yellow" if r["score"] >= 0.65 else "red"
        table.add_row(
            str(i),
            f"[{score_color}]{r['score']:.3f}[/{score_color}]",
            r["source_type"],
            r["source_name"],
            r["section_path"],
        )

    console.print(table)

    if args.no_generate:
        return

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    answerer = Answerer()
    out = answerer.answer(args.query, results)

    console.print()
    console.print(Rule("[bold green]Answer[/bold green]", characters="-"))
    console.print()
    console.print(out["answer"])
    console.print()

    # Sources block — only show cited chunks
    cited_indices = out["citations_used"]  # 1-based
    if cited_indices and not out["abstained"]:
        console.print(Rule("[bold]Sources[/bold]", characters="-"))
        for idx in cited_indices:
            if 1 <= idx <= len(results):
                r = results[idx - 1]
                console.print(
                    f"  [[bold]{idx}[/bold]] {r['source_name']} :: "
                    f"{r['section_path']} :: [dim]{r['chunk_id']}[/dim]"
                )
        console.print()

    # Footer
    console.print(
        f"[dim]model: {out['model']} | "
        f"in: {out['input_tokens']} tok | "
        f"out: {out['output_tokens']} tok[/dim]"
    )
    console.print()


if __name__ == "__main__":
    main()
