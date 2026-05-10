"""
run_no_rag_ablation.py -- No-RAG ablation for doubleml-rag eval.

For each non-abstain question, generates an answer from Claude Sonnet 4.5
using only the question (no retrieved chunks), then scores it with the
same quality judge used for RAG answers.

Augments eval/results.jsonl in-place (atomic write via .tmp).
Idempotent: skips questions that already have no_rag_answer.

Usage:
    uv run python scripts/run_no_rag_ablation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import anthropic
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from doubleml_rag.eval.judges import judge_answer_quality

RESULTS_PATH = ROOT / "eval" / "results.jsonl"
TMP_PATH     = ROOT / "eval" / "results.jsonl.tmp"
MODEL        = "claude-sonnet-4-5"
MAX_TOKENS   = 1024

_SYSTEM = (
    "Answer the user's question based on your knowledge. "
    "Be specific and accurate. If you don't know the answer, say so honestly."
)

console = Console(highlight=False)


def load_results() -> list[dict]:
    return [json.loads(l) for l in open(RESULTS_PATH, encoding="utf-8") if l.strip()]


def write_results_atomic(results: list[dict]) -> None:
    with open(TMP_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    os.replace(TMP_PATH, RESULTS_PATH)


def generate_no_rag(question: str, client: anthropic.Anthropic) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text.strip()


def main() -> None:
    console.print(Rule("[bold blue]No-RAG ablation[/bold blue]", characters="-"))

    results = load_results()
    client  = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    to_run = [
        r for r in results
        if not r.get("should_abstain") and "no_rag_answer" not in r
    ]
    already = sum(1 for r in results if "no_rag_answer" in r)
    console.print(
        f"Loaded [bold]{len(results)}[/bold] results. "
        f"To run: [bold]{len(to_run)}[/bold]  |  Already done: [dim]{already}[/dim]"
    )
    console.print()

    for r in results:
        if r.get("should_abstain") or "no_rag_answer" in r:
            continue

        qid = r["id"]
        question = r["question"]

        try:
            no_rag_answer = generate_no_rag(question, client)
        except Exception as exc:
            console.print(f"  [red]{qid}[/red]  generation error: {exc}")
            continue

        # Quality judge — no retrieved chunks for no-RAG (pass empty list)
        try:
            quality = judge_answer_quality(question, [], no_rag_answer)
        except Exception as exc:
            console.print(f"  [red]{qid}[/red]  judge error: {exc}")
            quality = {"score": None, "reason": str(exc), "judge": "answer_quality"}

        r["no_rag_answer"]  = no_rag_answer
        r["no_rag_quality"] = quality

        rag_score    = (r.get("judge_quality") or {}).get("score", "n/a")
        no_rag_score = quality.get("score", "n/a")
        delta_str    = ""
        if isinstance(rag_score, (int, float)) and isinstance(no_rag_score, (int, float)):
            delta = rag_score - no_rag_score
            delta_str = f"  delta={delta:+.0f}"

        console.print(
            f"  [cyan]{qid}[/cyan]  rag={rag_score}  no_rag={no_rag_score}{delta_str}"
        )

    write_results_atomic(results)
    console.print()
    console.print("[green]Written atomically to eval/results.jsonl[/green]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    console.print()
    console.print(Rule("[bold]Ablation summary[/bold]", characters="-"))

    scored = [
        r for r in results
        if not r.get("should_abstain")
        and isinstance((r.get("judge_quality") or {}).get("score"), (int, float))
        and isinstance((r.get("no_rag_quality") or {}).get("score"), (int, float))
    ]

    rag_scores    = [r["judge_quality"]["score"]    for r in scored]
    no_rag_scores = [r["no_rag_quality"]["score"]   for r in scored]
    deltas        = [a - b for a, b in zip(rag_scores, no_rag_scores)]

    mean_rag    = sum(rag_scores)    / len(rag_scores)
    mean_no_rag = sum(no_rag_scores) / len(no_rag_scores)

    console.print(f"Overall mean RAG quality : [bold]{mean_rag:.2f}[/bold]  (n={len(rag_scores)})")
    console.print(f"Overall mean no-RAG quality: [bold]{mean_no_rag:.2f}[/bold]  (n={len(no_rag_scores)})")
    console.print(f"Overall mean delta (RAG - no-RAG): [bold]{mean_rag - mean_no_rag:+.2f}[/bold]")

    # Per-category
    console.print()
    from collections import defaultdict
    cat_data: dict[str, dict] = defaultdict(lambda: {"rag": [], "no_rag": []})
    for r in scored:
        cat = r.get("category", "unknown")
        cat_data[cat]["rag"].append(r["judge_quality"]["score"])
        cat_data[cat]["no_rag"].append(r["no_rag_quality"]["score"])

    cat_table = Table(show_lines=False)
    cat_table.add_column("Category",    style="bold")
    cat_table.add_column("N",           justify="right")
    cat_table.add_column("RAG mean",    justify="right")
    cat_table.add_column("no-RAG mean", justify="right")
    cat_table.add_column("Delta",       justify="right")

    for cat in sorted(cat_data):
        rag_v    = cat_data[cat]["rag"]
        norag_v  = cat_data[cat]["no_rag"]
        mr       = sum(rag_v)   / len(rag_v)
        mn       = sum(norag_v) / len(norag_v)
        d        = mr - mn
        color    = "green" if d > 0 else "red" if d < 0 else "white"
        cat_table.add_row(
            cat, str(len(rag_v)),
            f"{mr:.2f}", f"{mn:.2f}",
            f"[{color}]{d:+.2f}[/{color}]",
        )
    console.print(cat_table)

    # Top 3 / bottom 3 by delta
    ranked = sorted(zip(deltas, scored), key=lambda x: x[0], reverse=True)

    def _show_trio(title: str, items: list) -> None:
        console.print()
        console.print(f"[bold]{title}[/bold]")
        t = Table(show_lines=False)
        t.add_column("ID",       style="bold cyan")
        t.add_column("Category")
        t.add_column("Question (first 80 chars)", no_wrap=False)
        t.add_column("RAG",    justify="right")
        t.add_column("no-RAG", justify="right")
        t.add_column("Delta",  justify="right")
        for delta, r in items:
            color = "green" if delta > 0 else "red" if delta < 0 else "white"
            t.add_row(
                r["id"], r.get("category", ""),
                r["question"][:80],
                str(r["judge_quality"]["score"]),
                str(r["no_rag_quality"]["score"]),
                f"[{color}]{delta:+.0f}[/{color}]",
            )
        console.print(t)

    _show_trio("Top 3 (RAG wins biggest)",         ranked[:3])
    _show_trio("Bottom 3 (RAG didn't help / lost)", ranked[-3:])
    console.print()


if __name__ == "__main__":
    main()
