"""
rejudge_faithfulness.py -- Re-run the faithfulness judge on existing eval/results.jsonl.

Re-retrieves chunks from ChromaDB (same k=10) and re-judges faithfulness using the
stored answer. Replaces judge_faithfulness in-place and writes atomically.

Usage:
    uv run python scripts/rejudge_faithfulness.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from doubleml_rag.eval.judges import judge_faithfulness
from doubleml_rag.retrieval.retriever import Retriever

RESULTS_PATH = ROOT / "eval" / "results.jsonl"
TMP_PATH = ROOT / "eval" / "results.jsonl.tmp"

console = Console(highlight=False)


def load_results() -> list[dict]:
    results = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def write_results_atomic(results: list[dict]) -> None:
    with open(TMP_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    TMP_PATH.replace(RESULTS_PATH)


def old_score_repr(r: dict) -> str:
    jf = r.get("judge_faithfulness")
    if not isinstance(jf, dict):
        return "none"
    if "faithfulness_score" in jf:
        return str(jf["faithfulness_score"])
    if "score" in jf:
        return f"{jf['score']} (old-scale)"
    return "?"


def main() -> None:
    console.print(Rule("[bold blue]Rejudge faithfulness[/bold blue]", characters="-"))

    results = load_results()
    console.print(f"Loaded [bold]{len(results)}[/bold] results from {RESULTS_PATH.relative_to(ROOT)}")
    console.print()

    retriever = Retriever()

    for r in results:
        qid = r["id"]

        # Skip only questions that are *designed* to have no answer
        if r.get("should_abstain"):
            r["judge_faithfulness"] = {
                "faithfulness_score": None,
                "reason": "abstention_question_no_answer_to_score",
                "unsupported_claims": [],
            }
            console.print(f"  [dim]{qid}[/dim]  skip (should_abstain)")
            continue

        answer = r.get("answer", "")
        if not answer:
            console.print(f"  [dim]{qid}[/dim]  skip (no answer stored)")
            continue

        old = old_score_repr(r)

        try:
            chunks = retriever.retrieve(r["question"], k=10)
        except Exception as exc:
            console.print(f"  [red]{qid}[/red]  retrieval error: {exc}")
            continue

        try:
            new_judge = judge_faithfulness(r["question"], chunks, answer)
        except Exception as exc:
            console.print(f"  [red]{qid}[/red]  judge error: {exc}")
            continue

        r["judge_faithfulness"] = new_judge
        new_score = new_judge.get("faithfulness_score")
        n_unsupported = len(new_judge.get("unsupported_claims") or [])
        suffix = f"  unsupported={n_unsupported}" if new_score != 1.0 else ""
        console.print(f"  [cyan]{qid}[/cyan]  old={old}  new={new_score}{suffix}")

    write_results_atomic(results)
    console.print()
    console.print("[green]Written atomically to eval/results.jsonl[/green]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _ABSTAIN_PHRASES = re.compile(
        r"does not contain enough information"
        r"|cannot answer"
        r"|can't answer"
        r"|unable to answer"
        r"|not enough information"
        r"|insufficient (context|information)",
        re.IGNORECASE,
    )

    scored = [
        r for r in results
        if isinstance(r.get("judge_faithfulness"), dict)
        and r["judge_faithfulness"].get("faithfulness_score") is not None
    ]
    null_scored = [
        r for r in results
        if isinstance(r.get("judge_faithfulness"), dict)
        and r["judge_faithfulness"].get("faithfulness_score") is None
    ]
    scores = [r["judge_faithfulness"]["faithfulness_score"] for r in scored]

    # False abstentions: should_abstain=false but system abstained
    false_abstentions = [
        r for r in results
        if not r.get("should_abstain")
        and bool(_ABSTAIN_PHRASES.search(r.get("answer") or ""))
    ]

    console.print()
    console.print(Rule("[bold]Score distribution (0 / 0.5 / 1)[/bold]", characters="-"))
    dist: dict[float, int] = defaultdict(int)
    for s in scores:
        dist[s] += 1

    dist_table = Table(show_lines=False)
    dist_table.add_column("Score", style="bold", justify="right")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("Pct", justify="right")
    for sv in sorted(dist):
        pct = 100 * dist[sv] / len(scores)
        dist_table.add_row(str(sv), str(dist[sv]), f"{pct:.0f}%")
    console.print(dist_table)

    avg = sum(scores) / len(scores) if scores else float("nan")
    console.print(f"\nValid faithfulness scores : [bold]{len(scored)}[/bold]  (mean={avg:.3f})")
    console.print(f"Null scores (should_abstain): [bold]{len(null_scored)}[/bold]")
    console.print(f"False abstentions            : [bold]{len(false_abstentions)}[/bold]"
                  f"  (should_abstain=false, but system said 'does not contain...')")

    console.print()
    console.print(Rule("[bold]Per-category mean faithfulness[/bold]", characters="-"))
    cat_scores: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        cat_scores[r.get("category", "unknown")].append(
            r["judge_faithfulness"]["faithfulness_score"]
        )
    cat_table = Table(show_lines=False)
    cat_table.add_column("Category", style="bold")
    cat_table.add_column("N", justify="right")
    cat_table.add_column("Mean faithful", justify="right")
    for cat in sorted(cat_scores):
        vals = cat_scores[cat]
        cat_table.add_row(cat, str(len(vals)), f"{sum(vals)/len(vals):.3f}")
    console.print(cat_table)

    n_with_unsupported = sum(
        1 for r in scored if r["judge_faithfulness"].get("unsupported_claims")
    )
    console.print(f"\nAnswers with unsupported claims: [bold]{n_with_unsupported}[/bold] / {len(scored)}")

    # ------------------------------------------------------------------
    # Abstention precision / recall
    # ------------------------------------------------------------------
    console.print()
    console.print(Rule("[bold]Abstention precision / recall[/bold]", characters="-"))

    system_abstained = [r for r in results if r.get("abstained")]
    n_sys = len(system_abstained)
    n_correct = sum(1 for r in system_abstained if r.get("should_abstain"))
    n_should = sum(1 for r in results if r.get("should_abstain"))

    precision = n_correct / n_sys if n_sys else float("nan")
    recall = n_correct / n_should if n_should else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")

    abs_table = Table(show_lines=False)
    abs_table.add_column("Metric", style="bold")
    abs_table.add_column("Value", justify="right")
    abs_table.add_row("System abstained (total)", str(n_sys))
    abs_table.add_row("  of which: correctly should_abstain", str(n_correct))
    abs_table.add_row("  of which: false abstentions", str(n_sys - n_correct))
    abs_table.add_row("Total should_abstain questions", str(n_should))
    abs_table.add_row("Abstention precision", f"{precision:.3f}")
    abs_table.add_row("Abstention recall", f"{recall:.3f}")
    abs_table.add_row("Abstention F1", f"{f1:.3f}")
    console.print(abs_table)
    console.print()


if __name__ == "__main__":
    main()
