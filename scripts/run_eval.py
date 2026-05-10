"""
run_eval.py -- Eval harness for doubleml-rag golden set.

Runs all 50 questions through retrieval + generation + judges and writes
structured results to eval/results.jsonl.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --resume      # skip already-evaluated question IDs
    uv run python scripts/run_eval.py --no-generate # retrieval metrics only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ruamel.yaml import YAML
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from doubleml_rag.retrieval.retriever import Retriever
from doubleml_rag.generation.answerer import Answerer
from doubleml_rag.eval.metrics import compute_retrieval_metrics
from doubleml_rag.eval.judges import judge_faithfulness, judge_answer_quality, judge_abstention

GOLDEN_PATH = ROOT / "eval" / "golden.yaml"
RESULTS_PATH = ROOT / "eval" / "results.jsonl"
K = 10

console = Console(highlight=False)
yaml = YAML()


def load_questions() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return yaml.load(f)


def load_done_ids(resume: bool) -> set[str]:
    if not resume or not RESULTS_PATH.exists():
        return set()
    done: set[str] = set()
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_result(result: dict) -> None:
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def safe_judge(fn, *args, **kwargs) -> dict:
    """Call a judge function; return an error dict on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"error": str(exc), "judge": fn.__name__}


def run_question(
    q: dict,
    retriever: Retriever,
    answerer: Answerer | None,
    no_generate: bool,
) -> dict:
    qid = q["id"]
    question = q["question"]
    should_abstain = q.get("should_abstain", False)
    gt_ids: list[str] = q.get("ground_truth_chunk_ids") or []

    result: dict = {
        "id": qid,
        "category": q.get("category", ""),
        "question": question,
        "should_abstain": should_abstain,
        "ground_truth_chunk_ids": gt_ids,
        "error": None,
    }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    try:
        chunks = retriever.retrieve(question, k=K)
    except Exception as exc:
        result["error"] = f"retrieval: {exc}"
        return result

    retrieved_ids = [c["chunk_id"] for c in chunks]
    result["retrieved_chunk_ids"] = retrieved_ids
    result["top_scores"] = [round(c["score"], 4) for c in chunks]

    # Retrieval metrics (only for non-abstain questions with ground truth)
    if not should_abstain and gt_ids:
        result["retrieval_metrics"] = compute_retrieval_metrics(retrieved_ids, gt_ids)
    else:
        result["retrieval_metrics"] = None

    if no_generate:
        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    try:
        gen_out = answerer.answer(question, chunks)
    except Exception as exc:
        result["error"] = f"generation: {exc}"
        return result

    answer = gen_out["answer"]
    system_abstained = gen_out["abstained"]

    result["answer"] = answer
    result["abstained"] = system_abstained
    result["citations_used"] = gen_out["citations_used"]
    result["model"] = gen_out["model"]
    result["input_tokens"] = gen_out["input_tokens"]
    result["output_tokens"] = gen_out["output_tokens"]

    # ------------------------------------------------------------------
    # Judges
    # ------------------------------------------------------------------
    # Abstention judge (always run)
    result["judge_abstention"] = safe_judge(
        judge_abstention, question, answer, system_abstained, should_abstain
    )

    # Faithfulness + quality — only for non-abstain questions where the
    # system actually provided an answer
    if not should_abstain and not system_abstained:
        result["judge_faithfulness"] = safe_judge(
            judge_faithfulness, question, chunks, answer
        )
        result["judge_quality"] = safe_judge(
            judge_answer_quality, question, chunks, answer
        )
    else:
        result["judge_faithfulness"] = None
        result["judge_quality"] = None

    return result


def print_summary(results: list[dict]) -> None:
    console.print()
    console.print(Rule("[bold]Aggregate Summary[/bold]", characters="-"))

    total = len(results)
    errors = sum(1 for r in results if r.get("error"))
    non_abstain = [r for r in results if not r.get("should_abstain")]
    abstain_q = [r for r in results if r.get("should_abstain")]

    # Retrieval metrics
    recall_vals = [
        r["retrieval_metrics"]["recall_at_k"]
        for r in non_abstain
        if r.get("retrieval_metrics") and r["retrieval_metrics"]["recall_at_k"] is not None
    ]
    mrr_vals = [
        r["retrieval_metrics"]["mrr"]
        for r in non_abstain
        if r.get("retrieval_metrics") and r["retrieval_metrics"]["mrr"] is not None
    ]

    avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else float("nan")
    avg_mrr = sum(mrr_vals) / len(mrr_vals) if mrr_vals else float("nan")

    # Abstention accuracy
    abstention_results = [r for r in results if r.get("judge_abstention")]
    abstention_correct = sum(
        1 for r in abstention_results
        if r["judge_abstention"].get("correct") is True
    )
    abstention_acc = abstention_correct / len(abstention_results) if abstention_results else float("nan")

    # Generation judge scores (1-5 scale)
    faith_scores = [
        r["judge_faithfulness"]["score"]
        for r in results
        if r.get("judge_faithfulness") and isinstance(r["judge_faithfulness"].get("score"), (int, float))
    ]
    quality_scores = [
        r["judge_quality"]["score"]
        for r in results
        if r.get("judge_quality") and isinstance(r["judge_quality"].get("score"), (int, float))
    ]

    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else float("nan")
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else float("nan")

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Metric", style="bold", min_width=32)
    table.add_column("Value", justify="right", min_width=10)

    table.add_row("Questions evaluated", str(total))
    table.add_row("Errors", f"[red]{errors}[/red]" if errors else "0")
    table.add_row("", "")
    table.add_row("-- Retrieval (non-abstain) --", "")
    table.add_row(f"  Recall@{K}", f"{avg_recall:.3f}" if recall_vals else "n/a")
    table.add_row("  MRR", f"{avg_mrr:.3f}" if mrr_vals else "n/a")
    table.add_row("", "")
    table.add_row("-- Abstention --", "")
    table.add_row("  Accuracy", f"{abstention_acc:.3f}" if abstention_results else "n/a")
    table.add_row(
        f"  Correct ({abstention_correct}/{len(abstention_results)})",
        f"  (should_abstain: {len(abstain_q)} q)",
    )
    table.add_row("", "")
    table.add_row("-- Generation judges (1-5) --", "")
    table.add_row(
        f"  Faithfulness ({len(faith_scores)} q)",
        f"{avg_faith:.2f}" if faith_scores else "n/a",
    )
    table.add_row(
        f"  Answer quality ({len(quality_scores)} q)",
        f"{avg_quality:.2f}" if quality_scores else "n/a",
    )

    console.print(table)

    # Per-category breakdown
    categories = sorted({r.get("category", "unknown") for r in results})
    if len(categories) > 1:
        console.print()
        console.print(Rule("[bold]By Category[/bold]", characters="-"))
        cat_table = Table(show_lines=False)
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("N", justify="right")
        cat_table.add_column("Recall@K", justify="right")
        cat_table.add_column("MRR", justify="right")
        cat_table.add_column("Faithful", justify="right")
        cat_table.add_column("Quality", justify="right")

        for cat in categories:
            cat_results = [r for r in results if r.get("category") == cat]
            n = len(cat_results)

            cat_recall = [
                r["retrieval_metrics"]["recall_at_k"]
                for r in cat_results
                if r.get("retrieval_metrics") and r["retrieval_metrics"]["recall_at_k"] is not None
            ]
            cat_mrr = [
                r["retrieval_metrics"]["mrr"]
                for r in cat_results
                if r.get("retrieval_metrics") and r["retrieval_metrics"]["mrr"] is not None
            ]
            cat_faith = [
                r["judge_faithfulness"]["score"]
                for r in cat_results
                if r.get("judge_faithfulness") and isinstance(r["judge_faithfulness"].get("score"), (int, float))
            ]
            cat_qual = [
                r["judge_quality"]["score"]
                for r in cat_results
                if r.get("judge_quality") and isinstance(r["judge_quality"].get("score"), (int, float))
            ]

            cat_table.add_row(
                cat,
                str(n),
                f"{sum(cat_recall)/len(cat_recall):.3f}" if cat_recall else "-",
                f"{sum(cat_mrr)/len(cat_mrr):.3f}" if cat_mrr else "-",
                f"{sum(cat_faith)/len(cat_faith):.2f}" if cat_faith else "-",
                f"{sum(cat_qual)/len(cat_qual):.2f}" if cat_qual else "-",
            )

        console.print(cat_table)

    console.print()
    console.print(f"Results written to [bold]{RESULTS_PATH.relative_to(ROOT)}[/bold]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run doubleml-rag eval harness")
    parser.add_argument("--resume", action="store_true", help="Skip already-evaluated question IDs")
    parser.add_argument("--no-generate", action="store_true", help="Retrieval metrics only, skip generation+judges")
    args = parser.parse_args()

    console.print(Rule("[bold blue]doubleml-rag eval harness[/bold blue]", characters="-"))

    questions = load_questions()
    done_ids = load_done_ids(args.resume)

    if args.resume and done_ids:
        console.print(f"[dim]Resuming: {len(done_ids)} questions already done, skipping.[/dim]")

    to_run = [q for q in questions if q["id"] not in done_ids]
    console.print(
        f"Running [bold]{len(to_run)}[/bold] questions "
        f"({'retrieval only' if args.no_generate else 'retrieval + generation + judges'})"
    )
    console.print()

    retriever = Retriever()
    answerer = None if args.no_generate else Answerer()

    # Clear results file if not resuming
    if not args.resume:
        RESULTS_PATH.write_text("", encoding="utf-8")

    all_results: list[dict] = []
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(to_run))

        for q in to_run:
            progress.update(task, description=f"[cyan]{q['id']}[/cyan] {q['question'][:50]}")
            try:
                result = run_question(q, retriever, answerer, args.no_generate)
            except Exception as exc:
                result = {
                    "id": q["id"],
                    "category": q.get("category", ""),
                    "question": q["question"],
                    "should_abstain": q.get("should_abstain", False),
                    "ground_truth_chunk_ids": q.get("ground_truth_chunk_ids") or [],
                    "error": f"unhandled: {exc}\n{traceback.format_exc()[-300:]}",
                }

            append_result(result)
            all_results.append(result)
            progress.advance(task)

    # Load all results for summary (including previously-done if resuming)
    if args.resume and done_ids:
        loaded: list[dict] = []
        with open(RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        loaded.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        all_results = loaded

    elapsed = time.time() - t0
    console.print(f"\n[dim]Completed in {elapsed:.0f}s[/dim]")

    print_summary(all_results)


if __name__ == "__main__":
    main()
