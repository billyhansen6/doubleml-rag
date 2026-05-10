"""
label_ground_truth.py — Interactive CLI for labeling ground-truth chunk IDs.

For each question in eval/golden.yaml that lacks ground_truth_chunk_ids and
is not a should_abstain, runs retrieval and prompts the user to select
which retrieved chunks are correct answers.

Usage:
    uv run python scripts/label_ground_truth.py               # interactive
    uv run python scripts/label_ground_truth.py --dump-mode   # write eval/labeling_dump.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ruamel.yaml import YAML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from doubleml_rag.retrieval.retriever import Retriever

GOLDEN_PATH = ROOT / "eval" / "golden.yaml"
DUMP_PATH = ROOT / "eval" / "labeling_dump.txt"
K = 10

console = Console(highlight=False)
yaml = YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False
yaml.width = 120


def load_questions() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return yaml.load(f)


def save_questions(questions: list[dict]) -> None:
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        yaml.dump(questions, f)


def display_question(q: dict, idx: int, total: int) -> None:
    header = f"[bold]{q['id']}[/bold]  [{q['category']}]  ({idx}/{total})"
    body = Text(q["question"])
    if q.get("notes"):
        body.append(f"\n\n[dim]{q['notes']}[/dim]")
    console.print(Panel(body, title=header, border_style="blue"))


def display_chunks(chunks: list[dict]) -> None:
    table = Table(show_lines=True, expand=True)
    table.add_column("Rank", justify="right", width=5, style="bold")
    table.add_column("Score", justify="right", width=7)
    table.add_column("Type", width=7)
    table.add_column("Source / Section", no_wrap=False)
    table.add_column("Preview", no_wrap=False)

    for i, c in enumerate(chunks, start=1):
        score_color = (
            "green" if c["score"] >= 0.75
            else "yellow" if c["score"] >= 0.60
            else "red"
        )
        source_cell = f"{c['source_name']}\n[dim]{c['section_path']}[/dim]"
        preview = c["text"][:200].replace("\n", " ")
        if len(c["text"]) > 200:
            preview += "..."
        table.add_row(
            str(i),
            f"[{score_color}]{c['score']:.3f}[/{score_color}]",
            c["source_type"],
            source_cell,
            preview,
        )

    console.print(table)


def prompt_user() -> str:
    console.print(
        "[bold cyan]Enter rank numbers of correct chunks (comma-separated, e.g. '1,3'), "
        "'n' for none/abstain-after-all, 's' to skip and revisit, or 'q' to save and quit:[/bold cyan] ",
        end="",
    )
    return input().strip()


def parse_ranks(raw: str) -> list[int]:
    ranks = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ranks.append(int(part))
    return ranks


def dump_mode() -> None:
    """Non-interactive: retrieve for every unlabeled question and write a flat text file."""
    retriever = Retriever()
    questions = load_questions()

    to_dump = [
        q for q in questions
        if not q.get("should_abstain", False)
        and not q.get("ground_truth_chunk_ids")
    ]

    lines: list[str] = []
    SEP = "=" * 64

    for q in to_dump:
        try:
            chunks = retriever.retrieve(q["question"], k=K)
        except Exception as exc:
            chunks = []
            lines.append(f"[RETRIEVAL ERROR: {exc}]")

        lines.append(SEP)
        lines.append(f"QUESTION ID: {q['id']}")
        lines.append(f"CATEGORY: {q['category']}")
        lines.append(f"QUESTION: {q['question']}")
        if q.get("notes"):
            lines.append(f"NOTES: {q['notes']}")
        lines.append("")
        lines.append("RETRIEVED CHUNKS:")

        for i, c in enumerate(chunks, start=1):
            preview = c["text"][:400].replace("\n", " ")
            if len(c["text"]) > 400:
                preview += "..."
            lines.append(
                f"[{i}] score={c['score']:.3f} type={c['source_type']} source={c['source_name']}"
            )
            lines.append(f"    Section: {c['section_path']}")
            lines.append(f"    Chunk ID: {c['chunk_id']}")
            lines.append(f"    Preview: {preview}")
            lines.append("")

        lines.append(SEP)
        lines.append("")

    output = "\n".join(lines)
    DUMP_PATH.write_text(output, encoding="utf-8")
    console.print(f"Wrote {len(to_dump)} questions to {DUMP_PATH.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground truth labeler for eval/golden.yaml")
    parser.add_argument(
        "--dump-mode",
        action="store_true",
        help="Non-interactive: write retrieval results to eval/labeling_dump.txt",
    )
    args = parser.parse_args()

    if args.dump_mode:
        dump_mode()
        return

    console.rule("[bold blue]Ground Truth Labeler[/bold blue]", characters="-")

    retriever = Retriever()
    questions = load_questions()

    # Questions that need labeling: not should_abstain AND no ground truth yet
    to_label = [
        (i, q) for i, q in enumerate(questions)
        if not q.get("should_abstain", False)
        and not q.get("ground_truth_chunk_ids")
    ]

    console.print(
        f"[green]{len(to_label)}[/green] questions need labeling  |  "
        f"[dim]{sum(1 for q in questions if q.get('should_abstain'))} abstain (auto-skipped) | "
        f"{sum(1 for q in questions if q.get('ground_truth_chunk_ids'))} already labeled[/dim]"
    )
    console.print()

    reclassified = 0
    labeled_this_session = 0
    skipped = 0

    for progress_idx, (q_idx, q) in enumerate(to_label, start=1):
        display_question(q, progress_idx, len(to_label))

        # Retrieve
        try:
            chunks = retriever.retrieve(q["question"], k=K)
        except Exception as exc:
            console.print(f"[red]Retrieval failed: {exc}[/red]")
            continue

        display_chunks(chunks)

        while True:
            raw = prompt_user()

            if raw.lower() == "q":
                save_questions(questions)
                console.print("[green]Saved.[/green]")
                _print_summary(questions, labeled_this_session, reclassified, skipped)
                return

            if raw.lower() == "s":
                skipped += 1
                console.print("[dim]Skipped.[/dim]")
                break

            if raw.lower() == "n":
                questions[q_idx]["should_abstain"] = True
                questions[q_idx]["category"] = "should_abstain"
                questions[q_idx]["ground_truth_chunk_ids"] = []
                questions[q_idx]["notes"] = (
                    questions[q_idx].get("notes", "") +
                    " [reclassified as should_abstain during labeling]"
                ).strip()
                reclassified += 1
                console.print("[yellow]Reclassified as should_abstain.[/yellow]")
                break

            ranks = parse_ranks(raw)
            if not ranks:
                console.print("[red]Could not parse input. Try again.[/red]")
                continue

            invalid = [r for r in ranks if r < 1 or r > len(chunks)]
            if invalid:
                console.print(f"[red]Invalid ranks: {invalid}. Must be 1-{len(chunks)}.[/red]")
                continue

            selected_ids = [chunks[r - 1]["chunk_id"] for r in ranks]
            questions[q_idx]["ground_truth_chunk_ids"] = selected_ids
            labeled_this_session += 1
            console.print(
                f"[green]Saved {len(selected_ids)} chunk(s) for {q['id']}.[/green]"
            )
            break

        console.print()

    # All questions processed
    save_questions(questions)
    console.print("[green]All questions processed. Saved.[/green]")
    _print_summary(questions, labeled_this_session, reclassified, skipped)


def _print_summary(
    questions: list[dict],
    labeled: int,
    reclassified: int,
    skipped: int,
) -> None:
    console.rule("[bold]Session summary[/bold]", characters="-")
    total = len(questions)
    have_gt = sum(1 for q in questions if q.get("ground_truth_chunk_ids"))
    abstain = sum(1 for q in questions if q.get("should_abstain"))
    still_empty = total - have_gt - abstain

    console.print(f"  Total questions      : {total}")
    console.print(f"  Have ground truth    : [green]{have_gt}[/green]")
    console.print(f"  should_abstain       : {abstain}")
    console.print(f"  Still unlabeled      : [yellow]{still_empty}[/yellow]")
    console.print(f"  --- This session ---")
    console.print(f"  Labeled              : [green]{labeled}[/green]")
    console.print(f"  Reclassified abstain : [yellow]{reclassified}[/yellow]")
    console.print(f"  Skipped              : [dim]{skipped}[/dim]")
    console.print()


if __name__ == "__main__":
    main()
