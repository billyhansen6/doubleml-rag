"""
collect_corpus.py — Acquisition-only script for doubleml-rag.

Downloads all source materials into data/raw/ in clean, parseable formats.
No parsing, chunking, or embedding logic here.

Usage:
    uv run python scripts/collect_corpus.py
"""

import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from git import Repo
from rich.console import Console
from rich.table import Table

console = Console(highlight=False)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"

PAPERS = [
    {
        "url": "https://arxiv.org/pdf/1608.00060",
        "filename": "chernozhukov_2018_dml.pdf",
        "label": "Chernozhukov et al. 2018 (DML)",
    },
    {
        "url": "https://arxiv.org/pdf/2104.03220",
        "filename": "bach_2022_doubleml_python.pdf",
        "label": "Bach et al. 2022 (DoubleML Python)",
    },
]

BOOK_URL = "https://matheusfacure.github.io/python-causality-handbook/"
BOOK_DIR = DATA_RAW / "book"
PAPERS_DIR = DATA_RAW / "papers"
DOCS_REPO_DIR = DATA_RAW / "doubleml_docs" / "repo"
DOCS_CLONE_URL = "https://github.com/DoubleML/doubleml-docs"

HEADERS = {"User-Agent": "doubleml-rag-research/0.1"}


# ---------------------------------------------------------------------------
# 1. DoubleML documentation
# ---------------------------------------------------------------------------

def collect_doubleml_docs() -> tuple[int, float]:
    """Clone doubleml-docs (shallow) and count .rst files in doc/."""
    DOCS_REPO_DIR.mkdir(parents=True, exist_ok=True)

    if (DOCS_REPO_DIR / ".git").exists():
        console.print("[yellow]docs:[/yellow] repo already present - skipping clone")
    else:
        console.print("[cyan]docs:[/cyan] cloning doubleml-docs (depth=1)...")
        Repo.clone_from(DOCS_CLONE_URL, DOCS_REPO_DIR, depth=1)
        console.print("[green]docs:[/green] clone complete")

    doc_dir = DOCS_REPO_DIR / "doc"
    rst_files = list(doc_dir.rglob("*.rst")) if doc_dir.exists() else []
    console.print(f"[green]docs:[/green] found {len(rst_files)} .rst files in doc/")

    total_bytes = sum(f.stat().st_size for f in rst_files)
    return len(rst_files), float(total_bytes)


# ---------------------------------------------------------------------------
# 2. Foundational papers
# ---------------------------------------------------------------------------

def collect_papers() -> tuple[int, float]:
    """Download arXiv PDFs; skip if already present; verify >100 KB."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    total_bytes = 0

    for paper in PAPERS:
        dest = PAPERS_DIR / paper["filename"]

        if dest.exists():
            console.print(f"[yellow]papers:[/yellow] {paper['filename']} already present - skipping")
            total_bytes += dest.stat().st_size
            downloaded += 1
            continue

        console.print(f"[cyan]papers:[/cyan] downloading {paper['label']}...")
        try:
            resp = requests.get(paper["url"], headers=HEADERS, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)

            size = dest.stat().st_size
            if size < 100_000:
                dest.unlink()
                raise ValueError(
                    f"{paper['filename']} is only {size} bytes — expected a real PDF (>100 KB)"
                )

            console.print(
                f"[green]papers:[/green] saved {paper['filename']} ({size / 1_048_576:.2f} MB)"
            )
            total_bytes += size
            downloaded += 1

        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]papers:[/red] failed to download {paper['filename']}: {exc}")

    return downloaded, float(total_bytes)


# ---------------------------------------------------------------------------
# 3. Causal Inference for the Brave and True
# ---------------------------------------------------------------------------

def _extract_chapter_urls(html: str, base_url: str) -> list[str]:
    """Parse landing-page.html and return absolute chapter URLs in order.

    The book uses a simple flat structure: all chapter links are relative .html
    hrefs on the landing page. We collect every link that ends in .html and is
    not a meta/index page, preserving document order.
    """
    soup = BeautifulSoup(html, "html.parser")
    base = base_url.rstrip("/") + "/"

    # Pages to exclude from chapter list
    exclude = {"index.html", "landing-page.html"}

    chapter_urls: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Only relative .html links that aren't anchors or external
        if not href.endswith(".html") or href.startswith("http") or href.startswith("#"):
            continue
        if href in exclude or href in seen:
            continue
        seen.add(href)
        chapter_urls.append(base + href)

    return chapter_urls


def collect_book() -> tuple[int, float]:
    """Download Causal Inference for the Brave and True HTML chapters."""
    BOOK_DIR.mkdir(parents=True, exist_ok=True)

    # The root URL is a meta-refresh redirect; the real ToC is on landing-page.html
    landing_url = BOOK_URL.rstrip("/") + "/landing-page.html"
    console.print("[cyan]book:[/cyan] fetching landing page...")
    try:
        resp = requests.get(landing_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]book:[/red] could not fetch landing page: {exc}")
        return 0, 0.0

    chapter_urls = _extract_chapter_urls(resp.text, BOOK_URL)
    console.print(f"[cyan]book:[/cyan] extracted {len(chapter_urls)} chapter URLs from ToC")

    if len(chapter_urls) < 15 or len(chapter_urls) > 40:
        console.print(
            f"[bold yellow]book WARNING:[/bold yellow] expected 15-40 chapters, "
            f"got {len(chapter_urls)} - check URL extraction logic"
        )

    total_bytes = 0
    saved = 0

    for idx, url in enumerate(chapter_urls, start=1):
        slug = url.rstrip("/").split("/")[-1].removesuffix(".html")
        filename = f"{idx:02d}-{slug}.html"
        dest = BOOK_DIR / filename

        if dest.exists():
            console.print(f"[yellow]book:[/yellow] {filename} already present - skipping")
            total_bytes += dest.stat().st_size
            saved += 1
            continue

        console.print(f"[cyan]book:[/cyan] downloading chapter {idx:02d}: {slug}...")
        try:
            time.sleep(1)
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            total_bytes += dest.stat().st_size
            saved += 1
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]book:[/red] failed {url}: {exc}")

    console.print(f"[green]book:[/green] {saved} chapters saved")
    return saved, float(total_bytes)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt_mb(total_bytes: float) -> str:
    return f"{total_bytes / 1_048_576:.2f} MB"


def print_summary(results: dict[str, tuple[int, float]]) -> None:
    table = Table(title="Corpus acquisition summary", show_lines=True)
    table.add_column("Source", style="bold")
    table.add_column("Files", justify="right")
    table.add_column("Size on disk", justify="right")

    grand_files = 0
    grand_bytes = 0.0
    for name, (count, nbytes) in results.items():
        table.add_row(name, str(count), _fmt_mb(nbytes))
        grand_files += count
        grand_bytes += nbytes

    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", str(grand_files), _fmt_mb(grand_bytes))
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold blue]doubleml-rag corpus acquisition[/bold blue]", characters="-")

    console.rule("[dim]1 / 3  DoubleML docs[/dim]", characters="-")
    docs_count, docs_bytes = collect_doubleml_docs()

    console.rule("[dim]2 / 3  Foundational papers[/dim]", characters="-")
    papers_count, papers_bytes = collect_papers()

    console.rule("[dim]3 / 3  Causal Inference book[/dim]", characters="-")
    book_count, book_bytes = collect_book()

    console.rule(characters="-")
    print_summary(
        {
            "DoubleML docs (.rst)": (docs_count, docs_bytes),
            "arXiv papers (.pdf)": (papers_count, papers_bytes),
            "CI Brave & True (.html)": (book_count, book_bytes),
        }
    )


if __name__ == "__main__":
    main()
