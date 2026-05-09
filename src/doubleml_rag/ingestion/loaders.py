"""
loaders.py — Parse raw source files into section-level dicts.

Each returned dict has:
    text             : str   — cleaned section text
    source_type      : str   — "docs" | "paper" | "book"
    source_name      : str   — logical name for the source
    original_path    : str   — path relative to DATA_RAW root
    section_hierarchy: list[str] — headings from page title down
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATA_RAW = Path(__file__).resolve().parents[4] / "data" / "raw"


def _rel(path: Path) -> str:
    """Return path relative to data/raw/, using forward slashes."""
    try:
        return path.relative_to(_DATA_RAW).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# 1. RST loader
# ---------------------------------------------------------------------------

_RST_UNDERLINE_CHARS = set("=-~+#^*<>_")

# Sphinx directives whose entire indented body we discard
_STRIP_DIRECTIVES = re.compile(
    r"^\.\. (?:toctree|raw|image|figure|grid|tab-set|tab-item|"
    r"currentmodule|autosummary|autoclass|autofunction|automethod|"
    r"deprecated|versionadded|versionchanged|include|only|"
    r"rubric|contents|substitution-def|replace|"
    r"|\|[^|]+\|)\b",
    re.IGNORECASE,
)

# Inline RST roles → keep the display text / label
_INLINE_ROLE = re.compile(r":[a-z_]+:`([^`]*)`")
# Anonymous hyperlinks, bare URLs in angle brackets
_BARE_HYPERLINK = re.compile(r"<https?://[^>]+>")
# RST field list markers like ":param foo:"
_FIELD_LIST = re.compile(r"^:\w[^:]*:", re.MULTILINE)


def _is_underline(line: str, min_len: int = 2) -> bool:
    stripped = line.rstrip()
    return (
        len(stripped) >= min_len
        and len(set(stripped)) == 1
        and stripped[0] in _RST_UNDERLINE_CHARS
    )


def _clean_rst_text(text: str) -> str:
    """Light cleanup of RST markup, preserving prose and code."""
    # Inline roles: :ref:`Display Text <target>` → Display Text
    text = re.sub(r":[a-z_]+:`([^<`]+)\s*<[^>]+>`", r"\1", text)
    # Inline roles: :py:func:`name` → name
    text = re.sub(r":[a-z_:]+:`([^`]+)`", r"\1", text)
    # Bare angle-bracket URLs
    text = _BARE_HYPERLINK.sub("", text)
    # RST substitution references
    text = re.sub(r"\|[^|]+\|", "", text)
    # Trailing underscores on hyperlink targets: `link text`_
    text = re.sub(r"`([^`]+)`_+", r"\1", text)
    return text


def _strip_directive_blocks(lines: list[str]) -> list[str]:
    """Remove Sphinx directive blocks (directive + its indented body)."""
    out: list[str] = []
    skip_indent: str | None = None  # indentation level of the block being skipped

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # If we're inside a directive body, skip until dedent
        if skip_indent is not None:
            indent = len(line) - len(line.lstrip())
            if stripped == "" or indent > len(skip_indent):
                i += 1
                continue
            else:
                skip_indent = None

        # Detect start of a stripped directive
        if stripped.startswith(".. ") and _STRIP_DIRECTIVES.match(stripped):
            # capture the directive's indentation level
            skip_indent = line[: len(line) - len(line.lstrip())] + "  "
            i += 1
            continue

        # Detect .. math:: or .. note:: etc. that we KEEP (just strip the directive line)
        if stripped.startswith(".. math::") or stripped.startswith(".. note::") or \
           stripped.startswith(".. warning::") or stripped.startswith(".. code-block::"):
            # Skip only the directive header line; keep body
            i += 1
            continue

        out.append(line)
        i += 1

    return out


def _parse_rst_sections(text: str, filepath: Path) -> list[dict]:
    """Split an RST file into sections with heading hierarchy."""
    lines = text.splitlines()
    lines = _strip_directive_blocks(lines)

    # First pass: identify (line_index, heading_text, underline_char) triples.
    # RST heading styles:
    #   overline + text + underline  (all same char)
    #   text + underline
    headings: list[tuple[int, str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # overline style
        if (
            _is_underline(line)
            and i + 2 < len(lines)
            and _is_underline(lines[i + 2])
            and line.rstrip()[0] == lines[i + 2].rstrip()[0]
            and lines[i + 1].strip()
        ):
            headings.append((i + 1, lines[i + 1].strip(), line.rstrip()[0]))
            i += 3
            continue
        # underline style
        if (
            i + 1 < len(lines)
            and lines[i].strip()
            and _is_underline(lines[i + 1])
            and len(lines[i + 1].rstrip()) >= len(lines[i].rstrip()) - 2
        ):
            headings.append((i, lines[i].strip(), lines[i + 1].rstrip()[0]))
            i += 2
            continue
        i += 1

    if not headings:
        # No headings found — treat whole file as one section
        full_text = _clean_rst_text("\n".join(lines)).strip()
        if full_text:
            return [
                {
                    "text": full_text,
                    "source_type": "docs",
                    "source_name": "doubleml_docs",
                    "original_path": _rel(filepath),
                    "section_hierarchy": [filepath.stem],
                }
            ]
        return []

    # Map underline char → level (ordered by first appearance)
    char_to_level: dict[str, int] = {}
    for _, _, char in headings:
        if char not in char_to_level:
            char_to_level[char] = len(char_to_level) + 1

    # Build section list: each heading owns the lines from its heading line
    # (inclusive) up to the next heading line (exclusive).
    # heading_line_indices: set of line indices that are heading text lines
    heading_line_set = {ln for ln, _, _ in headings}
    # also skip the overline/underline rows themselves
    underline_rows: set[int] = set()
    for ln, _, _ in headings:
        if ln > 0 and _is_underline(lines[ln - 1]):
            underline_rows.add(ln - 1)
        if ln + 1 < len(lines) and _is_underline(lines[ln + 1]):
            underline_rows.add(ln + 1)

    # Build hierarchy stack
    results: list[dict] = []
    stack: list[tuple[int, str]] = []  # (level, heading_text)

    for idx, (line_idx, heading_text, char) in enumerate(headings):
        level = char_to_level[char]

        # Pop stack to current level
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading_text))
        hierarchy = [h for _, h in stack]

        # Collect body lines: from line_idx+2 (skip underline) to next heading start
        body_start = line_idx + 1
        if body_start < len(lines) and _is_underline(lines[body_start]):
            body_start += 1

        if idx + 1 < len(headings):
            next_heading_line = headings[idx + 1][0]
            # Also skip overline of next heading if present
            if next_heading_line > 0 and _is_underline(lines[next_heading_line - 1]):
                body_end = next_heading_line - 1
            else:
                body_end = next_heading_line
        else:
            body_end = len(lines)

        body_lines = [
            l for j, l in enumerate(lines[body_start:body_end], start=body_start)
            if j not in heading_line_set and j not in underline_rows
        ]
        body_text = _clean_rst_text("\n".join(body_lines)).strip()

        if body_text:
            results.append(
                {
                    "text": body_text,
                    "source_type": "docs",
                    "source_name": "doubleml_docs",
                    "original_path": _rel(filepath),
                    "section_hierarchy": hierarchy,
                }
            )

    return results


def load_rst_files(repo_doc_dir: Path) -> list[dict]:
    """Load all .rst files under repo_doc_dir and parse into sections."""
    rst_files = sorted(repo_doc_dir.rglob("*.rst"))
    docs: list[dict] = []
    for path in rst_files:
        # Skip Jinja/Sphinx template files — they contain {{ }} template syntax
        if "_templates" in path.parts or "_static" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            # Quick sanity check: skip files that look like Jinja templates
            if "{{" in text and "}}" in text and "{%" in text:
                continue
            sections = _parse_rst_sections(text, path)
            docs.extend(sections)
        except Exception as exc:
            print(f"[loaders] WARNING: skipping {path}: {exc}")
    return docs


# ---------------------------------------------------------------------------
# 2. PDF loader
# ---------------------------------------------------------------------------

# Numbered section pattern: "1.", "1.1", "A.", "A.1" at start of a line
_SECTION_NUM = re.compile(r"^(?:[A-Z]|\d+)(?:\.\d+)*\.?\s{1,4}[A-Z]")
# Very short ALL-CAPS line that looks like a heading (≥3 words, no terminal period)
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z\s,/\-:]{10,80}$")


def _is_pdf_heading(line: str, body_size: float, line_max_size: float) -> bool:
    """Return True if this line looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    if line_max_size > body_size + 1.5:
        return True
    if _SECTION_NUM.match(stripped):
        return True
    return False


def _get_line_max_size(page_chars: list[dict], y_top: float, y_tol: float = 2.0) -> float:
    """Return the max font size for characters near a given y position."""
    sizes = [
        c["size"]
        for c in page_chars
        if abs(c["top"] - y_top) <= y_tol and c["text"].strip()
    ]
    return max(sizes) if sizes else 0.0


def load_pdf_papers(papers_dir: Path) -> list[dict]:
    """Extract text from PDF papers, split by detected headings."""
    docs: list[dict] = []

    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        source_name = pdf_path.stem
        sections: list[dict] = []
        current_heading: list[str] = [source_name]
        current_lines: list[str] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Determine body font size: most common size rounded to 1 decimal
                size_counts: dict[float, int] = {}
                for page in pdf.pages[:5]:
                    for c in page.chars:
                        s = round(c["size"], 1)
                        size_counts[s] = size_counts.get(s, 0) + 1
                body_size = max(size_counts, key=lambda k: size_counts[k])

                for page in pdf.pages:
                    # Build line → max_font_size map using char positions
                    # Group chars by rounded top coordinate
                    lines_by_top: dict[float, list[dict]] = {}
                    for c in page.chars:
                        top = round(c["top"], 1)
                        lines_by_top.setdefault(top, []).append(c)

                    page_text = page.extract_text(x_tolerance=2, y_tolerance=3)
                    if not page_text:
                        continue

                    for raw_line in page_text.splitlines():
                        stripped = raw_line.strip()
                        if not stripped:
                            current_lines.append("")
                            continue

                        # Find the max font size for chars matching this line's text
                        # Approximate: look for the first top-coord whose chars produce this text
                        line_max_size = 0.0
                        for top_key, chars in lines_by_top.items():
                            line_text = "".join(c["text"] for c in chars).strip()
                            if line_text and stripped.startswith(line_text[:10]):
                                line_max_size = max(c["size"] for c in chars if c["text"].strip())
                                break

                        if _is_pdf_heading(stripped, body_size, line_max_size):
                            # Save previous section
                            body = "\n".join(current_lines).strip()
                            if body:
                                sections.append(
                                    {
                                        "text": body,
                                        "source_type": "paper",
                                        "source_name": source_name,
                                        "original_path": _rel(pdf_path),
                                        "section_hierarchy": list(current_heading),
                                    }
                                )
                            current_heading = [source_name, stripped]
                            current_lines = []
                        else:
                            current_lines.append(raw_line)

                # Flush last section
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append(
                        {
                            "text": body,
                            "source_type": "paper",
                            "source_name": source_name,
                            "original_path": _rel(pdf_path),
                            "section_hierarchy": list(current_heading),
                        }
                    )

        except Exception as exc:
            print(f"[loaders] WARNING: could not parse {pdf_path}: {exc}")
            continue

        docs.extend(sections)

    return docs


# ---------------------------------------------------------------------------
# 3. Book HTML loader
# ---------------------------------------------------------------------------

_DISCARD_TAGS = {"script", "style", "nav", "header", "footer", "aside"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4"}


def _node_to_text(node: "Tag | NavigableString") -> str:
    """Recursively extract clean, readable text from an HTML node."""
    if isinstance(node, NavigableString):
        return str(node)

    if node.name in _DISCARD_TAGS:
        return ""

    # Code block → preserve verbatim
    if node.name in {"pre", "code"}:
        return node.get_text()

    # Math span → keep the LaTeX source
    if node.name == "span" and "math" in (node.get("class") or []):
        return node.get_text()

    # Images → note alt text
    if node.name == "img":
        alt = node.get("alt", "").strip()
        return f"[image: {alt}]" if alt else ""

    parts = [_node_to_text(child) for child in node.children]
    sep = "\n" if node.name in {"p", "li", "div", "blockquote", "td", "th", "section"} else ""
    return sep.join(parts)


def _extract_html_sections(
    element: Tag,
    parent_hierarchy: list[str] | None = None,
) -> list[tuple[list[str], str]]:
    """
    Recursively walk <section> elements.

    Jupyter Book renders each heading inside its own <section> element:
        <section id="why-bother">
          <h2>Why Bother?</h2>
          <p>...</p>
          <section id="...">...</section>
        </section>

    We extract the heading → build the hierarchy → collect prose children
    (everything except nested <section> tags), then recurse.
    """
    if parent_hierarchy is None:
        parent_hierarchy = []

    sections: list[tuple[list[str], str]] = []

    for child in element.children:
        if not isinstance(child, Tag):
            continue
        if child.name in _DISCARD_TAGS:
            continue

        if child.name == "section":
            # Find the heading inside this section (direct child only)
            heading_tag = next(
                (t for t in child.children if isinstance(t, Tag) and t.name in _HEADING_TAGS),
                None,
            )
            if heading_tag:
                heading_text = heading_tag.get_text(strip=True).rstrip("#").strip()
            else:
                heading_text = child.get("id", "section").replace("-", " ").title()

            hierarchy = parent_hierarchy + [heading_text]

            # Collect direct prose children (not nested sections, not the heading itself)
            prose_parts: list[str] = []
            for sub in child.children:
                if not isinstance(sub, Tag):
                    if isinstance(sub, NavigableString) and str(sub).strip():
                        prose_parts.append(str(sub).strip())
                    continue
                if sub.name == "section":
                    continue  # handled by recursion
                if sub is heading_tag:
                    continue  # already captured as hierarchy label
                if sub.name in _DISCARD_TAGS:
                    continue
                text = _node_to_text(sub).strip()
                if text:
                    prose_parts.append(text)

            prose = "\n\n".join(prose_parts).strip()
            if prose:
                sections.append((hierarchy, prose))

            # Recurse into nested sections
            sections.extend(_extract_html_sections(child, hierarchy))

        elif child.name in {"div", "article", "main"}:
            # Transparent container — pass through without adding to hierarchy
            sections.extend(_extract_html_sections(child, parent_hierarchy))

    return sections


def load_book_html(book_dir: Path) -> list[dict]:
    """Parse each chapter HTML and split into h1/h2/h3 sections."""
    docs: list[dict] = []

    for html_path in sorted(book_dir.glob("*.html")):
        try:
            html = html_path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")

            main = soup.find("main", id="main-content") or soup.find("main")
            if main is None:
                continue

            for section_hierarchy, text in _extract_html_sections(main, []):
                docs.append(
                    {
                        "text": text,
                        "source_type": "book",
                        "source_name": "causal_inference_book",
                        "original_path": _rel(html_path),
                        "section_hierarchy": section_hierarchy,
                    }
                )
        except Exception as exc:
            print(f"[loaders] WARNING: skipping {html_path}: {exc}")

    return docs
