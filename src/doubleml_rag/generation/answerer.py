"""
answerer.py — Generate grounded, cited answers using Claude.
"""

from __future__ import annotations

import os
import re

import anthropic

_SYSTEM_PROMPT = """\
You are a precise research assistant answering questions about causal inference \
and the DoubleML framework. You have been given a numbered list of context chunks \
retrieved from primary sources (papers, documentation, and a textbook).

Rules you must follow without exception:
1. Answer using ONLY the information in the provided context chunks. Do not draw on \
outside knowledge of DoubleML, causal inference, statistics, machine learning, or \
any other topic.
2. Every factual claim must end with a citation marker referencing the chunk(s) that \
support it, e.g. [1], [3], or [1, 4]. If a sentence draws on multiple chunks, cite \
all of them.
3. If the context does not contain sufficient information to answer the question, \
respond with exactly: "The provided context does not contain enough information to \
answer this question." Do not guess, infer, or supplement with general knowledge.
4. Be concise and direct. No preamble ("Great question!", "Based on the context...", \
"Certainly!"), no filler, no summaries of what you are about to do. Start with \
the answer.
5. If multiple chunks support the same claim, cite all of them.\
"""

# Phrases that indicate the model declined to answer
_ABSTAIN_PATTERNS = re.compile(
    r"does not contain enough information"
    r"|cannot answer"
    r"|can't answer"
    r"|unable to answer"
    r"|not enough information"
    r"|insufficient (context|information)",
    re.IGNORECASE,
)

# Extract [N] or [N, M, ...] citation markers
_CITATION_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def _parse_citations(text: str) -> list[int]:
    found: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            try:
                found.add(int(part.strip()))
            except ValueError:
                pass
    return sorted(found)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source_name", "unknown")
        section = chunk.get("section_path", "")
        text = chunk.get("text", "")
        parts.append(f"[{i}] Source: {source} | Section: {section}\n{text}")
    return "\n\n".join(parts)


class Answerer:
    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    def answer(self, query: str, retrieved_chunks: list[dict]) -> dict:
        context_block = _format_context(retrieved_chunks)
        user_message = (
            f"Context chunks:\n\n{context_block}\n\n"
            f"---\n\nQuestion: {query}"
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text
        citations = _parse_citations(raw)
        abstained = bool(_ABSTAIN_PATTERNS.search(raw))

        return {
            "answer": raw,
            "citations_used": citations,
            "abstained": abstained,
            "raw_response": raw,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
