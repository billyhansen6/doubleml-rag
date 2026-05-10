"""
judges.py -- LLM-as-judge functions using claude-haiku-4-5.

Three judges:
  judge_faithfulness   -- does the answer stay grounded in the retrieved chunks?
  judge_answer_quality -- how well does the answer address the question?
  judge_abstention     -- did the model correctly abstain (or not)?
"""

from __future__ import annotations

import json
import os
import re

import anthropic

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 512


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def _call(system: str, user: str) -> str:
    resp = _client().messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def _parse_json(raw: str) -> dict:
    """Extract the first JSON object from the response, tolerating markdown fences."""
    # Strip ```json ... ``` fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    # Find first {...}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: try parsing the whole thing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "parse_failed", "raw": raw[:200]}


_VALID_FAITH_SCORES = {0.0, 0.5, 1.0}


def _coerce_faith_score(value) -> float:
    """Coerce any numeric value to the nearest valid faithfulness score."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v in _VALID_FAITH_SCORES:
        return v
    # Coerce to nearest valid value
    return min(_VALID_FAITH_SCORES, key=lambda s: abs(s - v))


# ---------------------------------------------------------------------------
# Judge 1: Faithfulness
# ---------------------------------------------------------------------------
_FAITHFULNESS_SYSTEM = """\
Determine whether every factual claim in the ANSWER is supported by the RETRIEVED CHUNKS \
provided. Score on a strict 3-point scale.

Score anchors:
1.0 — Every factual claim in the answer is directly supported by content in the retrieved \
chunks. Restating, paraphrasing, or organizing chunk content is fine.
0.5 — Most claims are supported but the answer contains 1-2 minor extrapolations beyond \
what the chunks strictly say, OR the answer combines chunks in a way that goes slightly \
beyond either source individually.
0.0 — Significant claims in the answer are not supported by any retrieved chunk. The answer \
relies on outside knowledge.

Respond ONLY with a JSON object — no markdown fences, no extra text:
{"faithfulness_score": <0.0 | 0.5 | 1.0>, "unsupported_claims": [<list of strings, each a \
specific claim from the answer NOT supported by chunks; empty list if score is 1.0>], \
"reasoning": "<2-3 sentence explanation of the scoring decision>"}\
"""


def judge_faithfulness(question: str, context_chunks: list[dict], answer: str) -> dict:
    context_text = "\n\n".join(
        f"[{i}] {c.get('source_name', '')} | {c.get('section_path', '')}\n{c.get('text', '')}"
        for i, c in enumerate(context_chunks, start=1)
    )
    user_msg = (
        f"RETRIEVED CHUNKS:\n{context_text}\n\n"
        f"ANSWER:\n{answer}"
    )
    raw = _call(_FAITHFULNESS_SYSTEM, user_msg)

    parsed = _parse_json(raw)

    if "error" in parsed:
        return {
            "judge": "faithfulness",
            "faithfulness_score": None,
            "unsupported_claims": [],
            "reasoning": "parse_error",
            "raw": raw[:300],
        }

    score = _coerce_faith_score(parsed.get("faithfulness_score"))
    return {
        "judge": "faithfulness",
        "faithfulness_score": score,
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "reasoning": parsed.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Judge 2: Answer Quality
# ---------------------------------------------------------------------------
_QUALITY_SYSTEM = """\
You are an impartial judge evaluating the quality of an answer to a question about
causal inference and the DoubleML framework.

Evaluate on: correctness, completeness, and clarity — but ONLY relative to what the
provided context can support. Do not penalise the answer for omitting information that
is not in the context.

Respond ONLY with a JSON object — no markdown fences, no extra text:
{"score": <integer 1-5>, "reason": "<one sentence>"}

Scoring rubric:
5 = Excellent: accurate, complete given context, clearly written.
4 = Good: accurate and mostly complete, minor gaps or verbosity.
3 = Acceptable: addresses the question but with notable gaps or imprecision.
2 = Poor: misses key aspects or is confusing.
1 = Unacceptable: wrong, irrelevant, or not an answer.\
"""


def judge_answer_quality(question: str, context_chunks: list[dict], answer: str) -> dict:
    context_text = "\n\n".join(
        f"[{i}] {c.get('source_name', '')} | {c.get('section_path', '')}\n{c.get('text', '')[:400]}"
        for i, c in enumerate(context_chunks, start=1)
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"Context chunks:\n{context_text}\n\n"
        f"Answer to evaluate:\n{answer}"
    )
    raw = _call(_QUALITY_SYSTEM, user_msg)
    result = _parse_json(raw)
    result["judge"] = "answer_quality"
    return result


# ---------------------------------------------------------------------------
# Judge 3: Abstention correctness
# ---------------------------------------------------------------------------
_ABSTENTION_SYSTEM = """\
You are an impartial judge evaluating whether a RAG system correctly decided to
answer or abstain on a given question.

You will be told the EXPECTED behaviour (should_abstain: true/false) and what the
system actually did.

Respond ONLY with a JSON object — no markdown fences, no extra text:
{"correct": <true|false>, "label": "<answered|abstained>", "reason": "<one sentence>"}

"label" reflects what the system DID (not what it should have done).\
"""


def judge_abstention(question: str, answer: str, system_abstained: bool, should_abstain: bool) -> dict:
    user_msg = (
        f"Question: {question}\n\n"
        f"Expected behaviour: {'ABSTAIN (do not answer)' if should_abstain else 'ANSWER the question'}\n"
        f"System response:\n{answer}\n\n"
        f"Did the system abstain? {'YES' if system_abstained else 'NO'}"
    )
    raw = _call(_ABSTENTION_SYSTEM, user_msg)
    result = _parse_json(raw)
    result["judge"] = "abstention"
    result["should_abstain"] = should_abstain
    result["system_abstained"] = system_abstained
    # Deterministic correctness (don't rely on the judge for this)
    result["correct"] = (system_abstained == should_abstain)
    return result
