"""
embedder.py — Embed chunks via Voyage AI.

Adds an "embedding" field (list[float]) to each chunk dict.

Free-tier limits: 3 RPM, 10K TPM.
Uses token-aware batching + a sliding-window rate limiter so we never
exceed either limit.
"""

from __future__ import annotations

import os
import time
from collections import deque

import voyageai

_DEFAULT_MODEL = "voyage-3"

# Rate-limit guard-rails.
# Default values are tuned for the paid tier (high RPM/TPM).
# The free tier has 3 RPM / 10K TPM — if you hit those limits, lower
# _MAX_TOKENS_PER_REQUEST to 6_000 and _TPM_BUDGET to 9_000.
_TOKEN_SAFETY_FACTOR = 1.5   # tiktoken undercounts vs Voyage's tokenizer
_RPM_LIMIT = 300             # paid tier: effectively unlimited for our workload
_TPM_BUDGET = 4_000_000      # paid tier: very high — won't be the bottleneck
_MAX_TOKENS_PER_REQUEST = 80_000   # Voyage hard limit is ~120K
_WINDOW_SECONDS = 62
_RATE_LIMIT_RETRY_SLEEP = 30  # seconds to sleep on an unexpected RateLimitError


def _build_token_batches(chunks: list[dict]) -> list[list[dict]]:
    """
    Group chunks into batches where each batch stays under
    _MAX_TOKENS_PER_REQUEST tokens.
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for chunk in chunks:
        t = chunk["token_count"]
        if current and current_tokens + t > _MAX_TOKENS_PER_REQUEST:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += t

    if current:
        batches.append(current)

    return batches


class _RateLimiter:
    """Sliding-window rate limiter that enforces RPM and TPM budgets."""

    def __init__(self) -> None:
        # deque of (monotonic_time, tokens_used)
        self._log: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self._log and self._log[0][0] < now - _WINDOW_SECONDS:
            self._log.popleft()

    def wait_if_needed(self, tokens: int) -> None:
        # Apply safety factor to account for Voyage tokenizer being larger
        budgeted_tokens = int(tokens * _TOKEN_SAFETY_FACTOR)
        while True:
            now = time.monotonic()
            self._prune(now)

            tokens_in_window = sum(t for _, t in self._log)
            requests_in_window = len(self._log)

            rpm_ok = requests_in_window < _RPM_LIMIT
            tpm_ok = tokens_in_window + budgeted_tokens <= _TPM_BUDGET

            if rpm_ok and tpm_ok:
                break

            # Calculate how long to wait until the oldest entry falls out
            if self._log:
                oldest_ts = self._log[0][0]
                sleep_s = max(1.0, (_WINDOW_SECONDS - (now - oldest_ts)) + 2)
            else:
                sleep_s = 5.0

            print(
                f"  [rate limit] rpm={requests_in_window}/{_RPM_LIMIT}, "
                f"tpm~{tokens_in_window}/{_TPM_BUDGET} (budget={budgeted_tokens}) "
                f"— sleeping {sleep_s:.0f}s..."
            )
            time.sleep(sleep_s)

    def record(self, tokens: int) -> None:
        self._log.append((time.monotonic(), tokens))


def embed_chunks(
    chunks: list[dict],
    model: str = _DEFAULT_MODEL,
    batch_size: int = 0,  # ignored — we use token-aware batching
) -> list[dict]:
    """
    Embed each chunk's text with Voyage AI and attach the vector.

    Returns the same list with "embedding" added in-place.
    Automatically respects free-tier 3 RPM / 10K TPM limits.
    """
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    vo = voyageai.Client(api_key=api_key)

    batches = _build_token_batches(chunks)
    total_batches = len(batches)
    limiter = _RateLimiter()

    all_embeddings: list[list[float]] = []
    chunks_done = 0

    for batch_num, batch_chunks in enumerate(batches):
        texts = [c["text"] for c in batch_chunks]
        batch_tokens = sum(c["token_count"] for c in batch_chunks)

        limiter.wait_if_needed(batch_tokens)

        # Retry up to 3 times on RateLimitError with exponential backoff
        for attempt in range(3):
            try:
                result = vo.embed(texts, model=model, input_type="document")
                break
            except voyageai.error.RateLimitError:
                sleep_s = _RATE_LIMIT_RETRY_SLEEP * (2 ** attempt)
                print(
                    f"  [RateLimitError] unexpected server-side throttle "
                    f"— backing off {sleep_s}s (attempt {attempt + 1}/3)..."
                )
                time.sleep(sleep_s)
        else:
            raise RuntimeError(
                f"Batch {batch_num + 1} failed after 3 retries due to rate limiting. "
                "Consider adding VOYAGE_API_KEY payment method to unlock higher limits."
            )

        all_embeddings.extend(result.embeddings)
        limiter.record(batch_tokens)

        chunks_done += len(batch_chunks)
        print(
            f"Embedded batch {batch_num + 1}/{total_batches} "
            f"({chunks_done}/{len(chunks)} chunks, ~{batch_tokens} tokens)"
        )

    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    return chunks
