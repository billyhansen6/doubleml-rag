"""
metrics.py -- Retrieval evaluation metrics: recall@k and MRR.
"""

from __future__ import annotations


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
) -> dict:
    """
    Compute recall@k and MRR for a single question.

    Parameters
    ----------
    retrieved_ids    : ordered list of chunk_ids returned by the retriever
    ground_truth_ids : list of correct chunk_ids from golden.yaml

    Returns
    -------
    dict with keys: recall_at_k, mrr, k, num_relevant
    """
    if not ground_truth_ids:
        return {"recall_at_k": None, "mrr": None, "k": len(retrieved_ids), "num_relevant": 0}

    gt_set = set(ground_truth_ids)
    k = len(retrieved_ids)

    # Recall@k: fraction of ground-truth chunks found in top-k results
    hits = sum(1 for cid in retrieved_ids if cid in gt_set)
    recall_at_k = hits / len(gt_set)

    # MRR: reciprocal rank of the first relevant chunk
    mrr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gt_set:
            mrr = 1.0 / rank
            break

    return {
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "k": k,
        "num_relevant": len(gt_set),
    }
