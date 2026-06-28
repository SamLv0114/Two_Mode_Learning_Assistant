"""
Offline ranking evaluation metrics
"""
from typing import List, Optional
import numpy as np


def _dcg(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    gains = (2 ** relevances - 1)
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(gains / discounts))


def compute_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    group_sizes: Optional[List[int]] = None
) -> float:
    """Compute mean NDCG@k; if group_sizes provided, average across groups."""
    if y_true.size == 0:
        return 0.0

    def ndcg_for_group(truth: np.ndarray, pred: np.ndarray) -> float:
        order = np.argsort(-pred)
        top_truth = truth[order][:k]
        ideal = np.sort(truth)[::-1][:k]
        denom = _dcg(ideal)
        return _dcg(top_truth) / denom if denom > 0 else 0.0

    if not group_sizes:
        return ndcg_for_group(y_true, y_pred)

    scores = []
    idx = 0
    for size in group_sizes:
        if size <= 0:
            continue
        truth = y_true[idx:idx + size]
        pred = y_pred[idx:idx + size]
        scores.append(ndcg_for_group(truth, pred))
        idx += size
    return float(np.mean(scores)) if scores else 0.0


def compute_mrr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_sizes: Optional[List[int]] = None
) -> float:
    """Compute mean reciprocal rank for relevance > 0."""
    if y_true.size == 0:
        return 0.0

    def mrr_for_group(truth: np.ndarray, pred: np.ndarray) -> float:
        order = np.argsort(-pred)
        ranked_truth = truth[order]
        for idx, rel in enumerate(ranked_truth, start=1):
            if rel > 0:
                return 1.0 / idx
        return 0.0

    if not group_sizes:
        return mrr_for_group(y_true, y_pred)

    scores = []
    idx = 0
    for size in group_sizes:
        if size <= 0:
            continue
        truth = y_true[idx:idx + size]
        pred = y_pred[idx:idx + size]
        scores.append(mrr_for_group(truth, pred))
        idx += size
    return float(np.mean(scores)) if scores else 0.0
