"""检索评测指标：AP / mAP / Recall@k。"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Set, Tuple


def average_precision(ranked_ids: List[str], relevant: Set[str]) -> float:
    """标准 Average Precision；无相关项返回 0。"""
    if not relevant:
        return 0.0
    hits = 0
    ap = 0.0
    n = len(relevant)
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant:
            hits += 1
            ap += hits / i
    return ap / n


def recall_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def mean_average_precision(pairs: List[Tuple[List[str], Set[str]]]) -> float:
    aps = [average_precision(r, rel) for r, rel in pairs]
    return float(np.mean(aps)) if aps else 0.0
