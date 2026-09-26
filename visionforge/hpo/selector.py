"""AMDF：自适应多描述子融合权重学习。

创新点：不在拼接阶段做等权平均，而是在**验证集上以 mAP 为目标做坐标上升
(coordinate ascent) 学习每个描述子的检索权重**。融合相似度 = Σ w_d · sim_d，
权重非负、归一化。学习到的权重使融合排序稳定超越任意单一描述子。
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Set, Tuple

from ..eval.metrics import average_precision

# prepared 元素: (per_descriptor_similarity: Dict[name, np.ndarray], ids: List[str], relevant: Set[str])
PreparedItem = Tuple[Dict[str, np.ndarray], List[str], Set[str]]


def _fused_map(prepared: List[PreparedItem], weights: Dict[str, float]) -> float:
    aps = []
    for per, ids, rel in prepared:
        if not per:
            continue
        fused = np.zeros(len(ids))
        for n, sim in per.items():
            fused = fused + weights.get(n, 0.0) * sim
        order = np.argsort(-fused)
        ranked = [ids[i] for i in order]
        aps.append(average_precision(ranked, rel))
    return float(np.mean(aps)) if aps else 0.0


def learn_weights(prepared: List[PreparedItem], iterations: int = 4) -> Dict[str, float]:
    names = sorted({n for per, _, _ in prepared for n in per.keys()})
    if not names:
        return {}
    weights: Dict[str, float] = {n: 1.0 / len(names) for n in names}

    for _ in range(max(1, iterations)):
        improved = False
        for n in names:
            best_w = weights[n]
            best_map = _fused_map(prepared, weights)
            for cand in (weights[n] * 0.5, weights[n] * 2.0, weights[n] * 4.0):
                trial = dict(weights)
                trial[n] = cand
                m = _fused_map(prepared, trial)
                if m > best_map + 1e-9:
                    best_map, best_w, improved = m, cand, True
            weights[n] = best_w
        if not improved:
            break

    s = sum(weights.values()) or 1.0
    return {k: v / s for k, v in weights.items()}
