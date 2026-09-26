"""检索索引：为每个描述子维护一个归一化向量矩阵，余弦相似度即点积。"""
from __future__ import annotations

import numpy as np
from typing import Dict, List

from ..core.config import Config
from ..core.types import ImageRecord
from ..domain.descriptors import build_descriptors


class ForgeIndex:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.descriptors = build_descriptors(config)
        self.ids: List[str] = []
        self.categories: Dict[str, str] = {}
        self.vectors: Dict[str, np.ndarray] = {d.name: [] for d in self.descriptors}

    def build(self, records: List[ImageRecord]) -> "ForgeIndex":
        for rec in records:
            self.ids.append(rec.id)
            if rec.category is not None:
                self.categories[rec.id] = rec.category
            for d in self.descriptors:
                self.vectors[d.name].append(d.extract(rec.image))
        for name in list(self.vectors.keys()):
            if self.vectors[name]:
                self.vectors[name] = np.stack(self.vectors[name])
            else:
                self.vectors[name] = np.zeros((0, 1), dtype=np.float32)
        return self

    def per_descriptor_similarity(self, vec_by_name: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, vec in vec_by_name.items():
            M = self.vectors.get(name)
            if M is None or M.shape[0] == 0:
                continue
            out[name] = M @ vec  # 向量已 L2 归一化 => 余弦相似度
        return out

    def memory_mb(self) -> float:
        total = sum(v.nbytes for v in self.vectors.values())
        return total / 1e6
