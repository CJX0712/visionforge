"""VisionForgePipeline：构建索引 + AMDF 权重学习 + 检索 + 基准评测。"""
from __future__ import annotations

import json
import platform
import time
from typing import Dict, List, Optional

import numpy as np

from ..core.config import Config
from ..core.errors import PipelineError
from ..core.types import (
    BenchmarkResult,
    ImageRecord,
    QueryResult,
    RankedMatch,
)
from ..data.synthetic import generate_dataset
from ..domain.descriptors import build_descriptors
from ..eval.metrics import average_precision, recall_at_k
from ..hpo.selector import learn_weights
from ..training.index import ForgeIndex


class VisionForgePipeline:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.descriptors = build_descriptors(self.config)
        if not self.descriptors:
            raise PipelineError("无可用描述子（numpy 缺失？）", code="E500")
        self.index: Optional[ForgeIndex] = None
        self.weights: Dict[str, float] = {
            d.name: 1.0 / len(self.descriptors) for d in self.descriptors
        }

    def _extract(self, image) -> Dict[str, np.ndarray]:
        return {d.name: d.extract(image) for d in self.descriptors}

    def _fuse(self, per: Dict[str, np.ndarray]) -> np.ndarray:
        """AMDF 自适应多描述子融合：融合相似度 = Σ w_d · sim_d。

        权重 w_d 由验证集上以 mAP 为目标的坐标上升学习得到（非负、归一化），
        自动放大高区分度描述子、抑制噪声描述子 —— 使融合排序稳定优于单描述子拼接。
        """
        fused = np.zeros(len(self.index.ids))
        for n, sim in per.items():
            fused = fused + self.weights.get(n, 0.0) * sim
        return fused

    def build(self, records: List[ImageRecord], learn: bool = True, val_fraction: float = 0.3):
        # 阶段 A：在训练画廊上学习 AMDF 权重
        if learn and len(records) >= 4:
            rng = np.random.default_rng(self.config.random_state)
            order = np.arange(len(records))
            rng.shuffle(order)
            n_val = max(1, int(len(records) * val_fraction))
            val_set = set(order[:n_val].tolist())
            gallery_a = [r for i, r in enumerate(records) if i not in val_set]
            val_recs = [r for i, r in enumerate(records) if i in val_set]
            idx_a = ForgeIndex(self.config).build(gallery_a)
            prepared = []
            for r in val_recs:
                rel = {g.id for g in gallery_a if g.category == r.category}
                if rel:
                    per = idx_a.per_descriptor_similarity(self._extract(r.image))
                    prepared.append((per, idx_a.ids, rel))
            if prepared:
                self.weights = learn_weights(prepared, iterations=self.config.fusion_iterations)
        # 阶段 B：在全部数据上建最终索引
        self.index = ForgeIndex(self.config).build(records)
        return self

    def query(self, image, k: Optional[int] = None) -> QueryResult:
        if self.index is None:
            raise PipelineError("请先调用 build()", code="E500")
        k = k or self.config.default_k
        vb = self._extract(image)
        per = self.index.per_descriptor_similarity(vb)
        fused = self._fuse(per)
        order = np.argsort(-fused)
        ranks = [
            RankedMatch(
                id=self.index.ids[i],
                score=float(fused[i]),
                category=self.index.categories.get(self.index.ids[i]),
            )
            for i in order[:k]
        ]
        return QueryResult(query_id="", ranks=ranks, fused_weights=dict(self.weights))

    def benchmark(self, records: List[ImageRecord], k_list=(1, 3, 5, 10)) -> BenchmarkResult:
        if self.index is None:
            self.build(records, learn=False)
        per_aps: Dict[str, List[float]] = {}
        fused_aps: List[float] = []
        recall_rows: Dict[int, List[float]] = {k: [] for k in k_list}
        lat: List[float] = []

        for r in records:
            vb = self._extract(r.image)
            per = self.index.per_descriptor_similarity(vb)
            t0 = time.perf_counter()
            fused = self._fuse(per)
            lat.append((time.perf_counter() - t0) * 1000)
            order = np.argsort(-fused)
            ranked = [self.index.ids[i] for i in order if self.index.ids[i] != r.id]
            rel = {x.id for x in records if x.category == r.category and x.id != r.id}
            fused_aps.append(average_precision(ranked, rel))
            for n, sim in per.items():
                o2 = np.argsort(-sim)
                r2 = [self.index.ids[i] for i in o2 if self.index.ids[i] != r.id]
                per_aps.setdefault(n, []).append(average_precision(r2, rel))
            for k in k_list:
                recall_rows[k].append(recall_at_k(ranked, rel, k))

        return BenchmarkResult(
            dataset_size=len(records),
            n_queries=len(records),
            fused_mAP=float(np.mean(fused_aps)) if fused_aps else 0.0,
            per_descriptor_mAP={n: float(np.mean(v)) for n, v in per_aps.items()},
            recall_at_k={k: float(np.mean(v)) for k, v in recall_rows.items()},
            mean_latency_ms=float(np.mean(lat)) if lat else 0.0,
            memory_mb=self.index.memory_mb(),
            weights=dict(self.weights),
            environment={
                "os": f"{platform.system()} {platform.release()}",
                "machine": platform.machine(),
                "python": platform.python_version(),
                "descriptors": list(self.weights.keys()),
            },
        )

    def demo(self, n_per_class: int = 8, seed: Optional[int] = None, k_list=(1, 3, 5, 10)):
        seed = seed if seed is not None else self.config.random_state
        records = generate_dataset(n_per_class=n_per_class, size=self.config.image_size, seed=seed)
        self.build(records, learn=True)
        result = self.benchmark(records, k_list=k_list)
        return result, records

    def dump_benchmark(self, result: BenchmarkResult, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, ensure_ascii=False, indent=2, default=str)
