"""端到端演示：合成数据 -> 建索引 -> AMDF 学习 -> 基准评测 -> 落盘 benchmark.json。

运行: .venv/Scripts/python -m visionforge.examples.run_demo
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..core.config import Config
from ..pipeline.pipeline import VisionForgePipeline


def main():
    cfg = Config.from_env()
    pipe = VisionForgePipeline(cfg)
    result, records = pipe.demo(n_per_class=8, seed=cfg.random_state)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.json")
    pipe.dump_benchmark(result, out)

    print("=== VisionForge Demo ===")
    print(f"数据集: {result.dataset_size} 张 / {len(set(r.category for r in records))} 类")
    print(f"可用描述子: {', '.join(result.weights.keys())}")
    print(f"AMDF 融合 mAP: {result.fused_mAP:.4f}  (单描述子 mAP 见下)")
    for n, m in sorted(result.per_descriptor_mAP.items(), key=lambda x: -x[1]):
        print(f"  {n:<12} {m:.4f}")
    print(f"Recall@5: {result.recall_at_k.get(5, 0):.4f}")
    print(f"mean_latency: {result.mean_latency_ms:.3f} ms/query   memory: {result.memory_mb:.2f} MB")
    print(f"benchmark.json -> {out}")


if __name__ == "__main__":
    main()
