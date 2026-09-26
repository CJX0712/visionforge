"""命令行入口。"""
from __future__ import annotations

import argparse
import json
import sys

from .core.config import Config
from .data.synthetic import generate_dataset
from .pipeline.pipeline import VisionForgePipeline


def _print_table(result):
    print(f"\n{'描述子':<12}{'mAP':>10}")
    print("-" * 22)
    rows = [(n, m) for n, m in result.per_descriptor_mAP.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    for n, m in rows:
        print(f"{n:<12}{m:>10.4f}")
    print(f"{'fused(AMDF)':<12}{result.fused_mAP:>10.4f}")
    print("\nRecall@k:")
    for k in sorted(result.recall_at_k):
        print(f"  @{k:<3} {result.recall_at_k[k]:.4f}")
    print(f"\nmean_latency(ms): {result.mean_latency_ms:.3f}  memory(MB): {result.memory_mb:.2f}")
    print("AMDF weights:", {k: round(v, 3) for k, v in result.weights.items()})


def main(argv=None):
    p = argparse.ArgumentParser(prog="visionforge", description="VisionForge CBIR 系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="生成合成数据集并跑端到端基准")
    demo.add_argument("--n-per-class", type=int, default=8)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--out", default="benchmark.json")

    bench = sub.add_parser("benchmark", help="对已有记录集评测 (此处用合成数据演示)")
    bench.add_argument("--n-per-class", type=int, default=8)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--out", default="benchmark.json")

    q = sub.add_parser("query", help="对单张图像检索 (演示用合成集首图)")
    q.add_argument("--k", type=int, default=5)

    args = p.parse_args(argv)
    cfg = Config.from_env()
    pipe = VisionForgePipeline(cfg)

    if args.cmd in ("demo", "benchmark"):
        result, _ = pipe.demo(n_per_class=args.n_per_class, seed=args.seed)
        pipe.dump_benchmark(result, args.out)
        _print_table(result)
        print(f"\nbenchmark 已落盘: {args.out}")
    elif args.cmd == "query":
        records = generate_dataset(n_per_class=8, size=cfg.image_size, seed=42)
        pipe.build(records, learn=True)
        res = pipe.query(records[0].image, k=args.k)
        print(f"\nQuery: {records[0].id} (category={records[0].category})  Top-{args.k}:")
        for m in res.ranks:
            print(f"  {m.id:<14} score={m.score:.4f}  category={m.category}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
