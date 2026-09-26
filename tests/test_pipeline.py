"""Pipeline 端到端单测。"""
import numpy as np

from visionforge.core.config import Config
from visionforge.data.synthetic import generate_dataset
from visionforge.pipeline.pipeline import VisionForgePipeline


def test_build_and_query():
    recs = generate_dataset(n_per_class=4, seed=7)
    pipe = VisionForgePipeline(Config())
    pipe.build(recs, learn=True)
    res = pipe.query(recs[0].image, k=5)
    assert len(res.ranks) == 5
    assert res.ranks[0].id == recs[0].id  # 自匹配排第一
    assert abs(sum(pipe.weights.values()) - 1.0) < 1e-6


def test_benchmark_runs():
    recs = generate_dataset(n_per_class=6, seed=11)
    pipe = VisionForgePipeline(Config())
    result = pipe.benchmark(recs)
    assert result.fused_mAP >= 0.0
    assert "color_hist" in result.per_descriptor_mAP
    assert result.mean_latency_ms >= 0.0


def test_amdf_no_crash_on_small():
    recs = generate_dataset(n_per_class=2, seed=3)
    pipe = VisionForgePipeline(Config())
    pipe.build(recs, learn=False)
    r = pipe.benchmark(recs)
    assert np.isfinite(r.fused_mAP)
