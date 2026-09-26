"""描述子单测。"""
import numpy as np

from visionforge.core.config import Config
from visionforge.domain.descriptors import build_descriptors, ColorHistDescriptor
from visionforge.data.synthetic import generate_dataset


def test_color_hist_always_available():
    desc = ColorHistDescriptor(Config())
    assert desc.available is True
    recs = generate_dataset(n_per_class=2, seed=1)
    v = desc.extract(recs[0].image)
    assert v.ndim == 1
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4  # L2 归一化


def test_build_descriptors_returns_available_only():
    descs = build_descriptors(Config())
    names = [d.name for d in descs]
    assert "color_hist" in names
    recs = generate_dataset(n_per_class=2, seed=2)
    for d in descs:
        v = d.extract(recs[0].image)
        assert v.ndim == 1
        assert float(np.linalg.norm(v)) > 0


def test_extract_length_stable():
    descs = build_descriptors(Config())
    recs = generate_dataset(n_per_class=3, seed=3)
    for d in descs:
        lens = {len(d.extract(r.image)) for r in recs}
        assert len(lens) == 1  # 同描述子输出长度固定
