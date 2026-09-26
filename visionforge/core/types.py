"""核心数据类型定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ImageRecord:
    """一条图像记录。"""

    id: str
    image: Any = None  # PIL.Image.Image
    category: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DescriptorOutput:
    name: str
    vector: np.ndarray
    available: bool = True


@dataclass
class RankedMatch:
    id: str
    score: float
    category: Optional[str] = None


@dataclass
class QueryResult:
    query_id: str
    ranks: List[RankedMatch]
    fused_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    dataset_size: int
    n_queries: int
    fused_mAP: float
    per_descriptor_mAP: Dict[str, float]
    recall_at_k: Dict[int, float]
    mean_latency_ms: float
    memory_mb: float
    weights: Dict[str, float]
    environment: Dict[str, Any] = field(default_factory=dict)
