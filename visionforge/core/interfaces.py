"""模块间契约 (Protocol)。"""
from __future__ import annotations

from typing import Any, List, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class Descriptor(Protocol):
    name: str
    available: bool

    def extract(self, image: Any) -> np.ndarray: ...


@runtime_checkable
class SimilarityIndex(Protocol):
    def add(self, id: str, vector: np.ndarray) -> None: ...

    def query(self, vector: np.ndarray, k: int) -> List[Tuple[str, float]]: ...
