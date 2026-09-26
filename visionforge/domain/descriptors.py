"""描述子实现：每个描述子输出 L2 归一化固定长度向量（余弦相似度可直接点积）。

可用性：color_hist 永远可用（纯 numpy/pillow）；hog/lbp 依赖 scikit-image；
orb/sift 依赖 opencv。任一缺失则自动从可用集合剔除，纯 numpy 兜底保证 demo 可跑。
"""
from __future__ import annotations

import numpy as np
from typing import Any, List

from ..core.config import Config
from ..core.errors import DescriptorError
from .registry import HAVE_CV2, HAVE_SKIMAGE, HAVE_PIL


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v.astype(np.float32)


def _fallback_vec(gray: np.ndarray, dim: int) -> np.ndarray:
    """无关键点时的确定性兜底向量：灰度图横向降采样到 dim 长度，保证非零且等长的可复现。"""
    from PIL import Image

    arr = np.asarray(Image.fromarray(gray).resize((dim, 1)), dtype=np.float32)
    return arr.flatten().astype(np.float32)


def _to_pil(image: Any):
    from PIL import Image

    if hasattr(image, "resize"):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise DescriptorError("unsupported image type")


def _resize(image: Any, size: int):
    return _to_pil(image).resize((size, size))


class ColorHistDescriptor:
    """3D 颜色直方图（纯 numpy/pillow，永远可用）。"""

    name = "color_hist"
    available = True

    def __init__(self, config: Config) -> None:
        self.size = config.image_size
        self.bins = 8

    def extract(self, image: Any) -> np.ndarray:
        arr = np.asarray(_resize(image, self.size).convert("RGB"), dtype=np.float32)
        h = arr.reshape(-1, 3)
        hist, _ = np.histogramdd(h, bins=self.bins, range=[(0, 256)] * 3)
        hist = hist.flatten().astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return _l2(hist)


class HogDescriptor:
    """方向梯度直方图（scikit-image）。"""

    name = "hog"
    available = HAVE_SKIMAGE

    def __init__(self, config: Config) -> None:
        self.size = config.image_size
        self.orientations = config.hog_orientations

    def extract(self, image: Any) -> np.ndarray:
        if not self.available:
            raise DescriptorError("scikit-image 不可用")
        from skimage.feature import hog

        gray = np.asarray(_resize(image, self.size).convert("L"), dtype=np.float32)
        v = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        return _l2(v.astype(np.float32))


class LbpDescriptor:
    """局部二值模式纹理直方图（scikit-image）。"""

    name = "lbp"
    available = HAVE_SKIMAGE

    def __init__(self, config: Config) -> None:
        self.size = config.image_size
        self.P = config.lbp_points
        self.R = config.lbp_radius

    def extract(self, image: Any) -> np.ndarray:
        if not self.available:
            raise DescriptorError("scikit-image 不可用")
        from skimage.feature import local_binary_pattern

        gray = np.asarray(_resize(image, self.size).convert("L"), dtype=np.float32)
        lbp = local_binary_pattern(gray, P=self.P, R=self.R, method="uniform")
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return _l2(hist)


class OrbDescriptor:
    """ORB 关键点均值池化（opencv）。"""

    name = "orb"
    available = HAVE_CV2

    def __init__(self, config: Config) -> None:
        self.size = config.image_size
        self.dim = 32

    def extract(self, image: Any) -> np.ndarray:
        if not self.available:
            raise DescriptorError("opencv 不可用")
        import cv2

        gray = np.asarray(_resize(image, self.size).convert("L"))
        orb = cv2.ORB_create()
        _kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            return _l2(_fallback_vec(gray, self.dim))
        return _l2(des.astype(np.float32).mean(axis=0))


class SiftDescriptor:
    """SIFT 关键点均值池化（opencv）。"""

    name = "sift"
    available = HAVE_CV2

    def __init__(self, config: Config) -> None:
        self.size = config.image_size
        self.dim = 128

    def extract(self, image: Any) -> np.ndarray:
        if not self.available:
            raise DescriptorError("opencv 不可用")
        import cv2

        gray = np.asarray(_resize(image, self.size).convert("L"))
        sift = cv2.SIFT_create()
        _kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            return _l2(_fallback_vec(gray, self.dim))
        return _l2(des.astype(np.float32).mean(axis=0))


def build_descriptors(config: Config) -> List[Any]:
    """返回当前环境可用的描述子实例列表（顺序固定）。"""
    all_desc = [
        ColorHistDescriptor(config),
        HogDescriptor(config),
        LbpDescriptor(config),
        OrbDescriptor(config),
        SiftDescriptor(config),
    ]
    return [d for d in all_desc if d.available]


def available_names(config: Config) -> List[str]:
    return [d.name for d in build_descriptors(config)]
