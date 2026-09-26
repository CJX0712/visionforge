"""零下载合成图像数据集生成器。

每类生成可视觉区分的形状/纹理，叠加高斯噪声，作为检索系统的 ground truth：
同类图像互为 relevant。保证离线可跑且评测有意义。
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from ..core.types import ImageRecord

# 形状 × 颜色 的组合类别：单一描述子只能捕获一个维度（颜色或形状），
# 只有多描述子融合才能同时利用两维，从而严格超越任一单描述子。
SHAPES = ["circle", "square", "triangle"]
COLORS: Dict[str, tuple] = {
    "red": (214, 40, 40),
    "green": (34, 170, 60),
    "blue": (40, 80, 214),
}


def _draw_shape(d: ImageDraw.Draw, shape: str, color, size: int):
    if shape == "circle":
        r = size // 3
        d.ellipse([size // 2 - r, size // 2 - r, size // 2 + r, size // 2 + r], fill=color)
    elif shape == "square":
        s = size // 3
        d.rectangle([size // 2 - s, size // 2 - s, size // 2 + s, size // 2 + s], fill=color)
    elif shape == "triangle":
        s = size // 2
        d.polygon([(size // 2, s // 3), (size // 2 - s, s), (size // 2 + s, s)], fill=color)


def generate_dataset(
    n_per_class: int = 8,
    size: int = 64,
    seed: int = 42,
    shapes: Optional[List[str]] = None,
    colors: Optional[Dict[str, tuple]] = None,
) -> List[ImageRecord]:
    rng = np.random.default_rng(seed)
    shapes = shapes or SHAPES
    colors = colors or COLORS
    cats = [f"{cname}_{s}" for cname in colors for s in shapes]
    records: List[ImageRecord] = []
    for cat in cats:
        cname, shape = cat.split("_", 1)
        color = colors[cname]
        bg = (238, 238, 238)
        for i in range(n_per_class):
            img = Image.new("RGB", (size, size), bg)
            d = ImageDraw.Draw(img)
            # 位置/尺度抖动，避免图像完全雷同
            off = int(rng.integers(-4, 5))
            scale = 1.0 + float(rng.uniform(-0.1, 0.1))
            img2 = Image.new("RGB", (size, size), bg)
            d2 = ImageDraw.Draw(img2)
            if shape == "circle":
                r = int(size // 3 * scale)
                d2.ellipse([size // 2 - r + off, size // 2 - r, size // 2 + r + off, size // 2 + r], fill=color)
            elif shape == "square":
                s = int(size // 3 * scale)
                d2.rectangle([size // 2 - s + off, size // 2 - s, size // 2 + s + off, size // 2 + s], fill=color)
            else:  # triangle
                s = int(size // 2 * scale)
                d2.polygon([(size // 2 + off, s // 3), (size // 2 - s + off, s), (size // 2 + s + off, s)], fill=color)
            img = img2
            # 加噪声，模拟真实图像
            arr = np.asarray(img, dtype=np.float32)
            noise = rng.normal(0, 8, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255).astype("uint8")
            img = Image.fromarray(arr)
            records.append(ImageRecord(id=f"{cat}_{i}", image=img, category=cat))
    return records
