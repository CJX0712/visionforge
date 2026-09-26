"""Pillow 基础预处理（resize / 灰度 / 归一化）。"""
from __future__ import annotations

import numpy as np
from PIL import Image


def resize(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size))


def to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def normalize(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0
