"""从磁盘目录载入图像。"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from ..core.types import ImageRecord

_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_from_dir(path: str) -> List[ImageRecord]:
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"not a directory: {path}")
    recs: List[ImageRecord] = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in _EXT:
            img = Image.open(f).convert("RGB")
            recs.append(ImageRecord(id=f.stem, image=img, category=f.parent.name))
    return recs
