"""配置：支持 ENV_VISIONFORGE_* 覆盖。"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Dict

_ENV_PREFIX = "VISIONFORGE_"


@dataclass
class Config:
    image_size: int = 64
    device: str = "cpu"
    random_state: int = 42
    default_k: int = 10
    lbp_points: int = 24
    lbp_radius: int = 3
    hog_orientations: int = 9
    use_gpu: bool = False
    fusion_iterations: int = 4

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in fields(cls):
            env_name = _ENV_PREFIX + f.name.upper()
            if env_name not in os.environ:
                continue
            raw = os.environ[env_name]
            try:
                if f.name in (
                    "image_size",
                    "random_state",
                    "default_k",
                    "lbp_points",
                    "lbp_radius",
                    "hog_orientations",
                    "fusion_iterations",
                ):
                    setattr(cfg, f.name, int(raw))
                elif f.name in ("use_gpu",):
                    setattr(cfg, f.name, raw.strip().lower() in ("1", "true", "yes", "on"))
                else:
                    setattr(cfg, f.name, raw)
            except Exception:
                pass
        return cfg

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self.__class__)}
