"""顶级开源依赖可用性探测（离线兜底依据）。"""
import importlib


def _have(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


HAVE_CV2 = _have("cv2")
HAVE_SKIMAGE = _have("skimage")
HAVE_PIL = _have("PIL")

# 选型依据（近 6 个月活跃、社区热度靠前）：opencv-python / scikit-image / pillow / numpy
SELECTION_TABLE = [
    ("opencv-python", "5.0.0", "特征提取 ORB/SIFT", "Apache-2.0", "HAVE_CV2", HAVE_CV2),
    ("scikit-image", "0.26.0", "HOG / LBP 纹理描述子", "BSD-3", "HAVE_SKIMAGE", HAVE_SKIMAGE),
    ("pillow", "12.3.0", "图像预处理/合成数据", "HPND", "HAVE_PIL", HAVE_PIL),
    ("numpy", "2.5.3", "向量化/相似度/兜底描述子", "BSD-3", "always", True),
]
