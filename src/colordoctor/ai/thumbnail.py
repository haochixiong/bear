"""给 AI 用的临时缩略图（最长边 768 像素，写入系统临时目录；不会触碰用户原图）。"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image


def make_thumbnail(region: np.ndarray, max_side: int = 768) -> str:
    img = Image.fromarray(np.ascontiguousarray(region[..., :3]))
    img.thumbnail((max_side, max_side))
    fd, path = tempfile.mkstemp(prefix="colordoctor_", suffix=".jpg")
    os.close(fd)
    img.save(path, quality=90)
    return path
