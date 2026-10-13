"""AI 抠皮肤：MediaPipe 多类人像分割（Apache-2.0），脸部皮肤和身体皮肤分开。

- 模型：selfie_multiclass_256x256.onnx（约 16 MB，输入 256×256 RGB 0–1，输出 6 类 logits：
  背景 / 头发 / 身体皮肤 / 脸部皮肤 / 衣服 / 其他）。
- 推理：onnxruntime（可选依赖）。Windows 装 onnxruntime-directml 即可用显卡（N/A/I 卡都行），否则用 CPU。
- 模型文件随安装包分发或由用户放进配置目录的 models/，**运行时不会下载**。
- 256×256 的结果放大到原图后边缘较粗：再与肤色规则蒙版求交，去掉眉眼、嘴唇和衣服边缘；
  如果肤色规则去掉太多（例如强烈的彩色环境光），只用 AI 结果。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = "selfie_multiclass_256x256.onnx"
MODEL_SHA256 = "13961350bb2099688eb96182ffe26f24eb12c77b1df278a2ec95bbdc923136e1"
MODEL_URL = "https://huggingface.co/FreeHugsForRobots/ps-skin-seg/resolve/main/selfie_multiclass_256x256.onnx"
CLASSES = ("背景", "头发", "身体皮肤", "脸部皮肤", "衣服", "其他")
BODY, FACE = 2, 3
INPUT = 256
PREFERRED_PROVIDERS = ("DmlExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
COLOR_KEEP_MIN = 0.6  # 肤色规则至少保留 AI 皮肤的 60%，才用两者的交集


class SegmentationUnavailable(RuntimeError):
    """没有安装 onnxruntime 或找不到模型；message 是给用户看的中文说明。"""


@dataclass
class SkinSegmentation:
    face: np.ndarray  # bool (H, W)
    body: np.ndarray
    labels: np.ndarray = field(repr=False)  # uint8 (H, W) 类别（0–5）
    provider: str = ""
    info: dict = field(default_factory=dict)

    def part(self, name: str | None) -> np.ndarray:
        if name in ("脸部", "face"):
            return self.face
        if name in ("身体", "body", "手臂/腿"):
            return self.body
        return self.face | self.body


def model_candidates() -> list[Path]:
    from colordoctor.core.skin_presets import user_config_dir

    out = []
    if os.environ.get("COLORDOCTOR_SEG_MODEL"):
        out.append(Path(os.environ["COLORDOCTOR_SEG_MODEL"]))
    out.append(user_config_dir() / "models" / MODEL_NAME)
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    out.append(base / "models" / MODEL_NAME)
    return out


def find_model() -> Path | None:
    return next((p for p in model_candidates() if p.is_file()), None)


def _ort():
    try:
        import onnxruntime
    except ImportError as e:
        raise SegmentationUnavailable("AI 抠皮肤需要安装 onnxruntime（Windows 显卡加速装 onnxruntime-directml）") from e
    return onnxruntime


def available() -> tuple[bool, str]:
    try:
        ort = _ort()
    except SegmentationUnavailable as e:
        return False, str(e)
    if find_model() is None:
        from colordoctor.core.skin_presets import user_config_dir

        return False, f"找不到模型文件 {MODEL_NAME}，请放到：{user_config_dir() / 'models'}"
    provs = [p for p in PREFERRED_PROVIDERS if p in ort.get_available_providers()]
    return True, provs[0] if provs else "CPUExecutionProvider"


_session_cache: dict = {}


def _session():
    ort = _ort()
    path = find_model()
    if path is None:
        raise SegmentationUnavailable(available()[1])
    key = str(path)
    if key not in _session_cache:
        provs = [p for p in PREFERRED_PROVIDERS if p in ort.get_available_providers()] or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _session_cache[key] = ort.InferenceSession(key, sess_options=opts, providers=provs)
    return _session_cache[key]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _upsample(prob: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(prob.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))


def segment_skin(rgb: np.ndarray, refine_with_color: bool = True, session=None) -> SkinSegmentation:
    """rgb：(H, W, 3) uint8 sRGB。返回脸部 / 身体皮肤蒙版（与原图同尺寸）。"""
    sess = session or _session()
    h, w = rgb.shape[:2]
    x = np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., :3])).resize((INPUT, INPUT), Image.BILINEAR), np.float32) / 255.0
    inp = sess.get_inputs()[0].name
    logits = sess.run(None, {inp: x[None]})[0][0]  # (256, 256, 6)
    prob = _softmax(logits.astype(np.float32))
    labels_small = prob.argmax(-1).astype(np.uint8)
    face = _upsample(prob[..., FACE], w, h) > 0.5
    body = _upsample(prob[..., BODY], w, h) > 0.5
    labels = np.asarray(Image.fromarray(labels_small).resize((w, h), Image.NEAREST))
    info = {"provider": (sess.get_providers() or [""])[0] if hasattr(sess, "get_providers") else "",
            "ai_face": round(float(face.mean()), 4), "ai_body": round(float(body.mean()), 4), "refined": False}
    if refine_with_color and (face.any() or body.any()):
        from colordoctor.core.skin_mask import skin_mask

        both = face | body
        color, _ = skin_mask(rgb, within=both)
        kept = float((color & both).sum()) / max(1, int(both.sum()))
        info["color_keep"] = round(kept, 3)
        if kept >= COLOR_KEEP_MIN:
            face &= color
            body &= color
            info["refined"] = True
    return SkinSegmentation(face, body, labels, info["provider"], info)
