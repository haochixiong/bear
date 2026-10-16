"""本机桥接服务：Photoshop UXP 插件 <-> 分析引擎。

- 只监听 127.0.0.1（回环地址），不对外网开放，不发起任何网络请求。
- 插件用 imaging.getPixels 读出像素（已转换到 sRGB、8 位），以二进制 POST 过来；
  这里分析后返回结论 + 图层方案；蒙版按需以原始灰度字节取回（按文档尺寸缩放）。
- 每个请求必须带 X-ColorDoctor-Client 头：浏览器网页跨域发自定义头需要预检，
  而本服务从不返回 CORS 许可头，所以普通网页无法调用它。

协议（全部 JSON，除像素和蒙版外）：
  GET  /health
  POST /analyze?mode=dirty|skin|palette|reference[&preset=&part=&style=&family=&tone_style=&tone_target=&ref=]
       头：X-Width X-Height X-Components(3|4) [X-Doc-Width X-Doc-Height X-Layer-Kind X-Layer-Name]
       体：width*height*components 字节（行优先、交错 RGB(A)）
       [&precision=full|fast][&progressive=1]：原图精度且图很大时先返回快速结果（precision="preview"，
       带 refine 令牌），原图精查在后台继续，用 GET /refined?token=&wait=秒 取（没算完返回 202 {"pending": true}）
  POST /reference   同上的像素格式，存为参考图，返回 {"ref": id}
  GET  /mask?session=&layer=[&w=&h=][&left=&top=&doc_w=&doc_h=]  ->  灰度字节（255 = 作用）；
       给出 doc_w/doc_h 时返回整幅文档大小，图层范围以外为 0
  POST /acr-preset  体：{"name":..., "params": {...}} -> .xmp 文本
  GET  /calibration/chart  色卡像素（RGB）；GET /calibration/plan 要测的 (通道, 滑块, 数值) 列表
  POST /calibration/sample?slider=base | ?channel=&slider=&value=  上传色卡像素（未调整 / 调整后）
  POST /calibration/fit  拟合并保存换算表
  POST /multipoint  {"points": [{"name", "lab"}]}  多点取色对比（第一个点为基准）
  POST /reference-file（X-File-Name）  参考图文件本身（JPG / PNG / TIFF / WebP / RAW），返回 ref
"""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from colordoctor import __version__
from colordoctor.core import acr_channels, acr_preset, analysis, layer_plan, pixel_guard
from colordoctor.core.color_models import ColorContext
from colordoctor.core.report_json import summary, to_jsonable

HOST = "127.0.0.1"
DEFAULT_PORT = 47321
CLIENT_HEADER = "X-ColorDoctor-Client"
MAX_BODY = 400 * 1024 * 1024  # 约 1 亿像素的 RGBA
MAX_SESSIONS = 16
# 插件和服务的协议版本：加了接口 / 参数就加 1。插件发现服务的版本比自己要求的旧，就提示更新桌面版
PROTOCOL = 4  # 1：基础分析；2：/multipoint、/reference-file；3：影调风格（tone_style / tone_target / tone）；
# 4：分析精度（precision=full|fast）、先出快速结果再精查（progressive=1 + GET /refined）
PREVIEW_FROM = 20_000_000  # progressive=1 时，比这大的图先按约 1200 万像素出结果，再在后台按原图精查
MODES = ("dirty", "skin", "palette", "reference", "patch")


@dataclass
class Session:
    plan: layer_plan.LayerPlan
    shape: tuple[int, int]  # 分析图像（裁剪前）的 (h, w)
    offset: tuple[int, int]  # 裁剪偏移
    doc_size: tuple[int, int]  # 文档 (w, h)
    opaque: np.ndarray | None = None  # (h, w) bool：图层不透明的地方；None = 全部不透明


class Store:
    def __init__(self, limit: int = MAX_SESSIONS):
        self.limit = limit
        self._d: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, value) -> str:
        key = uuid.uuid4().hex[:12]
        with self._lock:
            self._d[key] = value
            while len(self._d) > self.limit:
                self._d.popitem(last=False)
        return key

    def get(self, key: str):
        """取出并刷新“最近使用”：正在用的结果（例如反复生成图层）不会因为别的页面分析了几次就被挤掉。"""
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v


def photoshop_context(layer_name: str = "") -> ColorContext:
    return ColorContext(source="photoshop", converted=True, device=layer_name or None,
                        warnings=("像素由 Photoshop 转换到 sRGB 后传入",))


def decode_pixels(body: bytes, width: int, height: int, comps: int) -> np.ndarray:
    if comps not in (3, 4) or width <= 0 or height <= 0:
        raise ValueError("X-Width / X-Height / X-Components 无效")
    if len(body) != width * height * comps:
        raise ValueError(f"像素字节数不符：收到 {len(body)}，应为 {width * height * comps}")
    return np.frombuffer(body, np.uint8).reshape(height, width, comps)


def run_analysis(prep, mode: str, q: dict, refs: Store, layer_name: str = ""):
    ctx = photoshop_context(layer_name)
    rgb, valid = prep.rgb, prep.valid
    if mode == "dirty":
        return analysis.scan_image(rgb, ctx, valid=valid)
    if mode == "skin":
        part = q.get("part") or None
        manual = None
        if q.get("seg") == "ai":
            from colordoctor.core import segmentation as sg

            try:
                manual = sg.segment_skin(rgb).part(part)
                if valid is not None:
                    manual &= valid
            except sg.SegmentationUnavailable as e:
                part = None  # 肤色规则分不出脸和身体，不能再标“脸部”
                ctx = dataclasses.replace(ctx, warnings=(*ctx.warnings, f"{e}；已改用肤色规则抠皮肤（不区分脸和身体）"))
            else:
                if not manual.any():  # 和桌面版一样：改用肤色规则，并说明（而不是直接报错）
                    manual = None
                    ctx = dataclasses.replace(ctx, warnings=(*ctx.warnings, f"AI 没有在画面里找到{part or ''}皮肤，已改用肤色规则抠皮肤（不区分脸和身体）"))
                    part = None
        if manual is None and valid is not None:
            from colordoctor.core.skin_mask import skin_mask

            manual = skin_mask(rgb, within=valid)[0]
            if int(manual.sum()) < 30:  # 透明图层里没找到皮肤：直接说清楚，而不是“蒙版内没有像素”
                raise pixel_guard.BlockedError("没有在这个图层里找到皮肤。请确认选中的是人像图层，或把分析对象改为「合并可见图层」。")
        if q.get("preset"):
            from colordoctor.core import skin_presets as sp

            if q["preset"] not in {p.key for p in sp.load_presets()}:
                raise ValueError(f"不认识的肤色风格：{q['preset']}")
        return analysis.analyze_skin_region(rgb, ctx, preset_key=q.get("preset") or None, part=part, manual_mask=manual)
    if mode == "palette":
        from colordoctor.core import tone_class as tcl

        from colordoctor.core import palette as pl
        from colordoctor.core.color_spaces import HUE_SECTORS

        tone_style, tone_target = q.get("tone_style") or "", q.get("tone_target") or ""
        if q.get("style") and q["style"] not in {s["key"] for s in pl.load_styles()}:
            raise ValueError(f"不认识的颜色模板：{q['style']}")
        if q.get("family") and q["family"] not in {name for name, _, _ in HUE_SECTORS}:
            raise ValueError(f"不认识的颜色：{q['family']}")
        if tone_style and tone_style not in tcl.TONE_STYLES:
            raise ValueError(f"不认识的影调风格：{tone_style}")
        if tone_target and tone_target not in tcl.TONE_NAMES:
            raise ValueError(f"不认识的影调：{tone_target}")
        return analysis.analyze_palette_report(prep.filled_for_statistics(), ctx, style_key=q.get("style") or None,
                                               family=q.get("family") or None, tone_style=tone_style, tone_target=tone_target)
    if mode == "reference":
        ref = refs.get(q.get("ref", ""))
        if ref is None:
            raise ValueError("找不到参考图，请重新选择参考图")
        return analysis.compare_reference(prep.filled_for_statistics(), ref, ctx)
    raise ValueError(f"未知模式：{mode}")


MAX_POINTS = 50


def multipoint_request(req: dict) -> dict:
    """多点对比（PS 颜色取样器）：{"points": [{"name": .., "lab": [L, a, b]}, ...]}，第一个点是基准。"""
    from colordoctor.core.multipoint import compare_points

    pts = req.get("points")
    if not isinstance(pts, list) or not 2 <= len(pts) <= MAX_POINTS:
        raise ValueError(f"需要 2–{MAX_POINTS} 个点")
    labs, names = [], []
    for i, p in enumerate(pts):
        raw = p.get("lab") if isinstance(p, dict) else None
        try:
            if not isinstance(raw, list) or len(raw) != 3:
                raise TypeError
            lab = [float(v) for v in raw]
        except (TypeError, ValueError) as e:
            raise ValueError(f"第 {i + 1} 个点的 Lab 无效（应为 [L, a, b]）") from e
        # L 0–100，a / b 在 ±200 内（PS 的 Lab 范围是 ±128）
        if not all(np.isfinite(lab)) or not 0 <= lab[0] <= 100 or abs(lab[1]) > 200 or abs(lab[2]) > 200:
            raise ValueError(f"第 {i + 1} 个点的 Lab 超出范围")
        labs.append(lab)
        names.append(str(p.get("name") or f"点 {i + 1}")[:40])
    res = compare_points(labs, names)
    return {"summary": res.summary, "legend": res.legend, "rows": [dataclasses.asdict(r) for r in res.rows]}


REF_MAX_PIXELS = 100_000_000  # 参考图和插件读文档一样按原图尺寸分析；超过这么多像素的超大图才缩小（与插件的 MAX_READ_PIXELS 相同）


_ref_cache: OrderedDict[str, tuple[str, dict]] = OrderedDict()  # 文件内容 → (ref id, 结果)：同一张参考图不重复解码
_ref_cache_lock = threading.Lock()


def reference_from_file(data: bytes, name: str, refs: Store) -> dict:
    import hashlib

    from PIL import Image

    from colordoctor.core.color_management import ImageLoadError, alpha_of_bytes, load_image_bytes

    key = hashlib.sha1(data).hexdigest() + "|" + name
    with _ref_cache_lock:
        hit = _ref_cache.get(key)
        if hit is not None and refs.get(hit[0]) is not None:  # 插件每次分析都会重新上传，RAW 不必每次都解码
            return {**hit[1], "ref": hit[0]}
    try:
        src = load_image_bytes(data, name)
    except ImageLoadError as e:
        raise ValueError(str(e)) from e
    px = src.pixels
    alpha = alpha_of_bytes(data)
    if alpha is not None and alpha.shape == px.shape[:2]:
        px = np.dstack([px, alpha])  # 透明的地方不算进参考图（与从 PS 图层读取一致）
    h, w = px.shape[:2]
    if h * w > REF_MAX_PIXELS:
        k = (REF_MAX_PIXELS / (h * w)) ** 0.5
        mode = "RGBA" if px.shape[2] == 4 else "RGB"
        px = np.asarray(Image.fromarray(np.ascontiguousarray(px), mode=mode).resize(
            (max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.BOX))
    try:
        prep = pixel_guard.prepare(px)
    except pixel_guard.BlockedError as e:
        # pixel_guard 的说法针对图层；这里是文件
        msg = str(e).replace("当前图层是", "这张参考图是").replace("（图层为空或尺寸为 0）", "（图片太小）")
        raise pixel_guard.BlockedError(msg) from e
    notes = [*src.context.warnings, *prep.notes]
    out = {"name": name, "width": int(w), "height": int(h), "notes": notes}
    ref = refs.put(prep.filled_for_statistics())
    with _ref_cache_lock:
        _ref_cache[key] = (ref, out)
        while len(_ref_cache) > MAX_SESSIONS:
            _ref_cache.popitem(last=False)
    return {**out, "ref": ref}


_prep_cache: OrderedDict = OrderedDict()  # 像素内容 -> (像素, pixel_guard.prepare 的结果)
_prep_lock = threading.Lock()


def _prepared(pixels: np.ndarray):
    """同一张图（插件换个设置再分析、或先快速再精查，像素内容完全一样）用同一份准备好的数组：
    分析引擎按数组记住了颜色编号，这样第二次不用重新编号。"""
    from colordoctor.core.pixel_table import content_key

    key = content_key(pixels)
    with _prep_lock:
        hit = _prep_cache.get(key)
        if hit is not None:
            _prep_cache.move_to_end(key)
            return hit
    prep = pixel_guard.prepare(pixels)
    with _prep_lock:
        _prep_cache[key] = prep
        while len(_prep_cache) > 2:
            _prep_cache.popitem(last=False)
    return prep


def analyze_request(pixels: np.ndarray, mode: str, q: dict, sessions: Store, refs: Store,
                    doc_size: tuple[int, int] | None = None, layer_kind: str = "", layer_name: str = "") -> dict:
    """服务端核心逻辑（与 HTTP 解耦，便于测试）。不能分析时抛 pixel_guard.BlockedError。"""
    pixel_guard.check_layer_kind(layer_kind, layer_name)
    if mode == "patch":
        lab = None
        if q.get("lab"):  # 前景色：插件直接给 PS 的 Lab（和文档的颜色配置文件无关），比按文档 RGB 读更准
            try:
                lab = tuple(float(v) for v in q["lab"].split(","))
            except ValueError as e:
                raise ValueError("lab 参数应为 L,a,b") from e
            if len(lab) != 3 or not (0 <= lab[0] <= 100):
                raise ValueError("lab 参数应为 L,a,b")
        return patch_values(pixels, q.get("preset") or "", lab=lab)
    prep = _prepared(pixels)
    precision = q.get("precision") or "full"
    if precision not in ("full", "fast", "preview"):
        raise ValueError(f"不认识的分析精度：{precision}")
    from colordoctor.core.sampling import FAST_MAX, FULL_RES_MAX, PREVIEW_MAX, resolution

    limit = {"full": FULL_RES_MAX, "fast": FAST_MAX, "preview": PREVIEW_MAX}[precision]
    with resolution(limit, precision):
        report = run_analysis(prep, mode, q, refs, layer_name)
    plan = layer_plan.plan_from_report(report, prep.rgb.shape)
    h, w = pixels.shape[:2]
    opaque = None
    if prep.valid is not None:  # 分析的是带透明区域的图层：蒙版在透明处必须为 0，否则调整会作用到下面的图层
        opaque = np.zeros((h, w), bool)
        ox, oy = prep.offset
        vh, vw = prep.valid.shape
        opaque[oy:oy + vh, ox:ox + vw] = prep.valid
    sid = sessions.put(Session(plan, (h, w), prep.offset, doc_size or (w, h), opaque))
    plan_json = plan.to_json()
    # 每条问题带上序号和修它的图层 id（图层的 finding 就是这个序号，顺序与 report.findings 相同）
    findings_json = layer_plan.annotate_findings(to_jsonable(list(report.findings)), plan)
    if opaque is not None:
        for ly in plan_json["layers"]:
            ly["has_mask"] = True  # 全图类的调整（如曲线）也要限制在这个图层的不透明范围内
    return {
        "session": sid,
        "mode": mode,
        "layer": layer_name,
        "notes": list(prep.notes) + [w for w in report.context.warnings if w not in photoshop_context().warnings],
        "summary": summary(report),
        "findings": findings_json,
        "adjustments": to_jsonable(list(report.adjustments)),
        "plan": plan_json,
        "acr_params": acr_preset.params_for_report(report),
        "acr_calibrated": acr_channels.is_calibrated(),
        "tone": tone_json(report),
        "precision": report.precision,
    }


class Refine:
    """后台的原图精查：完成后 result / error 之一有值。"""

    def __init__(self):
        self.done = threading.Event()
        self.result: dict | None = None
        self.error: tuple[int, str] | None = None


def analyze_progressive(pixels: np.ndarray, mode: str, q: dict, sessions: Store, refs: Store, refines: Store,
                        doc_size=None, layer_kind: str = "", layer_name: str = "") -> dict:
    """progressive=1：原图精度的大图先按约 1200 万像素出结果（马上返回），原图精查在后台线程继续。"""
    q = {k: v for k, v in q.items() if k != "progressive"}
    big = pixels.shape[0] * pixels.shape[1] > PREVIEW_FROM
    if mode == "patch" or (q.get("precision") or "full") != "full" or not big:
        return analyze_request(pixels, mode, q, sessions, refs, doc_size, layer_kind, layer_name)
    quick = analyze_request(pixels, mode, {**q, "precision": "preview"}, sessions, refs, doc_size, layer_kind, layer_name)
    job = Refine()
    token = refines.put(job)

    def run():
        try:
            job.result = analyze_request(pixels, mode, q, sessions, refs, doc_size, layer_kind, layer_name)
        except pixel_guard.BlockedError as e:
            job.error = (422, str(e))
        except ValueError as e:
            job.error = (400, str(e))
        except Exception as e:  # noqa: BLE001 —— 后台线程里的错误交给 /refined 报告
            job.error = (500, f"原图精查失败：{e}")
        finally:
            job.done.set()

    threading.Thread(target=run, name="cd-refine", daemon=True).start()
    return {**quick, "refine": token}


def tone_json(report) -> dict | None:
    """整图模式：现在的影调、目标影调和各影调风格离现在多远（插件画九宫格、给提示用）。"""
    from colordoctor.core import tone_class as tcl

    p = report.palette
    if p is None or p.tone_class is None:
        return None
    return tcl.to_json(p.tone_class, p.tone_target, p.tone_also, p.tone_label)


def patch_values(pixels: np.ndarray, preset_key: str = "", lab=None) -> dict:
    """取色：一小块像素（PS 选区或吸管范围）的平均颜色、肤色快速规则、所属 ACR 通道、纯度。

    preset_key：肤色风格；给出时按这个风格自己的快速规则判断（和「人像」页、桌面版一致），否则用全局规则。
    lab：直接给出的 Lab（D50，PS 前景色）；给出时数值以它为准，不经过 8 位 RGB 的取整。
    选区有羽化（半选中的像素）时按选中程度加权平均，和 PS 里“羽化选区”的效果一致。
    """
    from colordoctor.core import acr_channels as ac
    from colordoctor.core import color_spaces as cs
    from colordoctor.core.portrait_analysis import quick_rule_check
    from colordoctor.core.sampling import color_values
    from colordoctor.core.skin_presets import global_quick_rule

    import dataclasses

    px = pixels.reshape(-1, pixels.shape[2])
    weights = None
    if pixels.shape[2] == 4:
        keep = px[:, 3] > 0
        weights = px[keep, 3].astype(np.float64) / 255.0  # 羽化边缘按选中程度计入
        px = px[keep]
    if px.shape[0] == 0:
        raise pixel_guard.BlockedError("取样范围内没有不透明像素。")
    rgb = px[:, :3]
    mean = np.average(rgb.astype(np.float64), axis=0, weights=weights)
    if lab is not None:
        exact = np.array(lab, dtype=np.float64)
        vals = color_values(np.clip(cs.lab_to_srgb_float(exact), 0, 1) * 255.0)
        vals = dataclasses.replace(vals, lab=tuple(round(float(v), 4) for v in exact),
                                   lch=tuple(round(float(v), 4) for v in cs.lab_to_lch(exact)))
    else:
        vals = color_values(mean)
    core = rgb if weights is None else rgb[weights >= 0.5]  # 快速规则 / 纯度只看选中一半以上的像素
    core = core if core.shape[0] else rgb
    lab = cs.srgb8_to_lab(core) if lab is None else np.array([lab], dtype=np.float64)  # 每个像素都算
    preset = None
    if preset_key:
        from colordoctor.core import skin_presets as sp

        if preset_key not in {p.key for p in sp.load_presets()}:
            raise ValueError(f"不认识的肤色风格：{preset_key}")
        preset = sp.get_preset(preset_key)
    qr = quick_rule_check(lab, preset.quick_rule if preset is not None and preset.quick_rule else global_quick_rule())
    w = ac.memberships(np.array([vals.hsb[0]]))[0] if vals.lch[1] >= 4 else np.zeros(len(ac.NAMES))
    channels = [{"channel": n, "weight": round(float(x), 3)} for n, x in zip(ac.NAMES, w, strict=True) if x > 0.01]
    purity = None
    if vals.lch[1] >= 8:
        lch = cs.lab_to_lch(lab)
        chrom = lch[..., 1] >= 6
        if chrom.any():
            d = np.abs(((lch[..., 2][chrom] - vals.lch[2] + 180) % 360) - 180)
            purity = round(float((d <= 30).mean()), 4)
    return {"mode": "patch", "pixels": int(px.shape[0]), "values": to_jsonable(vals), "quick_rule": to_jsonable(qr),
            "quick_rule_preset": preset.name if preset is not None and preset.quick_rule else "",
            "acr_channels": channels, "purity": purity}


class NotFound(LookupError):
    """会话 / 图层不存在（返回 404）。"""


MAX_SIDE = 30_000  # 单边像素上限（PS 文档最大 300000，但这里只接受常见尺寸，防止内存耗尽）
MAX_MASK_PIXELS = 300_000_000
MAX_CAL_SAMPLES = 200


def _check_size(w: int, h: int, what: str) -> None:
    if w <= 0 or h <= 0:
        raise ValueError(f"{what}尺寸无效：{w}×{h}")
    if w > MAX_SIDE or h > MAX_SIDE or w * h > MAX_MASK_PIXELS:
        raise ValueError(f"{what}尺寸太大：{w}×{h}")


def mask_bytes(sess: Session, layer_id: str, width: int | None = None, height: int | None = None,
               place: tuple[int, int, int, int] | None = None) -> tuple[bytes, int, int]:
    """返回 (灰度字节, 宽, 高)。

    width/height：蒙版对应区域（分析时的图层范围）缩放到的尺寸，默认是分析时报告的文档尺寸。
    place：(left, top, 文档宽, 文档高)。给出时返回整幅文档大小的蒙版，图层范围以外为 0（不作用），
    插件就能把它写到文档 (0, 0)，不用担心范围外保持白色。
    """
    m = sess.plan.full_mask(layer_id)
    h, w = sess.shape
    full = np.full((h, w), 255, np.uint8) if m is None else np.zeros((h, w), np.uint8)
    if m is not None:
        ox, oy = sess.offset
        mh, mw = m.shape
        full[oy:oy + mh, ox:ox + mw] = m
    if sess.opaque is not None:
        full[~sess.opaque] = 0
    tw = sess.doc_size[0] if width is None else int(width)
    th = sess.doc_size[1] if height is None else int(height)
    _check_size(tw, th, "蒙版")
    out = layer_plan.resize_mask(full, tw, th)
    if place is None:
        return out.tobytes(), tw, th
    left, top, dw, dh = (int(v) for v in place)
    _check_size(dw, dh, "文档")
    canvas = np.zeros((dh, dw), np.uint8)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(dw, left + tw), min(dh, top + th)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = out[y0 - top:y1 - top, x0 - left:x1 - left]
    return canvas.tobytes(), dw, dh


class Handler(BaseHTTPRequestHandler):
    server_version = "ColorDoctorBridge/" + __version__
    timeout = 60  # 客户端发得太慢 / Content-Length 不实时，不让线程永远卡住
    sessions: Store
    refs: Store
    calibration: dict
    cal_lock: threading.Lock

    def log_message(self, fmt, *args):  # 不在控制台刷屏
        pass

    # ---------------------------------------------------------------- 工具
    def _send(self, code: int, payload, ctype="application/json; charset=utf-8", headers: dict | None = None):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        origin = self.headers.get("Origin", "")
        if origin.startswith(("http://", "https://")):
            return False
        # 防 DNS 重绑定：只接受以回环地址访问
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        return bool(self.headers.get(CLIENT_HEADER))

    def _body(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError as e:
            raise ValueError("Content-Length 无效") from e
        if n < 0 or n > MAX_BODY:
            raise ValueError("请求体大小无效或图像太大")
        return self.rfile.read(n)

    def _json(self) -> dict:
        try:
            req = json.loads(self._body() or b"{}")
        except json.JSONDecodeError as e:
            raise ValueError("请求不是有效的 JSON") from e
        if not isinstance(req, dict):
            raise ValueError("请求体应为 JSON 对象")
        return req

    def _int(self, name: str, default=None):
        v = self.headers.get(name)
        if v in (None, ""):
            return default
        try:
            return int(v)
        except ValueError as e:
            raise ValueError(f"{name} 不是整数") from e

    def _pixels(self) -> np.ndarray:
        w, h = self._int("X-Width", 0), self._int("X-Height", 0)
        _check_size(w, h, "图像")
        return decode_pixels(self._body(), w, h, self._int("X-Components", 4))

    def _guarded(self, fn):
        try:
            fn()
        except pixel_guard.BlockedError as e:
            self._send(422, {"blocked": str(e)})
        except NotFound as e:
            self._send(404, {"error": str(e)})
        except KeyError as e:
            self._send(400, {"error": f"缺少参数：{e.args[0] if e.args else ''}"})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # 分析出错也要回一个响应，插件才能显示真实原因
            self._send(500, {"error": f"分析服务内部错误：{type(e).__name__}: {e}"})

    # ---------------------------------------------------------------- 路由
    def do_OPTIONS(self):  # 浏览器预检：一律拒绝
        self._send(403, {"error": "forbidden"})

    def do_GET(self):
        if not self._authorized():
            return self._send(403, {"error": "forbidden"})
        self._guarded(self._get)
        return None

    def do_POST(self):
        if not self._authorized():
            return self._send(403, {"error": "forbidden"})
        self._guarded(self._post)
        return None

    def _get(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/health":
            return self._send(200, {"ok": True, "version": __version__, "protocol": PROTOCOL, "modes": list(MODES)})
        if url.path == "/options":
            from colordoctor.core import palette as pl
            from colordoctor.core import skin_presets as sp
            from colordoctor.core import tone_class as tcl
            from colordoctor.core.color_spaces import HUE_SECTORS

            return self._send(200, {
                "styles": [{"key": st["key"], "name": st["name"], "category": st.get("category", ""),
                            "description": st.get("description", ""), "tone_pair": st.get("tone_pair", [])}
                           for st in pl.load_styles()],
                "tone_styles": [{"key": k, **{f: v[f] for f in ("name", "target", "look", "scene")}, "also": list(v["also"])}
                                for k, v in tcl.TONE_STYLES.items()],
                "tones": [{"name": n, "desc": tcl.TONE_DESC[n]} for n in tcl.TONE_NAMES],
                "presets": [{"key": pr.key, "name": pr.name} for pr in sp.load_presets()],
                "families": [name for name, _, _ in HUE_SECTORS],
            })
        if url.path == "/mask":
            sess = self.sessions.get(q.get("session", ""))
            if sess is None:
                return self._send(404, {"error": "分析结果已过期，请重新分析"})

            def num(k):
                try:
                    return int(q[k]) if k in q else None
                except ValueError as e:
                    raise ValueError(f"参数 {k} 不是整数") from e

            place = None
            if ("doc_w" in q) != ("doc_h" in q):
                raise ValueError("doc_w 和 doc_h 必须同时提供")
            if "doc_w" in q:
                place = (num("left") or 0, num("top") or 0, num("doc_w"), num("doc_h"))
            try:
                sess.plan.get(q.get("layer", ""))
            except KeyError as e:
                raise NotFound("没有这个图层，请重新分析") from e
            data, w, h = mask_bytes(sess, q.get("layer", ""), num("w"), num("h"), place)
            return self._send(200, data, "application/octet-stream", {"X-Width": w, "X-Height": h})
        if url.path == "/refined":
            job = self.refines.get(q.get("token", ""))
            if job is None:
                return self._send(404, {"error": "原图精查的结果已过期，请重新分析"})
            try:
                wait = min(60.0, max(0.0, float(q.get("wait", "25"))))
            except ValueError as e:
                raise ValueError("wait 应为秒数") from e
            if not job.done.wait(wait):
                return self._send(202, {"pending": True})
            if job.error is not None:
                code, msg = job.error
                return self._send(code, {"blocked": msg} if code == 422 else {"error": msg})
            return self._send(200, job.result)
        if url.path == "/calibration/chart":
            from colordoctor.core import acr_calibration as cal

            img, _ = cal.chart()
            return self._send(200, img.tobytes(), "application/octet-stream",
                              {"X-Width": img.shape[1], "X-Height": img.shape[0], "X-Components": 3})
        if url.path == "/calibration/plan":
            from colordoctor.core import acr_calibration as cal

            steps = [{"channel": n, "slider": s_, "value": v} for n, _ in cal.ac.CHANNELS for s_ in cal.SLIDERS for v in cal.DEFAULT_VALUES]
            return self._send(200, {"steps": steps})
        return self._send(404, {"error": "not found"})

    def _post(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/analyze":
            mode = q.get("mode", "dirty")
            dw, dh = self._int("X-Doc-Width"), self._int("X-Doc-Height")
            if dw is not None or dh is not None:
                _check_size(dw or 0, dh or 0, "文档")
            res = analyze_progressive(self._pixels(), mode, q, self.sessions, self.refs, self.refines,
                                      (dw, dh) if dw and dh else None,
                                      self.headers.get("X-Layer-Kind", ""), _header_text(self.headers.get("X-Layer-Name", "")))
            return self._send(200, res)
        if url.path == "/reference":
            prep = pixel_guard.prepare(self._pixels())
            return self._send(200, {"ref": self.refs.put(prep.filled_for_statistics()), "notes": list(prep.notes)})
        if url.path == "/reference-file":
            # 插件直接选的参考图文件（JPG / PNG / TIFF / WebP / 相机 RAW），不用先在 PS 里打开
            name = _header_text(self.headers.get("X-File-Name", "")) or "参考图"
            return self._send(200, reference_from_file(self._body(), name, self.refs))
        if url.path == "/calibration/sample":
            from colordoctor.core import acr_calibration as cal

            img = self._pixels()[..., :3].copy()
            chart_shape = cal.chart()[0].shape
            if img.shape != chart_shape:
                raise ValueError(f"色卡尺寸不对：{img.shape[1]}×{img.shape[0]}，应为 {chart_shape[1]}×{chart_shape[0]}")
            with self.cal_lock:
                if q.get("slider") == "base":
                    self.calibration.clear()
                    self.calibration["base"] = img
                else:
                    if q.get("slider") not in cal.SLIDERS:
                        raise ValueError("slider 应为 hue / sat / lum / base")
                    if q.get("channel") not in cal.ac.NAMES:
                        raise ValueError(f"未知的混色器通道：{q.get('channel')}")
                    samples = self.calibration.setdefault("samples", [])
                    if len(samples) >= MAX_CAL_SAMPLES:
                        raise ValueError("校准样本太多")
                    samples.append((q["channel"], q["slider"], int(q["value"]), img))
                count = len(self.calibration.get("samples", []))
            return self._send(200, {"ok": True, "count": count})
        if url.path == "/multipoint":
            return self._send(200, multipoint_request(self._json()))
        if url.path == "/calibration/fit":
            from colordoctor.core import acr_calibration as cal

            req = self._json()
            with self.cal_lock:
                if "base" not in self.calibration:
                    raise ValueError("还没有上传未调整的色卡")
                if not self.calibration.get("samples"):
                    raise ValueError("还没有任何调整后的色卡，不会覆盖已有的校准")
                channels = cal.fit(self.calibration["base"], self.calibration["samples"])
            if not channels:
                raise ValueError("没有得到有效的校准结果，未保存")
            path = cal.store(channels, str(req.get("acr_version", "")))
            return self._send(200, {"channels": channels, "saved": path})
        if url.path == "/acr-preset":
            req = self._json()
            params = req.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params 应为对象")
            try:
                xmp = acr_preset.build_xmp(str(req.get("name") or "ColorDoctor"), params)
            except (TypeError, AttributeError, IndexError) as e:
                raise ValueError(f"ACR 参数格式不对：{e}") from e
            return self._send(200, xmp.encode("utf-8"), "application/rdf+xml; charset=utf-8")
        return self._send(404, {"error": "not found"})


def _header_text(v: str) -> str:
    """图层名可能含中文：插件用 encodeURIComponent 传。"""
    from urllib.parse import unquote

    return unquote(v or "")


class _ExclusiveServer(ThreadingHTTPServer):
    """端口被占用时必须绑定失败（桌面版据此知道已经有一个服务在跑）。

    标准库默认打开 SO_REUSEADDR：在 Windows 上这会让第二个程序也绑上同一个端口，两个服务各收一部分请求，
    插件拿到的会话 / 蒙版时有时无，旧版本也可能来回答。
    """

    allow_reuse_address = False

    def server_bind(self):
        import socket

        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows：独占端口
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"sessions": Store(), "refs": Store(4), "refines": Store(8), "calibration": {},
                                                "cal_lock": threading.Lock()})
    srv = _ExclusiveServer((HOST, port), handler)
    srv.daemon_threads = True
    return srv


def serve(port: int = DEFAULT_PORT) -> None:
    srv = make_server(port)
    print(f"ColorDoctor 本机服务已启动：http://{HOST}:{srv.server_address[1]}（只接受本机连接，Ctrl+C 退出）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
