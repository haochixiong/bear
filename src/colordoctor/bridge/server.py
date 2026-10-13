"""本机桥接服务：Photoshop UXP 插件 <-> 分析引擎。

- 只监听 127.0.0.1（回环地址），不对外网开放，不发起任何网络请求。
- 插件用 imaging.getPixels 读出像素（已转换到 sRGB、8 位），以二进制 POST 过来；
  这里分析后返回结论 + 图层方案；蒙版按需以原始灰度字节取回（按文档尺寸缩放）。
- 每个请求必须带 X-ColorDoctor-Client 头：浏览器网页跨域发自定义头需要预检，
  而本服务从不返回 CORS 许可头，所以普通网页无法调用它。

协议（全部 JSON，除像素和蒙版外）：
  GET  /health
  POST /analyze?mode=dirty|skin|palette|reference[&preset=&part=&style=&family=&ref=]
       头：X-Width X-Height X-Components(3|4) [X-Doc-Width X-Doc-Height X-Layer-Kind X-Layer-Name]
       体：width*height*components 字节（行优先、交错 RGB(A)）
  POST /reference   同上的像素格式，存为参考图，返回 {"ref": id}
  GET  /mask?session=&layer=[&w=&h=][&left=&top=&doc_w=&doc_h=]  ->  灰度字节（255 = 作用）；
       给出 doc_w/doc_h 时返回整幅文档大小，图层范围以外为 0
  POST /acr-preset  体：{"name":..., "params": {...}} -> .xmp 文本
  GET  /calibration/chart  色卡像素（RGB）；GET /calibration/plan 要测的 (通道, 滑块, 数值) 列表
  POST /calibration/sample?slider=base | ?channel=&slider=&value=  上传色卡像素（未调整 / 调整后）
  POST /calibration/fit  拟合并保存换算表
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
MAX_SESSIONS = 8
MODES = ("dirty", "skin", "palette", "reference", "patch")


@dataclass
class Session:
    plan: layer_plan.LayerPlan
    shape: tuple[int, int]  # 分析图像（裁剪前）的 (h, w)
    offset: tuple[int, int]  # 裁剪偏移
    doc_size: tuple[int, int]  # 文档 (w, h)


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
        with self._lock:
            return self._d.get(key)


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
                ctx = dataclasses.replace(ctx, warnings=(*ctx.warnings, f"{e}；已改用肤色规则抠皮肤"))
            else:
                if not manual.any():
                    raise pixel_guard.BlockedError("AI 没有在画面里找到这个部位的皮肤。可以换成「肤色规则」，或改选部位。")
        if manual is None and valid is not None:
            from colordoctor.core.skin_mask import skin_mask

            manual = skin_mask(rgb, within=valid)[0]
        return analysis.analyze_skin_region(rgb, ctx, preset_key=q.get("preset") or None, part=part, manual_mask=manual)
    if mode == "palette":
        return analysis.analyze_palette_report(prep.filled_for_statistics(), ctx, style_key=q.get("style") or None,
                                               family=q.get("family") or None)
    if mode == "reference":
        ref = refs.get(q.get("ref", ""))
        if ref is None:
            raise ValueError("找不到参考图，请重新选择参考图")
        return analysis.compare_reference(prep.filled_for_statistics(), ref, ctx)
    raise ValueError(f"未知模式：{mode}")


def analyze_request(pixels: np.ndarray, mode: str, q: dict, sessions: Store, refs: Store,
                    doc_size: tuple[int, int] | None = None, layer_kind: str = "", layer_name: str = "") -> dict:
    """服务端核心逻辑（与 HTTP 解耦，便于测试）。不能分析时抛 pixel_guard.BlockedError。"""
    pixel_guard.check_layer_kind(layer_kind, layer_name)
    if mode == "patch":
        return patch_values(pixels)
    prep = pixel_guard.prepare(pixels)
    report = run_analysis(prep, mode, q, refs, layer_name)
    plan = layer_plan.plan_from_report(report, prep.rgb.shape)
    h, w = pixels.shape[:2]
    sid = sessions.put(Session(plan, (h, w), prep.offset, doc_size or (w, h)))
    return {
        "session": sid,
        "mode": mode,
        "layer": layer_name,
        "notes": list(prep.notes),
        "summary": summary(report),
        "findings": to_jsonable(list(report.findings)),
        "adjustments": to_jsonable(list(report.adjustments)),
        "plan": plan.to_json(),
        "acr_params": acr_preset.params_for_report(report),
        "acr_calibrated": acr_channels.is_calibrated(),
    }


def patch_values(pixels: np.ndarray) -> dict:
    """取色：一小块像素（PS 选区或吸管范围）的平均颜色、肤色快速规则、所属 ACR 通道、纯度。"""
    from colordoctor.core import acr_channels as ac
    from colordoctor.core import color_spaces as cs
    from colordoctor.core.portrait_analysis import quick_rule_check
    from colordoctor.core.sampling import color_values
    from colordoctor.core.skin_presets import global_quick_rule

    px = pixels.reshape(-1, pixels.shape[2])
    if pixels.shape[2] == 4:
        px = px[px[:, 3] >= pixel_guard.ALPHA_MIN]
    if px.shape[0] == 0:
        raise pixel_guard.BlockedError("取样范围内没有不透明像素。")
    rgb = px[:, :3]
    vals = color_values(rgb.astype(np.float64).mean(axis=0))
    lab = cs.srgb8_to_lab(rgb[:: max(1, rgb.shape[0] // 50_000)])
    qr = quick_rule_check(lab, global_quick_rule())
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
            "acr_channels": channels, "purity": purity}


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
    ly = sess.plan.get(layer_id)
    h, w = sess.shape
    full = np.full((h, w), 255, np.uint8) if ly.mask is None else np.zeros((h, w), np.uint8)
    if ly.mask is not None:
        ox, oy = sess.offset
        mh, mw = ly.mask.shape
        full[oy:oy + mh, ox:ox + mw] = ly.mask
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
        except KeyError as e:
            self._send(404, {"error": f"找不到：{e.args[0] if e.args else ''}"})
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
            return self._send(200, {"ok": True, "version": __version__, "modes": list(MODES)})
        if url.path == "/options":
            from colordoctor.core import palette as pl
            from colordoctor.core import skin_presets as sp
            from colordoctor.core.color_spaces import HUE_SECTORS

            return self._send(200, {
                "styles": [{"key": st["key"], "name": st["name"], "category": st.get("category", "")} for st in pl.load_styles()],
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
            if "doc_w" in q:
                place = (num("left") or 0, num("top") or 0, num("doc_w"), num("doc_h"))
            data, w, h = mask_bytes(sess, q.get("layer", ""), num("w"), num("h"), place)
            return self._send(200, data, "application/octet-stream", {"X-Width": w, "X-Height": h})
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
            res = analyze_request(self._pixels(), mode, q, self.sessions, self.refs,
                                  (dw, dh) if dw and dh else None,
                                  self.headers.get("X-Layer-Kind", ""), _header_text(self.headers.get("X-Layer-Name", "")))
            return self._send(200, res)
        if url.path == "/reference":
            prep = pixel_guard.prepare(self._pixels())
            return self._send(200, {"ref": self.refs.put(prep.filled_for_statistics()), "notes": list(prep.notes)})
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
                    samples = self.calibration.setdefault("samples", [])
                    if len(samples) >= MAX_CAL_SAMPLES:
                        raise ValueError("校准样本太多")
                    samples.append((q["channel"], q["slider"], int(q["value"]), img))
                count = len(self.calibration.get("samples", []))
            return self._send(200, {"ok": True, "count": count})
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
            xmp = acr_preset.build_xmp(str(req.get("name") or "ColorDoctor"), params)
            return self._send(200, xmp.encode("utf-8"), "application/rdf+xml; charset=utf-8")
        return self._send(404, {"error": "not found"})


def _header_text(v: str) -> str:
    """图层名可能含中文：插件用 encodeURIComponent 传。"""
    from urllib.parse import unquote

    return unquote(v or "")


def make_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"sessions": Store(), "refs": Store(4), "calibration": {},
                                                "cal_lock": threading.Lock()})
    srv = ThreadingHTTPServer((HOST, port), handler)
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
