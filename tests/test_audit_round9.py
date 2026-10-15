"""第九轮审计（调整动作与实际、PS / 桌面版兼容、性能与指标）的回归测试。"""

import json
import socket
import threading
import urllib.request

import numpy as np
import pytest
from synth import face_scene

from colordoctor.bridge import server as bs


# ------------------------------------------------------------------ 兼容性：会话、版本、端口
def test_store_keeps_recently_used_sessions():
    """正在用的结果（反复生成图层 / 取蒙版）不会因为别的页面又分析了几次就被挤掉。"""
    s = bs.Store(3)
    keep = s.put("keep")
    for i in range(10):
        s.put(i)
        assert s.get(keep) == "keep"


def test_health_reports_protocol_version():
    srv = bs.make_server(0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/health", headers={bs.CLIENT_HEADER: "t"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assert json.loads(r.read())["protocol"] >= 3
    finally:
        srv.shutdown()
        srv.server_close()


def test_second_server_on_the_same_port_fails():
    """端口被占用时第二个服务必须绑定失败（Windows 上默认的 SO_REUSEADDR 会让两个服务同时监听）。"""
    srv = bs.make_server(0)
    try:
        with pytest.raises(OSError):
            bs.make_server(srv.server_address[1])
        assert srv.allow_reuse_address is False
    finally:
        srv.server_close()


def test_skin_patch_quick_rule_follows_the_chosen_preset():
    """肤色取色按「人像肤色」页选的肤色风格判断，和那一页、桌面版一致（以前总用全局规则）。"""
    px = np.full((4, 4, 3), (192, 148, 119), np.uint8)  # Lab 约 (65, 14, 22)
    glob = bs.patch_values(px)
    nat = bs.patch_values(px, "natural")
    assert nat["quick_rule_preset"] and not glob["quick_rule_preset"]
    assert nat["quick_rule"]["b_range"] != glob["quick_rule"]["b_range"]
    with pytest.raises(ValueError):
        bs.patch_values(px, "nosuch")


def test_ai_skin_without_skin_falls_back_like_the_desktop(monkeypatch):
    from colordoctor.core import segmentation as sg

    class Empty:
        def part(self, _p):
            return np.zeros((200, 200), bool)

    monkeypatch.setattr(sg, "segment_skin", lambda rgb: Empty())
    img = np.dstack([face_scene()[0], np.full((200, 200), 255, np.uint8)])
    r = bs.analyze_request(img, "skin", {"seg": "ai", "part": "脸部"}, bs.Store(), bs.Store(), layer_kind="pixel")
    assert any("已改用肤色规则" in n for n in r["notes"])


# ------------------------------------------------------------------ 性能与指标
def test_results_do_not_depend_on_resolution():
    """同一张图的 45 MP 原图和 PS 缩小到 2048 的图（插件传来的就是这种），脏色扫描结论一致（以前 20 分 vs 0.6 分）。"""
    from PIL import Image

    from colordoctor.core import analysis as an
    from colordoctor.core.color_management import synthetic_context

    rng = np.random.default_rng(1)
    big = np.clip(np.array([70, 130, 200], float) + rng.normal(0, 4, (3000, 4500, 3)), 0, 255).astype(np.uint8)
    small = np.asarray(Image.fromarray(big).resize((2048, 1365), Image.Resampling.BOX))
    r1 = an.scan_image(big, synthetic_context())
    r2 = an.scan_image(small, synthetic_context())
    assert abs(r1.overall_risk - r2.overall_risk) < 8 and r1.overall_severity == r2.overall_severity


def test_grid_downsample_is_an_area_average():
    from colordoctor.core.sampling import grid_downsample

    img = np.zeros((1000, 1000, 3), np.uint8)
    img[::2, ::2] = 255  # 1/4 的像素是白点
    sub, step = grid_downsample(img, 10_000)
    assert step == 10 and abs(float(sub.mean()) - 255 / 4) < 2  # 以前只取每格中间一个像素：全黑或全白
    sub, _ = grid_downsample(np.zeros((3, 200_000, 3), np.uint8), 1000)
    assert sub.size > 0


def test_transparent_area_does_not_dilute_the_scan():
    from colordoctor.core import contamination as ct

    rng = np.random.default_rng(0)
    img = np.clip(np.array([60, 110, 190], float) + rng.normal(0, 2, (300, 300, 3)), 0, 255).astype(np.uint8)
    img[rng.random((300, 300)) < 0.03] = (230, 200, 40)
    full = ct.analyze(img, mode="scan")
    pad = np.zeros((300, 600, 3), np.uint8)  # 不超过取样上限：两次分析用的是同一批像素
    pad[:, :300] = img
    valid = np.zeros((300, 600), bool)
    valid[:, :300] = True
    half = ct.analyze(pad, mode="scan", valid=valid)
    assert abs(half.opposite_fraction - full.opposite_fraction) < 0.2 * full.opposite_fraction + 1e-4


def test_saturated_colour_is_not_counted_as_clipped():
    from colordoctor.core import region_statistics as rs

    st = rs.compute_stats(np.full((50, 50, 3), (0, 40, 200), np.uint8), with_clusters=False)
    assert st.clipped_fraction == 0


def test_colourfulness_labels_use_the_nearest_anchor():
    from colordoctor.core import palette as pl

    label = lambda m: next(n for lim, n in pl.COLORFULNESS_LEVELS if m < lim)  # noqa: E731
    assert label(82) == "很鲜艳" and label(109) == "极鲜艳" and label(45) == "中等"


# ------------------------------------------------------------------ 调整图层在 PS 里的实际效果
def test_tone_curves_stay_monotonic_in_ps_spline():
    """PS 的曲线是平滑样条：生成的影调曲线在样条下也不往回走、不把暗部压成一片死黑。"""
    from synth import highkey_face_scene, lowkey_face_scene

    from colordoctor.core import analysis as an
    from colordoctor.core import tone_class as tc
    from colordoctor.core.color_management import synthetic_context

    for img in (face_scene()[0], lowkey_face_scene()[0], highkey_face_scene()[0]):
        for key in tc.TONE_STYLES:
            rep = an.analyze_palette_report(img, synthetic_context(), tone_style=key)
            for a in rep.findings[0].adjustments:
                if a.layer:
                    pts = a.layer["params"]["channels"]["composite"]
                    lut = tc.spline_lut(pts)
                    assert np.diff(lut).min() > -1.0, (key, pts)  # 不往回走（明暗不翻转）
                    assert lut[pts[1][0] // 2] > 0 or pts[1][1] == 0, (key, pts)  # 暗部不被压成一片死黑
                    assert all(abs(lut[x] - y) < 1 for x, y in pts), (key, pts)  # 卡片上的点就是实际曲线经过的点


def test_ps_saturation_increase_is_not_linear():
    """PS 色相/饱和度加饱和度时色度 ×1/(1−s)：想 ×1.4 只要 +29（以前给 +40，实际 ×1.67）。"""
    from colordoctor.core import native_layers as nl
    from colordoctor.core import ps_models as pm

    assert round(nl.sat_for_ratio(1.4)) == 29 and round(nl.sat_for_ratio(0.7)) == -30
    assert abs(pm.sat_multiplier(nl.sat_for_ratio(1.4)) - 1.4) < 1e-9


def test_luminosity_tone_curve_is_generated_last():
    """影调曲线（明度混合）放在最上面：下面的颜色图层都是按原图的明暗算的。"""
    from colordoctor.core import analysis as an
    from colordoctor.core import layer_plan as lp
    from colordoctor.core.color_management import synthetic_context

    img = face_scene()[0]
    rep = an.analyze_palette_report(img, synthetic_context(), style_key="teal_orange", tone_style="japanese")
    layers = lp.plan_from_report(rep, img.shape).layers
    assert layers[-1].blend == "luminosity" and all(ly.blend != "luminosity" for ly in layers[:-1])


def test_skin_dull_patch_uses_huesat_not_vibrance():
    from colordoctor.core import native_layers as nl

    kind, p, text = nl.skin_patch_layer("缺色发灰", 0.0, -0.2, (66.0, 10.0, 14.0))
    assert kind == "hueSaturation" and "饱和度" in text
