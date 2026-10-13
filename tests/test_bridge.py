"""PS 插件桥接：图层方案、图层检查、本机 HTTP 服务。"""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest
from synth import flat, lch_color, noisy, skin_region, sprinkle

from colordoctor.bridge import server as bs
from colordoctor.core import analysis as an
from colordoctor.core import color_spaces as cs
from colordoctor.core import layer_plan as lp
from colordoctor.core import pixel_guard as pg
from colordoctor.core.color_management import synthetic_context

BLUE = lch_color(55, 40, 265)
YELLOW = lch_color(80, 60, 90)


def dirty_sky(h=200, w=300):
    img = noisy(flat(BLUE, h, w), 2)
    right = sprinkle(img[:, 150:], YELLOW, 0.06, seed=3)
    img[:, 150:] = right
    return img


def yellow_patch_skin():
    img = skin_region(68, 14, 19, h=200, w=200)
    lab = cs.srgb8_to_lab(img[120:180, 20:90])
    L = lab[..., 0]
    patch = cs.lch_to_lab(np.stack([L, np.full_like(L, 25.0), np.full_like(L, 70.0)], -1))
    img[120:180, 20:90] = cs.lab_to_srgb8(patch)
    return img


# ------------------------------------------------------------------ 蒙版工具
def test_feather_and_resize():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    f = lp.feather(m, 6)
    assert f.dtype == np.uint8 and f[20, 20] == 255 and f[0, 0] == 0
    assert 0 < f[10, 20] < 255  # 边缘是渐变
    big = lp.resize_mask(f, 400, 200)
    assert big.shape == (200, 400) and big[100, 200] == 255 and big[0, 0] == 0


def test_ps_range_mapping():
    assert lp.ps_range_for_lab_hue(90)[1] == "黄色"
    assert lp.ps_range_for_lab_hue(265)[1] == "蓝色"
    assert lp.ps_range_for_lab_hue(30)[1] == "红色"


# ------------------------------------------------------------------ 图层方案
def test_dirty_sky_plan_masks_only_the_specks():
    img = dirty_sky()
    rep = an.scan_image(img, synthetic_context())
    plan = lp.plan_from_report(rep, img.shape)
    opp = [ly for ly in plan.layers if ly.id == "opposite"]
    assert opp, [ly.reason for ly in plan.layers]
    ly = opp[0]
    assert ly.kind == "hueSaturation"
    rng = ly.params["ranges"][0]
    assert rng["localRange"] == 2 and rng["saturation"] < 0  # 黄色 降饱和
    assert ly.mask.shape == img.shape[:2]
    left, right = ly.mask[:, :140].mean(), ly.mask[:, 160:].mean()
    assert right > 5 * max(left, 1)  # 只作用在有杂色的右半边
    js = plan.to_json()
    assert js["layers"][0]["has_mask"] and "mask" not in js["layers"][0]
    json.dumps(js)


def test_clean_image_gives_empty_plan():
    img = noisy(flat(BLUE, 120, 160), 1)
    plan = lp.plan_from_report(an.scan_image(img, synthetic_context()), img.shape)
    assert plan.layers == []


def test_skin_yellow_patch_plan():
    img = yellow_patch_skin()
    rep = an.analyze_skin_region(img, synthetic_context(), auto_mask=False)
    plan = lp.plan_from_report(rep, img.shape)
    yellow = [ly for ly in plan.layers if "偏黄" in ly.name]
    assert yellow, [ly.name for ly in plan.layers]
    ly = yellow[0]
    assert ly.kind == "hueSaturation"
    r = ly.params["ranges"][0]
    assert r["hue"] < 0  # 往红拉回
    # 范围以问题块的 PS 色相为中心（而不是固定的“黄色”范围，肤色只落在它的边缘）
    from colordoctor.core import acr_channels as ac

    center = ac.lab_hue_to_hsb_hue(70.0, L=68, C=25)
    lo, hi = r["beginSustain"], r["endSustain"]
    assert (center - lo) % 360 <= 25 and (hi - center) % 360 <= 25
    assert ly.mask[150, 55] > 200 and ly.mask[30, 150] == 0


# ------------------------------------------------------------------ 图层检查
def test_guard_blocks_adjustment_and_empty_layers():
    with pytest.raises(pg.BlockedError, match="调整图层"):
        pg.check_layer_kind("curves", "曲线 1")
    pg.check_layer_kind("pixel")
    empty = np.zeros((50, 50, 4), np.uint8)
    with pytest.raises(pg.BlockedError, match="空白"):
        pg.prepare(empty)
    solid = np.full((50, 50, 3), 120, np.uint8)
    with pytest.raises(pg.BlockedError, match="纯色"):
        pg.prepare(solid)


def test_guard_crops_partially_transparent_layer():
    rgba = np.zeros((100, 100, 4), np.uint8)
    rgba[20:60, 30:90, :3] = dirty_sky(40, 60)
    rgba[20:60, 30:90, 3] = 255
    p = pg.prepare(rgba)
    assert p.offset == (30, 20) and p.rgb.shape == (40, 60, 3)
    assert p.notes


# ------------------------------------------------------------------ 服务核心逻辑
def test_analyze_request_and_mask_bytes():
    img = dirty_sky()
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    sessions, refs = bs.Store(), bs.Store()
    res = bs.analyze_request(rgba, "dirty", {}, sessions, refs, doc_size=(600, 400), layer_kind="pixel", layer_name="背景")
    assert res["plan"]["layers"] and res["findings"]
    json.dumps(res, ensure_ascii=False)
    sess = sessions.get(res["session"])
    data, w, h = bs.mask_bytes(sess, "opposite")
    assert (w, h) == (600, 400) and len(data) == 600 * 400
    m = np.frombuffer(data, np.uint8).reshape(400, 600)
    assert m[:, 320:].mean() > m[:, :280].mean()


def test_analyze_request_blocks_adjustment_layer():
    with pytest.raises(pg.BlockedError):
        bs.analyze_request(np.zeros((10, 10, 4), np.uint8), "dirty", {}, bs.Store(), bs.Store(), layer_kind="hueSaturation")


def test_store_evicts_old_sessions():
    st = bs.Store(limit=2)
    a = st.put(1)
    st.put(2)
    st.put(3)
    assert st.get(a) is None


# ------------------------------------------------------------------ HTTP（仅本机回环）
@pytest.fixture
def bridge():
    srv = bs.make_server(0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _req(url, data=None, headers=None, method=None):
    h = {bs.CLIENT_HEADER: "test"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_http_requires_client_header_and_rejects_browser_origin(bridge):
    req = urllib.request.Request(bridge + "/health")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 403
    code, _, _ = _req(bridge + "/health", headers={"Origin": "https://evil.example"})
    assert code == 403
    code, _, body = _req(bridge + "/health")
    assert code == 200 and json.loads(body)["ok"]


def test_http_analyze_and_fetch_mask(bridge):
    img = dirty_sky()
    h, w = img.shape[:2]
    hdr = {"X-Width": str(w), "X-Height": str(h), "X-Components": "3", "X-Doc-Width": str(w * 2), "X-Doc-Height": str(h * 2),
           "X-Layer-Kind": "pixel", "X-Layer-Name": urllib.parse.quote("天空")}
    code, _, body = _req(bridge + "/analyze?mode=dirty", img.tobytes(), hdr)
    assert code == 200, body
    res = json.loads(body)
    assert res["layer"] == "天空"
    lid = res["plan"]["layers"][0]["id"]
    code, headers, data = _req(f"{bridge}/mask?session={res['session']}&layer={lid}")
    assert code == 200 and len(data) == w * 2 * h * 2 and headers["X-Width"] == str(w * 2)


def test_http_blocked_layer_returns_422(bridge):
    hdr = {"X-Width": "10", "X-Height": "10", "X-Components": "4"}
    code, _, body = _req(bridge + "/analyze?mode=dirty", bytes(400), hdr)
    assert code == 422 and "空白" in json.loads(body)["blocked"]
    code, _, body = _req(bridge + "/analyze?mode=dirty", bytes(300), hdr)
    assert code == 400  # 字节数不符


def test_http_acr_preset_and_calibration(bridge, tmp_path, monkeypatch):
    from colordoctor.core import acr_calibration as cal
    from colordoctor.core import acr_channels as ac

    monkeypatch.setattr("colordoctor.core.skin_presets.user_config_dir", lambda: tmp_path)
    body = json.dumps({"name": "测试", "params": {"hsl": {"橙色": {"hue": -4}}}}).encode()
    code, _, x = _req(bridge + "/acr-preset", body, {"Content-Type": "application/json"})
    assert code == 200 and b'crs:HueAdjustmentOrange="-4"' in x
    code, h, data = _req(bridge + "/calibration/chart")
    w, hh = int(h["X-Width"]), int(h["X-Height"])
    img = np.frombuffer(data, np.uint8).reshape(hh, w, 3)
    hdr = {"X-Width": str(w), "X-Height": str(hh), "X-Components": "3"}
    assert _req(bridge + "/calibration/sample?slider=base", img.tobytes(), hdr)[0] == 200
    _, patches = cal.chart()
    out = img.copy()
    for p in patches:
        if p.channel == "蓝色":
            x, y, pw, ph = p.rect
            lab = cs.srgb8_to_lab(out[y:y + ph, x:x + pw])
            lab[..., 0] += 10
            out[y:y + ph, x:x + pw] = cs.lab_to_srgb8(lab)
    assert _req(bridge + "/calibration/sample?channel=%E8%93%9D%E8%89%B2&slider=lum&value=50", out.tobytes(), hdr)[0] == 200
    code, _, body = _req(bridge + "/calibration/fit", b"{}")
    assert code == 200, body
    res = json.loads(body)
    assert res["channels"]["蓝色"]["lum_per_unit"] == pytest.approx(0.2, abs=0.04)
    assert (tmp_path / "acr_calibration.json").exists()
    ac.set_calibration({})
    code, _, body = _req(bridge + "/calibration/plan")
    assert len(json.loads(body)["steps"]) == 8 * 3 * 4


def test_patch_mode_returns_values_even_for_solid_color():
    from synth import lch_color as lc

    px = np.zeros((5, 5, 4), np.uint8)
    px[..., :3] = lc(68, 24.6, 53)
    px[..., 3] = 255
    res = bs.analyze_request(px, "patch", {}, bs.Store(), bs.Store(), layer_kind="pixel")
    assert res["values"]["lab"][0] == pytest.approx(68, abs=1)
    assert res["quick_rule"] is not None and res["purity"] == 1.0
    assert {c["channel"] for c in res["acr_channels"]} <= {"红色", "橙色", "黄色"}


def test_http_options(bridge):
    code, _, body = _req(bridge + "/options")
    res = json.loads(body)
    assert code == 200 and len(res["styles"]) >= 28 and res["presets"] and "绿" in res["families"]


def test_reference_plan_has_curve_and_acr_params():
    from synth import lch_color as lc

    mine = noisy(flat(lc(60, 30, 250), 120, 160), 2)
    mine[:, :60] = lc(70, 25, 60)
    ref = noisy(flat(lc(35, 30, 205), 120, 160), 2)
    ref[:, :60] = lc(55, 35, 55)
    refs = bs.Store()
    rid = refs.put(ref)
    res = bs.analyze_request(mine, "reference", {"ref": rid}, bs.Store(), refs, layer_kind="pixel")
    kinds = {ly["id"]: ly["kind"] for ly in res["plan"]["layers"]}
    assert kinds.get("ref_curve") == "curves", kinds
    assert "curve" in res["acr_params"] and res["acr_params"]["curve"]["composite"]
    from colordoctor.core import acr_preset as ap

    assert "ToneCurvePV2012" in ap.build_xmp("t", res["acr_params"])


def test_mask_bytes_full_document_placement():
    img = dirty_sky()
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    sessions = bs.Store()
    res = bs.analyze_request(rgba, "dirty", {}, sessions, bs.Store(), doc_size=(300, 200))
    sess = sessions.get(res["session"])
    data, w, h = bs.mask_bytes(sess, "opposite", 300, 200, place=(50, 40, 500, 400))
    m = np.frombuffer(data, np.uint8).reshape(400, 500)
    assert (w, h) == (500, 400)
    assert m[:40].max() == 0 and m[:, :50].max() == 0 and m[240:].max() == 0  # 图层范围外为 0
    assert m[40:240, 50:350].max() == 255
    with pytest.raises(ValueError):
        bs.mask_bytes(sess, "opposite", -5, 10)
    with pytest.raises(ValueError):
        bs.mask_bytes(sess, "opposite", 60000, 60000)


def test_http_rejects_bad_host_and_bad_sizes(bridge):
    port = bridge.rsplit(":", 1)[1]
    code, _, _ = _req(bridge + "/health", headers={"Host": f"evil.example:{port}"})
    assert code == 403
    hdr = {"X-Width": "-5", "X-Height": "10", "X-Components": "3"}
    code, _, body = _req(bridge + "/analyze?mode=dirty", b"", hdr)
    assert code == 400
    code, _, body = _req(bridge + "/acr-preset", b"[]", {"Content-Type": "application/json"})
    assert code == 400 and "JSON" in json.loads(body)["error"]
    code, _, body = _req(bridge + "/calibration/fit", b"{}")
    assert code == 400


def test_http_internal_errors_return_500(bridge, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(bs, "analyze_request", boom)
    img = dirty_sky(20, 20)
    hdr = {"X-Width": "20", "X-Height": "20", "X-Components": "3"}
    code, _, body = _req(bridge + "/analyze?mode=dirty", img.tobytes(), hdr)
    assert code == 500 and "RuntimeError" in json.loads(body)["error"]


def test_calibration_rejects_wrong_size_and_empty_fit(bridge):
    hdr = {"X-Width": "10", "X-Height": "10", "X-Components": "3"}
    code, _, body = _req(bridge + "/calibration/sample?slider=base", bytes(300), hdr)
    assert code == 400 and "色卡尺寸" in json.loads(body)["error"]


def test_hsb_shift_differs_from_lab_degrees():
    # 绿色区域：Lab 10° 约等于 HSB 20°；肤色区域：Lab 10° 只有 HSB 5° 左右
    assert abs(lp.hsb_shift(130, 140, 55, 40)) > 14
    assert abs(lp.hsb_shift(50, 60, 68, 25)) < 8


def test_transparent_pixels_do_not_count_as_grey():
    from colordoctor.core import palette as pl

    rgba = np.zeros((100, 100, 4), np.uint8)
    yy, xx = np.mgrid[0:100, 0:100]
    disc = (yy - 50) ** 2 + (xx - 50) ** 2 < 40 ** 2
    rgba[disc, :3] = lch_color(50, 60, 30)
    rgba[disc, 3] = 255
    prep = pg.prepare(rgba)
    assert prep.valid is not None and not prep.valid.all()
    filled = prep.filled_for_statistics()
    grey_before = [c.family for c in pl.ratio_631(cs.srgb8_to_lab(prep.rgb))[0]]
    fams = [c.family for c in pl.ratio_631(cs.srgb8_to_lab(filled))[0]]
    assert "灰" in grey_before and "灰" not in fams, fams
    res = bs.analyze_request(rgba, "dirty", {}, bs.Store(), bs.Store())
    assert res["plan"]["layers"] == []


def test_http_error_codes_round2(bridge):
    code, _, _ = _req(bridge + "/mask?session=nope&layer=x")
    assert code == 404
    code, _, body = _req(bridge + "/acr-preset", json.dumps({"params": {"hsl": {"红色": 5}}}).encode())
    assert code == 400
    code, _, body = _req(bridge + "/acr-preset", json.dumps({"params": {"curve": {"composite": []}}}).encode())
    assert code in (200, 400)
    from colordoctor.core import acr_calibration as cal

    img, _ = cal.chart()
    h = {"X-Width": str(img.shape[1]), "X-Height": str(img.shape[0]), "X-Components": "3"}
    code, _, body = _req(bridge + "/calibration/sample?slider=hue&value=10", img.tobytes(), h)
    assert code == 400 and "通道" in json.loads(body)["error"]
    code, _, body = _req(bridge + "/calibration/sample?slider=hue&value=10&channel=%E7%B4%AB%E7%BA%A2", img.tobytes(), h)
    assert code == 400


def test_skin_fallback_note_and_no_part_label(monkeypatch):
    from synth import face_scene

    from colordoctor.core import segmentation as sg

    def boom(*a, **k):
        raise sg.SegmentationUnavailable("找不到模型文件")

    monkeypatch.setattr(sg, "segment_skin", boom)
    img, _ = face_scene(160, 160)
    res = bs.analyze_request(img, "skin", {"seg": "ai", "part": "脸部"}, bs.Store(), bs.Store())
    assert any("肤色规则" in n for n in res["notes"])
    assert not any("脸部" in f["problem"] for f in res["findings"])


def test_large_image_plan_is_fast_and_masks_align():
    import time

    img = noisy(flat(BLUE, 3000, 4000), 2)
    img[:, 2000:] = sprinkle(img[:, 2000:], YELLOW, 0.05, seed=4)
    rep = an.scan_image(img, synthetic_context())
    t = time.perf_counter()
    plan = lp.plan_from_report(rep, img.shape)
    dt = time.perf_counter() - t
    assert dt < 5.0, dt
    full = plan.full_mask("opposite")
    assert full.shape == img.shape[:2]
    assert full[:, 2200:].mean() > 5 * max(full[:, :1800].mean(), 1)
