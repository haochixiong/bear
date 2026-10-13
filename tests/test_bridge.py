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
    assert r["localRange"] == 2 and r["hue"] < 0  # 黄色往红拉回
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
