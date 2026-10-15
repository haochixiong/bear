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
