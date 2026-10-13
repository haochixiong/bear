"""用 Node 跑 PS 插件的 JS（假的 photoshop 模块 + 真实的本机服务）。没有 Node 时跳过。"""

import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from colordoctor.bridge import server as bs

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="没有安装 Node.js")
def test_plugin_js_against_bridge():
    srv = bs.make_server(0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        r = subprocess.run([NODE, str(ROOT / "photoshop-plugin" / "test" / "run.js"), str(srv.server_address[1])],
                           capture_output=True, text=True, timeout=120)
    finally:
        srv.shutdown()
        srv.server_close()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PLUGIN JS OK" in r.stdout


def test_manifest_is_valid_json_and_loopback_only():
    import json

    m = json.loads((ROOT / "photoshop-plugin" / "manifest.json").read_text(encoding="utf-8"))
    assert m["manifestVersion"] == 5 and m["host"][0]["app"] == "PS"
    assert m["requiredPermissions"]["network"]["domains"] == ["http://127.0.0.1:47321"]
