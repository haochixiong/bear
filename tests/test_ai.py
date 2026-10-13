"""AI 点评：命令行会话、DeepSeek、Claude API（全部用本地假服务/假命令测试，不联网）。"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from colordoctor.ai import prompts
from colordoctor.ai import providers as pv
from colordoctor.core import analysis as an
from colordoctor.core import report_json as rj
from colordoctor.core.color_management import synthetic_context
from synth import skin_region


def _summary():
    return rj.summary(an.analyze_skin_region(skin_region(), synthetic_context(), preset_key="japanese"))


def test_prompt_contains_numbers_and_structure():
    p = prompts.build_prompt(_summary())
    assert "```json" in p and "Lab" in p and "调整步骤" in p


# ------------------------------------------------------------------ 本机命令行会话（用假命令代替 claude/codex）
FAKE = """
import sys
data = sys.stdin.read()
print("收到", len(data), "字；参数", " ".join(sys.argv[1:]))
"""


def test_cli_provider_with_fake_command(tmp_path):
    script = tmp_path / "fake_ai.py"
    script.write_text(FAKE, encoding="utf-8")
    cfg = pv.AIConfig(provider="claude_code", command=f'"{sys.executable}" "{script}" -p')
    r = pv.ask(prompts.build_prompt(_summary()), cfg)
    assert r.ok and "收到" in r.text


def test_codex_image_flag_is_inserted(tmp_path):
    cfg = pv.AIConfig(provider="codex", command=f'"{sys.executable}" exec --x -')
    cmd = pv.build_command(cfg, "/tmp/a.jpg")
    assert cmd[1:5] == ["exec", "-i", "/tmp/a.jpg", "--x"]


def test_missing_cli_gives_friendly_error():
    r = pv.ask("hi", pv.AIConfig(provider="codex", command="definitely-not-installed-xyz exec -"))
    assert not r.ok and "找不到命令" in r.text


def test_cli_nonzero_exit(tmp_path):
    script = tmp_path / "bad.py"
    script.write_text("import sys; sys.stderr.write('未登录'); sys.exit(3)", encoding="utf-8")
    r = pv.ask("hi", pv.AIConfig(provider="claude_code", command=f'"{sys.executable}" "{script}"'))
    assert not r.ok and "未登录" in r.text


# ------------------------------------------------------------------ 本地假 HTTP 服务
class _Server:
    def __init__(self, reply, status=200):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                outer.requests.append((self.path, dict(self.headers), json.loads(self.rfile.read(n))))
                body = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_deepseek_request_and_parse():
    srv = _Server({"choices": [{"message": {"role": "assistant", "content": "肤色偏黄，建议色温往蓝。"}}]})
    try:
        cfg = pv.AIConfig(provider="deepseek", api_key="sk-test", base_url=srv.url + "/chat/completions")
        r = pv.ask("测试", cfg, image_path="/tmp/ignored.jpg")  # DeepSeek 只发文字
        assert r.ok and "色温" in r.text
        path, headers, body = srv.requests[0]
        assert headers["Authorization"] == "Bearer sk-test"
        assert body["model"] == "deepseek-chat" and body["messages"][0]["role"] == "system"
        assert "image" not in json.dumps(body)
    finally:
        srv.close()


def test_deepseek_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    r = pv.ask("x", pv.AIConfig(provider="deepseek"))
    assert not r.ok and "API Key" in r.text


def test_deepseek_401():
    srv = _Server({"error": "bad key"}, status=401)
    try:
        r = pv.ask("x", pv.AIConfig(provider="deepseek", api_key="k", base_url=srv.url))
        assert not r.ok and "401" in r.text
    finally:
        srv.close()


MSG = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [{"type": "text", "text": "高光偏黄，建议颜色分级高光往蓝。"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 10},
}


def test_claude_api_request_shape_and_image(tmp_path):
    pytest.importorskip("anthropic")
    img = tmp_path / "t.jpg"
    from PIL import Image

    Image.new("RGB", (8, 8), (200, 150, 120)).save(img)
    srv = _Server(MSG)
    try:
        cfg = pv.AIConfig(provider="claude_api", api_key="test", base_url=srv.url)
        r = pv.ask("测试", cfg, image_path=str(img))
        assert r.ok and "颜色分级" in r.text
        path, headers, body = srv.requests[0]
        assert path.split("?")[0].endswith("/v1/messages")
        assert body["model"] == "claude-opus-5-5"
        assert body["fallbacks"] == "default"
        assert "server-side-fallback-2026-07-01" in headers.get("anthropic-beta", "")
        assert body["messages"][0]["content"][0]["type"] == "image"
    finally:
        srv.close()


def test_claude_api_refusal():
    pytest.importorskip("anthropic")
    srv = _Server({**MSG, "content": [], "stop_reason": "refusal"})
    try:
        r = pv.ask("x", pv.AIConfig(provider="claude_api", api_key="t", base_url=srv.url))
        assert not r.ok and "拒绝" in r.text
    finally:
        srv.close()


def test_windows_quoted_command_path_is_unquoted(monkeypatch):
    import shutil

    from colordoctor.ai import providers as pv

    monkeypatch.setattr(pv.sys, "platform", "win32")
    seen = []
    monkeypatch.setattr(shutil, "which", lambda a: seen.append(a) or a)
    cfg = pv.AIConfig(provider="claude_code", command='"C:\\Program Files\\nodejs\\claude.cmd" -p')
    cmd = pv.build_command(cfg, None)
    assert seen[0] == "C:\\Program Files\\nodejs\\claude.cmd" and cmd[1] == "-p"


def test_cli_timeout_kills_process(tmp_path):
    import sys

    from colordoctor.ai import providers as pv

    script = tmp_path / "slow.py"
    script.write_text("import time, sys\nsys.stdin.read()\ntime.sleep(30)\n")
    cfg = pv.AIConfig(provider="codex", command=f'"{sys.executable}" "{script}"', timeout=1.0)
    res = pv.ask("x", cfg)
    assert not res.ok and "停止" in res.text
