"""AI 点评的几种方式（全部可选）：

- claude_code：调用本机已登录的 Claude Code 命令行会话（claude -p），无需在本工具里保存 API Key；
- codex：调用本机已登录的 Codex 命令行会话（codex exec）；
- deepseek：DeepSeek API（OpenAI 兼容接口，仅文字）；
- claude_api：Claude API（官方 anthropic SDK，可附图）。

命令行方式的命令模板可在设置中修改（{image} 会被替换为临时缩略图路径）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from colordoctor.ai.prompts import SYSTEM

PROVIDERS = {
    "claude_code": "本机 Claude Code 会话（claude -p）",
    "codex": "本机 Codex 会话（codex exec）",
    "deepseek": "DeepSeek API",
    "claude_api": "Claude API",
}
SUPPORTS_IMAGE = {"claude_code": True, "codex": True, "deepseek": False, "claude_api": True}

DEFAULT_COMMANDS = {
    "claude_code": "claude -p",
    "codex": "codex exec --skip-git-repo-check -",
}
CLAUDE_MODEL = "claude-opus-5-5"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


@dataclass
class AIConfig:
    provider: str = "claude_code"
    send_image: bool = False
    command: str = ""  # 覆盖默认命令（命令行方式）
    api_key: str = ""  # DeepSeek / Claude API 的密钥（留空则读环境变量）
    model: str = ""
    base_url: str = ""  # 可选：自定义接口地址（测试或代理）
    timeout: float = 300.0


@dataclass
class AIResult:
    ok: bool
    text: str
    provider: str


class AIError(Exception):
    pass


def ask(prompt: str, cfg: AIConfig, image_path: str | None = None) -> AIResult:
    if cfg.provider not in PROVIDERS:
        raise AIError(f"未知的 AI 方式：{cfg.provider}")
    if image_path and not SUPPORTS_IMAGE[cfg.provider]:
        image_path = None
    try:
        if cfg.provider in ("claude_code", "codex"):
            text = _run_cli(prompt, cfg, image_path)
        elif cfg.provider == "deepseek":
            text = _deepseek(prompt, cfg)
        else:
            text = _claude_api(prompt, cfg, image_path)
    except AIError as e:
        return AIResult(False, str(e), cfg.provider)
    return AIResult(True, text.strip(), cfg.provider)


# ------------------------------------------------------------------ 本机命令行会话
def _cli_env() -> dict:
    """运行命令行工具用的环境变量。

    macOS 从“访达”打开的 .app 只拿到最小的 PATH（/usr/bin:/bin:/usr/sbin:/sbin），找不到 Homebrew / npm 装的
    claude、codex，npm 装的命令开头的 #!/usr/bin/env node 也找不到 node：把常见的安装目录补到 PATH 后面。"""
    env = dict(os.environ)
    if sys.platform.startswith("win"):
        return env
    import glob

    home = os.path.expanduser("~")
    extra = ["/opt/homebrew/bin", "/usr/local/bin", f"{home}/.local/bin", f"{home}/.npm-global/bin",
             f"{home}/.volta/bin", f"{home}/.bun/bin", f"{home}/.claude/local",
             *sorted(glob.glob(f"{home}/.nvm/versions/node/*/bin"), reverse=True)]
    have = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    env["PATH"] = os.pathsep.join(have + [d for d in extra if d not in have and os.path.isdir(d)])
    return env


def build_command(cfg: AIConfig, image_path: str | None) -> list[str]:
    import shlex

    template = cfg.command.strip() or DEFAULT_COMMANDS[cfg.provider]
    args = shlex.split(template, posix=not sys.platform.startswith("win"))
    # Windows 下 posix=False 会保留引号："C:\Program Files\...\claude.cmd" -> 去掉两端引号
    args = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a for a in args]
    if image_path:
        if "{image}" in template:
            args = [a.replace("{image}", image_path) for a in args]
        elif cfg.provider == "codex":
            args = args[:2] + ["-i", image_path] + args[2:]
        elif cfg.provider == "claude_code":
            args = args + ["--allowedTools", "Read"]
    exe = shutil.which(args[0], path=_cli_env()["PATH"])
    if exe is None:
        raise AIError(f"找不到命令“{args[0]}”。请先安装并登录对应的命令行工具，或在设置中填写完整路径")
    return [exe] + args[1:]


def _run_cli(prompt: str, cfg: AIConfig, image_path: str | None) -> str:
    cmd = build_command(cfg, image_path)
    full = SYSTEM + "\n\n" + prompt
    if image_path and cfg.provider == "claude_code":
        full += f"\n\n（图片缩略图在：{image_path}，请先查看它再回答）"
    win = sys.platform.startswith("win")
    flags = (0x08000000 | 0x00000200) if win else 0  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", creationflags=flags, start_new_session=not win,
            env=_cli_env(), cwd=tempfile.gettempdir(),  # 从访达打开时当前目录是 /，不让命令行工具在那里干活
        )
    except OSError as e:
        raise AIError(f"无法启动命令：{e}") from e
    try:
        out, err = proc.communicate(full, timeout=cfg.timeout)
    except subprocess.TimeoutExpired as e:
        _kill_tree(proc)
        raise AIError(f"等待 {cfg.timeout:.0f} 秒仍未返回，已停止") from e
    if proc.returncode != 0:
        raise AIError(f"命令返回错误（{proc.returncode}）：{(err or out).strip()[:800]}")
    return out


def _kill_tree(proc: subprocess.Popen) -> None:
    """超时后连同子进程一起结束（npm 的 .cmd 包装会再启动 node.exe，只杀 cmd.exe 会让管道一直不关）。"""
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=15,
                           creationflags=0x08000000)
        else:
            import os
            import signal

            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.communicate(timeout=5)
    except Exception:
        pass


# ------------------------------------------------------------------ DeepSeek（OpenAI 兼容接口，仅文字）
def _deepseek(prompt: str, cfg: AIConfig) -> str:
    import urllib.error
    import urllib.request

    key = cfg.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise AIError("没有 DeepSeek API Key：请在设置中填写，或设置环境变量 DEEPSEEK_API_KEY")
    body = json.dumps(
        {
            "model": cfg.model or DEEPSEEK_MODEL,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        cfg.base_url or DEEPSEEK_URL,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code == 401:
            raise AIError("DeepSeek API Key 无效（401）") from e
        if e.code == 429:
            raise AIError("DeepSeek 请求过于频繁或余额不足（429），请稍后再试") from e
        raise AIError(f"DeepSeek 返回错误 {e.code}：{detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise AIError(f"无法连接 DeepSeek：{e}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise AIError(f"DeepSeek 返回格式无法识别：{str(data)[:300]}") from e


# ------------------------------------------------------------------ Claude API（官方 SDK）
def _claude_api(prompt: str, cfg: AIConfig, image_path: str | None) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise AIError("未安装 anthropic：请运行 pip install anthropic") from e
    kwargs = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    client = anthropic.Anthropic(timeout=cfg.timeout, **kwargs)
    content = []
    if image_path:
        import base64

        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    content.append({"type": "text", "text": prompt})
    try:
        # 服务端拒答时自动换模型重试（server-side fallback）
        resp = client.beta.messages.create(
            model=cfg.model or CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError as e:
        raise AIError("Claude API 认证失败：请检查 API Key（或环境变量 ANTHROPIC_API_KEY）") from e
    except anthropic.PermissionDeniedError as e:
        raise AIError("Claude API 权限不足") from e
    except anthropic.RateLimitError as e:
        raise AIError("Claude API 请求过于频繁，请稍后再试") from e
    except anthropic.BadRequestError as e:
        raise AIError(f"Claude API 请求无效：{e.message}") from e
    except anthropic.APIStatusError as e:
        raise AIError(f"Claude API 错误 {e.status_code}：{e.message}") from e
    except anthropic.APIConnectionError as e:
        raise AIError("无法连接 Claude API，请检查网络") from e
    if resp.stop_reason == "refusal":
        raise AIError("模型拒绝回答该请求")
    text = "".join(b.text for b in resp.content if b.type == "text")
    if resp.stop_reason == "max_tokens":
        text += "\n\n（回答因长度上限被截断）"
    return text
