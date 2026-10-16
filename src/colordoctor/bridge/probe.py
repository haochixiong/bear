"""桌面版启动时问一下本机端口上是不是已经有 ColorDoctor 服务在跑（只连 127.0.0.1，不连外网）。

单独放在这里：server.py 本身只接受连接、从不主动发请求（有测试检查）。
"""

from __future__ import annotations

import http.client
import json

from colordoctor.bridge.server import CLIENT_HEADER, DEFAULT_PORT, HOST


def running_version(port: int = DEFAULT_PORT, timeout: float = 1.0) -> str | None:
    """这个端口上已经在跑的 ColorDoctor 服务的版本；没有服务、或者是别的程序时返回 None。"""
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        conn.request("GET", "/health", headers={CLIENT_HEADER: "desktop"})
        r = conn.getresponse()
        if r.status != 200:
            return None
        data = json.loads(r.read(4096).decode("utf-8"))
        return str(data["version"]) if isinstance(data, dict) and data.get("ok") else None
    except (OSError, ValueError, KeyError, http.client.HTTPException):
        return None
    finally:
        conn.close()
