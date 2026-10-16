"""把 Photoshop 插件打成 .ccx 安装包（就是一个 zip：manifest.json 在最外层）。Windows / Mac 通用。

用法：python packaging/build_plugin.py  -> dist/ColorDoctor-plugin-<版本>.ccx
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "photoshop-plugin"
SKIP_DIRS = {"test", "node_modules", ".git"}
SKIP_SUFFIX = {".md"}


def build(out_dir: Path = ROOT / "dist") -> Path:
    manifest = json.loads((PLUGIN / "manifest.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ColorDoctor-plugin-{manifest['version']}.ccx"
    files = sorted(p for p in PLUGIN.rglob("*") if p.is_file()
                   and not (set(p.relative_to(PLUGIN).parts[:-1]) & SKIP_DIRS) and p.suffix not in SKIP_SUFFIX)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(PLUGIN).as_posix(), date_time=(2024, 1, 1, 0, 0, 0))  # 固定时间：同样的内容打出同样的包
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    return out


if __name__ == "__main__":
    path = build()
    print(path)
    sys.exit(0)
