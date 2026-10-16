"""程序入口：python -m colordoctor"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


def _install_excepthook(app):
    from PyQt6.QtWidgets import QMessageBox

    from colordoctor.core.skin_presets import user_config_dir

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            d = user_config_dir()
            d.mkdir(parents=True, exist_ok=True)
            with (d / "error.log").open("a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass
        try:
            if sys.__stderr__ is not None:  # 打包后的窗口程序没有控制台
                sys.__stderr__.write(text)
        except Exception:
            pass
        try:
            QMessageBox.critical(None, "出现错误", f"{exc}\n\n详细信息已写入 error.log")
        except Exception:
            pass

    sys.excepthook = hook


def selftest() -> int:
    """打包后自检：核心算法、内置数据文件、界面能否创建。

    结果写入当前目录的 selftest.log（Windows 窗口程序没有控制台输出），
    成功时内容为 SELFTEST OK，返回码 0。
    """
    import os

    log = os.environ.get("COLORDOCTOR_SELFTEST_LOG", "selftest.log")
    try:
        _selftest_body()
        extras = _selftest_optional()
        msg, code = "SELFTEST OK" + "".join(f"\n{x}" for x in extras), 0
    except Exception:
        msg, code = "SELFTEST FAILED\n" + traceback.format_exc(), 1
    try:
        Path(log).write_text(msg + "\n", encoding="utf-8")
    except OSError:
        pass
    if sys.stdout is not None:
        print(msg)
    return code


def _selftest_body() -> None:
    import os

    import numpy as np

    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from colordoctor.core import analysis, skin_presets
    from colordoctor.core.color_management import synthetic_context

    presets = skin_presets.builtin_presets()
    assert len(presets) >= 12, "内置肤色风格缺失"
    from colordoctor.core import palette

    assert len(palette.load_styles()) >= 32, "内置整图颜色模板缺失"
    img = np.empty((120, 160, 3), np.uint8)
    img[:] = (40, 80, 200)
    img.reshape(-1, 3)[::20] = (230, 200, 40)
    rep = analysis.analyze_region(img, synthetic_context())
    assert rep.contamination.opposite_fraction > 0.04, "脏色检测异常"
    app = QApplication.instance() or QApplication([])
    import tempfile

    from colordoctor.ui.main_window import MainWindow

    tmp = tempfile.mkdtemp()
    win = MainWindow(settings=QSettings(os.path.join(tmp, "s.ini"), QSettings.Format.IniFormat))
    win.show()
    for _ in range(20):  # 不进入 exec()：只需让窗口完成布局和首次绘制
        app.processEvents()
    win.grab()
    win.close()
    win.deleteLater()


def _selftest_optional() -> list[str]:
    """可选组件：装了 / 打进包里就必须真的能用（动态库缺失、模型没打包会在这里暴露）。"""
    import importlib.util

    out = []
    if importlib.util.find_spec("rawpy") is not None:
        import rawpy

        out.append(f"rawpy {rawpy.__version__} (LibRaw {'.'.join(map(str, rawpy.libraw_version))})")
    else:
        out.append("rawpy: 未安装（不支持相机 RAW）")
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort

        from colordoctor.core import segmentation as sg

        model = sg.find_model()
        if getattr(sys, "frozen", False):
            assert model is not None, "AI 抠皮肤模型没有打进安装包"
        if model is not None:
            ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])  # 能加载 = 动态库和模型都没问题
        out.append(f"onnxruntime {ort.__version__}，模型 {'已找到' if model else '未找到'}")
    else:
        out.append("onnxruntime: 未安装（没有 AI 抠皮肤）")
    return out


def _start_bridge(win) -> None:
    """同时启动给 PS 插件用的本机服务（只监听 127.0.0.1）。端口被占用（已有一个在运行）时跳过。"""
    import threading

    from colordoctor.bridge.probe import running_version
    from colordoctor.bridge.server import DEFAULT_PORT, make_server

    if not win.setting_bool("bridge/enabled", True):
        return
    try:
        srv = make_server(DEFAULT_PORT)
    except OSError:
        v = running_version(DEFAULT_PORT)
        if v:  # 另一个 ColorDoctor 已经在提供服务，插件照样能用
            win.statusBar().showMessage(f"PS 插件服务由另一个已打开的 ColorDoctor（{v}）提供", 8000)
        else:
            win.statusBar().showMessage(
                f"PS 插件服务没能启动：端口 {DEFAULT_PORT} 被其他程序占用。关掉占用它的程序后重新打开本工具", 0)
        return
    threading.Thread(target=srv.serve_forever, daemon=True, name="colordoctor-bridge").start()
    win.statusBar().showMessage(f"PS 插件服务已就绪：127.0.0.1:{DEFAULT_PORT}（只接受本机连接）", 8000)


CLI_COMMANDS = ("analyze", "presets", "styles", "tones", "serve", "bench")
CLI_FLAGS = ("-h", "--help", "--version")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    argv = [a for a in argv if not a.startswith("-psn_")]  # 旧版 macOS 从 Finder 打开时会多传一个进程号参数
    if "--selftest" in argv:
        return selftest()
    # 只有认识的命令才走命令行；Qt 自己的参数（-platform、-style）和 Xcode 等传的 -NS... 参数照常打开界面
    if len(argv) > 1 and (argv[1] in CLI_COMMANDS or argv[1] in CLI_FLAGS):
        from colordoctor.cli import run

        return run(argv[1:])
    # 界面相关的模块放在这里才导入：命令行 / 服务在没有图形库的机器上也能用
    from PyQt6.QtWidgets import QApplication

    from colordoctor.ui.main_window import MainWindow
    from colordoctor.ui.theme import STYLESHEET

    app = QApplication.instance() or QApplication(argv)
    from colordoctor.ui import gc_guard

    gc_guard.install(app)  # 垃圾回收只在主线程做（见 gc_guard）
    app.setApplicationName("ColorDoctor")
    app.setOrganizationName("ColorDoctor")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    _install_excepthook(app)
    win = MainWindow()
    win.show()
    _start_bridge(win)
    for a in argv[1:]:
        from colordoctor.core.color_management import OTHER_EXTS, RAW_EXTS

        if Path(a).suffix.lower() in (".jpg", ".jpeg", *OTHER_EXTS, *RAW_EXTS) and Path(a).exists():
            win.open_jpg(a)
            break
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
