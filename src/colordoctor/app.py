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
            (d / "error.log").open("a", encoding="utf-8").write(text + "\n")
        except OSError:
            pass
        sys.__stderr__.write(text)
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
        msg, code = "SELFTEST OK", 0
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

    assert len(palette.load_styles()) >= 28, "内置整图风格模板缺失"
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


def _start_bridge(win) -> None:
    """同时启动给 PS 插件用的本机服务（只监听 127.0.0.1）。端口被占用（已有一个在运行）时跳过。"""
    import threading

    from colordoctor.bridge.server import DEFAULT_PORT, make_server

    if win.settings.value("bridge/enabled", True, type=bool) is False:
        return
    try:
        srv = make_server(DEFAULT_PORT)
    except OSError:
        return
    threading.Thread(target=srv.serve_forever, daemon=True, name="colordoctor-bridge").start()
    win.statusBar().showMessage(f"PS 插件服务已就绪：127.0.0.1:{DEFAULT_PORT}（只接受本机连接）", 8000)


def main(argv: list[str] | None = None) -> int:
    from PyQt6.QtWidgets import QApplication

    from colordoctor.ui.main_window import MainWindow
    from colordoctor.ui.theme import STYLESHEET

    argv = list(sys.argv if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    if len(argv) > 1 and argv[1] in ("analyze", "presets", "styles", "serve"):
        from colordoctor.cli import run

        return run(argv[1:])
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("ColorDoctor")
    app.setOrganizationName("ColorDoctor")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    _install_excepthook(app)
    win = MainWindow()
    win.show()
    _start_bridge(win)
    for a in argv[1:]:
        if Path(a).suffix.lower() in (".jpg", ".jpeg") and Path(a).exists():
            win.open_jpg(a)
            break
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
