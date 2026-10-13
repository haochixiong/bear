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


def main(argv: list[str] | None = None) -> int:
    from PyQt6.QtWidgets import QApplication

    from colordoctor.ui.main_window import MainWindow
    from colordoctor.ui.theme import STYLESHEET

    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("ColorDoctor")
    app.setOrganizationName("ColorDoctor")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    _install_excepthook(app)
    win = MainWindow()
    win.show()
    for a in argv[1:]:
        if Path(a).suffix.lower() in (".jpg", ".jpeg") and Path(a).exists():
            win.open_jpg(a)
            break
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
