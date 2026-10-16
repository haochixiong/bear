"""在后台线程运行分析，避免大选区时界面卡住。"""

from __future__ import annotations

import traceback

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal


def _log(text: str) -> str:
    try:
        from colordoctor.core.skin_presets import user_config_dir

        d = user_config_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / "error.log"
        with path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        return str(path)
    except OSError:
        return ""


class _Signals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _Task(QRunnable):
    def __init__(self, fn, args, kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:  # 把错误交给界面显示，而不是让程序崩溃
            msg = str(e) or e.__class__.__name__
            logged = _log(traceback.format_exc())
            # 弹窗只给简短说明；技术细节写进 error.log（给开发者看）
            self._emit("failed", f"{msg}\n\n{'详细信息已写入 ' + logged if logged else ''}".strip())
        else:
            self._emit("done", result)
        finally:
            # 算完就放掉选区像素（大图一份上百 MB）；任务对象本身可能还被界面留着一会儿
            self.fn = self.args = self.kwargs = None

    def _emit(self, name, value):
        try:
            getattr(self.signals, name).emit(value)
        except RuntimeError:
            pass  # 窗口在分析完成前已关闭：结果直接丢弃


def run_in_background(fn, *args, on_done=None, on_failed=None, **kwargs) -> _Task:
    task = _Task(fn, args, kwargs)
    if on_done:
        task.signals.done.connect(on_done)
    if on_failed:
        task.signals.failed.connect(on_failed)
    QThreadPool.globalInstance().start(task)
    return task
