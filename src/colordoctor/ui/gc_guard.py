"""只在界面主线程里做 Python 的循环垃圾回收。

自动回收会在“正好分配内存的那个线程”里触发；分析线程分配得很多，回收常在那里发生。
如果被回收的循环里有 Qt 对象（关掉的窗口、后台任务的信号对象……），它的 C++ 析构就跑在
分析线程里，和主线程的事件循环抢同一份数据，Windows 上会随机崩溃（abort / access violation）。
做法和 pyqtgraph 的 GarbageCollector 一样：关掉自动回收，由主线程的定时器按原来的阈值回收。
"""

from __future__ import annotations

import gc

from PyQt6.QtCore import QObject, QTimer


class MainThreadGC(QObject):
    def __init__(self, parent=None, interval_ms: int = 1000):
        super().__init__(parent)
        self.threshold = gc.get_threshold()
        gc.disable()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check)
        self.timer.start(interval_ms)

    def check(self) -> None:
        c0, c1, c2 = gc.get_count()
        t0, t1, t2 = self.threshold
        if c0 > t0:
            gc.collect(0)
            if c1 > t1:
                gc.collect(1)
                if c2 > t2:
                    gc.collect(2)


_guard = None


def install(app) -> MainThreadGC:
    """在 QApplication 建好后调用一次；返回的对象挂在 app 上，随程序存活。"""
    global _guard
    if _guard is None:
        _guard = MainThreadGC(app)
    return _guard
