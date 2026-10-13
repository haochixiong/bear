"""本次会话的取样历史（只在内存中，关闭程序即清空）。"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMenu

from colordoctor.ui.session import HistoryItem


class HistoryPanel(QListWidget):
    item_activated = pyqtSignal(int)
    set_reference = pyqtSignal(int)
    set_problem = pyqtSignal(int)
    use_as_target = pyqtSignal(int)
    remove_item = pyqtSignal(int)
    clear_all = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.currentItemChanged.connect(self._changed)
        self.ref_action_text = "设为参考区"  # 仿色模式下改为“设为参考作品”

    def set_items(self, items: list[HistoryItem], reference_id: int | None = None, ref_image_id: int | None = None):
        self.blockSignals(True)
        self.clear()
        for h in items:
            pm = QPixmap(18, 18)
            pm.fill(QColor(*h.swatch))
            text = f"{h.time_text}  {h.label}"
            if reference_id == h.id:
                text += "  [参考区]"
            if ref_image_id == h.id:
                text += "  [参考作品]"
            it = QListWidgetItem(QIcon(pm), text)
            it.setData(Qt.ItemDataRole.UserRole, h.id)
            it.setToolTip(Qt.convertFromPlainText(h.source_label) if hasattr(Qt, "convertFromPlainText") else h.source_label)
            self.addItem(it)
        self.blockSignals(False)

    def highlight(self, item_id: int | None):
        """让列表的选中项跟着当前显示的结果走（不触发重新显示）。"""
        self.blockSignals(True)
        self.setCurrentRow(-1)
        for i in range(self.count()):
            if item_id is not None and int(self.item(i).data(Qt.ItemDataRole.UserRole)) == item_id:
                self.setCurrentRow(i)
                break
        self.blockSignals(False)

    def keyPressEvent(self, ev):
        it = self.currentItem()
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and it is not None:
            self.remove_item.emit(int(it.data(Qt.ItemDataRole.UserRole)))
            return
        super().keyPressEvent(ev)

    def _changed(self, cur, _prev):
        if cur is not None:
            self.item_activated.emit(int(cur.data(Qt.ItemDataRole.UserRole)))

    def _menu(self, pos):
        it = self.itemAt(pos)
        m = QMenu(self)
        if it is not None:
            hid = int(it.data(Qt.ItemDataRole.UserRole))
            m.addAction(self.ref_action_text, lambda: self.set_reference.emit(hid))
            m.addAction("作为问题区与参考区对比", lambda: self.set_problem.emit(hid))
            m.addAction("用它的肤色作为目标风格", lambda: self.use_as_target.emit(hid))
            m.addSeparator()
            m.addAction("删除（Delete）", lambda: self.remove_item.emit(hid))
        m.addAction("清空历史", self.clear_all.emit)
        m.exec(self.mapToGlobal(pos))
