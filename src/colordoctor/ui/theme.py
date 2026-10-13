"""界面配色：全部使用中性灰，避免界面颜色干扰对照片颜色的判断。"""

BG = "#2a2a2a"
PANEL = "#323232"
PANEL_2 = "#3a3a3a"
BORDER = "#4a4a4a"
TEXT = "#e4e4e4"
TEXT_DIM = "#a8a8a8"
ACCENT = "#d8d8d8"
WHEEL_BG = "#242424"
WHEEL_GRID = "#3c3c3c"

SEVERITY_COLORS = {
    "正常": "#8fa58f",
    "轻微": "#b9a86f",
    "明显": "#c38a5a",
    "较强": "#c46464",
}

# 问题位置叠加层
OVERLAY_DIM = (0, 0, 0, 120)
OVERLAY_GRAYING = (255, 255, 255, 170)
OVERLAY_BOX = "#ffffff"

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolButton, QPushButton {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 5px 10px;
}}
QToolButton:checked, QPushButton:checked {{ background: #5a5a5a; border-color: {ACCENT}; }}
QToolButton:hover, QPushButton:hover {{ border-color: {ACCENT}; }}
QComboBox, QLineEdit, QSpinBox {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 6px;
}}
QTabWidget::pane {{ border: 1px solid {BORDER}; }}
QTabBar::tab {{ background: {PANEL}; padding: 6px 12px; border: 1px solid {BORDER}; }}
QTabBar::tab:selected {{ background: {PANEL_2}; }}
QTextBrowser, QListWidget, QTableWidget {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QStatusBar {{ color: {TEXT_DIM}; }}
QCheckBox {{ spacing: 6px; }}
QSplitter::handle {{ background: {BORDER}; }}
"""
