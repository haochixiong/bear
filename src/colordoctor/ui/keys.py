"""界面文字里的按键名：macOS 上 Qt 把 Ctrl 快捷键映射到 ⌘，文字也跟着写 ⌘。"""

import sys

MAC = sys.platform == "darwin"
CMD = "⌘" if MAC else "Ctrl+"  # 例：f"{CMD}Z 撤销" -> ⌘Z / Ctrl+Z
