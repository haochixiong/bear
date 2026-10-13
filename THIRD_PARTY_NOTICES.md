# 第三方说明

## RNV Color Picker（参考）

- 仓库：https://github.com/RNVizion/rnv-color-picker
- 审计版本：`0ef558a8eea0c1ff6ccb70051017b8c02eae12c4`
- 许可证：MIT，Copyright (c) 2025-2026 RNV

本项目参考了它的屏幕取色交互思路（全屏覆盖层、放大镜、十字准星），也参考了它的 PyInstaller 打包方式。本项目**没有复制它的源代码**，所有代码均为独立编写。阶段 0 的审计记录见 `docs/stage0_audit.md`。

为表示尊重，下面附上原许可证全文：

```
MIT License

Copyright (c) 2025-2026 RNV

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 运行时依赖

| 依赖 | 许可证 | 用途 |
|---|---|---|
| PyQt6 / Qt 6 | GPL v3（PyQt6）/ LGPL v3（Qt） | 界面 |
| NumPy | BSD-3-Clause | 数值计算 |
| Pillow（含 LittleCMS） | MIT-CMU（HPND） / MIT | JPG 读取、ICC 转换 |

**注意**：PyQt6 采用 GPL v3 许可。如果要闭源分发本程序，需要购买 PyQt6 商业许可，或改用 PySide6（LGPL）。
