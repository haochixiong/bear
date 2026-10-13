# Windows 交接说明（给在 Windows 上继续开发的 Codex / 开发者）

本项目的核心算法和界面已在 Linux（offscreen）上完成并通过测试。下面这些部分**只能在真实的 Windows 桌面上验证**。请按顺序执行，并把结果（通过/失败、实际数值、截图）记录在本文末尾的表格里。

## 0. 环境

```bat
git checkout claude/cool-ptolemy-0vmqtd
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
python -m pytest -q
```

预期：全部通过。Linux 上的基线为 **301 passed**。

如果在 Windows 上失败，请先确认失败的测试是不是只与平台相关，例如路径或字体。**不要为了让测试通过而放宽算法阈值。**

## 1. 必须验证的平台相关代码

| # | 内容 | 代码位置 | 怎么验证 | 预期 |
|---|---|---|---|---|
| 1 | 读取显示器 ICC | `src/colordoctor/capture/windows_display.py`：`monitor_icc_path`，通过 `CreateDCW` 和 `GetICMProfileW` | `python -c "from PyQt6.QtWidgets import QApplication; a=QApplication([]); from colordoctor.capture import screen_capture as s, windows_display as w; [print(i.name, w.monitor_icc_path(i.name)) for i in s.list_screens()]"` | 每块屏幕打印的 .icc/.icm 路径，与“Windows 颜色管理 → 设备”中该显示器的默认配置文件一致。注意 `QScreen.name()` 应形如 `\\.\DISPLAY1`；如果不是，需要改用 `EnumDisplayMonitors` + `GetMonitorInfoW` 取设备名 |
| 2 | 屏幕取样与 Photoshop 一致 | `ui/main_window.py`：`_on_capture_selected` / `lock_sample` | 在 Photoshop 中打开一张 sRGB 测试图，缩放 100%。信息面板（F8）的吸管取样设为 5×5。分别在**普通屏**和**广色域/校色屏**上，用本工具“框选屏幕 → 单击”读取同一个点 | RGB 与 Photoshop 信息面板相差 ≤ 2；Lab 相差 ≤ 1。在广色域屏上，如果把下拉框改成“未做色彩管理的软件”，读数应明显偏离（这说明 ICC 反转换生效了） |
| 3 | 全局快捷键 | `capture/hotkeys.py`：`RegisterHotKey` + `nativeEventFilter` | 让 Photoshop 处于前台，按 Ctrl+Alt+Shift+Z（需要先开启实时取色）和 Ctrl+Alt+Shift+X | 两个快捷键都能触发。**需要确认** PyQt6 中 `nativeEventFilter` 的返回值类型：目前返回 `(bool, 0)`，如果报错，改为返回 `(bool, None)` 或 sip.voidptr。如果和某个 Photoshop 自定义快捷键冲突，状态栏会提示，此时修改默认值 |
| 4 | 多显示器 / 混合缩放 | `capture/geometry.py`、`capture/screen_capture.py`、`ui/capture_overlay.py` | 两块屏幕，分别设为 100% 和 150% 缩放；把 Photoshop 窗口拖到副屏，重复第 2 项；用框选层在两块屏幕上各框一次 | 覆盖层和截图完全对齐（放大镜中心就是光标下的像素）；读数与 Photoshop 一致。Qt6 在混合 DPI 下的逻辑坐标可能不连续，如果出现错位，修 `freeze_all_screens` / `_ScreenOverlay.setGeometry` |
| 5 | 实时取色 | `capture/screen_capture.py`：`grab_around_cursor` | 打开“实时取色”，在 125% 和 150% 缩放下，把光标移到 Photoshop 中 1 像素宽的线条上 | 读数与光标下的像素一致，CPU 占用不明显 |
| 6 | 打包 | `packaging/build_windows.bat`、`packaging/colordoctor.spec` | 运行 `packaging\build_windows.bat` | 生成 `dist\ColorDoctor\ColorDoctor.exe`，`selftest.log` 内容为 `SELFTEST OK`；双击启动没有控制台窗口；在没有安装 Python 的机器上也能运行 |
| 7 | 高 DPI 界面 | 全部 UI | 在 150% 和 200% 缩放下打开 | 文字没有被截断，色轮清晰 |
| 8 | JPG | `core/color_management.py` | 分别用相机直出 JPG（带 EXIF 方向）、Photoshop 导出的 Adobe RGB JPG、sRGB JPG 打开、取样 | 方向与 Photoshop 一致；Adobe RGB 图提示“已转换到 sRGB”；读数与 Photoshop（文档转换到 sRGB 后）的信息面板一致 |

### 1b. 第四版新增、需要在 Windows 上确认的项

| # | 内容 | 代码位置 | 怎么验证 | 预期 |
|---|---|---|---|---|
| 9 | 本机 Claude Code 会话 | `ai/providers.py`：`_run_cli`，默认命令 `claude -p`，提示词通过 stdin 传入 | ⚙ 设置 → 启用 AI 点评 → 方式选 Claude Code；对任意分析点“AI 点评” | 弹出中文点评。如果 `claude -p` 不读取 stdin，就改为把提示词作为参数传入；Windows 上 npm 安装的是 `claude.cmd`，`shutil.which` 应能找到。勾选“发送缩略图”时会追加 `--allowedTools Read`，请确认它能读到临时图片 |
| 10 | 本机 Codex 会话 | 同上，默认命令 `codex exec --skip-git-repo-check -`，图片用 `-i 路径` | 同上，方式选 Codex | 弹出点评。如果参数名在当前 Codex 版本中不同，请修改 `DEFAULT_COMMANDS` / `build_command`，并更新 `tests/test_ai.py` |
| 11 | DeepSeek / Claude API | `ai/providers.py` | 填入真实 Key 后测试一次 | 正常返回；Key 错误时显示中文提示 |
| 12 | 套索、`[` `]` 切换取样大小 | `ui/selection_widget.py`、`ui/capture_overlay.py` | 人像肤色模式勾选“套索”，在 PS 画面上圈皮肤；框选层中按 `[` `]` | 套索选区与圈出的范围一致；放大镜中的取样框随之变化 |
| 13 | 皮肤蒙版编辑 | `ui/mask_editor.py` | 对一张真实人像分析后点“编辑皮肤蒙版”，用画笔擦掉头发、用魔棒补上漏掉的皮肤 | 编辑流畅（大图时画笔不卡顿）；“用这个蒙版重新分析”后结果更新 |
| 14 | 真实人像的自动抠皮肤 | `core/skin_mask.py` | 分别用浅肤色、深肤色、强侧光、彩色环境光的人像测试 | 头发、眉眼、嘴唇、背景基本被排除；漏选或多选可以用蒙版编辑修正；把失败样例记录下来，用于调整 `HUE_TOL` / `SAT_RATIO` / `DARK_RATIO` |

## 2. 建议完成的校准（提高“调整参数”精度）

`core/ps_models.py` 中，色彩平衡、色相/饱和度、可选颜色的模型是近似实现，与 Photoshop 有差异。校准方法：

1. 在 Photoshop 中新建一个 sRGB 文档，放入 6 个肤色色块，例如 `#C79C84`、`#DBBAAB`、`#CA9F81`、`#916F5E`、`#E9CEC6`、`#D6A88C`。
2. 对每个工具，分别应用若干组已知参数（例如色彩平衡中间调：青-红 ±10/±20，洋红-绿 ±10，黄-蓝 ±10，勾选保留明度），然后导出 PNG 或用信息面板读取 RGB。
3. 新建 `tests/data/ps_calibration.json`，记录“输入 RGB、工具、参数、Photoshop 输出 RGB”。
4. 新增一个测试，比较 `ps_models` 的输出和 Photoshop 的输出（ΔE00）；调整模型，使平均 ΔE00 < 1.5。
5. 可选颜色是误差最大的工具。校准之后，可以把 `recommendations.skin_adjustments` 中可选颜色的输出从“方向 + 起步值”升级为“具体数值”。

## 3. 可选增强（第一版不要求）

- 检测 Windows HDR / 自动颜色管理是否开启（`DisplayConfigGetDeviceInfo`），开启时在界面上提示。
- 应用图标（`packaging/colordoctor.spec` 中的 `icon=`）。
- 混色器/颜色分级滑块换算常数（`core/acr_channels.py` 中的 `HUE_PER_UNIT`、`LUM_PER_UNIT`）：在 ACR 中对色块拉动滑块并测量，校准这两个常数。
- 首次运行时的简短引导。

## 4. 不要做的事

- **不要**改为直接把屏幕像素当作 sRGB（参见 README 的“色彩管理”一节）。
- **不要**引入网络请求。`tests/test_acceptance.py` 会检查源码中是否导入了网络模块。
- **不要**让任何代码写入用户图片。`tests/test_acceptance.py` 会检查。

## 5. 验证记录

| # | 结果 | 实测数值 / 备注 | 日期 |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |
| 6 |  |  |  |
| 7 |  |  |  |
| 8 |  |  |  |
| 9 |  |  |  |
| 10 |  |  |  |
| 11 |  |  |  |
| 12 |  |  |  |
| 13 |  |  |  |
| 14 |  |  |  |
