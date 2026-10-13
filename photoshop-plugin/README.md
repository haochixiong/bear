# ColorDoctor Photoshop 插件（UXP）

面板只负责读取像素、显示结果、建调整图层和蒙版；分析在本机服务里完成。服务只监听 127.0.0.1，不联网。

## 安装（开发模式）

1. 启动本机分析服务（二选一）：
   - 打开 ColorDoctor 桌面版：启动时自动开启服务，状态栏会提示（设置里可以关闭）；
   - 或者用 Python 包的命令行：`colordoctor serve`（打包版是窗口程序，没有控制台，请用第一种方式）。
2. 安装 Adobe **UXP Developer Tool**（Creative Cloud → 所有应用 → UXP Developer Tools），打开 Photoshop Beta。
3. UXP Developer Tool → Add Plugin → 选本目录的 `manifest.json` → Load。
4. Photoshop：增效工具 → ColorDoctor。

## 使用

1. 在图层面板选中要分析的像素图层，或在面板顶部改为「合并可见图层」。
2. 点「分析当前图层」。插件不会自动分析：
   - 选中调整图层或空白图层时，按钮变灰并说明原因。
   - 图层分析后又被修改时，会提示「结果可能过期」。
3. 每条问题都可以「生成此层（带蒙版）」或「只生成选区」。底部「一键生成全部」把所有调整图层放进一个组。整个过程是一步历史记录。
4. 「ACR 参数」可以直接改数值，然后：
   - 导出 `.xmp` 预设；
   - 或点「生成 Camera Raw 图层」：复制图层 → 转为智能对象 → 打开 Camera Raw。在里面套用刚导出的预设，或直接调整即可。

## 文件

| 文件 | 作用 |
|---|---|
| `manifest.json` | UXP manifest v5。网络权限只开放 `http://127.0.0.1:47321` |
| `index.html` `styles.css` | 面板界面（对应原型的 色彩 / 人像 / 仿色） |
| `js/main.js` | 面板逻辑 |
| `js/ps.js` | Photoshop 操作：图层检查、`imaging.getPixels`、batchPlay 建调整图层、`putLayerMask` / `putSelection` |
| `js/bridge.js` | 与本机服务通信 |
| `test/run.js` | 在 Node 里用假的 photoshop 模块和真实服务跑一遍（`pytest tests/test_plugin_js.py`） |
