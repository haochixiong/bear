# 第六阶段交接（PS 插件方向）— 给接手的本地 agent

> 每完成一块就更新本文件并推送。最新状态以本文件和 `git log` 为准。
> 设计依据：`docs/stage6_ps_plugin_design.md`（第 7 节是用户已确认的修改）；界面原型：https://claude.ai/artifact/V6wzgXjExY5kESVCaXkzM6

## 进度

| # | 模块 | 状态 | 位置 | 测试 |
|---|---|---|---|---|
| 1 | 本机桥接服务（127.0.0.1） | ✅ 完成 | `src/colordoctor/bridge/server.py`，命令 `colordoctor serve [--port 47321]` | `tests/test_bridge.py` |
| 2 | 图层方案（问题 → 调整图层 + 羽化蒙版） | ✅ 完成（脏色、发灰、肤色不均、颜色统一） | `src/colordoctor/core/layer_plan.py` | 同上 |
| 3 | 图层检查（调整图层 / 空白 / 纯色 / 半透明裁剪） | ✅ 完成（服务端兜底） | `src/colordoctor/core/pixel_guard.py` | 同上 |
| 4 | ACR 预设 .xmp 导出 + 滑块实测校准 | ✅ 完成（Python 端） | `core/acr_preset.py`（`build_xmp`、`params_from_adjustments`）、`core/acr_calibration.py`（色卡、拟合、保存）；`acr_channels` 的滑块换算按通道读取 `acr_calibration.json`；服务路由 `/acr-preset`、`/calibration/*` | `tests/test_acr.py`、`tests/test_bridge.py` |
| 5 | UXP 插件 | 🟡 代码完成，**未在真实 PS 上运行过** | `photoshop-plugin/`（见其 README）；新增服务模式 `patch`（取色）和 `/options` | `tests/test_plugin_js.py`（Node + 假 photoshop 模块 + 真实服务） |
| 6 | RAW（rawpy） | ✅ 完成（未用真实 RAW 文件测过） | `color_management.load_raw / load_image / is_raw`；桌面版“打开 JPG / RAW”、命令行 `--raw-preview`；设置里有“RAW 使用内嵌预览”开关 | `tests/test_raw.py`（假 rawpy 模块） |
| 7 | AI 抠皮肤（onnxruntime + MediaPipe 多类分割） | ✅ 完成（Linux CPU 上用 MediaPipe 示例人像验证过：约 1 秒，脸 / 身体 / 头发 / 衣服分得开） | `core/segmentation.py`；服务 `mode=skin&seg=ai`，插件“抠皮肤”下拉框；模型不在仓库里，`build_windows.bat` 构建时下载到 `packaging/models/` 并打包；运行时查找顺序见 `model_candidates()`；没有模型或 onnxruntime 时自动退回肤色规则 | `tests/test_segmentation.py`（假推理会话；设 `COLORDOCTOR_SEG_MODEL` 时也跑真实模型） |

## 已定的接口约定（插件按这个写）

- 服务：`http://127.0.0.1:47321`，每个请求必须带头 `X-ColorDoctor-Client: uxp`；带 `Origin: http(s)://…` 的请求一律 403（防网页调用）。
- `POST /analyze?mode=dirty|skin|palette|reference&preset=&part=&style=&family=&ref=`
  - 头：`X-Width` `X-Height` `X-Components`(3/4) `X-Doc-Width` `X-Doc-Height` `X-Layer-Kind`（UXP `layer.kind`）`X-Layer-Name`（encodeURIComponent）
  - 体：交错 RGB(A) 8 位字节（`imaging.getPixels` 取 `colorSpace:"RGB"`、`colorProfile:"sRGB IEC61966-2.1"`、`componentSize:8`；建议 `targetSize` 长边 ≤ 2048）
  - 200：`{session, summary, findings, adjustments, notes, plan:{group_name,width,height,layers:[{id,name,kind,params,reason,has_mask,mask_coverage,estimated}]}}`
  - 422：`{blocked: "中文原因"}` → 面板显示原因、按钮变灰
- `GET /mask?session=&layer=[&w=&h=]` → `w*h` 字节灰度，默认文档尺寸（`X-Doc-*`）；响应头 `X-Width/X-Height`
- `POST /reference`（同像素格式）→ `{ref}`；之后 `/analyze?mode=reference&ref=…`
- 图层 `kind` 与参数（PS 面板单位）：
  - `hueSaturation`：`{master:{hue,saturation,lightness}, ranges:[{localRange(1 红…6 洋红), beginRamp, beginSustain, endSustain, endRamp, hue, saturation, lightness}]}`
  - `colorBalance`：`{shadows:[青红,洋红绿,黄蓝], midtones:[…], highlights:[…], preserveLuminosity}`
  - `vibrance`：`{vibrance, saturation}`
  - `curves`：`{channels:{composite|red|green|blue: [[in,out],…]}}`

## 接手时先做

1. `pip install -e .[dev]`，`python -m pytest -q`（当前基线 **386 passed, 1 skipped**（需要 Node 才会跑插件 JS 测试））。
2. 按“进度”表从第一个 ⏳ 开始。每完成一项：补测试 → 全量测试 → 更新本表 → 提交推送。

## 校准流程（插件端待实现，见 UXP 部分）

1. `GET /calibration/chart` 取色卡像素 → 插件新建 sRGB 文档并 `putPixels`。
2. 读回未调整的色卡像素 → `POST /calibration/sample?slider=base`。
3. `GET /calibration/plan` 得到 96 步（8 通道 × 色相/饱和度/明亮度 × −100/−50/+50/+100）。每步：复制色卡图层 → 对副本应用 Camera Raw 滤镜，只设这一个滑块 → 读像素 → `POST /calibration/sample?channel=&slider=&value=` → 删除副本。
4. `POST /calibration/fit`（体 `{"acr_version": "…"}`）→ 保存到用户配置目录，之后所有混色器数值标为“已校准”。
- **未验证**：Camera Raw 滤镜的 batchPlay 描述符需要在 Windows 上用 Alchemist 录一次（录“滤镜 → Camera Raw 滤镜”，改一个 HSL 滑块后确定），把录到的键名填进插件的 `acrFilterDescriptor()`。
- `.xmp` 的字段名（`HueAdjustmentOrange`、`SplitToningShadowHue`、`ColorGradeBlending`、`ToneCurvePV2012` 等）按 ACR 预设格式写，**需要在 Windows 上把导出的预设放进 ACR 预设文件夹验证一次**。

## 插件：必须在 Windows PS Beta 上逐条确认

| # | 内容 | 位置 | 怎么验证 / 可能要改的地方 |
|---|---|---|---|
| P1 | 能加载 | `manifest.json` | UXP Developer Tool 加载无报错。如果网络权限的写法 `http://127.0.0.1:47321` 不被接受，改成 `"domains": "all"` 或 `["http://localhost:47321"]`，同时把 `js/bridge.js` 的 BASE 改成一致 |
| P2 | 图层检查 | `ps.js` `inspectTarget` | 分别选中像素图层、曲线调整图层、空白新图层、组、文字图层，确认提示正确。`layer.kind` 的实际取值请打印确认（目前按字符串比较；如果是枚举对象，改成 `constants.LayerKind.*`） |
| P3 | 读像素 | `ps.js` `readPixels` | 16 位、Lab、CMYK 文档也能读成 sRGB 8 位；`targetSize` 缩放正确；图层不在 (0,0) 时 `sourceBounds` 正确 |
| P4 | 建调整图层 | `ps.js` `makeAdjustmentLayer` | 色相/饱和度（尤其 `localRange` 黄色那一档的数值）、色彩平衡、自然饱和度、曲线的参数在属性面板里显示正确。如果描述符不对，用 Alchemist 录一次替换 |
| P5 | 写蒙版 | `ps.js` `putMask` | 调整图层自带的白色蒙版被替换成羽化蒙版，位置对齐；分析对象是局部图层时，`targetBounds` 以外的蒙版应为黑色（不确定，需要确认） |
| P6 | 编组 | `ps.js` `groupLayers` | `doc.createLayerGroup({fromLayers})` 能把新建的图层收进一个组 |
| P7 | 一步撤销 | `ps.js` `modal` | 一键生成后按一次 Ctrl+Z 全部撤销 |
| P8 | 过期提示 | `main.js` `refreshTarget` | 分析后在图层上画一笔，面板提示“可能过期” |
| P9 | Camera Raw | `ps.js` `cameraRawLayer` | 复制 → 智能对象 → 打开 Camera Raw 对话框。如果描述符名不是 `Adobe Camera Raw Filter`，用 Alchemist 录制后替换；录到带参数的描述符后，可以把 ACR 参数直接写进去，不再需要用户手动套预设，同时实现 `main.js` 里的 `calibrate()` |
| P10 | 导出 xmp | `main.js` `exportXmp` | 存下的 `.xmp` 能在 ACR 预设面板里导入，并且数值正确 |
| P11 | 分辨率不是 72 ppi | `ps.js` `inspectTarget` / `readPixels` | 用 300 ppi 的文档确认 `doc.width/height`、`layer.bounds` 都是像素（不是按 72 ppi 换算的点）；否则蒙版位置和大小会错 |
| P12 | 超大图 | `ps.js` `readPixels` | 长边 > 4096 的文档：返回的 `level` 与 `sourceBounds` 换算后，蒙版与画面对齐（已按 2^level 换算，需在真机确认） |
| P13 | 16 / 32 位文档 | `ps.js` `putMask` / `grayImageData` | 16 位文档的蒙版写成 0–32768，32 位写成 0–1 浮点，PS 里看到的灰度正确；32 位文档里色彩平衡等是否可用 |
| P14 | 不规则选区取色 | `ps.js` `readSelectionPixels` | 椭圆 / 套索 / 魔棒选区时 `imaging.getSelection` 返回的灰度与像素对齐；羽化选区的半透明部分按比例计入 |
| P15 | 有选区时生成 | `ps.js` `withoutSelection` | 有选区时点「生成」：选区先存成临时通道并取消，整体调整图层不带选区蒙版；生成后选区恢复、临时通道删除 |
| P16 | Camera Raw 图层 | `ps.js` `cameraRawLayer` | 分析单个图层后（当前选中的是调整图层组）点「生成 Camera Raw 图层」：复制的是分析过的图层；合并模式下盖印图层在最顶层；在 Camera Raw 里点「取消」时复制出的图层被撤回 |
| P18 | 可选颜色图层 | `ps.js` `selectiveColorDescriptor` | 生成的「可选颜色」图层在属性面板里方法为“相对”，红色 / 黄色等各项数值正确（描述符按 ScriptListener 的 colors / cyan / magenta / yellowColor / black 写，未录制核对） |
| P19 | 颜色取样器 | `ps.js` `readColorSamplers` | 用颜色取样器工具放 2–4 个点后点「多点对比」：`doc.colorSamplers[i].color.lab` 读得到，顺序与放置顺序一致 |
| P20 | 选参考图文件 | `main.js` `chooseRefFile` | `getFileForOpening` + `file.read({format: binary})` 能读 JPG / PNG / TIFF / RAW；大文件（50 MB RAW）不卡死 |
| P21 | 记住设置 | `main.js` `prefs` | 关闭再打开面板、重启 PS 后，上次的页面、分析对象、风格等恢复（UXP 的 localStorage） |
| P22 | 自定义颜色范围 | `native_layers.py` 色相/饱和度 | 一层里有多个范围时，每个范围占一个槽位（如“黄色”槽位但范围对准 22° 或 210°），确认 PS 接受远离默认颜色的 begin/end ramp |
| P23 | 调整幅度 | `native_layers.py` | 可选颜色约“每 1 Lab 单位 2.5%”（蜡黄肤色 → 红色 洋红 +21%）、颜色分级饱和度 30 ≈ 色彩平衡 18、白平衡 → 色彩平衡中间调，在真实照片上看是否过强 / 过弱 |
| P24 | 颜色曲线 + 明度曲线 | 肤色对比类建议 | 颜色曲线和明度曲线分成两层叠放，效果是否符合卡片描述 |
| P25 | 当前图层及以下 | `ps.js` `readModal(fn, info)` / `layersAbove` | 选中中间的一个调整图层 / 组内图层，分析：上方图层在读像素时被临时隐藏（`layer.visible = false`），`imaging.getPixels` 不带 layerID 读到的是隐藏后的合成画面；读完上方图层恢复显示、历史记录面板里没有多出步骤（`resumeHistory(id, false)`） |
| P26 | 选中组时生成 | `main.js` `generate` | 分析对象是图层组时，新调整图层会被建在组内最上面（PS 的行为），效果应与放在组上方相同（穿透模式）；组的混合模式不是“穿透”时请确认 |
| P17 | 非 RGB 文档 | `main.js` `generate` | CMYK / Lab / 灰度文档里点「生成」给出“请先转换为 RGB”的提示，不建图层；`doc.mode` 的实际取值请打印确认 |

## 还没做 / 建议下一步（按优先级）

1. **在 Windows PS Beta 上跑通插件**：逐条完成上面的 P1–P10。这一步最重要，插件代码还没在真实 Photoshop 里运行过。
2. **Camera Raw 描述符**（P9）：在 `ps.js` 里给 `ACR_DESCRIPTOR` 赋值（参数 → 描述符的函数）后：
   - 「校准 ACR 滑块」整套流程即可运行，流程已写好：建色卡文档 → 逐步复制图层、应用 Camera Raw、读像素、上传 → 拟合并保存；
   - 「生成 Camera Raw 图层」也可以改成直接写入参数。
3. **DirectML**：在 Windows 上确认 `onnxruntime-directml` 能选中 `DmlExecutionProvider`。
4. **风格模板的 recipe**：目前是文字区间，不能直接转成 ACR 参数。如果需要，可以给每个模板补一组“起点数值”。

已完成（上一版列在这里的）：
- 桌面版“AI 抠皮肤”勾选（没有模型时变灰并说明原因）；
- 设置里的“RAW 使用内嵌预览”和“允许 PS 插件连接”；
- 仿色的颜色分级参数和 RGB 复合曲线：进入 ACR 参数和图层方案（曲线图层）。

## 注意

- 不要让 `bridge` 以外的包导入网络模块（`tests/test_acceptance.py` 会检查）；`bridge` 只能监听 127.0.0.1。
- 插件只**新增**调整图层和蒙版，不改原图像素（用户已同意）。
- 分析只在用户点“分析当前图层”时运行，不要自动分析。
