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
| 6 | RAW（rawpy） | ✅ 完成（未用真实 RAW 文件测过） | `color_management.load_raw / load_image / is_raw`；桌面版“打开 JPG / RAW”、命令行 `--raw-preview`；设置键 `raw/use_preview`（界面上还没有开关） | `tests/test_raw.py`（假 rawpy 模块） |
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

1. `pip install -e .[dev]`，`python -m pytest -q`（当前基线 **363 passed, 1 skipped**（需要 Node 才会跑插件 JS 测试））。
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

## 还没做 / 建议下一步（按优先级）

1. **在 Windows PS Beta 上跑通插件**：逐条完成上面的 P1–P10。这一步最重要，插件代码还没在真实 Photoshop 里运行过。
2. **Camera Raw 描述符**（P9）：录到之后：
   - 实现“生成 Camera Raw 图层”时直接写入参数；
   - 实现 `main.js` 的 `calibrate()`，流程见上文“校准流程”。
3. **桌面版接入 AI 抠皮肤**：`ui/session.py` 的肤色分析目前只用肤色规则。可加一个“AI 抠皮肤”勾选，调用 `segmentation.segment_skin(region).part(part)` 作为 `manual_mask` 传入；蒙版编辑器照常可用。
4. **桌面版设置**：加“RAW 使用内嵌预览”（设置键 `raw/use_preview`）和“允许 PS 插件连接”（`bridge/enabled`）两个开关。
5. **DirectML**：在 Windows 上确认 `onnxruntime-directml` 能选中 `DmlExecutionProvider`，状态信息里会显示 provider 名称。
6. **仿色和风格的 ACR 参数**：
   - 现在 `acr_params` 只从“混色器 / 颜色分级”建议里收集；
   - 仿色的 `ChannelDiff` 滑块和风格模板的 recipe 也可以转成参数；
   - 曲线的点（`reference_match` 的 RGB 复合曲线）可以写进 `curve`。

## 注意

- 不要让 `bridge` 以外的包导入网络模块（`tests/test_acceptance.py` 会检查）；`bridge` 只能监听 127.0.0.1。
- 插件只**新增**调整图层和蒙版，不改原图像素（用户已同意）。
- 分析只在用户点“分析当前图层”时运行，不要自动分析。
