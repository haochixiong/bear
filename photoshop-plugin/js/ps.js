// Photoshop 端操作：读取当前图层信息与像素、检查能否分析、建调整图层、写图层蒙版。
// 所有修改都在 executeAsModal 里完成，并合并成一步历史记录（Ctrl+Z 一次撤销）。
const { app, action, core, imaging, constants } = require("photoshop");

// 按原图尺寸读像素（每个像素都检查，不先缩小）；只有超过这么多像素的超大文档才缩小读取（本机服务一次最多收 400 MB）
// 平台：Mac 上快捷键写 ⌘（PS 里 Ctrl 的位置在 Mac 是 Command）
const IS_MAC = (() => {
  try { return require("os").platform() === "darwin"; } catch (_) { /* 没有 os 模块 */ }
  try { return /mac/i.test(navigator.platform || ""); } catch (_) { return false; }
})();
const CMD = IS_MAC ? "⌘" : "Ctrl+";

const MAX_READ_PIXELS = 100_000_000;

// 没有像素、不能直接分析的图层类型（与 Python 端 pixel_guard.NO_PIXEL_KINDS 对应）
const NEED_MERGED_KINDS = { group: "图层组", text: "文字图层", smartObject: "智能对象", video: "视频图层", threeD: "3D 图层" };
// 能直接读像素的只有像素图层（含背景）；其余已知类型中，NEED_MERGED_KINDS 之外的都是调整 / 填充图层。
// 优先用 PS 自己的 constants.LayerKind 取值，避免名字写错（如 invert / inversion）。
const NO_PIXEL_KINDS = new Set([
  "brightnessContrast", "levels", "curves", "exposure", "vibrance", "hueSaturation", "colorBalance",
  "blackAndWhite", "photoFilter", "channelMixer", "colorLookup", "invert", "inversion", "posterize", "threshold",
  "gradientMap", "selectiveColor", "solidColor", "solidFill", "gradientFill", "pattern", "patternFill",
  "light", "clarity", "dehaze", "grain", "adjustment",
]);
try {
  for (const v of Object.values((constants && constants.LayerKind) || {})) {
    const k = String(v);
    if (k !== "pixel" && k !== "normal" && !NEED_MERGED_KINDS[k]) NO_PIXEL_KINDS.add(k);
  }
} catch (_) { /* 旧版本没有 LayerKind */ }

function kindName(layer) {
  const k = layer.kind;
  return typeof k === "string" ? k : String(k);
}

// 当前图层“上方”的所有图层（按叠放顺序；所在组之外更上面的也算），不含它的上级组
function layersAbove(doc, layer) {
  const out = [];
  let node = layer;
  while (node) {
    const parent = node.parent || null;
    const siblings = parent ? parent.layers : doc.layers;  // layers[0] 是最上面
    const idx = siblings ? siblings.findIndex((l) => l.id === node.id) : -1;
    if (idx > 0) out.push(...siblings.slice(0, idx));
    node = parent;
  }
  return out;
}

// 返回 {ok, reason, doc, layer, kind, bounds, stack}
// target: "stack" = 当前图层及以下看到的画面（默认；任何图层都可以，包括调整图层、组、透明图层）
//         "layer" = 只看当前图层自己的像素（抠出来的透明图层要单独分析时用）
//         "merged" = 合并可见图层（整个画面）
function inspectTarget(target) {
  const doc = app.activeDocument;
  if (!doc) return { ok: false, reason: "没有打开的文档。" };
  if (target === "merged") {
    return { ok: true, doc, layer: null, kind: "merged", label: "整个画面（合并可见）",
      bounds: { left: 0, top: 0, right: doc.width, bottom: doc.height } };
  }
  const layers = doc.activeLayers;
  if (!layers || layers.length !== 1) {
    return { ok: false, reason: "请在图层面板里只选中一个图层，或把分析对象改为「合并可见图层」。" };
  }
  const layer = layers[0];
  const kind = kindName(layer);
  if (target === "stack") {
    // 读的是“从最底下到这个图层为止”叠起来的画面：调整图层、组、透明图层都能分析，不新建盖印图层
    const hiddenAbove = layersAbove(doc, layer).filter((l) => l.visible !== false).length;
    const notes = [];
    if (layer.visible === false) notes.push("这个图层是隐藏的，画面里不包含它");
    if (hiddenAbove) notes.push(`上方还有 ${hiddenAbove} 个可见图层，分析时不计入，生成的调整图层也放在它们下面`);
    return { ok: true, doc, layer, kind: "stack", stack: true, label: `「${layer.name}」及以下的画面`, notes,
      bounds: { left: 0, top: 0, right: doc.width, bottom: doc.height } };
  }
  const label = `${layer.name}`;
  if (NO_PIXEL_KINDS.has(kind)) {
    return { ok: false, layer, kind, label,
      reason: `「${layer.name}」是调整图层或填充图层，本身没有像素。请把分析对象改为「到这一层的画面」，或选中像素图层。` };
  }
  if (NEED_MERGED_KINDS[kind]) {
    // 图层组不能栅格化，只能合并
    const fix = kind === "group" ? `或先合并这个组（${CMD}E）` : "或先栅格化（图层 → 栅格化）";
    return { ok: false, layer, kind, label,
      reason: `「${layer.name}」是${NEED_MERGED_KINDS[kind]}，请把分析对象改为「到这一层的画面」，${fix}。` };
  }
  const b = layer.boundsNoEffects || layer.bounds;
  if (!b || b.right - b.left < 2 || b.bottom - b.top < 2) {
    return { ok: false, layer, kind, label, reason: `「${layer.name}」是空白图层，没有可分析的像素。` };
  }
  return { ok: true, doc, layer, kind, label, bounds: { left: b.left, top: b.top, right: b.right, bottom: b.bottom } };
}

function historyStateId(doc) {
  try { return doc.activeHistoryState ? doc.activeHistoryState.id : null; } catch (_) { return null; }
}

function clipToCanvas(b, doc) {
  return { left: Math.max(0, b.left), top: Math.max(0, b.top), right: Math.min(doc.width, b.right), bottom: Math.min(doc.height, b.bottom) };
}

// 读取像素（sRGB、8 位、交错），按原图尺寸（超大文档才缩小到 MAX_READ_PIXELS）。必须在 executeAsModal 里调用。
async function readPixels(info, sourceBounds) {
  // 图层可能超出画布（移出去一部分）：只读画布内的部分，返回的坐标才能和蒙版对齐
  const b = clipToCanvas(sourceBounds || info.bounds, info.doc);
  const w = b.right - b.left;
  const h = b.bottom - b.top;
  if (w < 1 || h < 1) throw new Error("要分析的范围在画布之外。");
  const scale = Math.min(1, Math.sqrt(MAX_READ_PIXELS / (w * h)));
  const opts = {
    documentID: info.doc.id,
    sourceBounds: b,
    colorSpace: "RGB",
    colorProfile: "sRGB IEC61966-2.1",
    componentSize: 8,
    applyAlpha: false,
  };
  if (scale < 1) opts.targetSize = w >= h ? { width: Math.round(w * scale) } : { height: Math.round(h * scale) };
  if (info.layer && !info.stack) opts.layerID = info.layer.id;  // “及以下的画面”读合成结果（上方图层已临时隐藏）
  const res = await imaging.getPixels(opts);
  const img = res.imageData;
  const data = await img.getData({ chunky: true });
  // 用 targetSize 缩小读取时，PS 可能从金字塔缓存的第 level 级读：返回的 sourceBounds 是该级的坐标
  // （每级缩小一半），不是文档坐标。换算回文档坐标，再裁到画布内。
  const f = Math.pow(2, res.level || 0);
  const sb = res.sourceBounds || b;
  const docB = {
    left: Math.max(0, Math.round(sb.left * f)), top: Math.max(0, Math.round(sb.top * f)),
    right: Math.min(info.doc.width, Math.round(sb.right * f)), bottom: Math.min(info.doc.height, Math.round(sb.bottom * f)),
  };
  // 原图尺寸的数据可能有几百 MB：已经是 Uint8Array 时直接用，不再复制一份
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const out = { data: bytes, width: img.width, height: img.height, components: img.components, sourceBounds: docB, scaled: scale < 1 };
  img.dispose();
  return out;
}

function to255(arr, v) {
  // getData 在 16 位文档里返回 Uint16Array（0–32768），32 位返回 Float32Array（0–1）
  if (arr instanceof Uint16Array) return Math.min(255, Math.round((v * 255) / 32768));
  if (arr instanceof Float32Array) return Math.min(255, Math.round(v * 255));
  return v;
}

// 取色：只取选区真正选中的像素（套索 / 椭圆 / 魔棒选区不是矩形，不能用外接矩形的平均）
async function readSelectionPixels(info) {
  const sel = info.doc.selection && info.doc.selection.bounds;
  if (!sel || sel.right - sel.left < 1) throw new Error("请先用选框工具框选要取色的区域（或用吸管设置前景色）。");
  const b = clipToCanvas({ left: sel.left, top: sel.top, right: sel.right, bottom: sel.bottom }, info.doc);
  const px = await readPixels(info, b);
  let selImg = null;
  try {
    // 按实际读到的像素范围取选区：“只看这个图层”时选区超出图层的部分读不到像素，
    // 用框选的范围会把整块选区压缩到较小的像素块上，透明度权重对不上
    const pb = px.sourceBounds || b;
    const opts = { documentID: info.doc.id, sourceBounds: pb };
    if (px.width !== pb.right - pb.left || px.height !== pb.bottom - pb.top) opts.targetSize = { width: px.width, height: px.height };
    const res = await imaging.getSelection(opts);
    selImg = res.imageData;
    const s = await selImg.getData({ chunky: true });
    const sw = selImg.width;
    const sh = selImg.height;
    const n = px.width * px.height;
    const c = px.components;
    const out = new Uint8Array(n * 4);
    for (let y = 0; y < px.height; y++) {
      const sy = Math.min(sh - 1, Math.floor((y * sh) / px.height));
      for (let x = 0; x < px.width; x++) {
        const sx = Math.min(sw - 1, Math.floor((x * sw) / px.width));
        const i = y * px.width + x;
        const src = i * c;
        out[i * 4] = px.data[src];
        out[i * 4 + 1] = px.data[src + 1];
        out[i * 4 + 2] = px.data[src + 2];
        const a0 = c === 4 ? px.data[src + 3] : 255;
        out[i * 4 + 3] = Math.min(a0, to255(s, s[sy * sw + sx]));
      }
    }
    return { ...px, data: out, components: 4 };
  } catch (_) {
    return px;  // 读不到选区形状（旧版本）时退回外接矩形
  } finally {
    if (selImg) selImg.dispose();
  }
}

// 灰度蒙版按文档位深建 imageData（16 位文档用 0–32768，32 位用 0–1）
async function grayImageData(bytes, width, height, depth) {
  let buf = bytes;
  const d = String(depth || "");
  if (d.includes("16")) {
    buf = new Uint16Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) buf[i] = Math.round((bytes[i] * 32768) / 255);
  } else if (d.includes("32")) {
    buf = new Float32Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes[i] / 255;
  }
  const opts = { width, height, components: 1, chunky: true, colorSpace: "Grayscale" };
  if (!d.includes("32")) opts.colorProfile = "Gray Gamma 2.2";
  return imaging.createImageDataFromBuffer(buf, opts);
}

// PS「颜色取样器工具」放下的取样点（最多 10 个），按放置顺序；取样大小跟随吸管设置
function readColorSamplers() {
  const doc = app.activeDocument;
  if (!doc) throw new Error("没有打开的文档。");
  const list = doc.colorSamplers;
  const n = list ? list.length : 0;
  if (n < 2) throw new Error("请先用「颜色取样器工具」（吸管工具组里，快捷键 Shift+I 切换）在画面上点至少两个点。");
  const out = [];
  const skipped = [];
  for (let i = 0; i < n; i++) {
    let lab = null;
    try { lab = list[i].color && list[i].color.lab; } catch (_) { lab = null; }
    // 取样点在画布外（例如裁剪后）时读不到颜色：跳过并说明
    if (!lab || ![lab.l, lab.a, lab.b].every((v) => Number.isFinite(v))) { skipped.push(i + 1); continue; }
    const pos = list[i].position || {};
    out.push({ name: `取样点 ${i + 1}`, lab: [lab.l, lab.a, lab.b], x: Math.round(pos.x || 0), y: Math.round(pos.y || 0) });
  }
  if (out.length < 2) {
    throw new Error(`能读到颜色的取样点不足两个${skipped.length ? `（取样点 ${skipped.join("、")} 读不到颜色，可能在画布外）` : ""}。`);
  }
  out.skipped = skipped;
  return out;
}

function foregroundLab() {
  const c = app.foregroundColor;
  return { L: c.lab.l, a: c.lab.a, b: c.lab.b, hex: c.rgb.hexValue };
}

// ---------------------------------------------------------------- 调整图层描述符
const ORDINAL = [{ _ref: "adjustmentLayer", _enum: "ordinal", _value: "targetEnum" }];

function hueSatDescriptor(p) {
  const adj = [{ _obj: "hueSatAdjustmentV2", hue: p.master.hue, saturation: p.master.saturation, lightness: p.master.lightness }];
  for (const r of p.ranges || []) {
    adj.push({
      _obj: "hueSatAdjustmentV2", localRange: r.localRange,
      beginRamp: r.beginRamp, beginSustain: r.beginSustain, endSustain: r.endSustain, endRamp: r.endRamp,
      hue: r.hue, saturation: r.saturation, lightness: r.lightness,
    });
  }
  return { _obj: "hueSaturation", presetKind: { _enum: "presetKindType", _value: "presetKindCustom" }, colorize: false, adjustment: adj };
}

function colorBalanceDescriptor(p) {
  return { _obj: "colorBalance", shadowLevels: p.shadows, midtoneLevels: p.midtones, highlightLevels: p.highlights,
    preserveLuminosity: p.preserveLuminosity !== false };
}

function vibranceDescriptor(p) {
  return { _obj: "vibrance", vibrance: p.vibrance, saturation: p.saturation || 0 };
}

// batchPlay 里通道名用的是 PS 的字符串 ID：绿通道 'Grn ' 的字符串 ID 是 "grain"（不是 "green"），红、蓝就是 red / blue
const CURVE_CHANNEL_ID = { composite: "composite", red: "red", green: "grain", blue: "blue" };
function curvesDescriptor(p) {
  const adjustment = Object.entries(p.channels).map(([ch, pts]) => ({
    _obj: "curvesAdjustment",
    channel: { _ref: "channel", _enum: "channel", _value: CURVE_CHANNEL_ID[ch] || ch },
    curve: pts.map(([x, y]) => ({ _obj: "paint", horizontal: x, vertical: y })),
  }));
  return { _obj: "curves", presetKind: { _enum: "presetKindType", _value: "presetKindCustom" }, adjustment };
}

// 可选颜色（相对）：colors = {reds|yellows|greens|cyans|blues|magentas|whites|neutrals|blacks: {c, m, y, k}}（−100…100 %）
const SELECTIVE_COLORS = ["reds", "yellows", "greens", "cyans", "blues", "magentas", "whites", "neutrals", "blacks"];
// 可选颜色里“红色”'Rds ' 的字符串 ID 是 "radius"（PS 的老毛病：和半径共用一个 ID），其余就是英文名
const SELECTIVE_COLOR_ID = { reds: "radius" };
function selectiveColorDescriptor(p) {
  const pct = (v) => ({ _unit: "percentUnit", _value: Math.max(-100, Math.min(100, Math.round(v || 0))) });
  const colorCorrection = Object.entries(p.colors || {})
    .filter(([name]) => SELECTIVE_COLORS.includes(name))
    .map(([name, v]) => ({
      _obj: "colorCorrection", colors: { _enum: "colors", _value: SELECTIVE_COLOR_ID[name] || name },
      cyan: pct(v.c), magenta: pct(v.m), yellowColor: pct(v.y), black: pct(v.k),
    }));
  return { _obj: "selectiveColor", method: { _enum: "correctionMethod", _value: p.method === "absolute" ? "absolute" : "relative" }, colorCorrection };
}

const DESCRIPTORS = {
  hueSaturation: hueSatDescriptor,
  colorBalance: colorBalanceDescriptor,
  vibrance: vibranceDescriptor,
  curves: curvesDescriptor,
  selectiveColor: selectiveColorDescriptor,
};
const DEFAULT_TYPES = {
  hueSaturation: { _obj: "hueSaturation", presetKind: { _enum: "presetKindType", _value: "presetKindDefault" }, colorize: false },
  colorBalance: { _obj: "colorBalance", shadowLevels: [0, 0, 0], midtoneLevels: [0, 0, 0], highlightLevels: [0, 0, 0], preserveLuminosity: true },
  vibrance: { _obj: "vibrance" },
  curves: { _obj: "curves", presetKind: { _enum: "presetKindType", _value: "presetKindDefault" } },
  selectiveColor: { _obj: "selectiveColor", method: { _enum: "correctionMethod", _value: "relative" } },
};

// 在当前选中图层上方新建一个调整图层，返回其 layer id。必须在 executeAsModal 里调用。
function checkResults(results, what) {
  for (const r of results || []) {
    if (r && r._obj === "error") throw new Error(`${what}失败：${r.message || r.result || "Photoshop 返回错误"}`);
  }
}

function activeLayerId() {
  const ls = app.activeDocument && app.activeDocument.activeLayers;
  return ls && ls.length ? ls[0].id : null;
}

async function makeAdjustmentLayer(spec) {
  const build = DESCRIPTORS[spec.kind];
  if (!build) throw new Error(`不支持的图层类型：${spec.kind}`);
  const before = activeLayerId();
  // batchPlay 出错时不抛异常，而是返回 {_obj:"error"}；逐步检查，避免把原图层改名或写上蒙版
  checkResults(await action.batchPlay([
    { _obj: "make", _target: [{ _ref: "adjustmentLayer" }], using: { _obj: "adjustmentLayer", type: DEFAULT_TYPES[spec.kind] } },
  ], {}), `新建「${spec.name}」`);
  const id = activeLayerId();
  if (id == null || id === before) throw new Error(`新建「${spec.name}」失败：Photoshop 没有创建调整图层（文档模式或图层锁定？）`);
  checkResults(await action.batchPlay([
    { _obj: "set", _target: ORDINAL, to: build(spec.params) },
    // 影调曲线用“明度”混合模式：只改明暗，不改颜色
    { _obj: "set", _target: [{ _ref: "layer", _id: id }], to: { _obj: "layer", name: spec.name, ...(spec.blend === "luminosity" ? { mode: { _enum: "blendMode", _value: "luminosity" } } : {}) } },
  ], {}), `设置「${spec.name}」`);
  return id;
}

// 把灰度蒙版（Uint8Array，w*h）写进图层的用户蒙版，放在文档的 bounds 位置。
async function putMask(docId, layerId, bytes, width, height, bounds, depth) {
  const img = await grayImageData(bytes, width, height, depth);
  try {
    await imaging.putLayerMask({
      documentID: docId, layerID: layerId, imageData: img, replace: true,
      targetBounds: { left: bounds.left, top: bounds.top },
      commandName: "ColorDoctor 蒙版",
    });
  } finally {
    img.dispose();
  }
}

// 只生成选区（不建图层）：把蒙版写进一个 Alpha 通道再载入选区 —— 用 putSelection。
async function putSelection(docId, bytes, width, height, bounds, depth) {
  const img = await grayImageData(bytes, width, height, depth);
  try {
    await imaging.putSelection({ documentID: docId, imageData: img, replace: true,
      targetBounds: { left: bounds.left, top: bounds.top }, commandName: "ColorDoctor 选区" });
  } finally {
    img.dispose();
  }
}

async function groupLayers(doc, layerIds, name) {
  const layers = doc.layers ? collect(doc.layers).filter((l) => layerIds.includes(l.id)) : [];
  if (!layers.length) return null;
  const g = await doc.createLayerGroup({ name, fromLayers: layers });
  return g ? g.id : null;
}

// 把图层（或组）从所在的组里挪出来，放到它最外层祖先的上方。UXP 的 Layer.move + ElementPlacement。
async function liftOutOfGroups(doc, layerId) {
  const layer = findLayer(doc, layerId);
  if (!layer || !layer.parent) return;
  let outer = layer.parent;
  while (outer.parent) outer = outer.parent;
  const before = (constants && constants.ElementPlacement && constants.ElementPlacement.PLACEBEFORE) || "placeBefore";
  await layer.move(outer, before);
}

function collect(layers, out = []) {
  for (const l of layers) {
    out.push(l);
    if (l.layers) collect(l.layers, out);
  }
  return out;
}

// 在一次历史记录里执行 fn(ctx)
async function modal(name, fn, { interactive = false, docId = null, docSize = null } = {}) {
  return core.executeAsModal(async (ctx) => {
    const doc = app.activeDocument;
    if (!doc) throw new Error("没有打开的文档。");
    // 防止分析后切换了文档、或改了画布尺寸：在错误的文档里建图层 / 写蒙版
    if (docId != null && doc.id !== docId) throw new Error("当前文档不是刚才分析的那一个，请切回原文档或重新分析。");
    if (docSize && (doc.width !== docSize[0] || doc.height !== docSize[1])) {
      throw new Error("文档尺寸在分析后改变了（裁剪 / 画布大小 / 图像大小），请重新分析。");
    }
    const id = await ctx.hostControl.suspendHistory({ documentID: doc.id, name });
    let ok = false;
    try {
      const r = await fn(ctx, doc);
      ok = true;
      return r;
    } finally {
      // 出错时不提交：已经做了一半的修改整体撤回
      await ctx.hostControl.resumeHistory(id, ok);
    }
  }, { commandName: name, interactive });
}

// 只读操作（读像素）也需要 modal，但不进历史记录。
// info.stack 时：临时隐藏当前图层上方的可见图层，读到的合成画面就是“到这一层为止”的样子；
// 读完立即恢复，并且这一步的历史记录整体丢弃（resumeHistory(..., false)），文档不留任何改动。
async function readModal(fn, info = null) {
  if (!info || !info.stack) return core.executeAsModal(async () => fn(), { commandName: "ColorDoctor 读取像素" });
  return core.executeAsModal(async (ctx) => {
    const doc = info.doc;
    const hist = await ctx.hostControl.suspendHistory({ documentID: doc.id, name: "ColorDoctor 读取像素" });
    const hidden = [];
    try {
      for (const l of layersAbove(doc, info.layer)) {
        if (l.visible !== false) { l.visible = false; hidden.push(l); }
      }
      return await fn();
    } finally {
      for (const l of hidden) { try { l.visible = true; } catch (_) { /* 已被删除 */ } }
      await ctx.hostControl.resumeHistory(hist, false);
    }
  }, { commandName: "ColorDoctor 读取像素" });
}

// ---------------------------------------------------------------- Camera Raw（待在 Windows 上用 Alchemist 录制确认）
// TODO(Windows)：录制“滤镜 → Camera Raw 滤镜”，改一个混色器滑块后确定，把描述符的键名替换到这里。
// 目前没有可靠的公开文档，所以“生成 Camera Raw 图层”先只做：复制图层 → 转智能对象 → 打开 Camera Raw 对话框
// （dialogOptions: "display"），并把 .xmp 预设导出给用户在对话框里一键套用。
async function cameraRawLayer(name, { merged = false, stack = false, layerId = null, doc = null } = {}) {
  if (stack && layerId != null) {
    // 到这一层为止的画面：选中它，临时隐藏上方图层，盖印可见图层（新图层就在它上方），再恢复上方图层
    const d = doc || app.activeDocument;
    await selectLayer(layerId);
    const hidden = layersAbove(d, findLayer(d, layerId) || { id: layerId }).filter((l) => l.visible !== false);
    for (const l of hidden) l.visible = false;
    try {
      checkResults(await action.batchPlay([{ _obj: "mergeVisible", duplicate: true }], {}), "盖印画面");
    } finally {
      for (const l of hidden) { try { l.visible = true; } catch (_) { /* 已被删除 */ } }
    }
    checkResults(await action.batchPlay([
      { _obj: "newPlacedLayer" },
      { _obj: "set", _target: [{ _ref: "layer", _enum: "ordinal", _value: "targetEnum" }], to: { _obj: "layer", name } },
    ], {}), "准备 Camera Raw 图层");
    const res = await action.batchPlay([{ _obj: "Adobe Camera Raw Filter", _options: { dialogOptions: "display" } }], {});
    if (isCancel(res)) {
      const e = new Error("已取消，没有新建图层。");
      e.cancelled = true;
      throw e;
    }
    checkResults(res, "Camera Raw");
    return;
  }
  // 合并可见图层：在最顶层盖印可见图层（Ctrl+Alt+Shift+E），否则上面的图层会挡住效果；
  // 单个图层：先选中要处理的图层，再整层复制。不用「通过拷贝的图层」(Ctrl+J)：有选区时它只复制选区里的像素，
  // 当前选中的是调整图层时还会复制成调整图层。
  if (merged) await selectTopLayer(doc || app.activeDocument);
  else if (layerId != null) await selectLayer(layerId);
  const first = merged
    ? { _obj: "mergeVisible", duplicate: true }
    : { _obj: "duplicate", _target: [{ _ref: "layer", _enum: "ordinal", _value: "targetEnum" }], version: 5 };
  checkResults(await action.batchPlay([
    first,
    { _obj: "newPlacedLayer" },
    { _obj: "set", _target: [{ _ref: "layer", _enum: "ordinal", _value: "targetEnum" }], to: { _obj: "layer", name } },
  ], {}), "准备 Camera Raw 图层");
  // 打开 Camera Raw 对话框让用户确认（需要 executeAsModal 的 interactive 模式）。
  // 用户点「取消」时 batchPlay 返回错误：抛出，让 modal 不提交历史记录，复制出的图层一并撤回。
  const res = await action.batchPlay([{ _obj: "Adobe Camera Raw Filter", _options: { dialogOptions: "display" } }], {});
  if (isCancel(res)) {
    const e = new Error("已取消，没有新建图层。");
    e.cancelled = true;
    throw e;
  }
  checkResults(res, "Camera Raw");
}

// PS 对话框里点「取消」：batchPlay 返回 error，result 为 -128（userCanceledErr）
function isCancel(results) {
  return (results || []).some((r) => r && r._obj === "error" && (r.result === -128 || /cancel|取消/i.test(String(r.message || ""))));
}

// 在文档（含所有图层组）里按 id 找图层；找不到返回 null
function findLayer(doc, id) {
  const walk = (layers) => {
    for (const l of layers || []) {
      if (l.id === id) return l;
      const hit = walk(l.layers);
      if (hit) return hit;
    }
    return null;
  };
  return doc ? walk(doc.layers) : null;
}

function findDocument(id) {
  return (app.documents || []).find((d) => d.id === id) || null;
}

// Camera Raw 滤镜的参数描述符：录制前为 null。
// TODO(Windows)：用 Alchemist 录“滤镜 → Camera Raw 滤镜”（改一个混色器滑块后确定），
// 把录到的描述符改写成 params -> descriptor 的函数赋给 ACR_DESCRIPTOR，校准和“直接写入参数”就能用了。
let ACR_DESCRIPTOR = null;

function acrDescriptorReady() {
  return typeof ACR_DESCRIPTOR === "function";
}

async function applyCameraRaw(layer, params) {
  if (!acrDescriptorReady()) throw new Error("Camera Raw 滤镜的命令格式还没有录制（见 docs/STAGE6_HANDOFF.md P9）");
  await action.batchPlay([
    { _obj: "select", _target: [{ _ref: "layer", _id: layer.id }], makeVisible: false },
    { ...ACR_DESCRIPTOR(params), _options: { dialogOptions: "dontDisplay" } },
  ], {});
}

// 新建 sRGB 文档，把色卡像素放进一个像素图层。必须在 executeAsModal 里调用。
async function createChartDocument(px) {
  const doc = await app.createDocument({
    width: px.width, height: px.height, resolution: 72, mode: "RGBColorMode", fill: "white",
    name: "ColorDoctor 校准色卡", profile: "sRGB IEC61966-2.1",  // 色卡按 sRGB 测量，文档必须也是 sRGB
  });
  const layer = await doc.createPixelLayer({ name: "色卡" });
  const img = await imaging.createImageDataFromBuffer(px.data, {
    width: px.width, height: px.height, components: 3, chunky: true, colorSpace: "RGB", colorProfile: "sRGB IEC61966-2.1",
  });
  try {
    await imaging.putPixels({ documentID: doc.id, layerID: layer.id, imageData: img, replace: true });
  } finally {
    img.dispose();
  }
  return { doc, layer };
}

async function duplicateLayer(layer) {
  return layer.duplicate();
}

async function deleteLayer(layer) {
  await layer.delete();
}

async function closeDocument(doc) {
  try { await doc.closeWithoutSaving(); } catch (_) { /* 用户已手动关闭 */ }
}

// 新建调整图层时，PS 会自动把当前选区变成它的蒙版：没有蒙版的（整体）调整会被意外限制在选区里。
// 生成前把选区存成临时通道并取消选择，生成后恢复选区、删除临时通道。必须在 executeAsModal 里调用。
const TEMP_SELECTION = "ColorDoctor 临时选区";
async function withoutSelection(doc, fn) {
  let has = false;
  try { has = !!(doc.selection && doc.selection.bounds); } catch (_) { has = false; }
  if (!has) return fn();
  const selRef = [{ _ref: "channel", _property: "selection" }];
  checkResults(await action.batchPlay([
    { _obj: "duplicate", _target: selRef, name: TEMP_SELECTION },
    { _obj: "set", _target: selRef, to: { _enum: "ordinal", _value: "none" } },
  ], {}), "暂存选区");
  try {
    return await fn();
  } finally {
    checkResults(await action.batchPlay([
      { _obj: "set", _target: selRef, to: { _ref: "channel", _name: TEMP_SELECTION } },
      { _obj: "delete", _target: [{ _ref: "channel", _name: TEMP_SELECTION }] },
    ], {}), "恢复选区");
  }
}

// 「合并可见图层」的结果要作用于整个画面：调整图层必须建在最顶层，否则上面的图层会把效果挡住
async function selectTopLayer(doc) {
  const top = doc.layers && doc.layers.length ? doc.layers[0] : null;
  if (top) await selectLayer(top.id);
}

async function selectLayer(id) {
  checkResults(await action.batchPlay([{ _obj: "select", _target: [{ _ref: "layer", _id: id }], makeVisible: false }], {}),
    "选中原图层");
}

module.exports = {
  IS_MAC, CMD,
  selectLayer, selectTopLayer, withoutSelection, liftOutOfGroups, findLayer, findDocument, isCancel, clipToCanvas, acrDescriptorReady, applyCameraRaw, createChartDocument, duplicateLayer, deleteLayer, closeDocument,
  constants, inspectTarget, layersAbove, historyStateId, readPixels, readSelectionPixels, readColorSamplers, foregroundLab,
  makeAdjustmentLayer, putMask, putSelection, groupLayers, modal, readModal, cameraRawLayer,
  _descriptors: { hueSatDescriptor, colorBalanceDescriptor, vibranceDescriptor, curvesDescriptor, selectiveColorDescriptor },
};
