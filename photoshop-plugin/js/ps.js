// Photoshop 端操作：读取当前图层信息与像素、检查能否分析、建调整图层、写图层蒙版。
// 所有修改都在 executeAsModal 里完成，并合并成一步历史记录（Ctrl+Z 一次撤销）。
const { app, action, core, imaging, constants } = require("photoshop");

const MAX_SIDE = 2048; // 分析用的长边像素（蒙版会按文档尺寸放大回去）

// 没有像素、不能直接分析的图层类型（与 Python 端 pixel_guard.NO_PIXEL_KINDS 对应）
const NO_PIXEL_KINDS = new Set([
  "brightnessContrast", "levels", "curves", "exposure", "vibrance", "hueSaturation", "colorBalance",
  "blackAndWhite", "photoFilter", "channelMixer", "colorLookup", "invert", "posterize", "threshold",
  "gradientMap", "selectiveColor", "solidColor", "gradientFill", "pattern", "light", "clarity", "dehaze", "grain",
]);
const NEED_MERGED_KINDS = { group: "图层组", text: "文字图层", smartObject: "智能对象", video: "视频图层" };

function kindName(layer) {
  const k = layer.kind;
  return typeof k === "string" ? k : String(k);
}

// 返回 {ok, reason, doc, layer, kind, bounds}；target: "layer" | "merged"
function inspectTarget(target) {
  const doc = app.activeDocument;
  if (!doc) return { ok: false, reason: "没有打开的文档。" };
  if (target === "merged") {
    return { ok: true, doc, layer: null, kind: "merged", label: "合并可见图层（整个画面）",
      bounds: { left: 0, top: 0, right: doc.width, bottom: doc.height } };
  }
  const layers = doc.activeLayers;
  if (!layers || layers.length !== 1) {
    return { ok: false, reason: "请在图层面板里只选中一个图层，或把分析对象改为「合并可见图层」。" };
  }
  const layer = layers[0];
  const kind = kindName(layer);
  const label = `${layer.name}`;
  if (NO_PIXEL_KINDS.has(kind)) {
    return { ok: false, layer, kind, label,
      reason: `「${layer.name}」是调整图层或填充图层，本身没有像素。请选中像素图层，或把分析对象改为「合并可见图层」。` };
  }
  if (NEED_MERGED_KINDS[kind]) {
    return { ok: false, layer, kind, label,
      reason: `「${layer.name}」是${NEED_MERGED_KINDS[kind]}，请把分析对象改为「合并可见图层」，或先栅格化。` };
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

// 读取像素（sRGB、8 位、交错），长边缩到 MAX_SIDE。必须在 executeAsModal 里调用。
async function readPixels(info, sourceBounds) {
  const b = sourceBounds || info.bounds;
  const w = b.right - b.left;
  const h = b.bottom - b.top;
  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  const opts = {
    documentID: info.doc.id,
    sourceBounds: b,
    colorSpace: "RGB",
    colorProfile: "sRGB IEC61966-2.1",
    componentSize: 8,
    applyAlpha: false,
  };
  if (scale < 1) opts.targetSize = w >= h ? { width: Math.round(w * scale) } : { height: Math.round(h * scale) };
  if (info.layer) opts.layerID = info.layer.id;
  const res = await imaging.getPixels(opts);
  const img = res.imageData;
  const data = await img.getData({ chunky: true });
  const out = { data: new Uint8Array(data), width: img.width, height: img.height, components: img.components,
    sourceBounds: res.sourceBounds || b };
  img.dispose();
  return out;
}

async function readSelectionPixels(info) {
  const sel = info.doc.selection && info.doc.selection.bounds;
  if (!sel || sel.right - sel.left < 1) throw new Error("请先用选框工具框选要取色的区域（或用吸管设置前景色）。");
  return readPixels(info, { left: sel.left, top: sel.top, right: sel.right, bottom: sel.bottom });
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

function curvesDescriptor(p) {
  const adjustment = Object.entries(p.channels).map(([ch, pts]) => ({
    _obj: "curvesAdjustment",
    channel: { _ref: "channel", _enum: "channel", _value: ch === "composite" ? "composite" : ch },
    curve: pts.map(([x, y]) => ({ _obj: "paint", horizontal: x, vertical: y })),
  }));
  return { _obj: "curves", presetKind: { _enum: "presetKindType", _value: "presetKindCustom" }, adjustment };
}

const DESCRIPTORS = {
  hueSaturation: hueSatDescriptor,
  colorBalance: colorBalanceDescriptor,
  vibrance: vibranceDescriptor,
  curves: curvesDescriptor,
};
const DEFAULT_TYPES = {
  hueSaturation: { _obj: "hueSaturation", presetKind: { _enum: "presetKindType", _value: "presetKindDefault" }, colorize: false },
  colorBalance: { _obj: "colorBalance", shadowLevels: [0, 0, 0], midtoneLevels: [0, 0, 0], highlightLevels: [0, 0, 0], preserveLuminosity: true },
  vibrance: { _obj: "vibrance" },
  curves: { _obj: "curves", presetKind: { _enum: "presetKindType", _value: "presetKindDefault" } },
};

// 在当前选中图层上方新建一个调整图层，返回其 layer id。必须在 executeAsModal 里调用。
async function makeAdjustmentLayer(spec) {
  const build = DESCRIPTORS[spec.kind];
  if (!build) throw new Error(`不支持的图层类型：${spec.kind}`);
  await action.batchPlay([
    { _obj: "make", _target: [{ _ref: "adjustmentLayer" }], using: { _obj: "adjustmentLayer", type: DEFAULT_TYPES[spec.kind] } },
    { _obj: "set", _target: ORDINAL, to: build(spec.params) },
    { _obj: "set", _target: [{ _ref: "layer", _enum: "ordinal", _value: "targetEnum" }], to: { _obj: "layer", name: spec.name } },
  ], {});
  return app.activeDocument.activeLayers[0].id;
}

// 把灰度蒙版（Uint8Array，w*h）写进图层的用户蒙版，放在文档的 bounds 位置。
async function putMask(docId, layerId, bytes, width, height, bounds) {
  const img = await imaging.createImageDataFromBuffer(bytes, {
    width, height, components: 1, chunky: true, colorSpace: "Grayscale", colorProfile: "Gray Gamma 2.2",
  });
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
async function putSelection(docId, bytes, width, height, bounds) {
  const img = await imaging.createImageDataFromBuffer(bytes, {
    width, height, components: 1, chunky: true, colorSpace: "Grayscale", colorProfile: "Gray Gamma 2.2",
  });
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

function collect(layers, out = []) {
  for (const l of layers) {
    out.push(l);
    if (l.layers) collect(l.layers, out);
  }
  return out;
}

// 在一次历史记录里执行 fn(ctx)
async function modal(name, fn) {
  return core.executeAsModal(async (ctx) => {
    const doc = app.activeDocument;
    const id = await ctx.hostControl.suspendHistory({ documentID: doc.id, name });
    try {
      return await fn(ctx, doc);
    } finally {
      await ctx.hostControl.resumeHistory(id);
    }
  }, { commandName: name });
}

// 只读操作（读像素）也需要 modal，但不进历史记录
async function readModal(fn) {
  return core.executeAsModal(async () => fn(), { commandName: "ColorDoctor 读取像素" });
}

// ---------------------------------------------------------------- Camera Raw（待在 Windows 上用 Alchemist 录制确认）
// TODO(Windows)：录制“滤镜 → Camera Raw 滤镜”，改一个混色器滑块后确定，把描述符的键名替换到这里。
// 目前没有可靠的公开文档，所以“生成 Camera Raw 图层”先只做：复制图层 → 转智能对象 → 打开 Camera Raw 对话框
// （dialogOptions: "display"），并把 .xmp 预设导出给用户在对话框里一键套用。
async function cameraRawLayer(name) {
  await action.batchPlay([
    { _obj: "copyToLayer" },
    { _obj: "newPlacedLayer" },
    { _obj: "set", _target: [{ _ref: "layer", _enum: "ordinal", _value: "targetEnum" }], to: { _obj: "layer", name } },
  ], {});
  await action.batchPlay([{ _obj: "Adobe Camera Raw Filter", _options: { dialogOptions: "display" } }], {});
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
    name: "ColorDoctor 校准色卡",
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

module.exports = {
  acrDescriptorReady, applyCameraRaw, createChartDocument, duplicateLayer, deleteLayer,
  constants, inspectTarget, historyStateId, readPixels, readSelectionPixels, foregroundLab,
  makeAdjustmentLayer, putMask, putSelection, groupLayers, modal, readModal, cameraRawLayer,
  _descriptors: { hueSatDescriptor, colorBalanceDescriptor, vibranceDescriptor, curvesDescriptor },
};
