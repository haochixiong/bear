// 在 Node 里用假的 photoshop 模块跑插件的 bridge.js / ps.js，连真实的本机服务。
// 用法：node photoshop-plugin/test/run.js <端口>   （由 tests/test_plugin_js.py 启动服务后调用）
const Module = require("module");
const assert = require("assert");
const path = require("path");

const port = process.argv[2];
const calls = { batchPlay: [], masks: [], selections: [] };
let nextLayerId = 100;
const doc = {
  id: 1, name: "测试.jpg", width: 300, height: 200,
  activeLayers: [{ id: 7, name: "背景", kind: "pixel", bounds: { left: 0, top: 0, right: 300, bottom: 200 } }],
  activeHistoryState: { id: 11 },
  layers: [],
  async createLayerGroup(opts) { calls.group = opts; return { id: 999 }; },
};
const fakePs = {
  app: { activeDocument: doc, documents: [doc], foregroundColor: {} },
  constants: {},
  core: { executeAsModal: async (fn) => fn({ hostControl: { suspendHistory: async () => 1, resumeHistory: async () => {} } }) },
  action: {
    batchPlay: async (descs) => {
      calls.batchPlay.push(...descs);
      if (descs[0]._obj === "make") {
        const l = { id: nextLayerId++, name: "", kind: "hueSaturation" };
        doc.layers.push(l);
        doc.activeLayers = [l];
      }
      return [];
    },
    addNotificationListener: () => {},
  },
  imaging: {
    createImageDataFromBuffer: async (buf, o) => ({ buf, ...o, dispose() {} }),
    putLayerMask: async (o) => calls.masks.push(o),
    putSelection: async (o) => calls.selections.push(o),
  },
};
const origLoad = Module._load;
Module._load = function (req, ...rest) {
  if (req === "photoshop") return fakePs;
  if (req === "uxp") return {};
  return origLoad.call(this, req, ...rest);
};
// bridge.js 写死了 47321 端口：测试时改到临时端口
const realFetch = global.fetch;
global.fetch = (url, opts) => realFetch(String(url).replace("127.0.0.1:47321", `127.0.0.1:${port}`), opts);

const js = (f) => path.join(__dirname, "..", "js", f);
const bridge = require(js("bridge.js"));
const ps = require(js("ps.js"));

function dirtySky(w, h) {
  // 蓝色天空，右半边散布 6% 黄色杂点（与 Python 测试相同的思路）
  const data = new Uint8Array(w * h * 4);
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const n = Math.round((rnd() - 0.5) * 4);
      let c = [70 + n, 120 + n, 200 + n];
      if (x >= w / 2 && rnd() < 0.06) c = [225, 200, 60];
      data.set([...c, 255], i);
    }
  }
  return { data, width: w, height: h, components: 4 };
}

(async () => {
  // 1. 图层检查
  let info = ps.inspectTarget("layer");
  assert.ok(info.ok, info.reason);
  doc.activeLayers = [{ id: 8, name: "曲线 1", kind: "curves" }];
  info = ps.inspectTarget("layer");
  assert.ok(!info.ok && info.reason.includes("调整图层"));
  doc.activeLayers = [{ id: 9, name: "图层 3", kind: "pixel", bounds: { left: 0, top: 0, right: 0, bottom: 0 } }];
  assert.ok(ps.inspectTarget("layer").reason.includes("空白"));
  assert.ok(ps.inspectTarget("merged").ok);

  // 2. 服务：健康检查、空白图层被拦截
  assert.ok((await bridge.health()).ok);
  await assert.rejects(bridge.analyze({ data: new Uint8Array(400), width: 10, height: 10, components: 4 }, "dirty", {}),
    (e) => e.blocked && e.message.includes("空白"));

  // 3. 分析 → 生成图层（与面板的 generate() 相同步骤）
  const px = dirtySky(300, 200);
  const res = await bridge.analyze(px, "dirty", {}, { docWidth: 300, docHeight: 200, layerKind: "pixel", layerName: "背景" });
  assert.ok(res.plan.layers.length >= 1, JSON.stringify(res.plan));
  await ps.modal("ColorDoctor 校正", async (ctx, d) => {
    const made = [];
    for (const spec of res.plan.layers) {
      const id = await ps.makeAdjustmentLayer(spec);
      if (spec.has_mask) {
        const m = await bridge.fetchMask(res.session, spec.id, { left: 0, top: 0, right: 300, bottom: 200 }, 300, 200);
        assert.strictEqual(m.data.length, 300 * 200);
        await ps.putMask(d.id, id, m.data, m.width, m.height, { left: 0, top: 0 }, "bitDepth8");
      }
      made.push(id);
    }
  });
  const make = calls.batchPlay.find((c) => c._obj === "make");
  assert.strictEqual(make._target[0]._ref, "adjustmentLayer");
  const set = calls.batchPlay.find((c) => c._obj === "set" && c.to._obj === "hueSaturation");
  assert.ok(set, "应该设置色相/饱和度参数");
  const yellow = set.to.adjustment.find((a) => a.localRange === 2);
  assert.ok(yellow && yellow.saturation < 0, "黄色应降饱和");
  assert.ok(calls.masks.length >= 1 && calls.masks[0].imageData.components === 1);
  const m = calls.masks[0].imageData.buf;
  let left = 0, right = 0;
  for (let y = 0; y < 200; y++) for (let x = 0; x < 300; x++) (x < 140 ? (left += m[y * 300 + x]) : x > 160 ? (right += m[y * 300 + x]) : 0);
  assert.ok(right > 5 * Math.max(left, 1), "蒙版应集中在右半边");

  // 4. 其他描述符
  const cb = ps._descriptors.colorBalanceDescriptor({ shadows: [0, 0, 0], midtones: [-5, 0, -3], highlights: [0, 0, 0] });
  assert.deepStrictEqual(cb.midtoneLevels, [-5, 0, -3]);
  const cv = ps._descriptors.curvesDescriptor({ channels: { composite: [[0, 12], [255, 248]] } });
  assert.strictEqual(cv.adjustment[0].curve[0].vertical, 12);

  // 5. 取色、ACR 预设
  const patch = await bridge.analyze({ data: new Uint8Array([199, 156, 132]), width: 1, height: 1, components: 3 }, "patch", {}, { layerKind: "pixel" });
  assert.ok(Math.abs(patch.values.lab[0] - 68.3) < 1.5, JSON.stringify(patch.values.lab));
  const xmp = await bridge.acrPreset("测试", { hsl: { 绿色: { hue: 60 } } });
  assert.ok(xmp.includes('crs:HueAdjustmentGreen="+60"'));
  // 5b. batchPlay 返回 error 时不能把原图层改名
  const realBp = fakePs.action.batchPlay;
  fakePs.action.batchPlay = async () => [{ _obj: "error", message: "not available" }];
  doc.activeLayers = [{ id: 7, name: "背景", kind: "pixel" }];
  await assert.rejects(ps.makeAdjustmentLayer(res.plan.layers[0]), /失败/);
  assert.strictEqual(doc.activeLayers[0].name, "背景");
  fakePs.action.batchPlay = realBp;

  // 5c. 大图：PS 从金字塔第 1 级读取时 sourceBounds 是缩小一半的坐标，要换算回文档坐标
  const bigDoc = { id: 3, width: 6000, height: 4000 };
  fakePs.imaging.getPixels = async (o) => ({
    level: 1,
    sourceBounds: { left: o.sourceBounds.left / 2, top: o.sourceBounds.top / 2, right: o.sourceBounds.right / 2, bottom: o.sourceBounds.bottom / 2 },
    imageData: { width: 2048, height: 1365, components: 4, getData: async () => new Uint8Array(2048 * 1365 * 4), dispose() {} },
  });
  const big = await ps.readPixels({ doc: bigDoc, layer: null, bounds: { left: 0, top: 0, right: 6000, bottom: 4000 } });
  assert.deepStrictEqual(big.sourceBounds, { left: 0, top: 0, right: 6000, bottom: 4000 });
  const part = await ps.readPixels({ doc: bigDoc, layer: { id: 9 }, bounds: { left: 1000, top: 600, right: 5000, bottom: 3800 } });
  assert.deepStrictEqual(part.sourceBounds, { left: 1000, top: 600, right: 5000, bottom: 3800 });

  // 5d. 图层超出画布：只读画布内的部分
  let lastOpts = null;
  fakePs.imaging.getPixels = async (o) => {
    lastOpts = o;
    const w = o.sourceBounds.right - o.sourceBounds.left, h = o.sourceBounds.bottom - o.sourceBounds.top;
    return { level: 0, sourceBounds: o.sourceBounds,
      imageData: { width: w, height: h, components: 3, getData: async () => new Uint8Array(w * h * 3).fill(200), dispose() {} } };
  };
  const smallDoc = { id: 4, width: 100, height: 80, selection: { bounds: { left: 10, top: 10, right: 30, bottom: 20 } } };
  const out1 = await ps.readPixels({ doc: smallDoc, layer: { id: 1 }, bounds: { left: -50, top: -10, right: 60, bottom: 200 } });
  assert.deepStrictEqual(lastOpts.sourceBounds, { left: 0, top: 0, right: 60, bottom: 80 });
  assert.deepStrictEqual(out1.sourceBounds, { left: 0, top: 0, right: 60, bottom: 80 });

  // 5e. 不规则选区：只取选中的像素（选区外 alpha = 0）
  fakePs.imaging.getSelection = async (o) => {
    const w = o.sourceBounds.right - o.sourceBounds.left, h = o.sourceBounds.bottom - o.sourceBounds.top;
    const d = new Uint8Array(w * h);
    for (let i = 0; i < d.length; i++) d[i] = (i % w) < w / 2 ? 255 : 0;  // 左半边选中
    return { imageData: { width: w, height: h, getData: async () => d, dispose() {} } };
  };
  const selPx = await ps.readSelectionPixels({ doc: smallDoc, layer: null, bounds: null });
  assert.strictEqual(selPx.components, 4);
  assert.strictEqual(selPx.data[3], 255);
  assert.strictEqual(selPx.data[(selPx.width - 1) * 4 + 3], 0);

  // 5f. 16 位文档：蒙版写成 0–32768
  let made16 = null;
  fakePs.imaging.createImageDataFromBuffer = async (buf, o) => { made16 = buf; return { buf, ...o, dispose() {} }; };
  await ps.putMask(1, 2, new Uint8Array([0, 255]), 2, 1, { left: 0, top: 0 }, "bitDepth16");
  assert.ok(made16 instanceof Uint16Array && made16[1] === 32768);
  await ps.putMask(1, 2, new Uint8Array([0, 255]), 2, 1, { left: 0, top: 0 }, "bitDepth8");
  assert.ok(made16 instanceof Uint8Array);

  // 5g. 合并画面：先选中最顶层
  calls.batchPlay.length = 0;
  await ps.selectTopLayer({ layers: [{ id: 77 }, { id: 5 }] });
  assert.strictEqual(calls.batchPlay[0]._target[0]._id, 77);

  // 5h. Camera Raw 图层：单个图层先选中分析的图层再整层复制（不用 copyToLayer：有选区时只复制选区）；
  //     合并画面在最顶层盖印；用户取消 Camera Raw 时抛错（modal 不提交，复制出的图层撤回）
  calls.batchPlay.length = 0;
  await ps.cameraRawLayer("CR", { merged: false, layerId: 7 });
  assert.strictEqual(calls.batchPlay[0]._obj, "select");
  assert.strictEqual(calls.batchPlay[0]._target[0]._id, 7);
  assert.ok(calls.batchPlay.some((c) => c._obj === "duplicate"));
  assert.ok(!calls.batchPlay.some((c) => c._obj === "copyToLayer"));
  calls.batchPlay.length = 0;
  await ps.cameraRawLayer("CR", { merged: true, doc: { layers: [{ id: 88 }] } });
  assert.strictEqual(calls.batchPlay[0]._target[0]._id, 88);
  assert.ok(calls.batchPlay.some((c) => c._obj === "mergeVisible" && c.duplicate));
  const bp0 = fakePs.action.batchPlay;
  fakePs.action.batchPlay = async (descs) => (descs[0]._obj === "Adobe Camera Raw Filter" ? [{ _obj: "error", result: -128 }] : []);
  await assert.rejects(ps.cameraRawLayer("CR", { merged: false, layerId: 7 }), (e) => e.cancelled === true && /已取消/.test(e.message));
  assert.strictEqual(ps.findLayer({ layers: [{ id: 1, layers: [{ id: 9 }] }] }, 9).id, 9);
  assert.strictEqual(ps.findLayer({ layers: [{ id: 1 }] }, 9), null);
  fakePs.action.batchPlay = bp0;

  // 5i. 有选区时：先存成临时通道并取消选择，生成后恢复并删除临时通道；没有选区时不碰
  calls.batchPlay.length = 0;
  assert.strictEqual(await ps.withoutSelection({ selection: { bounds: null } }, async () => 3), 3);
  assert.strictEqual(calls.batchPlay.length, 0);
  const seq = [];
  const r5 = await ps.withoutSelection({ selection: { bounds: { left: 0, top: 0, right: 5, bottom: 5 } } }, async () => {
    seq.push(calls.batchPlay.map((c) => c._obj).join(","));
    return 4;
  });
  assert.strictEqual(r5, 4);
  assert.strictEqual(seq[0], "duplicate,set");
  assert.deepStrictEqual(calls.batchPlay.map((c) => c._obj), ["duplicate", "set", "set", "delete"]);
  calls.batchPlay.length = 0;
  await assert.rejects(ps.withoutSelection({ selection: { bounds: { left: 0, top: 0, right: 5, bottom: 5 } } }, async () => {
    throw new Error("boom");
  }), /boom/);
  assert.deepStrictEqual(calls.batchPlay.map((c) => c._obj), ["duplicate", "set", "set", "delete"]);  // 失败也恢复

  // 5j. 直接选参考图文件：不是图片时给出中文原因
  await assert.rejects(bridge.uploadReferenceFile("参考.jpg", new Uint8Array([1, 2, 3])), (e) => e.status === 400 && /图片/.test(e.message));

  // 5k. 可选颜色描述符；颜色取样器多点对比（真实服务）
  const sc = ps._descriptors.selectiveColorDescriptor({ method: "relative", colors: { reds: { c: 5, m: -3, y: -12, k: 0 }, bogus: { c: 1 } } });
  assert.strictEqual(sc._obj, "selectiveColor");
  assert.strictEqual(sc.method._value, "relative");
  assert.strictEqual(sc.colorCorrection.length, 1);
  assert.strictEqual(sc.colorCorrection[0].colors._value, "reds");
  assert.strictEqual(sc.colorCorrection[0].yellowColor._value, -12);
  doc.colorSamplers = [{ color: { lab: { l: 71, a: 13.8, b: 19.6 } }, position: { x: 10, y: 10 } },
    { color: { lab: { l: 67, a: 10.9, b: 24.9 } }, position: { x: 50, y: 80 } }];
  const pts = ps.readColorSamplers();
  assert.strictEqual(pts.length, 2);
  const mpres = await bridge.multipoint(pts.map((p) => ({ name: p.name, lab: p.lab })));
  assert.strictEqual(mpres.rows[0].judge, "基准");
  assert.ok(/偏黄/.test(mpres.rows[1].direction) && mpres.rows[1].words);
  doc.colorSamplers = [];
  assert.throws(() => ps.readColorSamplers(), /颜色取样器/);

  // 6. 校准用的色卡文档；Camera Raw 描述符未录制时应明确报错
  fakePs.app.createDocument = async (o) => ({ id: 2, ...o, createPixelLayer: async (lo) => ({ id: 55, ...lo }) });
  fakePs.imaging.putPixels = async (o) => { calls.putPixels = o; };
  const chart = await bridge.calibrationChart();
  const made = await ps.createChartDocument(chart);
  assert.strictEqual(made.layer.id, 55);
  assert.strictEqual(calls.putPixels.imageData.width, chart.width);
  assert.ok(!ps.acrDescriptorReady());
  await assert.rejects(ps.applyCameraRaw(made.layer, {}), /录制/);
  console.log("PLUGIN JS OK");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
