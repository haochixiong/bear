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
        const bytes = await bridge.fetchMask(res.session, spec.id, 300, 200);
        assert.strictEqual(bytes.length, 300 * 200);
        await ps.putMask(d.id, id, bytes, 300, 200, { left: 0, top: 0 });
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
