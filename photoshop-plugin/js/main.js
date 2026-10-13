// ColorDoctor 面板：只在用户点击时分析；只新增调整图层和蒙版，不改原图像素。
const { app, action } = require("photoshop");
const uxp = require("uxp");
const bridge = require("./bridge.js");
const ps = require("./ps.js");

const SUBS = {
  color: [["dirty", "脏色"], ["patch", "取色"], ["ratio", "比例"], ["style", "风格"]],
  portrait: [["skin", "皮肤均匀"], ["skinpatch", "肤色取色"]],
  reference: [["reference", "仿色"]],
};
const MODE_OF = { dirty: "dirty", patch: "patch", skinpatch: "patch", ratio: "palette", style: "palette", skin: "skin", reference: "reference" };

const state = {
  cat: "color", sub: "dirty", options: null,
  result: null,            // 最近一次分析返回
  analyzed: null,          // {docId, layerId, history, bounds, target}
  refId: null,
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- 分析对象栏
function refreshTarget() {
  const info = ps.inspectTarget($("target").value);
  $("targetName").textContent = info.label || "—";
  const st = $("status");
  if (!info.ok) {
    $("analyze").disabled = true;
    st.className = "status warn";
    st.textContent = info.reason;
    return info;
  }
  $("analyze").disabled = false;
  const a = state.analyzed;
  if (a && info.doc && a.docId === info.doc.id && a.layerId === (info.layer ? info.layer.id : null)) {
    const stale = a.history !== ps.historyStateId(info.doc);
    st.className = stale ? "status warn" : "status ok";
    st.textContent = stale ? "这个图层在上次分析后有改动，结果可能过期，请再点一次「分析当前图层」" : "已分析 · 下面的结果对应这个图层";
  } else {
    st.className = "status";
    st.textContent = "只在点击按钮时分析";
  }
  return info;
}

// ---------------------------------------------------------------- 分类与控件
function renderNav() {
  document.querySelectorAll(".cat").forEach((b) => b.classList.toggle("on", b.dataset.cat === state.cat));
  $("subs").innerHTML = SUBS[state.cat].map(([k, n]) => `<button data-sub="${k}" class="${k === state.sub ? "on" : ""}">${n}</button>`).join("");
  $("subs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.sub = b.dataset.sub; renderNav(); }));
  renderControls();
  $("analyze").textContent = MODE_OF[state.sub] === "patch" ? "读取选区颜色" : "分析当前图层";
}

function select(id, items, withNone) {
  const opts = (withNone ? [`<option value="">${withNone}</option>`] : []).concat(items.map((i) => `<option value="${esc(i.key)}">${esc(i.name)}</option>`));
  return `<select id="${id}">${opts.join("")}</select>`;
}

function renderControls() {
  const o = state.options || { styles: [], presets: [], families: [] };
  const c = $("controls");
  if (state.sub === "style") {
    const groups = {};
    o.styles.forEach((s) => (groups[s.category] = groups[s.category] || []).push(s));
    c.innerHTML = `<div class="row"><label>目标风格</label><select id="style">${Object.entries(groups).map(([g, ss]) =>
      `<optgroup label="${esc(g)}">${ss.map((s) => `<option value="${esc(s.key)}">${esc(s.name)}</option>`).join("")}</optgroup>`).join("")}</select></div>`;
  } else if (state.sub === "ratio") {
    c.innerHTML = `<div class="row"><label>颜色统一</label>${select("family", o.families.map((f) => ({ key: f, name: f })), "（只看比例）")}</div>`;
  } else if (state.sub === "skin") {
    c.innerHTML = `<div class="row"><label>抠皮肤</label><select id="seg"><option value="ai">AI 人像分割（显卡）</option><option value="color">肤色规则（不用 AI）</option></select></div>
      <div class="row"><label>肤色风格</label>${select("preset", o.presets)}</div>
      <div class="row"><label>部位</label><select id="part"><option value="脸部">脸部</option><option value="身体">身体</option><option value="">全部皮肤</option></select></div>`;
  } else if (MODE_OF[state.sub] === "patch") {
    c.innerHTML = `<div class="row"><button id="fg">读取前景色（先用吸管取色）</button></div>
      <div class="muted">或用选框框出一块区域，再点上面的「读取选区颜色」（相当于大取样）。</div>`;
    $("fg").addEventListener("click", readForeground);
  } else if (state.sub === "reference") {
    const docs = app.documents.filter((d) => !app.activeDocument || d.id !== app.activeDocument.id);
    c.innerHTML = `<div class="row"><label>参考图</label><select id="refDoc">${docs.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("") || '<option value="">（先在 PS 里打开参考图）</option>'}</select></div>`;
  } else {
    c.innerHTML = "";
  }
}

function query() {
  const q = {};
  const v = (id) => ($(id) ? $(id).value : "");
  if (state.sub === "style") q.style = v("style");
  if (state.sub === "ratio" && v("family")) q.family = v("family");
  if (state.sub === "skin") { q.preset = v("preset"); q.part = v("part"); q.seg = v("seg"); }
  if (state.sub === "reference") q.ref = state.refId || "";
  return q;
}

// ---------------------------------------------------------------- 分析
async function runAnalyze() {
  const info = refreshTarget();
  if (!info.ok) return;
  const mode = MODE_OF[state.sub];
  setBusy(true, mode === "patch" ? "正在读取…" : "正在分析…");
  try {
    let px;
    if (mode === "patch") {
      px = await ps.readModal(() => ps.readSelectionPixels(info));
    } else {
      px = await ps.readModal(() => ps.readPixels(info));
    }
    if (mode === "reference") {
      const refDoc = app.documents.find((d) => String(d.id) === ($("refDoc") || {}).value);
      if (!refDoc) throw new Error("请先在 PS 里打开参考图，并在上面选择它。");
      const refInfo = { ok: true, doc: refDoc, layer: null, bounds: { left: 0, top: 0, right: refDoc.width, bottom: refDoc.height } };
      const refPx = await ps.readModal(() => ps.readPixels(refInfo));
      state.refId = await bridge.uploadReference(refPx);
    }
    const sb = px.sourceBounds;
    const res = await bridge.analyze(px, mode, query(), {
      docWidth: sb.right - sb.left, docHeight: sb.bottom - sb.top, layerKind: info.kind, layerName: info.label,
    });
    state.result = res;
    state.analyzed = { docId: info.doc.id, layerId: info.layer ? info.layer.id : null, history: ps.historyStateId(info.doc), bounds: sb,
      docWidth: info.doc.width, docHeight: info.doc.height };
    renderResult();
  } catch (e) {
    renderError(e);
  } finally {
    setBusy(false);
    refreshTarget();
  }
}

async function readForeground() {
  const c = app.foregroundColor.rgb;
  const px = { data: new Uint8Array([c.red, c.green, c.blue]), width: 1, height: 1, components: 3 };
  try {
    state.result = await bridge.analyze(px, "patch", {}, { layerKind: "pixel" });
    renderResult();
  } catch (e) { renderError(e); }
}

function setBusy(busy, text) {
  $("analyze").disabled = busy;
  if (busy) { $("status").className = "status"; $("status").textContent = text; }
}

function renderError(e) {
  const blocked = e && e.blocked;
  $("results").innerHTML = `<div class="blocked">${blocked ? "<b>没有可分析的像素</b><br>" : ""}${esc(e.message || e)}</div>`;
  $("generateAll").disabled = true;
}

// ---------------------------------------------------------------- 结果
function renderResult() {
  const r = state.result;
  if (!r) return;
  if (r.mode === "patch") return renderPatch(r);
  const cards = [];
  (r.notes || []).forEach((n) => cards.push(`<div class="muted">${esc(n)}</div>`));
  const layers = (r.plan && r.plan.layers) || [];
  layers.forEach((ly, i) => cards.push(`
    <div class="card">
      <div class="title"><span>${i + 1}. ${esc(ly.name)}</span><span class="muted">${ly.estimated ? "估计" : ""}</span></div>
      <div class="line">${esc(ly.reason)}</div>
      <div class="row">
        <button class="small make" data-make="${esc(ly.id)}">生成此层${ly.has_mask ? "（带蒙版）" : ""}</button>
        ${ly.has_mask ? `<button class="small" data-sel="${esc(ly.id)}">只生成选区</button>` : ""}
      </div>
    </div>`));
  (r.findings || []).forEach((f) => cards.push(`
    <div class="card">
      <div class="title"><span>${esc(f.problem)}</span><span class="sev-${esc(f.severity)}">${esc(f.severity)}</span></div>
      ${(f.evidence || []).slice(0, 3).map((x) => `<div class="line">证据：${esc(x)}</div>`).join("")}
      ${(f.causes || []).slice(0, 2).map((x) => `<div class="line">可能原因：${esc(x)}</div>`).join("")}
      ${(f.adjustments || []).slice(0, 3).map((x) => `<div class="line">调整：${esc(x.tool)} · ${esc(x.instruction)}</div>`).join("")}
      <div class="muted">置信度 ${Math.round((f.confidence || 0) * 100)}%${(f.notes || []).length ? " · 注意：" + esc(f.notes[0]) : ""}</div>
    </div>`));
  if (!layers.length && !(r.findings || []).length) cards.push('<p class="hint">没有发现需要处理的问题。</p>');
  $("results").innerHTML = cards.join("");
  $("results").querySelectorAll("[data-make]").forEach((b) => b.addEventListener("click", () => generate([b.dataset.make])));
  $("results").querySelectorAll("[data-sel]").forEach((b) => b.addEventListener("click", () => makeSelection(b.dataset.sel)));
  $("generateAll").disabled = !layers.length;
  renderAcr(r.acr_params || {}, r.acr_calibrated);
}

function renderPatch(r) {
  const v = r.values;
  const qr = r.quick_rule;
  const qrOk = qr && qr.a_ok && qr.b_ok && qr.diff_ok;
  $("results").innerHTML = `
    <div class="swatch" style="background:${esc(v.hex)}"></div>
    <div class="values">
      <div>L* ${v.lab[0].toFixed(1)}</div><div>a* ${v.lab[1].toFixed(1)}</div><div>b* ${v.lab[2].toFixed(1)}</div>
      <div>C* ${v.lch[1].toFixed(1)}</div><div>h° ${v.lch[2].toFixed(1)}</div><div>${esc(v.hue_name)}</div>
    </div>
    <div class="card"><div class="line">RGB ${v.rgb.join(" ")} · ${esc(v.hex)}</div>
      <div class="line">HSB ${v.hsb.map((x) => Math.round(x)).join(" / ")}（= PS 拾色器）</div>
      <div class="muted">取样 ${r.pixels} 像素 · Lab 为 D50（与 PS 信息面板一致）</div></div>
    ${qr ? `<div class="card"><div class="title"><span>肤色快速规则</span><span>${qrOk ? "通过" : "不通过"}</span></div>
      <div class="line">a* ${qr.a.toFixed(1)}、b* ${qr.b.toFixed(1)}，|a−b| = ${Math.abs(qr.a - qr.b).toFixed(1)}</div></div>` : ""}
    ${r.acr_channels.length ? `<div class="card"><div class="title"><span>归哪个 ACR 滑块管</span></div>
      ${r.acr_channels.map((c) => `<div class="line">${esc(c.channel)} ${(c.weight * 100).toFixed(0)}%</div>`).join("")}</div>` : ""}
    ${r.purity != null ? `<div class="card"><div class="line">纯度：${(r.purity * 100).toFixed(0)}% 的像素与平均色相差在 30° 以内</div></div>` : ""}`;
  $("generateAll").disabled = true;
}

// ---------------------------------------------------------------- 生成图层
async function generate(ids) {
  const r = state.result;
  const a = state.analyzed;
  if (!r || !a) return;
  const doc = app.activeDocument;
  if (!doc || doc.id !== a.docId) { renderError(new Error("当前文档不是刚才分析的那一个，请重新分析。")); return; }
  const specs = r.plan.layers.filter((l) => ids.includes(l.id));
  try {
    // 先把所有蒙版取回来（整幅文档大小，分析范围外为 0）；取不到就什么都不建
    const masks = {};
    for (const spec of specs) {
      if (spec.has_mask) masks[spec.id] = await bridge.fetchMask(r.session, spec.id, a.bounds, a.docWidth, a.docHeight);
    }
    await ps.modal(ids.length > 1 ? "ColorDoctor 一键校正" : "ColorDoctor 校正", async (ctx, d) => {
      if (a.layerId != null) {
        await action.batchPlay([{ _obj: "select", _target: [{ _ref: "layer", _id: a.layerId }], makeVisible: false }], {});
      }
      const made = [];
      for (const spec of specs) {
        const id = await ps.makeAdjustmentLayer(spec);
        const m = masks[spec.id];
        if (m) await ps.putMask(d.id, id, m.data, m.width, m.height, { left: 0, top: 0 });
        made.push(id);
      }
      if (made.length > 1) await ps.groupLayers(d, made, r.plan.group_name);
    });
    $("status").className = "status ok";
    $("status").textContent = `已生成 ${specs.length} 个调整图层（Ctrl+Z 可一次撤销）`;
  } catch (e) {
    renderError(e);
  }
}

async function makeSelection(id) {
  const r = state.result;
  const a = state.analyzed;
  if (!r || !a) return;
  try {
    const m = await bridge.fetchMask(r.session, id, a.bounds, a.docWidth, a.docHeight);
    await ps.modal("ColorDoctor 选区", async (ctx, d) => {
      await ps.putSelection(d.id, m.data, m.width, m.height, { left: 0, top: 0 });
    });
  } catch (e) { renderError(e); }
}

// ---------------------------------------------------------------- ACR
const HSL = ["红色", "橙色", "黄色", "绿色", "浅绿色", "蓝色", "紫色", "洋红"];
const ZONES = [["shadow", "阴影"], ["midtone", "中间调"], ["highlight", "高光"]];

function renderAcr(p, calibrated) {
  const hsl = p.hsl || {};
  const g = p.grading || {};
  const num = (path, v) => `<input type="number" data-acr="${path}" value="${v == null ? 0 : v}">`;
  $("acrTable").innerHTML = `
    <table><tr><td></td><td>色相</td><td>饱和度</td><td>明亮度</td></tr>
    ${HSL.map((c) => `<tr><td>${c}</td><td>${num(`hsl.${c}.hue`, (hsl[c] || {}).hue)}</td><td>${num(`hsl.${c}.sat`, (hsl[c] || {}).sat)}</td><td>${num(`hsl.${c}.lum`, (hsl[c] || {}).lum)}</td></tr>`).join("")}
    </table>
    <table><tr><td>颜色分级</td><td>色相</td><td>饱和度</td></tr>
    ${ZONES.map(([k, n]) => `<tr><td>${n}</td><td>${num(`grading.${k}.hue`, (g[k] || {}).hue)}</td><td>${num(`grading.${k}.sat`, (g[k] || {}).sat)}</td></tr>`).join("")}
    </table>`;
  $("acrBadge").className = calibrated ? "chip ok" : "chip warn";
  $("acrBadge").textContent = calibrated ? "已校准" : "估计 · 未校准";
}

function acrParamsFromTable() {
  // 表格里只有混色器和颜色分级；曲线等其他参数沿用分析结果
  const base = (state.result && state.result.acr_params) || {};
  const out = {};
  for (const [k, v] of Object.entries(base)) if (k !== "hsl" && k !== "grading") out[k] = v;
  document.querySelectorAll("[data-acr]").forEach((inp) => {
    const v = Number(inp.value);
    if (!v) return;
    const [sec, key, field] = inp.dataset.acr.split(".");
    ((out[sec] = out[sec] || {})[key] = out[sec][key] || {})[field] = v;
  });
  return out;
}

async function exportXmp() {
  try {
    const xmp = await bridge.acrPreset("ColorDoctor", acrParamsFromTable());
    const file = await uxp.storage.localFileSystem.getFileForSaving("ColorDoctor.xmp", { types: ["xmp"] });
    if (!file) return;
    await file.write(xmp);
    $("status").className = "status ok";
    $("status").textContent = "已导出 ACR 预设：放进 Camera Raw 的预设文件夹，或在 ACR「预设」面板里导入";
  } catch (e) { renderError(e); }
}

async function acrLayer() {
  // 先导出预设，再复制图层 → 智能对象 → 打开 Camera Raw 对话框，由用户套用预设并确认。
  await exportXmp();
  try {
    const merged = $("target").value === "merged";
    await ps.modal("ColorDoctor Camera Raw", async () => ps.cameraRawLayer("ColorDoctor · Camera Raw", merged), { interactive: true });
  } catch (e) { renderError(e); }
}

// ---------------------------------------------------------------- ACR 滑块校准（需要 Camera Raw 描述符，见 ps.js 的 TODO）
async function calibrate() {
  if (!ps.acrDescriptorReady()) {
    $("results").innerHTML = `<div class="blocked">ACR 滑块校准需要 Camera Raw 滤镜的 batchPlay 命令格式。这一步要在 Windows 上用 Alchemist 录制一次（见 docs/STAGE6_HANDOFF.md 的 P9），录好之前无法自动校准。</div>`;
    return;
  }
  const progress = (t) => { $("status").className = "status"; $("status").textContent = t; };
  try {
    const chart = await bridge.calibrationChart();
    const steps = await bridge.calibrationPlan();
    // 校准在新建的色卡文档里进行，不进当前文档的历史记录；完成后关闭色卡文档
    await ps.readModal(async () => {
      const { doc, layer } = await ps.createChartDocument(chart);
      const info = { ok: true, doc, layer, bounds: { left: 0, top: 0, right: chart.width, bottom: chart.height } };
      await bridge.calibrationSample(await ps.readPixels(info), { slider: "base" });
      for (let i = 0; i < steps.length; i++) {
        const st = steps[i];
        progress(`正在校准 ${i + 1}/${steps.length}：${st.channel} ${st.slider} ${st.value}`);
        const dup = await ps.duplicateLayer(layer);
        await ps.applyCameraRaw(dup, { hsl: { [st.channel]: { [st.slider]: st.value } } });
        await bridge.calibrationSample(await ps.readPixels({ ...info, layer: dup }), st);
        await ps.deleteLayer(dup);
      }
      await ps.closeDocument(doc);
    });
    const res = await bridge.calibrationFit(app.version);
    progress(`校准完成：${Object.keys(res.channels).length} 个通道，已保存到 ${res.saved}`);
    renderAcr((state.result && state.result.acr_params) || {}, true);
  } catch (e) { renderError(e); }
}

// ---------------------------------------------------------------- 启动
async function init() {
  document.querySelectorAll(".cat").forEach((b) => b.addEventListener("click", () => {
    state.cat = b.dataset.cat;
    state.sub = SUBS[state.cat][0][0];
    renderNav();
  }));
  $("target").addEventListener("change", refreshTarget);
  $("analyze").addEventListener("click", runAnalyze);
  $("generateAll").addEventListener("click", () => state.result && generate(state.result.plan.layers.map((l) => l.id)));
  $("toggleAcr").addEventListener("click", () => $("acr").classList.toggle("hidden"));
  $("acrExport").addEventListener("click", exportXmp);
  $("acrLayer").addEventListener("click", acrLayer);
  $("calibrate").addEventListener("click", calibrate);
  try {
    await bridge.health();
    state.options = await bridge.options();
  } catch (e) {
    $("svc").className = "chip bad";
    $("svc").textContent = "分析服务未启动";
  }
  renderNav();
  renderAcr({}, false);
  refreshTarget();
  // 选中别的图层、做了修改时刷新“分析对象”栏（不会自动分析）
  action.addNotificationListener(["select", "make", "set", "delete", "historyStateChanged"], () => refreshTarget());
}

init();
