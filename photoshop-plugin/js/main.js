// ColorDoctor 面板：只在用户点击时分析；只新增调整图层和蒙版，不改原图像素。
const { app, action } = require("photoshop");
const uxp = require("uxp");
const bridge = require("./bridge.js");
const ps = require("./ps.js");

const SUBS = {
  color: [["dirty", "脏色"], ["patch", "取色"], ["ratio", "比例 6:3:1"], ["style", "风格"]],
  portrait: [["skin", "皮肤均匀"], ["skinpatch", "肤色取色"]],
  reference: [["reference", "仿色"]],
};
const SUB_NAME = Object.fromEntries(Object.values(SUBS).flat());
const MODE_OF = { dirty: "dirty", patch: "patch", skinpatch: "patch", ratio: "palette", style: "palette", skin: "skin", reference: "reference" };

const state = {
  cat: "color", sub: "dirty", options: null, busy: false,
  // 每个子页各保留自己最近一次的结果：切到别的页再切回来还在，不会把脏色的结果显示在皮肤页下
  // bySub[sub] = { result, analyzed, made: Set<图层方案 id>, stale: 已提醒过的历史状态 }
  bySub: {},
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cur = () => state.bySub[state.sub] || null;

function analyzeLabel() {
  if (MODE_OF[state.sub] === "patch") return "读取选区颜色";
  return $("target").value === "merged" ? "分析合并画面" : "分析当前图层";
}

// ---------------------------------------------------------------- 分析对象栏
function refreshTarget() {
  const info = ps.inspectTarget($("target").value);
  $("targetName").textContent = info.label || "—";
  $("analyze").textContent = analyzeLabel();
  if (state.sub === "reference") refreshRefDocs();
  const st = $("status");
  if (state.busy) return info;  // 进行中：状态栏显示进度，不覆盖
  updateButtons();
  const e = cur();
  const a = e && e.analyzed;
  const doc = app.activeDocument;
  if (a && a.justMade && doc && doc.id === a.docId) {
    // 刚生成的调整图层 / 组被选中：保留成功提示，不当成“不能分析”的错误
    if (!info.ok && info.layer && a.justMade.has(info.layer.id)) {
      st.className = "status ok";
      st.textContent = `${a.justMadeText}。要再次分析，请在图层面板里选回「${a.layerName}」`;
      $("analyze").disabled = true;
      return info;
    }
  }
  if (!info.ok) {
    $("analyze").disabled = true;
    st.className = "status warn";
    st.textContent = info.reason;
    return info;
  }
  $("analyze").disabled = false;
  if (a && info.doc && a.docId === info.doc.id && a.layerId === (info.layer ? info.layer.id : null)) {
    const stale = isStale(a, info.doc);
    st.className = stale ? "status warn" : "status ok";
    st.textContent = stale
      ? `文档在分析后有改动（不一定是这个图层），结果可能过期，可以再点一次「${analyzeLabel()}」`
      : "已分析 · 下面的结果对应这个对象";
  } else if (a) {
    st.className = "status";
    st.textContent = `下面的结果来自「${a.layerName}」（${a.docName}），不是当前选中的对象`;
  } else {
    st.className = "status";
    st.textContent = "只在点击按钮时分析";
  }
  return info;
}

function isStale(a, doc) {
  // 插件自己做的操作（生成图层、生成选区）以及撤销回到这些状态，都不算改动
  return !a.okHistory.has(ps.historyStateId(doc));
}

// ---------------------------------------------------------------- 分类与控件
function renderNav() {
  document.querySelectorAll(".cat").forEach((b) => b.classList.toggle("on", b.dataset.cat === state.cat));
  // 只有一个子页时不显示子页栏
  const subs = SUBS[state.cat];
  $("subs").classList.toggle("hidden", subs.length < 2);
  $("subs").innerHTML = subs.map(([k, n]) => `<button data-sub="${k}" class="${k === state.sub ? "on" : ""}">${n}</button>`).join("");
  $("subs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    if (state.busy) return;
    state.sub = b.dataset.sub;
    renderNav();
  }));
  renderControls();
  renderResult();
  refreshTarget();
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
    c.innerHTML = `<div class="row"><label>颜色统一</label>${select("family", o.families.map((f) => ({ key: f, name: f })), "（只看比例）")}</div>
      <div class="muted">看主色 / 辅色 / 点缀色的面积是否接近 6:3:1；选一个色系时还会生成把杂色往它统一的调整图层。</div>`;
  } else if (state.sub === "skin") {
    c.innerHTML = `<div class="row"><label>抠皮肤</label><select id="seg"><option value="ai">AI 人像分割（显卡）</option><option value="color">肤色规则（不分脸和身体）</option></select></div>
      <div class="row"><label>肤色风格</label>${select("preset", o.presets)}</div>
      <div class="row"><label>部位</label><select id="part"><option value="脸部">脸部</option><option value="身体">身体</option><option value="">全部皮肤</option></select></div>`;
    const syncPart = () => {
      const color = $("seg").value === "color";
      if (color) $("part").value = "";
      $("part").disabled = color;  // 肤色规则分不出脸和身体
    };
    $("seg").addEventListener("change", syncPart);
    syncPart();
  } else if (MODE_OF[state.sub] === "patch") {
    c.innerHTML = `<div class="row"><button id="fg">读取前景色（先用吸管取色）</button></div>
      <div class="muted">或用选框 / 套索框出一块区域，再点上面的「读取选区颜色」（相当于大取样）。</div>`;
    $("fg").addEventListener("click", readForeground);
  } else if (state.sub === "reference") {
    c.innerHTML = `<div class="row"><label>参考图</label><select id="refDoc"></select></div>
      <div class="muted">参考图要先在 PS 里打开（作为另一个文档）；当前文档是“要调的图”。</div>`;
    refreshRefDocs();
  } else {
    c.innerHTML = "";
  }
}

// 参考图列表跟着 PS 里打开 / 关闭的文档实时更新，保留原来的选择；不列出当前文档自己
function refreshRefDocs() {
  const sel = $("refDoc");
  if (!sel) return;
  const keep = sel.value;
  const active = app.activeDocument;
  const docs = app.documents.filter((d) => !active || d.id !== active.id);
  sel.innerHTML = docs.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("")
    || '<option value="">（先在 PS 里打开参考图）</option>';
  if (docs.some((d) => String(d.id) === keep)) sel.value = keep;
}

function query(sub) {
  const q = {};
  const v = (id) => ($(id) ? $(id).value : "");
  if (sub === "style") q.style = v("style");
  if (sub === "ratio" && v("family")) q.family = v("family");
  if (sub === "skin") {
    q.preset = v("preset"); q.seg = v("seg");
    q.part = q.seg === "ai" ? v("part") : "";  // 肤色规则分不出脸和身体：不标部位
  }
  return q;
}

function setControlsLocked(locked) {
  // 分析进行中不能切换页面、分析对象和参数：参数和结果必须属于同一个页面
  document.querySelectorAll(".cat, #subs button, #controls select, #controls button, #controls input, #target")
    .forEach((el) => { el.disabled = locked; });
  if (!locked && $("seg") && $("part")) $("part").disabled = $("seg").value === "color";
}

// ---------------------------------------------------------------- 分析
async function runAnalyze() {
  if (state.busy) return;
  const info = refreshTarget();
  if (!info.ok) return;
  // 开始时一次取定：页面、参数、参考图。分析过程中就算界面变化，也按点击时的设置算、结果归到这个页面
  const sub = state.sub;
  const mode = MODE_OF[sub];
  if (!state.options) await loadOptions();
  const q = query(sub);
  if (sub === "style" && !q.style) {
    statusError(new Error("风格列表还没加载出来（分析服务未启动？），请先启动 ColorDoctor 再选风格。"));
    return;
  }
  let refDoc = null;
  if (mode === "reference") {
    refDoc = app.documents.find((d) => String(d.id) === ($("refDoc") || {}).value);
    if (!refDoc) { statusError(new Error("请先在 PS 里打开参考图，并在上面的「参考图」里选择它。")); return; }
    if (refDoc.id === info.doc.id) { statusError(new Error("参考图不能是当前文档：请切到要调的图，再选参考图。")); return; }
  }
  setBusy(true, mode === "patch" ? "正在读取…" : "正在分析…");
  // 在读像素之前记下历史状态：分析过程中如果有改动，会被正确判为“可能过期”
  const historyBefore = ps.historyStateId(info.doc);
  try {
    const px = await ps.readModal(() => (mode === "patch" ? ps.readSelectionPixels(info) : ps.readPixels(info)));
    if (mode === "reference") {
      const refInfo = { ok: true, doc: refDoc, layer: null, bounds: { left: 0, top: 0, right: refDoc.width, bottom: refDoc.height } };
      const refPx = await ps.readModal(() => ps.readPixels(refInfo));
      q.ref = await bridge.uploadReference(refPx);
    }
    const sb = px.sourceBounds;
    const res = await bridge.analyze(px, mode, q, {
      docWidth: sb.right - sb.left, docHeight: sb.bottom - sb.top, layerKind: info.kind, layerName: info.label,
    });
    res.sub = sub;
    state.bySub[sub] = {
      result: res, made: new Set(),
      analyzed: {
        docId: info.doc.id, docName: info.doc.name, layerId: info.layer ? info.layer.id : null,
        layerName: info.layer ? info.label : "合并可见图层", okHistory: new Set([historyBefore]), bounds: sb,
        docWidth: info.doc.width, docHeight: info.doc.height, depth: String(info.doc.bitsPerChannel || ""),
        refName: refDoc ? refDoc.name : null,
      },
    };
  } catch (e) {
    // 失败：这个页面不再保留旧结果，避免用旧结果生成图层
    delete state.bySub[sub];
    setBusy(false);
    if (state.sub === sub) renderError(e);
    refreshTarget();
    return;
  }
  setBusy(false);
  if (state.sub === sub) renderResult();
  refreshTarget();
}

async function readForeground() {
  if (state.busy) return;
  const c = app.foregroundColor.rgb;
  const px = { data: new Uint8Array([c.red, c.green, c.blue]), width: 1, height: 1, components: 3 };
  const sub = state.sub;
  setBusy(true, "正在读取前景色…");
  try {
    const res = await bridge.analyze(px, "patch", {}, { layerKind: "pixel" });
    res.sub = sub;
    res.fromForeground = true;
    state.bySub[sub] = { result: res, analyzed: null, made: new Set() };
    setBusy(false);
    if (state.sub === sub) renderResult();
    $("status").className = "status ok";
    $("status").textContent = `已读取前景色 ${res.values ? res.values.hex : ""}`;
  } catch (e) {
    setBusy(false);
    statusError(e);
  }
}

function hasLayers(e) {
  return !!(e && e.analyzed && e.result && e.result.plan && e.result.plan.layers.length);
}

function hasAcrSource() {
  const e = cur();
  const fromResult = e && e.result && e.result.mode !== "patch";
  const edited = [...document.querySelectorAll("[data-acr]")].some((i) => i.dataset.set === "1" && Number(i.value));
  return !!(fromResult || edited);
}

function updateButtons() {
  const busy = state.busy;
  const e = cur();
  const offDoc = e && e.analyzed && (!app.activeDocument || app.activeDocument.id !== e.analyzed.docId);
  $("generateAll").disabled = busy || !hasLayers(e) || !!offDoc;
  if (e && hasLayers(e)) {
    const left = e.result.plan.layers.filter((l) => !e.made.has(l.id)).length;
    $("generateAll").textContent = left === 0 ? "全部已生成" : (e.made.size ? `生成其余 ${left} 层（带蒙版）` : "一键生成全部（带蒙版）");
    if (left === 0) $("generateAll").disabled = true;
  } else {
    $("generateAll").textContent = "一键生成全部（带蒙版）";
  }
  document.querySelectorAll("[data-make],[data-sel]").forEach((b) => { b.disabled = busy || !!offDoc; });
  const acrOk = !busy && hasAcrSource();
  $("acrExport").disabled = !acrOk;
  $("acrLayer").disabled = !acrOk;
  const calReady = ps.acrDescriptorReady();
  $("calibrate").disabled = busy || !calReady;
  $("calibrate").title = calReady ? "" : "这个版本还不能自动校准（需要先录制 Camera Raw 的命令格式），ACR 数值是经验估计";
}

function setBusy(busy, text) {
  state.busy = busy;
  $("analyze").disabled = busy;
  setControlsLocked(busy);
  updateButtons();
  if (busy) { $("status").className = "status"; $("status").textContent = text; }
}

function renderError(e) {
  const blocked = e && e.blocked;
  $("results").innerHTML = `<div class="blocked">${blocked ? "<b>无法分析</b><br>" : ""}${esc(e.message || e)}</div>`;
  renderAcr({}, false);
  updateButtons();
}

// ---------------------------------------------------------------- 结果
function resultHeader(e) {
  const a = e.analyzed;
  if (!a) return "";
  const ref = a.refName ? ` · 参考图「${esc(a.refName)}」` : "";
  return `<div class="muted">结果：${esc(SUB_NAME[e.result.sub] || "")} · 「${esc(a.layerName)}」 · ${esc(a.docName)}${ref}</div>`;
}

function renderResult() {
  const e = cur();
  if (!e || !e.result) {
    $("results").innerHTML = `<p class="hint">选好要分析的对象，点「${analyzeLabel()}」。</p>`;
    renderAcr({}, false);
    updateButtons();
    return;
  }
  const r = e.result;
  if (r.mode === "patch") return renderPatch(e);
  const cards = [resultHeader(e)];
  (r.notes || []).forEach((n) => cards.push(`<div class="muted">${esc(n)}</div>`));
  const layers = (r.plan && r.plan.layers) || [];
  if (layers.length) cards.push('<div class="strong">可生成的调整图层</div>');
  layers.forEach((ly, i) => {
    const done = e.made.has(ly.id);
    cards.push(`
    <div class="card">
      <div class="title"><span>${i + 1}. ${esc(ly.name)}</span><span class="muted">${done ? "已生成 ✓" : (ly.estimated ? "估计值" : "")}</span></div>
      <div class="line">${esc(ly.reason)}</div>
      <div class="row">
        <button class="small make" data-make="${esc(ly.id)}">${done ? "再生成一次" : `生成此层${ly.has_mask ? "（带蒙版）" : ""}`}</button>
        ${ly.has_mask ? `<button class="small" data-sel="${esc(ly.id)}" title="只在当前文档里建立这个问题区域的选区，不新建图层">只建选区</button>` : ""}
      </div>
    </div>`);
  });
  if ((r.findings || []).length) cards.push('<div class="strong">诊断</div>');
  (r.findings || []).forEach((f) => cards.push(`
    <div class="card">
      <div class="title"><span>${esc(f.problem)}</span><span class="sev-${esc(f.severity)}">${esc(f.severity)}</span></div>
      ${(f.evidence || []).slice(0, 3).map((x) => `<div class="line">证据：${esc(x)}</div>`).join("")}
      ${(f.causes || []).slice(0, 2).map((x) => `<div class="line">可能原因：${esc(x)}</div>`).join("")}
      ${(f.adjustments || []).slice(0, 3).map((x) => `<div class="line">调整：${esc(x.tool)} · ${esc(x.instruction)}</div>`).join("")}
      <div class="muted">置信度 ${Math.round((f.confidence || 0) * 100)}%${(f.notes || []).length ? " · 注意：" + esc(f.notes[0]) : ""}</div>
    </div>`));
  if (!layers.length && !(r.findings || []).length) cards.push('<p class="hint">没有发现需要处理的问题。</p>');
  else if (!layers.length) {
    cards.push('<p class="hint">这些建议没有对应的调整图层：可以按文字手动调整，或打开下方「ACR 参数」导出预设 / 生成 Camera Raw 图层。</p>');
  }
  $("results").innerHTML = cards.join("");
  $("results").querySelectorAll("[data-make]").forEach((b) => b.addEventListener("click", () => generate([b.dataset.make], true)));
  $("results").querySelectorAll("[data-sel]").forEach((b) => b.addEventListener("click", () => makeSelection(b.dataset.sel)));
  renderAcr(r.acr_params || {}, r.acr_calibrated);
  updateButtons();
}

function renderPatch(e) {
  const r = e.result;
  const v = r.values;
  const qr = r.quick_rule;
  const qrOk = qr && qr.a_ok && qr.b_ok && qr.diff_ok;
  const skin = r.sub === "skinpatch";  // 肤色快速规则只在“肤色取色”页显示
  const src = r.fromForeground ? "前景色" : (e.analyzed ? `「${esc(e.analyzed.layerName)}」的选区` : "选区");
  $("results").innerHTML = `
    <div class="muted">取色：${src}</div>
    <div class="swatch" style="background:${esc(v.hex)}"></div>
    <div class="values">
      <div>L* ${v.lab[0].toFixed(1)}</div><div>a* ${v.lab[1].toFixed(1)}</div><div>b* ${v.lab[2].toFixed(1)}</div>
      <div>C* ${v.lch[1].toFixed(1)}</div><div>h° ${v.lch[2].toFixed(1)}</div><div>${esc(v.hue_name)}</div>
    </div>
    <div class="card"><div class="line">RGB ${v.rgb.join(" ")} · ${esc(v.hex)}</div>
      <div class="line">HSB ${v.hsb.map((x) => Math.round(x)).join(" / ")}（= PS 拾色器）</div>
      <div class="muted">取样 ${r.pixels} 像素 · Lab 为 D50（与 PS 信息面板一致）</div></div>
    ${skin && qr ? `<div class="card"><div class="title"><span>肤色快速规则</span><span>${qrOk ? "通过" : "不通过"}</span></div>
      <div class="line">a* ${qr.a.toFixed(1)}、b* ${qr.b.toFixed(1)}，|a−b| = ${Math.abs(qr.a - qr.b).toFixed(1)}</div></div>` : ""}
    ${r.acr_channels.length ? `<div class="card"><div class="title"><span>归哪个 ACR 混色器通道管</span></div>
      ${r.acr_channels.map((c) => `<div class="line">${esc(c.channel)} ${(c.weight * 100).toFixed(0)}%</div>`).join("")}</div>` : ""}
    ${r.purity != null ? `<div class="card"><div class="line">纯度：${(r.purity * 100).toFixed(0)}% 的像素与平均色相差在 30° 以内</div></div>` : ""}`;
  renderAcr({}, false);  // 取色没有 ACR 调整参数
  updateButtons();
}

// ---------------------------------------------------------------- 生成图层
function statusError(e) {
  // 生成失败时只在状态栏提示，保留分析结果（不必重新分析）
  $("status").className = e && e.cancelled ? "status" : "status warn";
  $("status").textContent = (e && e.message) || String(e);
}

function checkStillValid(a) {
  if (!ps.findDocument(a.docId)) return "刚才分析的文档已经关闭，请重新分析。";
  const doc = app.activeDocument;
  if (!doc || doc.id !== a.docId) return `当前文档不是刚才分析的「${a.docName}」，请切回去或重新分析。`;
  if (doc.width !== a.docWidth || doc.height !== a.docHeight) return "文档尺寸在分析后改变了（裁剪 / 画布大小 / 图像大小），请重新分析。";
  if (a.layerId != null && !ps.findLayer(doc, a.layerId)) return `刚才分析的图层「${a.layerName}」已被删除，请重新分析。`;
  if (isStale(a, doc)) return "STALE";
  return null;
}

function markMade(e, ids, text) {
  const a = e.analyzed;
  ids.forEach((id) => e.made.add(id));
  const doc = app.activeDocument;
  a.okHistory.add(ps.historyStateId(doc));  // 插件自己新建的图层不算“文档改动”
  a.justMade = new Set(((doc && doc.activeLayers) || []).map((l) => l.id));
  a.justMadeText = text;
}

async function generate(ids, single) {
  const e = cur();
  if (!e || !e.analyzed || state.busy) return;
  const r = e.result;
  const a = e.analyzed;
  const problem = checkStillValid(a);
  if (problem === "STALE") {
    // 文档有改动：提醒一次；用户再点一次就用旧结果继续
    if (e.stale !== ps.historyStateId(app.activeDocument)) {
      e.stale = ps.historyStateId(app.activeDocument);
      statusError(new Error("文档在分析后有改动，结果可能已过期。建议重新分析；如果确定要用旧结果，再点一次生成。"));
      return;
    }
  } else if (problem) {
    statusError(new Error(problem));
    return;
  }
  const docMode = String((app.activeDocument && app.activeDocument.mode) || "");
  if (docMode && !/rgb/i.test(docMode)) {
    // 色彩平衡 / 自然饱和度等在 Lab、灰度里不可用，CMYK 里颜色范围和数值含义也不同
    statusError(new Error("校正图层按 RGB 设计：请先把文档转换为 RGB（图像 → 模式 → RGB 颜色），或在 RGB 副本上生成。"));
    return;
  }
  // 撤销过的图层不算“已生成”
  for (const id of [...e.made]) if (!e.madeLayers || !e.madeLayers[id] || !e.madeLayers[id].every((lid) => ps.findLayer(app.activeDocument, lid))) e.made.delete(id);
  // 一键生成只补还没生成的；单层按钮总是按用户的意思生成
  const todo = single ? ids : ids.filter((id) => !e.made.has(id));
  if (!todo.length) { statusError(new Error("全部已经生成过了。如需重新生成，请点各层的「再生成一次」。")); updateButtons(); return; }
  const specs = r.plan.layers.filter((l) => todo.includes(l.id));
  setBusy(true, "正在生成调整图层…");
  const madeIds = {};
  try {
    // 蒙版是整幅文档大小（大图每张几十 MB）：逐层取、用完即弃，不一次全部放在内存里。
    // 中途任何一步失败，modal 都不提交历史记录，已建的图层整体撤回。
    await ps.modal(specs.length > 1 ? "ColorDoctor 一键校正" : "ColorDoctor 校正", async (ctx, d) => {
      if (a.layerId != null) await ps.selectLayer(a.layerId);  // 分析的是单个图层：建在它上方
      else await ps.selectTopLayer(d);  // 分析的是合并画面：建在最顶层
      // 当前选区不能变成调整图层的蒙版：暂时取消，生成后恢复
      await ps.withoutSelection(d, async () => {
        const made = [];
        for (let i = 0; i < specs.length; i++) {
          const spec = specs[i];
          $("status").textContent = `正在生成 ${i + 1}/${specs.length}：${spec.name}`;
          let m = spec.has_mask ? await bridge.fetchMask(r.session, spec.id, a.bounds, a.docWidth, a.docHeight) : null;
          const id = await ps.makeAdjustmentLayer(spec);
          if (m) await ps.putMask(d.id, id, m.data, m.width, m.height, { left: 0, top: 0 }, a.depth);
          m = null;
          made.push(id);
          madeIds[spec.id] = [id];
        }
        if (made.length > 1) await ps.groupLayers(d, made, r.plan.group_name);
      });
    }, { docId: a.docId, docSize: [a.docWidth, a.docHeight] });
    e.madeLayers = { ...(e.madeLayers || {}), ...madeIds };
    setBusy(false);
    markMade(e, specs.map((s) => s.id), `已生成 ${specs.length} 个调整图层${specs.length > 1 ? `，放在「${r.plan.group_name}」组里` : ""}（Ctrl+Z 可一次撤销）`);
    if (state.sub === r.sub) renderResult();
    $("status").className = "status ok";
    $("status").textContent = a.justMadeText;
  } catch (err) {
    setBusy(false);
    statusError(err);
  }
}

async function makeSelection(id) {
  const e = cur();
  if (!e || !e.analyzed || state.busy) return;
  const r = e.result;
  const a = e.analyzed;
  const problem = checkStillValid(a);
  if (problem && problem !== "STALE") { statusError(new Error(problem)); return; }
  setBusy(true, "正在建立选区…");
  try {
    const m = await bridge.fetchMask(r.session, id, a.bounds, a.docWidth, a.docHeight);
    await ps.modal("ColorDoctor 选区", async (ctx, d) => {
      await ps.putSelection(d.id, m.data, m.width, m.height, { left: 0, top: 0 }, a.depth);
    }, { docId: a.docId, docSize: [a.docWidth, a.docHeight] });
    setBusy(false);
    a.okHistory.add(ps.historyStateId(app.activeDocument));  // 建选区不算“文档改动”
    $("status").className = "status ok";
    $("status").textContent = "已建立选区（蚂蚁线）：可以用它手动加调整图层或修饰";
  } catch (err) {
    setBusy(false);
    statusError(err);
  }
}

// ---------------------------------------------------------------- ACR
const HSL = ["红色", "橙色", "黄色", "绿色", "浅绿色", "蓝色", "紫色", "洋红"];
const ZONES = [["shadow", "阴影"], ["midtone", "中间调"], ["highlight", "高光"]];

function renderAcr(p, calibrated) {
  const hsl = p.hsl || {};
  const g = p.grading || {};
  // data-set：分析给出过这个值，或用户改过它 —— 这时 0 也要导出（例如颜色分级色相 0 = 红）
  const num = (path, v, lo, hi) => `<input type="number" min="${lo}" max="${hi}" data-acr="${path}" value="${v == null ? 0 : v}"${v == null ? "" : ' data-set="1"'}>`;
  const curve = p.curve && p.curve.composite;
  $("acrTable").innerHTML = `
    <table><tr><td>混色器</td><td>色相</td><td>饱和度</td><td>明亮度</td></tr>
    ${HSL.map((c) => `<tr><td>${c}</td><td>${num(`hsl.${c}.hue`, (hsl[c] || {}).hue, -100, 100)}</td><td>${num(`hsl.${c}.sat`, (hsl[c] || {}).sat, -100, 100)}</td><td>${num(`hsl.${c}.lum`, (hsl[c] || {}).lum, -100, 100)}</td></tr>`).join("")}
    </table>
    <table><tr><td>颜色分级</td><td>色相</td><td>饱和度</td></tr>
    ${ZONES.map(([k, n]) => `<tr><td>${n}</td><td>${num(`grading.${k}.hue`, (g[k] || {}).hue, 0, 359)}</td><td>${num(`grading.${k}.sat`, (g[k] || {}).sat, 0, 100)}</td></tr>`).join("")}
    </table>
    ${curve ? `<div class="muted">另含 RGB 曲线（仿色）：${curve.map(([x, y]) => `${x}→${y}`).join("，")}（导出时一并写入）</div>` : ""}`;
  $("acrTable").querySelectorAll("[data-acr]").forEach((inp) => inp.addEventListener("input", () => {
    inp.dataset.set = "1";
    const v = Number(inp.value);
    inp.classList.toggle("bad", Number.isFinite(v) && (v < Number(inp.min) || v > Number(inp.max)));
    updateButtons();
  }));
  $("acrBadge").className = calibrated ? "chip ok" : "chip warn";
  $("acrBadge").textContent = calibrated ? "已校准" : "估计 · 未校准";
}

function acrParamsFromTable() {
  // 表格里只有混色器和颜色分级；曲线等其他参数沿用当前页面的分析结果
  const e = cur();
  const base = (e && e.result && e.result.mode !== "patch" && e.result.acr_params) || {};
  const out = {};
  for (const [k, v] of Object.entries(base)) if (k !== "hsl" && k !== "grading") out[k] = v;
  document.querySelectorAll("[data-acr]").forEach((inp) => {
    const v = Number(inp.value);
    if (!Number.isFinite(v) || (!v && inp.dataset.set !== "1")) return;
    const [sec, key, field] = inp.dataset.acr.split(".");
    ((out[sec] = out[sec] || {})[key] = out[sec][key] || {})[field] = v;
  });
  return out;
}

function presetName() {
  // 多次导出时在 ACR 预设列表里分得清：模式 + 图层 + 日期时间
  const e = cur();
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const when = `${d.getMonth() + 1}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
  const what = e && e.result ? (SUB_NAME[e.result.sub] || "") : "手动";
  const layer = e && e.analyzed ? e.analyzed.layerName : "";
  return `ColorDoctor ${what}${layer ? " " + layer : ""} ${when}`.replace(/[\\/:*?"<>|]/g, "_");
}

// 返回保存的文件名；用户取消保存返回 null
async function exportXmp() {
  try {
    const name = presetName();
    const xmp = await bridge.acrPreset(name, acrParamsFromTable());
    const file = await uxp.storage.localFileSystem.getFileForSaving(`${name}.xmp`, { types: ["xmp"] });
    if (!file) return null;
    await file.write(xmp);
    $("status").className = "status ok";
    $("status").textContent = `已导出 ACR 预设「${name}」：在 Camera Raw 的「预设」面板 → ⋯ → 导入配置文件和预设。`
      + "数值是按当前画面算的调整量，如果原来的混色器 / 颜色分级已经调过，请在原数值上加减";
    return name;
  } catch (err) {
    statusError(err);  // 只提示，不清掉分析结果
    return null;
  }
}

async function acrLayer() {
  // 先保存预设，再复制图层 → 智能对象 → 打开 Camera Raw 对话框，由用户套用预设并确认。
  if (state.busy) return;
  const merged = $("target").value === "merged";
  const doc = app.activeDocument;
  const e = cur();
  const a = e && e.analyzed;
  let layerId = null;
  let layerName = "合并可见图层";
  if (!merged) {
    // 优先用这个页面分析过的图层（生成校正后当前选中的往往是调整图层 / 图层组）
    if (a && doc && a.docId === doc.id && a.layerId != null && ps.findLayer(doc, a.layerId)) {
      layerId = a.layerId;
      layerName = a.layerName;
    } else {
      const info = refreshTarget();
      if (!info.ok) { statusError(new Error(info.reason)); return; }
      layerId = info.layer.id;
      layerName = info.label;
    }
  }
  $("status").className = "status";
  $("status").textContent = "第 1 步：保存 ACR 预设文件（打开 Camera Raw 后要导入它）";
  const saved = await exportXmp();
  if (!saved) {
    if (!$("status").classList.contains("warn")) {
      $("status").className = "status";
      $("status").textContent = "没有保存预设，已取消（没有打开 Camera Raw）";
    }
    return;
  }
  setBusy(true, `第 2 步：复制「${layerName}」并打开 Camera Raw。在「预设」面板 → ⋯ → 导入配置文件和预设，选「${saved}.xmp」，再点确定`);
  try {
    await ps.modal("ColorDoctor Camera Raw", async (ctx, d) => ps.cameraRawLayer("ColorDoctor · Camera Raw", { merged, layerId, doc: d }),
      { interactive: true });
    setBusy(false);
    if (a && doc && a.docId === doc.id) {
      a.okHistory.add(ps.historyStateId(doc));  // 新建图层不算“文档改动”
      a.justMade = new Set((doc.activeLayers || []).map((l) => l.id));
      a.justMadeText = "已新建「ColorDoctor · Camera Raw」智能对象（双击图层里的 Camera Raw 滤镜可再改）";
    }
    $("status").className = "status ok";
    $("status").textContent = "已新建「ColorDoctor · Camera Raw」智能对象（双击图层里的 Camera Raw 滤镜可再改）";
  } catch (err) {
    setBusy(false);
    statusError(err);
  }
}

// ---------------------------------------------------------------- ACR 滑块校准（需要 Camera Raw 描述符，见 ps.js 的 TODO）
async function calibrate() {
  if (state.busy || !ps.acrDescriptorReady()) return;  // 按钮在不可用时已禁用并有说明
  const progress = (t) => { $("status").className = "status"; $("status").textContent = t; };
  setBusy(true, "正在准备校准…");  // 防止校准进行中再次点击，两轮样本混在一起
  try {
    const chart = await bridge.calibrationChart();
    const steps = await bridge.calibrationPlan();
    // 校准在新建的色卡文档里进行，不进当前文档的历史记录；完成后关闭色卡文档
    await ps.readModal(async () => {
      const { doc, layer } = await ps.createChartDocument(chart);
      try {
        const info = { ok: true, doc, layer, bounds: { left: 0, top: 0, right: chart.width, bottom: chart.height } };
        // 基准也要经过一次 Camera Raw（所有滑块为 0）：ACR 本身的默认处理不能算进滑块的效果里
        const base = await ps.duplicateLayer(layer);
        await ps.applyCameraRaw(base, {});
        await bridge.calibrationSample(await ps.readPixels({ ...info, layer: base }), { slider: "base" });
        await ps.deleteLayer(base);
        for (let i = 0; i < steps.length; i++) {
          const st = steps[i];
          progress(`正在校准 ${i + 1}/${steps.length}：${st.channel} ${st.slider} ${st.value}`);
          const dup = await ps.duplicateLayer(layer);
          await ps.applyCameraRaw(dup, { hsl: { [st.channel]: { [st.slider]: st.value } } });
          await bridge.calibrationSample(await ps.readPixels({ ...info, layer: dup }), st);
          await ps.deleteLayer(dup);
        }
      } finally {
        await ps.closeDocument(doc);  // 中途失败也关闭色卡文档
      }
    });
    // ACR 版本拿不到，记录 Photoshop 版本作参考（ACR 随 PS 更新）
    const res = await bridge.calibrationFit(`Photoshop ${app.version}`);
    setBusy(false);
    // 已有结果里的数值是按旧换算算的：提示重新分析，不直接把徽标改成“已校准”
    progress(`校准完成：${Object.keys(res.channels).length} 个通道。请重新分析，ACR 数值才会按校准结果计算`);
  } catch (err) {
    setBusy(false);
    statusError(err);
  }
}

// ---------------------------------------------------------------- 启动
// 服务可能在面板打开之后才启动：每次分析前没有选项就再试一次
async function loadOptions() {
  try {
    await bridge.health();
    state.options = await bridge.options();
    $("svc").className = "chip ok";
    $("svc").textContent = "本机分析 · 不上传";
    renderControls();
  } catch (e) {
    $("svc").className = "chip bad";
    $("svc").textContent = "分析服务未启动";
  }
}

async function init() {
  document.querySelectorAll(".cat").forEach((b) => b.addEventListener("click", () => {
    if (state.busy) return;
    state.cat = b.dataset.cat;
    state.sub = SUBS[state.cat][0][0];
    renderNav();
  }));
  $("target").addEventListener("change", refreshTarget);
  $("analyze").addEventListener("click", runAnalyze);
  $("generateAll").addEventListener("click", () => {
    const e = cur();
    if (e && e.result && e.result.plan) generate(e.result.plan.layers.map((l) => l.id), false);
  });
  $("toggleAcr").addEventListener("click", () => $("acr").classList.toggle("hidden"));
  $("acrExport").addEventListener("click", exportXmp);
  $("acrLayer").addEventListener("click", acrLayer);
  $("calibrate").addEventListener("click", calibrate);
  await loadOptions();
  renderNav();
  // 选中别的图层、做了修改、打开 / 关闭文档时刷新“分析对象”栏和参考图列表（不会自动分析）
  // 画笔涂抹时这些事件非常频繁：合并成每 200 ms 最多刷新一次
  let timer = null;
  action.addNotificationListener(["select", "make", "set", "delete", "historyStateChanged", "open", "close"], () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; refreshTarget(); }, 200);
  });
}

init();
