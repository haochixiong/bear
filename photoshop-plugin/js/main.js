// ColorDoctor 面板：只在用户点击时分析；只新增调整图层和蒙版，不改原图像素。
const { app, action } = require("photoshop");
const uxp = require("uxp");
const bridge = require("./bridge.js");
const ps = require("./ps.js");

const SUBS = {
  color: [["dirty", "脏色"], ["patch", "取色"], ["ratio", "比例 6:3:1"], ["style", "影调与颜色"]],
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
  refFile: null,  // 直接选的参考图文件 {name, data}
};

// 用户的选择（页面、分析对象、风格、肤色风格、抠皮肤方式、部位、统一色）在面板重开 / PS 重启后保留
const PREF_KEY = "colordoctor.prefs";
const prefs = (() => {
  try {
    const v = JSON.parse(localStorage.getItem(PREF_KEY) || "{}");
    return v && typeof v === "object" && !Array.isArray(v) ? v : {};  // 存坏了（数字、数组）就当没有
  } catch (_) { return {}; }
})();
const hasOwn = (obj, k) => typeof k === "string" && Object.prototype.hasOwnProperty.call(obj, k);
function savePref(k, v) {
  prefs[k] = v;
  try { localStorage.setItem(PREF_KEY, JSON.stringify(prefs)); } catch (_) { /* 存不了就算了 */ }
}
// 控件建好后：恢复上次的值，并在改动时记下
function remember(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const opts = Array.from(el.querySelectorAll("option"));
  if (prefs[id] != null && opts.some((o) => o.value === prefs[id])) el.value = prefs[id];
  el.addEventListener("change", () => savePref(id, el.value));
}

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cur = () => state.bySub[state.sub] || null;

function analyzeLabel() {
  if (MODE_OF[state.sub] === "patch") return "读取选区颜色";
  const t = $("target").value;
  return t === "merged" ? "分析合并画面" : (t === "stack" ? "分析到这一层的画面" : "分析当前图层");
}

// ---------------------------------------------------------------- 分析对象栏
function refreshTarget() {
  const info = ps.inspectTarget($("target").value);
  state.lastInfo = info;
  $("targetName").textContent = (info.label || "—") + ((info.notes || []).length ? `（${info.notes.join("；")}）` : "");
  $("analyze").textContent = analyzeLabel();
  if (state.sub === "reference") refreshRefDocs();
  toneHint();
  const st = $("status");
  if (state.busy) return info;  // 进行中：状态栏显示进度，不覆盖
  if (pruneMade(cur())) renderResult();  // 撤销了生成的图层：卡片上的“已生成 ✓”和按钮跟着恢复
  updateButtons();
  const e = cur();
  const a = e && e.analyzed;
  const doc = app.activeDocument;
  if (a && a.justMade && doc && doc.id === a.docId) {
    // 刚生成的调整图层 / 组被选中：保留成功提示，不当成“不能分析”的错误
    const sel = (doc.activeLayers || [])[0];
    if (sel && a.justMade.has(sel.id)) {
      st.className = "status ok";
      st.textContent = info.ok && info.stack
        ? `${a.justMadeText}。现在点「${analyzeLabel()}」会连同新图层一起复查效果`
        : `${a.justMadeText}。要再次分析，请在图层面板里选回「${a.layerName}」`;
      $("analyze").disabled = !info.ok;
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
  const sameObject = a && info.doc && a.docId === info.doc.id && a.layerId === (info.layer ? info.layer.id : null);
  if (sameObject && a.target === $("target").value) {
    const stale = isStale(a, info.doc);
    const changed = settingsChanged(e);
    st.className = stale || changed ? "status warn" : "status ok";
    st.textContent = changed
      ? `设置改过了：下面的结果仍是按“${a.settingsText}”算的。要按新设置，请再点「${analyzeLabel()}」`
      : (stale ? `文档在分析后有改动（不一定是这个图层），结果可能过期，可以再点一次「${analyzeLabel()}」`
        : "已分析 · 下面的结果对应这个对象和设置");
  } else if (sameObject) {
    st.className = "status warn";
    st.textContent = `下面的结果是按「${TARGET_NAME[a.target] || a.target}」分析的，现在选的是「${TARGET_NAME[$("target").value] || ""}」`;
  } else if (a) {
    st.className = "status";
    st.textContent = `下面的结果来自「${a.layerName}」（${a.docName}），不是当前选中的对象`;
  } else {
    st.className = "status";
    st.textContent = "只在点击按钮时分析";
  }
  return info;
}

const TARGET_NAME = { stack: "当前图层及以下的画面", layer: "只看当前图层", merged: "合并可见图层" };

// 结果是按哪些设置算的：分析时记下，设置改了就提示（不能拿旧设置的结果当新设置的用）
function settingsKey(sub) { return JSON.stringify(query(sub)); }
function settingsChanged(e) {
  return !!(e && e.analyzed && e.analyzed.settings != null && e.result && e.result.sub === state.sub
    && e.analyzed.settings !== settingsKey(state.sub));
}
function optName(list, key, fallback) {
  const it = (list || []).find((x) => x.key === key);
  return it ? it.name : (key || fallback);
}
function settingsText(sub, q) {
  const o = state.options || {};
  if (sub === "style") {
    const tone = q.tone_style ? `影调风格「${optName(o.tone_styles, q.tone_style)}」` : (q.tone_target ? `目标影调「${q.tone_target}」` : "不改影调");
    return `${tone} · ${q.style ? `颜色「${optName(o.styles, q.style)}」` : "不改颜色"}`;
  }
  if (sub === "ratio") return q.family ? `统一色「${q.family}」` : "只看比例";
  if (sub === "skin") return `肤色风格「${optName(o.presets, q.preset, "默认")}」 · ${q.part || "全部皮肤"} · ${q.seg === "color" ? "肤色规则" : "AI 分割"}`;
  return "";
}

// 生成过的图层被撤销 / 删掉：不再算“已生成”。有变化时返回 true
function pruneMade(e) {
  if (!e || !e.analyzed || !e.made || !e.made.size) return false;
  const doc = app.activeDocument;
  if (!doc || doc.id !== e.analyzed.docId) return false;
  let changed = false;
  for (const id of [...e.made]) {
    const ids = e.madeLayers && e.madeLayers[id];
    if (!ids || !ids.every((lid) => ps.findLayer(doc, lid))) { e.made.delete(id); changed = true; }
  }
  return changed;
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
    savePref("cat", state.cat);
    savePref("sub", state.sub);
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

// 影调风格 = 预制的目标影调（只管明暗和反差）；颜色模板只管颜色。两者互不绑定：港风可以配青橙，也可以配黑金
function toneStyleSelect(o) {
  const styles = (o.tone_styles || []).map((t) =>
    `<option value="s:${esc(t.key)}">${esc(t.name)} · ${esc(t.target)}</option>`).join("");
  const tones = (o.tones || []).map((t) => `<option value="t:${esc(t.name)}">${esc(t.name)}</option>`).join("");
  return `<select id="toneStyle"><option value="">（不改影调，只看）</option>
    <optgroup label="影调风格">${styles}</optgroup><optgroup label="直接选影调">${tones}</optgroup></select>`;
}

// 影调风格的说明：不熟悉“日系”“港风”指什么的人看这里；分析过后再说这张图要怎么改
function toneHint() {
  const el = $("toneHint");
  if (!el) return;
  if (!state.options) { el.textContent = "影调风格列表还没加载出来（本机分析服务未启动）。启动 ColorDoctor 后会自动出现。"; return; }
  const o = state.options;
  const v = $("toneStyle").value;
  let text;
  if (v.startsWith("s:")) {
    const t = (o.tone_styles || []).find((x) => x.key === v.slice(2));
    text = t ? `「${t.name}」= 目标 ${t.target}${t.also.length ? `（也可以是${t.also.join("、")}，往离现在最近的走）` : ""}：${t.look}。常用于${t.scene}。` : "";
  } else if (v.startsWith("t:")) {
    const t = (o.tones || []).find((x) => x.name === v.slice(2));
    text = t ? `目标 ${t.name}：${t.desc}。` : "";
  } else {
    text = "不改影调：只说明这张图现在是九种影调里的哪一种。想往某种影调走，在上面选一个影调风格。";
  }
  // 只有结果属于现在选中的文档 / 图层时，才说“这张图要怎么改”
  const e = state.bySub.style;
  const a = e && e.analyzed;
  const info = state.lastInfo;
  const tone = e && e.result && e.result.tone;
  const mine = tone && a && info && info.doc && a.docId === info.doc.id && a.target === $("target").value
    && a.layerId === (info.layer ? info.layer.id : null);
  if (mine && v.startsWith("s:")) {
    const s = (tone.suggestions || []).find((x) => x.key === v.slice(2));
    if (s) text += ` 这张图现在是${tone.name}：${s.change === "已经符合" ? "已经符合" : `要${s.change}（到${s.target}）`}。`;
  } else if (mine && v.startsWith("t:") && tone.paths && tone.paths[v.slice(2)]) {
    text += ` 这张图现在是${tone.name}：${tone.paths[v.slice(2)] === "已经是这种影调" ? "已经是这种影调" : `要${tone.paths[v.slice(2)]}`}。`;
  }
  el.textContent = text;
}

function colorHint() {
  const el = $("colorHint");
  if (!el) return;
  if (!state.options) { el.textContent = "颜色模板列表还没加载出来（本机分析服务未启动）。"; return; }
  const o = state.options || {};
  const st = (o.styles || []).find((x) => x.key === $("style").value);
  const names = Object.fromEntries((o.tone_styles || []).map((t) => [t.key, t.name]));
  const pair = st ? (st.tone_pair || []).map((k) => names[k]).filter(Boolean) : [];
  el.textContent = st ? `${st.description || ""}${pair.length ? ` 常配影调风格：${pair.join("、")}（只是常见搭配，可以随意换）。` : ""}`
    : "不改颜色：只看影调和整体色彩（缺色、6:3:1）。";
}

function renderControls() {
  const o = state.options || { styles: [], presets: [], families: [] };
  const c = $("controls");
  if (state.sub === "style") {
    const groups = {};
    o.styles.forEach((s) => (groups[s.category] = groups[s.category] || []).push(s));
    c.innerHTML = `<div class="row"><label>影调风格</label>${toneStyleSelect(o)}</div>
      <div class="muted" id="toneHint"></div>
      <div class="row"><label>颜色</label><select id="style"><option value="">（不改颜色）</option>${Object.entries(groups).map(([g, ss]) =>
      `<optgroup label="${esc(g)}">${ss.map((s) => `<option value="${esc(s.key)}">${esc(s.name)}</option>`).join("")}</optgroup>`).join("")}</select></div>
      <div class="muted" id="colorHint"></div>
      <div class="muted">影调和颜色分开选：影调只改明暗和反差，颜色只改颜色。想照着参考图调颜色，用「仿色」页。</div>`;
    remember("toneStyle");
    remember("style");
    $("toneStyle").addEventListener("change", toneHint);
    $("style").addEventListener("change", colorHint);
    if (!state.options) { $("toneStyle").disabled = true; $("style").disabled = true; }
    toneHint();
    colorHint();
  } else if (state.sub === "ratio") {
    c.innerHTML = `<div class="row"><label>颜色统一</label>${select("family", o.families.map((f) => ({ key: f, name: f })), "（只看比例）")}</div>
      <div class="muted">影调风格和颜色模板在「影调与颜色」页选；这里只看比例和统一色。</div>
      <div class="muted">看主色 / 辅色 / 点缀色的面积是否接近 6:3:1；选一个色系时还会生成把杂色往它统一的调整图层。</div>`;
    remember("family");
  } else if (state.sub === "skin") {
    c.innerHTML = `<div class="row"><label>抠皮肤</label><select id="seg"><option value="ai">AI 人像分割（显卡）</option><option value="color">肤色规则（不分脸和身体）</option></select></div>
      <div class="row"><label>肤色风格</label>${select("preset", o.presets)}</div>
      <div class="row"><label>部位</label><select id="part"><option value="脸部">脸部</option><option value="身体">身体</option><option value="">全部皮肤</option></select></div>`;
    const syncPart = () => {
      const color = $("seg").value === "color";
      if (color) $("part").value = "";
      $("part").disabled = color;  // 肤色规则分不出脸和身体
    };
    ["seg", "preset", "part"].forEach(remember);
    $("seg").addEventListener("change", syncPart);
    syncPart();
  } else if (MODE_OF[state.sub] === "patch") {
    c.innerHTML = `<div class="row"><button id="fg">读取前景色（先用吸管取色）</button></div>
      <div class="row"><button id="samplers">多点对比（读取颜色取样器）</button></div>
      <div class="muted">或用选框 / 套索框出一块区域，再点上面的「读取选区颜色」（相当于大取样）。多点对比：先用「颜色取样器工具」在画面上点几个位置（例如额头、两颊、下巴），第一个点是基准。</div>`;
    $("fg").addEventListener("click", readForeground);
    $("samplers").addEventListener("click", readSamplers);
  } else if (state.sub === "reference") {
    c.innerHTML = `<div class="row"><label>参考图</label><select id="refDoc"></select></div>
      <div class="row"><button id="refFile">选择文件…</button></div>
      <div class="muted">直接选参考图文件（JPG / PNG / TIFF / WebP / 相机 RAW），或者在 PS 里另外打开它后从上面的列表选（PSD 请用这种方式）。当前文档是“要调的图”。</div>`;
    $("refFile").addEventListener("click", chooseRefFile);
    refreshRefDocs();
  } else {
    c.innerHTML = "";
  }
}

// 参考图列表跟着 PS 里打开 / 关闭的文档实时更新，保留原来的选择；不列出当前文档自己
function refreshRefDocs() {
  const sel = $("refDoc");
  if (!sel) return;
  // 用户的选择单独记下：列表每次重建（换页、服务后启动、文档开关）都按它恢复，而不是跳回第一项
  if (!sel.dataset.bound) {
    sel.dataset.bound = "1";
    sel.addEventListener("change", () => { state.refChoice = sel.value; });
  }
  const keep = state.refChoice || sel.value;
  const active = app.activeDocument;
  const docs = app.documents.filter((d) => !active || d.id !== active.id);
  const file = state.refFile ? `<option value="file">文件：${esc(state.refFile.name)}</option>` : "";
  sel.innerHTML = file + docs.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("")
    || '<option value="">（选择文件，或先在 PS 里打开参考图）</option>';
  if (keep === "file" && state.refFile) sel.value = "file";
  else if (docs.some((d) => String(d.id) === keep)) sel.value = keep;
}

const REF_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "webp", "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2",
  "raf", "orf", "rw2", "dng", "pef", "srw", "3fr", "iiq", "erf", "kdc", "mos", "mrw", "x3f"];

// 直接选参考图文件：内容交给本机服务解码，不在 PS 里打开
async function chooseRefFile() {
  if (state.busy) return;
  let file;
  try {
    file = await uxp.storage.localFileSystem.getFileForOpening({ types: REF_TYPES });
  } catch (e) { statusError(e); return; }
  if (!file) return;
  setBusy(true, `正在读取参考图「${file.name}」…`);
  try {
    const data = await file.read({ format: uxp.storage.formats.binary });
    if (data.byteLength > 380 * 1024 * 1024) throw new Error("参考图文件太大（超过 380 MB），请另存为较小的 JPG / TIFF 再选。");
    const res = await bridge.uploadReferenceFile(file.name, data);  // 先读一次：格式不对马上提示
    state.refFile = { name: file.name, data };
    setBusy(false);
    state.refChoice = "file";
    refreshRefDocs();
    $("status").className = "status ok";
    $("status").textContent = `已选参考图「${file.name}」（${res.width}×${res.height}）${(res.notes || []).length ? "：" + res.notes[0] : ""}`;
  } catch (e) {
    setBusy(false);
    statusError(e);
  }
}

function query(sub) {
  const q = {};
  const v = (id) => ($(id) ? $(id).value : "");
  if (sub === "style") {
    if (v("style")) q.style = v("style");
    const t = v("toneStyle");
    if (t.startsWith("s:")) q.tone_style = t.slice(2);
    if (t.startsWith("t:")) q.tone_target = t.slice(2);
  }
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
  if (!state.options) {
    // 先锁住界面再连服务：避免连点两次、或等待时切到别的页面
    setBusy(true, "正在连接本机分析服务…");
    await loadOptions();
    setBusy(false);
    if (state.options && ["style", "ratio", "skin"].includes(sub)) {
      // 列表刚加载出来，会恢复上次的选择：先让用户看到再分析，而不是按用户没看到的设置直接算
      $("status").className = "status warn";
      $("status").textContent = `分析服务已连上，选项已加载（恢复了上次的选择）。请确认设置后再点「${analyzeLabel()}」。`;
      return;
    }
  }
  const q = query(sub);
  const settings = settingsKey(sub);
  if (sub === "style" && !state.options) {
    statusError(new Error("影调风格和颜色列表还没加载出来（分析服务未启动？），请先启动 ColorDoctor。"));
    return;
  }
  let refDoc = null;
  let refFile = null;
  if (mode === "reference") {
    const v = ($("refDoc") || {}).value;
    if (v === "file" && state.refFile) refFile = state.refFile;
    else {
      refDoc = app.documents.find((d) => String(d.id) === v);
      if (!refDoc) { statusError(new Error("请先选择参考图：点「选择文件…」，或在 PS 里打开参考图后从列表里选。")); return; }
      if (refDoc.id === info.doc.id) { statusError(new Error("参考图不能是当前文档：请切到要调的图，再选参考图。")); return; }
    }
  }
  setBusy(true, mode === "patch" ? "正在读取…" : "正在分析…");
  // 在读像素之前记下历史状态：分析过程中如果有改动，会被正确判为“可能过期”
  const historyBefore = ps.historyStateId(info.doc);
  try {
    // “及以下的画面”：读像素时临时隐藏上方图层（不留历史记录）
    const px = await ps.readModal(() => (mode === "patch" ? ps.readSelectionPixels(info) : ps.readPixels(info)), info);
    if (refFile) {
      // 每次分析都重新交给服务（服务只暂存最近几张参考图）
      q.ref = (await bridge.uploadReferenceFile(refFile.name, refFile.data)).ref;
    } else if (mode === "reference") {
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
        layerName: info.layer ? info.layer.name : "合并可见图层", target: $("target").value,
        okHistory: new Set([historyBefore]), bounds: sb,
        docWidth: info.doc.width, docHeight: info.doc.height, depth: String(info.doc.bitsPerChannel || ""),
        refName: refFile ? refFile.name : (refDoc ? refDoc.name : null),
        settings, settingsText: settingsText(sub, q),
      },
    };
  } catch (e) {
    // 失败：这个页面不再保留旧结果，避免用旧结果生成图层
    delete state.bySub[sub];
    setBusy(false);
    serviceDown(e);
    if (state.sub === sub) renderError(e);
    refreshTarget();  // 也会刷新影调说明（不再说“这张图现在是…”）
    return;
  }
  setBusy(false);
  if (state.sub === sub) {
    renderResult();
    toneHint();  // 有了这张图的影调，说明里加上“要怎么改”
  }
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

// PS 颜色取样器的多点对比：差别用文字说（几乎看不出 … 差别很大），不只给 ΔE 数字
async function readSamplers() {
  if (state.busy) return;
  const sub = state.sub;
  let pts;
  try { pts = ps.readColorSamplers(); } catch (err) { statusError(err); return; }
  setBusy(true, `正在对比 ${pts.length} 个取样点…`);
  try {
    const res = await bridge.multipoint(pts.map((p) => ({ name: p.name, lab: p.lab })));
    state.bySub[sub] = { result: { mode: "multipoint", sub, multipoint: res }, analyzed: null, made: new Set() };
    setBusy(false);
    if (state.sub === sub) renderResult();
    $("status").className = "status ok";
    $("status").textContent = res.summary + ((pts.skipped || []).length ? `（取样点 ${pts.skipped.join("、")} 读不到颜色，已跳过）` : "");
  } catch (err) {
    setBusy(false);
    statusError(err);
  }
}

const JUDGE_CLASS = { 基准: "", 一致: "ok", 略有差别: "warn", 不均: "bad" };

function renderMultipoint(r) {
  const m = r.multipoint;
  const rows = m.rows.map((x) => {
    const bar = x.de == null ? "—" : `${"▮".repeat(x.level + 1)}${"▯".repeat(5 - x.level)} ${esc(x.words)} <span class="muted">(${x.de.toFixed(1)})</span>`;
    return `<tr><td>${esc(x.name)}</td><td><span class="dot" style="background:rgb(${x.rgb.join(",")})"></span></td>
      <td>${bar}</td><td>${esc(x.direction)}</td><td class="${JUDGE_CLASS[x.judge] || ""}">${esc(x.judge)}</td></tr>`;
  }).join("");
  $("results").innerHTML = `
    <div class="card"><div class="title"><span>多点对比</span></div><div class="line">${esc(m.summary)}</div></div>
    <table class="mp"><tr><th>点</th><th>颜色</th><th>和基准的差别</th><th>偏向</th><th>判断</th></tr>${rows}</table>
    <div class="muted">第一个点是基准。${esc(m.legend)}</div>`;
  renderAcr({}, false);
  updateButtons();
}

function hasLayers(e) {
  return !!(e && e.analyzed && e.result && e.result.plan && e.result.plan.layers.length);
}

function hasAcrSource() {
  const e = cur();
  const fromResult = e && e.result && e.result.mode !== "patch" && e.result.mode !== "multipoint";
  const edited = [...document.querySelectorAll("[data-acr]")].some((i) => i.dataset.set === "1" && Number(i.value));
  return !!(fromResult || edited);
}

function updateButtons() {
  const busy = state.busy;
  const e = cur();
  const offDoc = e && e.analyzed && (!app.activeDocument || app.activeDocument.id !== e.analyzed.docId);
  // 校正图层按 RGB 设计：CMYK / Lab / 灰度文档先说明，不要等点了才报错
  const docMode = String((app.activeDocument && app.activeDocument.mode) || "");
  const nonRgb = !!docMode && !/rgb/i.test(docMode);
  const blockTitle = nonRgb ? "校正图层按 RGB 设计：请先把文档转换为 RGB（图像 → 模式 → RGB 颜色）" : "";
  $("generateAll").disabled = busy || !hasLayers(e) || !!offDoc || nonRgb;
  $("generateAll").title = blockTitle;
  if (e && hasLayers(e)) {
    const rest = e.result.plan.layers.filter((l) => !e.made.has(l.id));
    const mask = rest.some((l) => l.has_mask) ? "（带蒙版）" : "";
    $("generateAll").textContent = rest.length === 0 ? "全部已生成" : (e.made.size ? `生成其余 ${rest.length} 层${mask}` : `一键生成全部${mask}`);
    if (rest.length === 0) $("generateAll").disabled = true;
  } else {
    $("generateAll").textContent = "一键生成全部";
  }
  document.querySelectorAll("[data-make],[data-make-all]").forEach((b) => { b.disabled = busy || !!offDoc || nonRgb || b.dataset.done === "1"; b.title = blockTitle; });
  document.querySelectorAll("[data-sel]").forEach((b) => { b.disabled = busy || !!offDoc; });
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
  const set = a.settingsText ? ` · ${esc(a.settingsText)}` : "";
  return `<div class="muted">结果：${esc(SUB_NAME[e.result.sub] || "")} · 「${esc(a.layerName)}」 · ${esc(a.docName)}${ref}${set}</div>`;
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
  if (r.mode === "multipoint") return renderMultipoint(r);
  const cards = [resultHeader(e)];
  (r.notes || []).forEach((n) => cards.push(`<div class="muted">${esc(n)}</div>`));
  if (r.tone && r.sub === "style") cards.push(toneGrid(r.tone));
  const layers = (r.plan && r.plan.layers) || [];
  const findings = r.findings || [];
  // 每张问题卡片里放修它的调整图层（layer.finding = 问题的序号）；没挂到问题上的放在最后“其他调整”
  const owner = (ly) => (Number.isInteger(ly.finding) && ly.finding >= 0 && ly.finding < findings.length ? ly.finding : null);
  const layerHtml = (ly) => {
    const done = e.made.has(ly.id);
    return `
      <div class="fix">
        <div class="title"><span>${esc(ly.name)}</span><span class="muted">${done ? "已生成 ✓" : (ly.estimated ? "估计值" : "")}</span></div>
        <div class="line">${esc(ly.reason)}</div>
        <div class="row">
          <button class="small make" data-make="${esc(ly.id)}">${done ? "再生成一次" : `生成此层${ly.has_mask ? "（带蒙版）" : ""}`}</button>
          ${ly.has_mask ? `<button class="small" data-sel="${esc(ly.id)}" title="只在当前文档里建立这个问题区域的选区，不新建图层">只建选区</button>` : ""}
        </div>
      </div>`;
  };
  findings.forEach((f, i) => {
    const own = layers.filter((ly) => owner(ly) === i);
    if (r.sub !== "style" && String(f.problem).startsWith("影调：") && !own.length) return;  // 比例页不管影调
    const tips = (f.adjustments || []).filter((x) => !x.layer);
    cards.push(`
    <div class="card problem">
      <div class="title"><span>${esc(f.problem)}</span><span class="sev-${esc(f.severity)}">${esc(f.severity)}</span></div>
      ${(f.evidence || []).slice(0, 3).map((x) => `<div class="line">证据：${esc(x)}</div>`).join("")}
      ${(f.causes || []).slice(0, 2).map((x) => `<div class="line">可能原因：${esc(x)}</div>`).join("")}
      ${own.length ? `<div class="strong">修正${own.length > 1 ? `（${own.length} 层）` : ""}</div>${own.map(layerHtml).join("")}
        ${own.length > 1 ? (() => {
          const left = own.filter((ly) => !e.made.has(ly.id)).length;
          const label = left === 0 ? `这 ${own.length} 层都已生成 ✓` : (left < own.length ? `生成其余 ${left} 层` : `生成这 ${own.length} 层`);
          return `<div class="row"><button class="small make" data-make-all="${own.map((ly) => esc(ly.id)).join(",")}" data-finding="${i}"${left === 0 ? " data-done=\"1\"" : ""}>${label}</button></div>`;
        })() : ""}` : ""}
      ${(own.length ? tips : (f.adjustments || [])).slice(0, 3).map((x) => `<div class="line">${own.length ? "另外：" : "调整："}${esc(x.tool)} · ${esc(x.instruction)}</div>`).join("")}
      <div class="muted">置信度 ${Math.round((f.confidence || 0) * 100)}%${(f.notes || []).length ? " · 注意：" + esc(f.notes[0]) : ""}</div>
    </div>`);
  });
  const orphans = layers.filter((ly) => owner(ly) === null);
  if (orphans.length) {
    cards.push(`<div class="card"><div class="title"><span>${findings.length ? "其他调整" : "可生成的调整图层"}</span></div>${orphans.map(layerHtml).join("")}</div>`);
  }
  if (!layers.length && !findings.length) cards.push('<p class="hint">没有发现需要处理的问题。</p>');
  else if (!layers.length) {
    cards.push('<p class="hint">这些建议没有对应的调整图层：可以按文字手动调整，或打开下方「ACR 参数」导出预设 / 生成 Camera Raw 图层。</p>');
  }
  $("results").innerHTML = cards.join("");
  $("results").querySelectorAll("[data-make]").forEach((b) => b.addEventListener("click", () => generate([b.dataset.make], true)));
  // 卡片的“生成这 N 层”：只补还没生成的（单层按钮才是“再生成一次”）
  $("results").querySelectorAll("[data-make-all]").forEach((b) => b.addEventListener("click",
    () => generate(b.dataset.makeAll.split(","), false, Number(b.dataset.finding))));
  $("results").querySelectorAll("[data-sel]").forEach((b) => b.addEventListener("click", () => makeSelection(b.dataset.sel)));
  renderAcr(r.acr_params || {}, r.acr_calibrated);
  updateButtons();
}

// 九种影调的九宫格 + 明度直方图：实底 = 现在，橙框 = 目标；下面按“改动大小”列出各影调风格
const KEYS = ["高", "中", "低"];
const LENGTHS = ["长", "中", "短"];
function toneGrid(t) {
  const cell = (name) => {
    const cls = [name === t.name ? "now" : "", name === t.target && t.target !== t.name ? "goal" : "", name === t.near ? "near" : ""].join(" ");
    const tag = name === t.name ? "现在" : (name === t.target ? "目标" : "");
    const info = ((state.options && state.options.tones) || []).find((x) => x.name === name);
    return `<div class="tcell ${cls}" title="${esc(info ? info.desc : name)}">${esc(name)}${tag ? `<span>${tag}</span>` : ""}</div>`;
  };
  const grid = `<div class="tgrid"><div></div>${LENGTHS.map((n) => `<div class="thead">${n}调</div>`).join("")}
    ${KEYS.map((k) => `<div class="thead">${k}调</div>${LENGTHS.map((n) => cell(`${k}${n}调`)).join("")}`).join("")}</div>`;
  const max = Math.max(...t.histogram, 1e-6);
  const hist = `<div class="thist">${t.histogram.map((v) => `<i style="height:${Math.round(100 * v / max)}%"></i>`).join("")}</div>`;
  const groups = [[], [], []];
  (t.suggestions || []).forEach((s) => groups[Math.min(s.distance, 2)].push(s.name));
  const near = [["已经符合", groups[0]], ["稍改就到", groups[1]], ["变化较大", groups[2]]]
    .filter(([, g]) => g.length).map(([w, g]) => `${w}：${g.join("、")}`).join("；");
  const shares = t.shares.map((v) => `${Math.round(v * 100)}%`);
  return `<div class="card tone">
    <div class="title"><span>影调：${esc(t.name)}${t.near ? `（接近${esc(t.near)}）` : ""}</span>${t.target && t.target !== t.name ? `<span>→ ${esc(t.label)} ${esc(t.target)}</span>` : ""}</div>
    <div class="line">${esc(t.desc)}</div>
    ${grid}${hist}
    <div class="muted">暗部 ${shares[0]} · 中间调 ${shares[1]} · 亮部 ${shares[2]}；跨度 ${Math.round(t.span)} 级（≥${t.long_span || 170} 长调，&lt;${t.short_span || 90} 短调）</div>
    <div class="muted">各影调风格离现在多远：${esc(near)}</div>
  </div>`;
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

async function generate(ids, single, findingIndex) {
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
  if (settingsChanged(e)) {
    // 设置在分析后改过：提醒一次；用户再点一次就按旧结果生成
    const now = settingsKey(state.sub);
    if (e.settingsAck !== now) {
      e.settingsAck = now;
      statusError(new Error(`设置在分析后改过了：这些图层仍是按“${a.settingsText}”算的。要按新设置请重新分析；确定用旧结果就再点一次生成。`));
      return;
    }
  }
  const docMode = String((app.activeDocument && app.activeDocument.mode) || "");
  if (docMode && !/rgb/i.test(docMode)) {
    // 色彩平衡 / 自然饱和度等在 Lab、灰度里不可用，CMYK 里颜色范围和数值含义也不同
    statusError(new Error("校正图层按 RGB 设计：请先把文档转换为 RGB（图像 → 模式 → RGB 颜色），或在 RGB 副本上生成。"));
    return;
  }
  pruneMade(e);  // 撤销过的图层不算“已生成”
  // 一键生成只补还没生成的；单层按钮总是按用户的意思生成
  const todo = single ? ids : ids.filter((id) => !e.made.has(id));
  if (!todo.length) { statusError(new Error("全部已经生成过了。如需重新生成，请点各层的「再生成一次」。")); updateButtons(); return; }
  const specs = r.plan.layers.filter((l) => todo.includes(l.id));
  // 一张卡片的几层：组名用这个问题的名字，几次生成的组在图层面板里分得清
  const f = Number.isInteger(findingIndex) ? (r.findings || [])[findingIndex] : null;
  const groupName = f ? `ColorDoctor · ${String(f.problem).replace(/[：:（(].*$/, "").slice(0, 24)}` : r.plan.group_name;
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
        let top = made.length ? made[made.length - 1] : null;
        if (made.length > 1) top = (await ps.groupLayers(d, made, groupName)) || top;
        // 合并画面：最顶层可能是上次生成的组，新图层会被建进那个组里 —— 挪到组外的最顶层
        if (a.layerId == null && top != null) await ps.liftOutOfGroups(d, top);
      });
    }, { docId: a.docId, docSize: [a.docWidth, a.docHeight] });
    e.madeLayers = { ...(e.madeLayers || {}), ...madeIds };
    setBusy(false);
    markMade(e, specs.map((s) => s.id), `已生成 ${specs.length} 个调整图层${specs.length > 1 ? `，放在「${groupName}」组里` : ""}（Ctrl+Z 可一次撤销）`);
    if (state.sub === r.sub) renderResult();
    $("status").className = "status ok";
    $("status").textContent = a.justMadeText;
  } catch (err) {
    setBusy(false);
    serviceDown(err);
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
  const stack = $("target").value === "stack";
  const doc = app.activeDocument;
  const e = cur();
  const a = e && e.analyzed;
  let layerId = null;
  let layerName = "合并可见图层";
  if (stack && a && doc && a.docId === doc.id && a.target === "stack" && a.layerId != null && ps.findLayer(doc, a.layerId)) {
    // 用分析过的那一层：一键生成后选中的是新调整图层，盖印它会把校正算两次
    layerId = a.layerId;
    layerName = `${a.layerName}及以下的画面`;
  } else if (stack) {
    // 及以下的画面：在选中图层上方盖印“到这一层为止”的画面（上方图层临时隐藏），调整图层也可以
    const info = refreshTarget();
    if (!info.ok) { statusError(new Error(info.reason)); return; }
    layerId = info.layer.id;
    layerName = `${info.layer.name}及以下的画面`;
  } else if (!merged) {
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
  setBusy(true, `第 2 步：${stack ? "盖印" : "复制"}「${layerName}」并打开 Camera Raw。在「预设」面板 → ⋯ → 导入配置文件和预设，选「${saved}.xmp」，再点确定`);
  try {
    await ps.modal("ColorDoctor Camera Raw", async (ctx, d) => ps.cameraRawLayer("ColorDoctor · Camera Raw", { merged, stack, layerId, doc: d }),
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
// 连不上本机服务（status 0）：右上角的服务状态跟着变红
function serviceDown(err) {
  if (err && err.status === 0) {
    $("svc").className = "chip bad";
    $("svc").textContent = "分析服务未启动";
  }
}

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
    savePref("cat", state.cat);
    savePref("sub", state.sub);
    renderNav();
  }));
  // 恢复上次的大类 / 子页 / 分析对象
  if (hasOwn(SUBS, prefs.cat)) {
    state.cat = prefs.cat;
    state.sub = SUBS[state.cat].some(([k]) => k === prefs.sub) ? prefs.sub : SUBS[state.cat][0][0];
  }
  if (["stack", "layer", "merged"].includes(prefs.target)) $("target").value = prefs.target;
  $("target").addEventListener("change", () => { savePref("target", $("target").value); refreshTarget(); });
  $("analyze").addEventListener("click", runAnalyze);
  // 改了影调风格 / 颜色 / 统一色 / 肤色设置：状态栏马上说明“结果是按旧设置算的”
  $("controls").addEventListener("change", () => { if (!state.busy) refreshTarget(); });
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
