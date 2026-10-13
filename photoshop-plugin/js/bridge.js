// 与本机分析服务（colordoctor serve，只监听 127.0.0.1）通信。
const BASE = "http://127.0.0.1:47321";
const CLIENT = { "X-ColorDoctor-Client": "uxp" };

class BridgeError extends Error {
  constructor(message, { blocked = false, status = 0 } = {}) {
    super(message);
    this.blocked = blocked;
    this.status = status;
  }
}

async function call(path, { method = "GET", headers = {}, body } = {}) {
  let res;
  try {
    res = await fetch(BASE + path, { method, headers: { ...CLIENT, ...headers }, body });
  } catch (e) {
    throw new BridgeError("连不上本机分析服务。请先运行 ColorDoctor（或命令 colordoctor serve），再点一次。");
  }
  if (res.status === 422) {
    const j = await res.json();
    throw new BridgeError(j.blocked, { blocked: true, status: 422 });
  }
  if (!res.ok) {
    let msg = `分析服务出错（${res.status}）`;
    try { msg = (await res.json()).error || msg; } catch (_) { /* 非 JSON */ }
    throw new BridgeError(msg, { status: res.status });
  }
  return res;
}

function pixelHeaders(px, extra = {}) {
  return {
    "Content-Type": "application/octet-stream",
    "X-Width": String(px.width),
    "X-Height": String(px.height),
    "X-Components": String(px.components),
    ...extra,
  };
}

async function health() {
  return (await call("/health")).json();
}

// px: {data: Uint8Array, width, height, components}
async function analyze(px, mode, query, { docWidth, docHeight, layerKind, layerName } = {}) {
  const qs = new URLSearchParams({ mode, ...query }).toString();
  const res = await call(`/analyze?${qs}`, {
    method: "POST",
    headers: pixelHeaders(px, {
      "X-Doc-Width": String(docWidth || px.width),
      "X-Doc-Height": String(docHeight || px.height),
      "X-Layer-Kind": layerKind || "",
      "X-Layer-Name": encodeURIComponent(layerName || ""),
    }),
    body: px.data,
  });
  return res.json();
}

async function uploadReference(px) {
  const res = await call("/reference", { method: "POST", headers: pixelHeaders(px), body: px.data });
  return (await res.json()).ref;
}

// 返回整幅文档大小的蒙版：分析范围（bounds）以外为 0，写到文档 (0, 0) 即可
async function fetchMask(session, layerId, bounds, docWidth, docHeight) {
  const w = bounds.right - bounds.left;
  const h = bounds.bottom - bounds.top;
  const qs = new URLSearchParams({
    session, layer: layerId, w: String(w), h: String(h),
    left: String(bounds.left), top: String(bounds.top), doc_w: String(docWidth), doc_h: String(docHeight),
  }).toString();
  const res = await call(`/mask?${qs}`);
  return { data: new Uint8Array(await res.arrayBuffer()), width: Number(res.headers.get("X-Width")), height: Number(res.headers.get("X-Height")) };
}

async function options() {
  return (await call("/options")).json();
}

async function acrPreset(name, params) {
  const res = await call("/acr-preset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, params }),
  });
  return res.text();
}

async function calibrationChart() {
  const res = await call("/calibration/chart");
  return {
    data: new Uint8Array(await res.arrayBuffer()),
    width: Number(res.headers.get("X-Width")),
    height: Number(res.headers.get("X-Height")),
    components: 3,
  };
}

async function calibrationPlan() {
  return (await (await call("/calibration/plan")).json()).steps;
}

async function calibrationSample(px, { channel, slider, value }) {
  const qs = slider === "base" ? "slider=base"
    : new URLSearchParams({ channel, slider, value: String(value) }).toString();
  await call(`/calibration/sample?${qs}`, { method: "POST", headers: pixelHeaders(px), body: px.data });
}

async function calibrationFit(acrVersion) {
  const res = await call("/calibration/fit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ acr_version: acrVersion || "" }),
  });
  return res.json();
}

module.exports = {
  BridgeError, health, options, analyze, uploadReference, fetchMask, acrPreset,
  calibrationChart, calibrationPlan, calibrationSample, calibrationFit,
};
