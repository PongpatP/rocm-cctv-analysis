/* Detection dashboard: history charts from /api/stats, live feed from the
 * bridge WebSocket, and a live camera view with annotation overlay.
 * Charts are plain SVG — crosshair + all-series tooltip, no libraries. */

import { VideoRTC } from "../video-rtc.js";
import { iouDedupe, latestBatch } from "./boxes.js";

/* Chart palette: darker steps of the overlay class hues, validated for the
 * white surface (lightness band, CVD separation, ≥3:1 contrast). */
const CHART_COLORS = {
  person: "#d45a00",
  car: "#0b7fc2",
  truck: "#7a4fd8",
  motorcycle: "#d1216f",
  bicycle: "#178a4a",
};
const FALLBACKS = ["#8a6d00", "#0f7f7a", "#b3403c", "#5b5f9e", "#7d6a52"];

/* Bright variants used only on the dark video overlay. */
const OVERLAY_COLORS = {
  person: "#ff6a00", car: "#2ec5ff", truck: "#b085ff",
  motorcycle: "#ff4fa3", bicycle: "#3ddc84",
};

function chartColor(name, idx) {
  return CHART_COLORS[name] || FALLBACKS[idx % FALLBACKS.length];
}

const state = { classes: [], nvrs: [], liveCam: null, liveRecords: null, aspect: "16:9" };

/* ---- boot -------------------------------------------------------- */

async function init() {
  const cfg = await (await fetch("config.json")).json();
  state.classes = cfg.classes || [];
  state.nvrs = cfg.nvrs || [];
  state.aspect = cfg.aspect || "16:9";
  document.body.dataset.aspect = state.aspect;

  buildLiveCameraSelect();
  document.getElementById("range").addEventListener("change", loadStats);

  await loadStats();
  await loadFeedHistory();
  connectFeed();
  loadAiStatus();
  setInterval(loadStats, 60_000);
  setInterval(loadAiStatus, 15_000);   // AI health refresh
}

function esc(x) { return String(x ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

(function initLiveExpand() {
  const stage = document.querySelector(".live-stage");
  const btn = document.getElementById("live-expand");
  const go = () => {
    if (!stage) return;
    if (document.fullscreenElement) document.exitFullscreen();
    else stage.requestFullscreen?.();
  };
  if (btn) btn.onclick = go;
  if (stage) stage.addEventListener("dblclick", go);
})();

async function loadAiStatus() {
  const grid = document.getElementById("ai-grid");
  if (!grid) return;
  let d;
  try { d = await (await fetch("/api/ai/status")).json(); }
  catch { document.getElementById("ai-summary").textContent = "status unavailable"; return; }
  document.getElementById("ai-summary").textContent = `${d.up}/${d.total} running`;
  grid.replaceChildren(...(d.models || []).map((m) => {
    const el = document.createElement("div");
    el.className = "ai-item " + (m.up ? "ai-item--up" : "ai-item--down");
    el.innerHTML =
      `<span class="ai-item__dot"></span>` +
      `<div><div class="ai-item__role">${esc(m.role)}</div>` +
      `<div class="ai-item__name">${esc(m.name)}</div>` +
      `<div class="ai-item__meta">${esc(m.meta || "")}</div>` +
      `<div class="ai-item__out">${esc(m.out || "")}</div></div>` +
      `<span class="ai-item__state">${m.up ? "live" : "down"}</span>`;
    return el;
  }));
}


/* ---- stats + charts ------------------------------------------------ */

async function loadStats() {
  const hours = document.getElementById("range").value;
  const res = await fetch(`/api/stats?hours=${hours}`);
  if (!res.ok) return;
  const s = await res.json();

  document.getElementById("tile-total").textContent = compact(s.total);
  document.getElementById("tile-hour").textContent = compact(s.last_hour);
  document.getElementById("tile-busy").textContent =
    s.cameras[0] ? s.cameras[0].camera : "—";
  document.getElementById("tile-cams").textContent = s.cameras.length;
  document.getElementById("updated").textContent =
    "updated " + new Date().toLocaleTimeString(undefined, { hour12: false });

  renderLineChart(s);
  renderCameraBars(s.cameras);
  renderTable(s);
}

function compact(n) {
  if (n == null) return "—";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 10_000) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function seriesList(s) {
  // fixed order: configured classes first, then anything else seen
  const seen = new Set(Object.keys(s.totals));
  const order = [...state.classes.filter((c) => seen.has(c)),
    ...[...seen].filter((c) => !state.classes.includes(c)).sort()];
  return order;
}

function renderLineChart(s) {
  const holder = document.getElementById("line-chart");
  const series = seriesList(s);
  const W = holder.clientWidth || 900, H = 240;
  const M = { l: 44, r: 14, t: 10, b: 24 };
  const pts = s.buckets;
  const n = pts.length;
  const maxY = Math.max(1, ...pts.flatMap((b) => series.map((c) => b.classes[c] || 0)));

  const x = (i) => M.l + (i / Math.max(1, n - 1)) * (W - M.l - M.r);
  const y = (v) => M.t + (1 - v / maxY) * (H - M.t - M.b);

  const fmtT = (t) => {
    const d = new Date(t * 1000);
    return s.hours > 48
      ? d.toLocaleDateString(undefined, { month: "short", day: "2-digit" })
      : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false });
  };

  let g = "";
  const yticks = 4;
  for (let i = 0; i <= yticks; i++) {
    const v = (maxY / yticks) * i, yy = y(v);
    g += `<line x1="${M.l}" y1="${yy}" x2="${W - M.r}" y2="${yy}" stroke="#e9ebee"/>`
      + `<text x="${M.l - 6}" y="${yy + 3}" text-anchor="end" class="ax">${Math.round(v)}</text>`;
  }
  const xticks = Math.min(6, n);
  for (let i = 0; i < xticks; i++) {
    const idx = Math.round((i / Math.max(1, xticks - 1)) * (n - 1));
    g += `<text x="${x(idx)}" y="${H - 6}" text-anchor="middle" class="ax">${fmtT(pts[idx].t)}</text>`;
  }

  let lines = "", ends = "";
  series.forEach((cls, si) => {
    const col = chartColor(cls, si);
    const d = pts.map((b, i) =>
      `${i ? "L" : "M"}${x(i).toFixed(1)},${y(b.classes[cls] || 0).toFixed(1)}`).join("");
    lines += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round"/>`;
    const lastV = pts[n - 1]?.classes[cls] || 0;
    ends += `<text x="${W - M.r - 2}" y="${y(lastV) - 4}" text-anchor="end" class="endlab" fill="${col}">${cls}</text>`;
  });

  holder.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:${H}px">`
    + `<style>.ax{font:10px ui-monospace,monospace;fill:#676e76}`
    + `.endlab{font:600 10px -apple-system,sans-serif}</style>`
    + g + lines + ends
    + `<line id="xhair" x1="0" y1="${M.t}" x2="0" y2="${H - M.b}" stroke="#9aa1a9" stroke-dasharray="3 3" visibility="hidden"/>`
    + `</svg>`;

  const legend = document.getElementById("line-legend");
  legend.innerHTML = "";
  series.forEach((cls, si) => {
    const item = document.createElement("span");
    const sw = document.createElement("span");
    sw.className = "legend__swatch";
    sw.style.background = chartColor(cls, si);
    item.appendChild(sw);
    item.appendChild(document.createTextNode(cls));
    legend.appendChild(item);
  });

  // crosshair + all-series tooltip
  const svg = holder.querySelector("svg");
  const tip = document.getElementById("chart-tip");
  const xhair = svg.querySelector("#xhair");
  svg.addEventListener("pointermove", (e) => {
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.max(0, Math.min(n - 1,
      Math.round(((px - M.l) / (W - M.l - M.r)) * (n - 1))));
    const sx = x(i);
    xhair.setAttribute("x1", sx); xhair.setAttribute("x2", sx);
    xhair.setAttribute("visibility", "visible");

    tip.innerHTML = "";
    const title = document.createElement("div");
    title.className = "chart-tip__title";
    title.textContent = fmtT(pts[i].t);
    tip.appendChild(title);
    series.forEach((cls, si) => {
      const row = document.createElement("div");
      row.className = "chart-tip__row";
      const nameSpan = document.createElement("span");
      const sw = document.createElement("span");
      sw.className = "legend__swatch";
      sw.style.background = chartColor(cls, si);
      nameSpan.appendChild(sw);
      nameSpan.appendChild(document.createTextNode(" " + cls));
      const val = document.createElement("span");
      val.className = "val";
      val.textContent = pts[i].classes[cls] || 0;
      row.appendChild(nameSpan); row.appendChild(val);
      tip.appendChild(row);
    });
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tw - 8) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  svg.addEventListener("pointerleave", () => {
    tip.hidden = true;
    xhair.setAttribute("visibility", "hidden");
  });
}

function renderCameraBars(cameras) {
  const holder = document.getElementById("cam-bars");
  holder.innerHTML = "";
  const max = Math.max(1, ...cameras.map((c) => c.objects));
  for (const c of cameras.slice(0, 14)) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const name = document.createElement("span");
    name.className = "bar-row__name";
    name.textContent = c.camera;
    const track = document.createElement("div");
    track.className = "bar-row__track";
    const fill = document.createElement("div");
    fill.className = "bar-row__fill";
    fill.style.width = Math.max(1, (c.objects / max) * 100) + "%";
    track.appendChild(fill);
    const val = document.createElement("span");
    val.className = "bar-row__val";
    val.textContent = compact(c.objects);
    row.title = `${c.camera}: ${c.objects} objects`;
    row.append(name, track, val);
    holder.appendChild(row);
  }
  if (!cameras.length) holder.textContent = "No detections in range.";
}

function renderTable(s) {
  const series = seriesList(s);
  const table = document.getElementById("line-table");
  table.innerHTML = "";
  const head = table.insertRow();
  ["time", ...series].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    head.appendChild(th);
  });
  const step = Math.max(1, Math.floor(s.buckets.length / 24));
  for (let i = 0; i < s.buckets.length; i += step) {
    const b = s.buckets[i];
    const tr = table.insertRow();
    tr.insertCell().textContent =
      new Date(b.t * 1000).toLocaleString(undefined,
        { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false });
    series.forEach((c) => { tr.insertCell().textContent = b.classes[c] || 0; });
  }
}

/* ---- live feed ------------------------------------------------------ */

const feed = document.getElementById("feed");
const FEED_MAX = 100;

function addFeedRow(r, prepend = true) {
  const row = document.createElement("div");
  row.className = "feed__row";
  const t = document.createElement("span");
  t.textContent = (r.timestamp || "").slice(11, 19);
  const cam = document.createElement("span");
  cam.textContent = r.camera_id;
  const cls = document.createElement("span");
  cls.className = "feed__class";
  const sw = document.createElement("span");
  sw.className = "legend__swatch";
  sw.style.background = chartColor(r.class_name, 0);
  cls.append(sw, document.createTextNode(r.class_name));
  const conf = document.createElement("span");
  conf.textContent = ((r.confidence ?? 0) * 100 | 0) + "%";
  const track = document.createElement("span");
  track.textContent = r.track_id != null ? "#" + r.track_id : "—";
  row.append(t, cam, cls, conf, track);

  const headRow = feed.querySelector(".feed__row--head");
  if (prepend) headRow.after(row);
  else feed.appendChild(row);
  while (feed.children.length > FEED_MAX + 1) feed.lastChild.remove();
}

async function loadFeedHistory() {
  try {
    const items = await (await fetch("/api/history?limit=50")).json();
    for (const r of items) addFeedRow(r, false);
  } catch { /* bridge may be starting */ }
}

let feedRetry = 3000;

function connectFeed() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    feedRetry = 3000;
    document.getElementById("feed-led").className = "led led--ok";
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type !== "detections") return;
    // newest few only — the feed is a sample, not a firehose
    for (const r of msg.records.slice(-5)) addFeedRow(r);
    updateLiveOverlay(msg.records);
  };
  ws.onclose = () => {
    document.getElementById("feed-led").className = "led led--err";
    setTimeout(connectFeed, feedRetry);
    feedRetry = Math.min(feedRetry * 2, 30000);
  };
  ws.onerror = () => ws.close();
}

/* ---- live camera view ------------------------------------------------ */

class CamStream extends VideoRTC {
  constructor() {
    super();
    this.mode = "mse";
    this.media = "video";
  }
  oninit() {
    super.oninit();
    this.video.controls = false;
    this.video.muted = true;
    this.video.autoplay = true;
    // match the main page: stretch anamorphic CCTV streams to scene shape
    this.video.style.objectFit = state.aspect === "native" ? "contain" : "fill";
  }
}
customElements.define("cam-stream", CamStream);

let liveCamEl = null;

function buildLiveCameraSelect() {
  const sel = document.getElementById("live-camera");
  for (const nvr of state.nvrs) {
    for (let ch = 1; ch <= nvr.channels; ch++) {
      if (nvr.skip && nvr.skip.includes(ch)) continue;
      const key = `${nvr.id}_ch${String(ch).padStart(2, "0")}`;
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = `${nvr.name} · CH ${String(ch).padStart(2, "0")}`;
      sel.appendChild(opt);
    }
  }
  sel.addEventListener("change", () => switchLiveCamera(sel.value));
  if (sel.options.length) switchLiveCamera(sel.options[0].value);
}

function switchLiveCamera(key) {
  state.liveCam = key;
  state.liveRecords = null;
  const holder = document.getElementById("live-holder");
  holder.replaceChildren();
  liveCamEl = document.createElement("cam-stream");
  holder.appendChild(liveCamEl);
  liveCamEl.src = `api/ws?src=${key}_sub`;
}

function updateLiveOverlay(records) {
  const mine = iouDedupe(latestBatch(records.filter((r) => r.camera_id === state.liveCam)));
  if (!mine.length && !state.liveRecords) return;
  state.liveRecords = mine.length ? mine : null;

  const canvas = document.getElementById("live-anno");
  const video = liveCamEl?.video;
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!video || !mine.length) return;

  let rect;
  if (state.aspect !== "native") {
    rect = { x: 0, y: 0, w: canvas.width, h: canvas.height };
  } else {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    const scale = Math.min(canvas.width / vw, canvas.height / vh);
    rect = {
      x: (canvas.width - vw * scale) / 2,
      y: (canvas.height - vh * scale) / 2,
      w: vw * scale, h: vh * scale,
    };
  }

  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.lineWidth = 2;
  for (const r of mine) {
    const fw = r.frame?.width || 960, fh = r.frame?.height || 544;
    const b = r.bounding_box;
    const x = rect.x + (b.x / fw) * rect.w;
    const y = rect.y + (b.y / fh) * rect.h;
    const w = (b.w / fw) * rect.w, h = (b.h / fh) * rect.h;
    const col = OVERLAY_COLORS[r.class_name] || "#ffd23c";
    ctx.strokeStyle = col;
    ctx.strokeRect(x, y, w, h);
    const label = `${r.class_name} ${(r.confidence * 100) | 0}%` +
      (r.track_id != null ? ` #${r.track_id}` : "");
    const tw = ctx.measureText(label).width + 8;
    const ly = y >= 15 ? y - 15 : y + 1;
    ctx.fillStyle = col;
    ctx.fillRect(x - 1, ly, tw, 15);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 3, ly + 2);
  }
}

init();
