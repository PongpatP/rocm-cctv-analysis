/* Playback — control-room player: a continuous 24h scrub timeline (footage +
 * AI event markers), a video with live AI boxes, and a per-frame AI review. */

import { iouDedupe } from "./boxes.js";

const camSel = document.getElementById("camera");
const dateSel = document.getElementById("date");
const player = document.getElementById("player");
const emptyEl = document.getElementById("empty");
const playingEl = document.getElementById("playing");
const segCount = document.getElementById("seg-count");
const statusBar = document.getElementById("status-text");
const statusFoot = document.getElementById("status-foot");
const anno = document.getElementById("pb-anno");
const aiBtn = document.getElementById("pb-ai");
const playBtn = document.getElementById("pb-play");
const tlCanvas = document.getElementById("pb-tl");
const tlTip = document.getElementById("pb-tl-tip");
const tlScroll = document.getElementById("pb-tl-scroll");
const pbTimeline = document.getElementById("pb-timeline");
let pxPerHour = 160;                 // timeline zoom (px per hour); scrollable
const timelineWidth = () => Math.max(tlScroll.clientWidth, Math.round(24 * pxPerHour));

const DAY_MS = 86400000;
let segments = [];       // [{file, time, size, quality}] for camera+date
let currentIdx = -1;
let summary = {};        // minute_epoch_s -> {class: count}
let segStartMs = 0;      // absolute epoch ms of the playing segment's start
let dayStart = 0;        // epoch ms of 00:00 of the loaded date
let aspect = "16:9";
let dets = [];           // detections for the current segment
let aiOn = true;
let pbThr = 0.30;

const OVERLAY_COLORS = {
  person: "#ff6a00", car: "#2ec5ff", truck: "#b085ff",
  motorcycle: "#ff4fa3", bicycle: "#3ddc84",
};
const VEHICLE = ["car", "truck", "bus", "motorcycle", "bicycle"];

/* ---- time helpers ---- */
const fmtDate = (d) => `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;
// Files before 2026-07-11 were stamped in UTC; from then on in Asia/Bangkok.
function segEpochMs(dateStr, timeStr) {
  const y = +dateStr.slice(0, 4), m = +dateStr.slice(4, 6) - 1, d = +dateStr.slice(6, 8),
        h = +timeStr.slice(0, 2), mi = +timeStr.slice(2, 4), s = +timeStr.slice(4, 6);
  if (dateStr < "20260711") return Date.UTC(y, m, d, h, mi, s);
  return new Date(y, m, d, h, mi, s).getTime();
}
const p2 = (n) => String(n).padStart(2, "0");
function clock(ms) { const t = new Date(ms); return `${p2(t.getHours())}:${p2(t.getMinutes())}:${p2(t.getSeconds())}`; }
function segTimeDisplay(dateStr, timeStr) { return clock(segEpochMs(dateStr, timeStr)); }

/* ---- confidence filter ---- */
const thrEl = document.getElementById("pb-thr");
const thrVal = document.getElementById("pb-thr-val");
function setThr(v) { pbThr = Number(v); thrEl.value = pbThr; thrVal.textContent = pbThr.toFixed(2); }
setThr(0.30);
thrEl.addEventListener("input", () => setThr(thrEl.value));
fetch("/api/settings").then((r) => r.json())
  .then((s) => setThr(Math.max(0.10, s.min_confidence || 0.30))).catch(() => {});

aiBtn.addEventListener("click", () => {
  aiOn = !aiOn;
  aiBtn.classList.toggle("pb-toggle--on", aiOn);
  if (!aiOn) clearAnno();
});

/* ---- init ---- */
async function init() {
  const cfg = await (await fetch("config.json")).json();
  aspect = cfg.aspect || "16:9";
  document.body.dataset.aspect = aspect;
  for (const nvr of cfg.nvrs || []) {
    for (let ch = 1; ch <= nvr.channels; ch++) {
      if (nvr.skip && nvr.skip.includes(ch)) continue;
      const key = `${nvr.id}_ch${p2(ch)}`;
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = `${nvr.name} · CH ${p2(ch)}`;
      camSel.appendChild(opt);
    }
  }
  camSel.addEventListener("change", () => { announceCamera(); loadDates(); });
  dateSel.addEventListener("change", loadSegments);
  player.addEventListener("ended", () => playIndex(currentIdx + 1));
  player.addEventListener("pause", () => { playBtn.textContent = "▶"; onPause(); });
  player.addEventListener("play", () => { playBtn.textContent = "❚❚"; });
  playBtn.addEventListener("click", () => {
    if (currentIdx < 0) { seekToTime(segments.length ? segEpochMs(dateSel.value, segments[0].time) : 0); return; }
    player.paused ? player.play().catch(() => {}) : player.pause();
  });
  tlCanvas.addEventListener("click", onTimelineClick);
  tlCanvas.addEventListener("mousemove", onTimelineHover);
  tlCanvas.addEventListener("mouseleave", () => { tlTip.hidden = true; });
  document.getElementById("pb-zoom-in").addEventListener("click", () => setZoom(pxPerHour * 1.6));
  document.getElementById("pb-zoom-out").addEventListener("click", () => setZoom(pxPerHour / 1.6));
  window.addEventListener("resize", drawTimeline);
  requestAnimationFrame(tlLoop);
  await loadDates();
  announceCamera();
}

/* When the operator picks a camera, tell them in the review panel WHAT they are
   looking at — camera id and (roughly) its place — so they have context. */
let graphNodes = null;
async function cameraPlace(cam) {
  if (!graphNodes) {
    try {
      const g = await (await fetch("/api/reid/graph")).json();
      graphNodes = {};
      for (const n of g.nodes || []) graphNodes[n.camera] = n;
    } catch { graphNodes = {}; }
  }
  const n = graphNodes[cam];
  return n ? (n.confirmed_label || n.vlm_caption || "") : "";
}
async function announceCamera() {
  const cam = camSel.value; if (!cam) return;
  const label = camSel.options[camSel.selectedIndex]?.textContent || cam;
  const place = (await cameraPlace(cam)) || "";
  const el = newMsg({ role: "camera", who: "Now viewing", icon: "🎥" });
  el.querySelector(".pb-msg__body").textContent = place ? `${label} — ${place}` : label;
}

/* ---- data ---- */
async function loadDates() {
  const dates = await (await fetch(`/api/recordings/${camSel.value}/dates`)).json();
  dateSel.innerHTML = "";
  if (!dates.length) {
    dateSel.appendChild(Object.assign(document.createElement("option"), { textContent: "no recordings", value: "" }));
  } else {
    for (const d of dates.slice().reverse())
      dateSel.appendChild(Object.assign(document.createElement("option"), { value: d, textContent: fmtDate(d) }));
  }
  await loadSegments();
}

async function loadSegments() {
  segments = []; currentIdx = -1; summary = {};
  if (dateSel.value) {
    segments = await (await fetch(`/api/recordings/${camSel.value}?date=${dateSel.value}`)).json();
    dayStart = segEpochMs(dateSel.value, "000000");
    try {
      const s = await fetch(`/api/detections/summary?camera=${camSel.value}` +
        `&start_ms=${dayStart}&end_ms=${dayStart + DAY_MS}`);
      summary = await s.json();
    } catch { summary = {}; }
  }
  segCount.textContent = `${segments.length} segments`;
  const range = segments.length
    ? `${camSel.value} · ${fmtDate(dateSel.value)} · ${segTimeDisplay(dateSel.value, segments[0].time)} → ${segTimeDisplay(dateSel.value, segments[segments.length - 1].time)}`
    : "no footage for this camera / date";
  statusBar.textContent = range;
  statusFoot.textContent = range;
  stopPlayer();
  drawTimeline();
}

/* ---- scrub timeline ---- */
const timeToX = (tMs, W) => ((tMs - dayStart) / DAY_MS) * W;
const xToTime = (x, W) => dayStart + (x / W) * DAY_MS;

function drawTimeline() {
  const W = timelineWidth();
  tlCanvas.style.width = W + "px";
  const H = tlCanvas.clientHeight || 62;
  if (!W || !H) return;
  const dpr = window.devicePixelRatio || 1;
  tlCanvas.width = W * dpr; tlCanvas.height = H * dpr;
  const ctx = tlCanvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const top = 4, trackH = 26, evY = top + trackH + 4, labelY = H - 1;
  const segW = Math.max(1, (60000 / DAY_MS) * W);

  // footage lane base + hour grid
  ctx.fillStyle = "rgba(0,0,0,.045)";
  ctx.fillRect(0, top, W, trackH);
  ctx.font = "9px ui-monospace, monospace"; ctx.textBaseline = "bottom";
  for (let h = 0; h <= 24; h++) {
    const x = (h / 24) * W;
    ctx.strokeStyle = h % 6 === 0 ? "rgba(0,0,0,.20)" : "rgba(0,0,0,.07)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x + 0.5, top); ctx.lineTo(x + 0.5, top + trackH); ctx.stroke();
    if (h < 24 && h % 2 === 0) { ctx.fillStyle = "#94a0ac"; ctx.fillText(p2(h), x + 3, labelY); }
  }
  // footage present
  ctx.fillStyle = "#9db9d3";
  for (const s of segments) ctx.fillRect(timeToX(segEpochMs(dateSel.value, s.time), W), top, segW, trackH);

  // AI event markers per minute (stacked: person / vehicle / animal)
  for (const k in summary) {
    const x = timeToX((+k) * 1000, W);
    if (x < -2 || x > W) continue;
    const c = summary[k];
    const veh = VEHICLE.reduce((a, v) => a + (c[v] || 0), 0);
    if (c.person) { ctx.fillStyle = "#2563eb"; ctx.fillRect(x, evY, segW, 5); }
    if (veh)      { ctx.fillStyle = "#d97706"; ctx.fillRect(x, evY + 5, segW, 5); }
    if (c.dog || c.cat) { ctx.fillStyle = "#dc2626"; ctx.fillRect(x, evY + 10, segW, 5); }
  }

  // playhead
  if (currentIdx >= 0) {
    const x = timeToX(segStartMs + (player.currentTime || 0) * 1000, W);
    ctx.strokeStyle = "#0f1114"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, top - 3); ctx.lineTo(x, top + trackH + 3); ctx.stroke();
    ctx.fillStyle = "#0f1114";
    ctx.beginPath(); ctx.moveTo(x - 4, top - 3); ctx.lineTo(x + 4, top - 3); ctx.lineTo(x, top + 3); ctx.closePath(); ctx.fill();
  }
}

function tlLoop() {
  requestAnimationFrame(tlLoop);
  if (currentIdx >= 0 && !player.paused) {
    drawTimeline();
    const W = timelineWidth();
    const x = timeToX(segStartMs + (player.currentTime || 0) * 1000, W);
    const vw = tlScroll.clientWidth;
    if (x < tlScroll.scrollLeft + 60 || x > tlScroll.scrollLeft + vw - 60)
      tlScroll.scrollLeft = Math.max(0, x - vw / 2);
  }
}

function setZoom(px) {
  const W0 = timelineWidth();
  const centerT = xToTime(tlScroll.scrollLeft + tlScroll.clientWidth / 2, W0);
  pxPerHour = Math.max(60, Math.min(600, px));
  drawTimeline();
  const W = timelineWidth();
  tlScroll.scrollLeft = Math.max(0, timeToX(centerT, W) - tlScroll.clientWidth / 2);
}

function segIndexAt(tMs) {
  for (let i = 0; i < segments.length; i++) {
    const s = segEpochMs(dateSel.value, segments[i].time);
    if (tMs >= s && tMs < s + 60000) return i;
  }
  return -1;
}

function seekToTime(tMs) {
  if (!segments.length) return;
  let idx = segIndexAt(tMs);
  if (idx < 0) {
    let best = -1, bd = Infinity;
    for (let i = 0; i < segments.length; i++) {
      const d = Math.abs(segEpochMs(dateSel.value, segments[i].time) - tMs);
      if (d < bd) { bd = d; best = i; }
    }
    idx = best; if (idx < 0) return;
    tMs = segEpochMs(dateSel.value, segments[idx].time);
  }
  const off = Math.max(0, (tMs - segEpochMs(dateSel.value, segments[idx].time)) / 1000);
  if (idx === currentIdx) { player.currentTime = off; player.play().catch(() => {}); }
  else playIndex(idx, off);
}

function onTimelineClick(e) {
  const r = tlCanvas.getBoundingClientRect();
  seekToTime(xToTime(e.clientX - r.left, r.width));
}

function onTimelineHover(e) {
  const r = tlCanvas.getBoundingClientRect();
  const t = xToTime(e.clientX - r.left, r.width);
  const mk = Math.floor(t / 1000 / 60) * 60;
  const c = summary[mk];
  let extra = "";
  if (c) {
    const bits = [];
    if (c.person) bits.push(`person×${c.person}`);
    const veh = VEHICLE.reduce((a, v) => a + (c[v] || 0), 0);
    if (veh) bits.push(`vehicle×${veh}`);
    if (c.dog || c.cat) bits.push(`animal×${(c.dog || 0) + (c.cat || 0)}`);
    if (bits.length) extra = " · " + bits.join(", ");
  }
  tlTip.hidden = false;
  tlTip.style.left = `${e.clientX - pbTimeline.getBoundingClientRect().left}px`;
  tlTip.textContent = clock(t) + extra;
}

/* ---- player ---- */
const DET_LATENCY_MS = 400;
async function loadDetections(seg) {
  dets = [];
  segStartMs = segEpochMs(dateSel.value, seg.time);
  try {
    const res = await fetch(`/api/detections?camera=${camSel.value}` +
      `&start_ms=${segStartMs - 1000}&end_ms=${segStartMs + 65000}`);
    if (res.ok) { dets = await res.json(); for (const d of dets) d.t = d.ts - DET_LATENCY_MS; }
  } catch { /* overlay is best-effort */ }
}

function playIndex(idx, offset = 0) {
  if (idx < 0 || idx >= segments.length) { stopPlayer(); return; }
  currentIdx = idx;
  const seg = segments[idx];
  emptyEl.style.display = "none";
  player.style.objectFit = (seg.quality === "main" || aspect === "native") ? "contain" : "fill";
  player.src = `/api/recordings/file/${camSel.value}/${seg.file}`;
  if (offset > 0) player.addEventListener("loadedmetadata", () => { player.currentTime = offset; }, { once: true });
  player.play().catch(() => {});
  loadDetections(seg);
  playingEl.textContent = `${fmtDate(dateSel.value)} ${segTimeDisplay(dateSel.value, seg.time)} · ${idx + 1}/${segments.length}`;
  drawTimeline();
}

function stopPlayer() {
  currentIdx = -1;
  player.removeAttribute("src"); player.load();
  emptyEl.style.display = "";
  playingEl.textContent = "—";
  playBtn.textContent = "▶";
  dets = []; clearAnno();
}

/* ---- AI box overlay ---- */
function clearAnno() { const ctx = anno.getContext("2d"); ctx.clearRect(0, 0, anno.width, anno.height); }
function videoRect() {
  const ew = anno.clientWidth, eh = anno.clientHeight;
  if (!ew || !eh) return null;
  if (aspect !== "native") {
    // object-fit: contain -> letterboxed; map boxes to where the picture sits
    const vw = player.videoWidth, vh = player.videoHeight;
    if (!vw || !vh) return { x: 0, y: 0, w: ew, h: eh };
    const scale = Math.min(ew / vw, eh / vh), w = vw * scale, h = vh * scale;
    return { x: (ew - w) / 2, y: (eh - h) / 2, w, h };
  }
  const vw = player.videoWidth, vh = player.videoHeight;
  if (!vw || !vh) return null;
  const scale = Math.min(ew / vw, eh / vh), w = vw * scale, h = vh * scale;
  return { x: (ew - w) / 2, y: (eh - h) / 2, w, h };
}

function drawFrame() {
  requestAnimationFrame(drawFrame);
  if (!aiOn || !dets.length || (player.paused && !player.currentTime)) return;
  anno.width = anno.clientWidth; anno.height = anno.clientHeight;
  const ctx = anno.getContext("2d");
  ctx.clearRect(0, 0, anno.width, anno.height);
  const rect = videoRect(); if (!rect) return;
  const nowMs = segStartMs + player.currentTime * 1000;
  let latest = -Infinity;
  for (const d of dets) if (d.t <= nowMs && d.t > latest) latest = d.t;
  if (latest === -Infinity || nowMs - latest > 700) return;
  const visible = iouDedupe(dets.filter((d) => Math.abs(d.t - latest) < 60 && (d.confidence || 0) >= pbThr));
  ctx.font = "11px ui-monospace, monospace"; ctx.textBaseline = "top"; ctx.lineWidth = 2;
  for (const r of visible) {
    const fw = r.frame?.width || 960, fh = r.frame?.height || 544, b = r.bounding_box;
    const x = rect.x + (b.x / fw) * rect.w, y = rect.y + (b.y / fh) * rect.h;
    const w = (b.w / fw) * rect.w, h = (b.h / fh) * rect.h;
    const col = OVERLAY_COLORS[r.class_name] || "#ffd23c";
    ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
    const label = `${r.class_name} ${(r.confidence * 100) | 0}%` + (r.track_id != null ? ` #${r.track_id}` : "");
    const tw = ctx.measureText(label).width + 8, ly = y >= 15 ? y - 15 : y + 1;
    ctx.fillStyle = col; ctx.fillRect(x - 1, ly, tw, 15);
    ctx.fillStyle = "#111"; ctx.fillText(label, x + 3, ly + 2);
  }
}
requestAnimationFrame(drawFrame);

/* ---- Suggested action: three AIs review the paused frame ---- */
const suggestLog = document.getElementById("suggest-log");
let suggestBusy = false, lastSuggestAt = 0;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function onPause() {
  if (player.seeking || player.ended || !player.currentSrc) return;
  if (suggestBusy || Date.now() - lastSuggestAt < 1500) return;
  reviewFrame();
}
function grabFrame() {
  if (!player.videoWidth) return null;
  const c = document.createElement("canvas");
  c.width = player.videoWidth; c.height = player.videoHeight;
  c.getContext("2d").drawImage(player, 0, 0, c.width, c.height);
  try { return c.toDataURL("image/jpeg", 0.85); } catch { return null; }
}
function fmtClock() { const p = playingEl.textContent || ""; return p.length && p !== "—" ? p : "the clip"; }
function newMsg(m) {
  document.getElementById("suggest-idle")?.remove();
  const el = document.createElement("div");
  el.className = `pb-msg pb-msg--${m.role}` + (m.warn ? " pb-msg--warn" : "");
  const who = document.createElement("span");
  who.className = "pb-msg__who"; who.textContent = `${m.icon || ""} ${m.who}`.trim();
  const body = document.createElement("div"); body.className = "pb-msg__body";
  el.append(who, body); suggestLog.appendChild(el);
  suggestLog.scrollTop = suggestLog.scrollHeight;
  return el;
}
async function typeInto(el, text) {
  const body = el.querySelector(".pb-msg__body");
  el.classList.add("pb-msg--typing");
  const speed = Math.max(6, Math.min(22, 900 / Math.max(text.length, 1)));
  for (let i = 1; i <= text.length; i++) {
    body.textContent = text.slice(0, i);
    if (i % 3 === 0) suggestLog.scrollTop = suggestLog.scrollHeight;
    await sleep(speed);
  }
  el.classList.remove("pb-msg--typing");
  const t = document.createElement("span");
  t.className = "pb-msg__time";
  t.textContent = `${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · at ${fmtClock()}`;
  el.appendChild(t); suggestLog.scrollTop = suggestLog.scrollHeight;
}
async function reviewFrame() {
  const img = grabFrame(); if (!img) return;
  suggestBusy = true; lastSuggestAt = Date.now();
  const think = newMsg({ role: "vision", who: "Observer", icon: "👁️" });
  think.classList.add("pb-msg--thinking");
  think.querySelector(".pb-msg__body").textContent = "reading the frame…";
  let d;
  try {
    const tsMs = Number.isFinite(segStartMs) && Number.isFinite(player.currentTime)
      ? Math.round(segStartMs + player.currentTime * 1000) : null;
    const res = await fetch("/api/reid/playback/suggest", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: img, camera: camSel.value, ts_ms: tsMs }),
    });
    d = await res.json();
  } catch { d = { ok: false, error: "AI review service unreachable" }; }
  think.remove();
  if (!d.ok) {
    await typeInto(newMsg({ role: "sup", who: "Supervisor", icon: "🛡️", warn: true }),
      d.error || "could not read the frame");
    suggestBusy = false; return;
  }
  for (const m of d.messages || []) { await typeInto(newMsg(m), m.text); await sleep(180); }
  suggestBusy = false;
}

/* ---- text search: find moments on this camera, jump to them ---- */
const searchForm = document.getElementById("pb-search-form");
const searchInput = document.getElementById("pb-search");
searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  let q = searchInput.value.trim();
  if (!q || !dateSel.value) return;
  searchInput.value = "";
  // translate the query to English so it reads in English and searches the
  // English VLM captions well
  try {
    const tr = await (await fetch("/api/reid/translate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: q }),
    })).json();
    if (tr && tr.text) q = tr.text;
  } catch { /* keep original */ }
  // a cross-camera command — "which camera has …", "jump to the camera with …",
  // or a plain "move to the storage room" — searches the WHOLE site (every
  // camera) and switches to whichever one matches. A place name ("… room",
  // "… entrance", "… car park") counts too.
  if (/\b(cameras?|move to|go to|take me to|bring me to|switch to|navigate to|show me the|bring up|other (camera|room|area))\b/i.test(q)
      || /\b\w+\s+(room|entrance|car ?park|alley|gate|ward|lobby|corridor|hallway|hall)\b/i.test(q)) {
    await crossCameraSearch(q); return;
  }
  newMsg({ role: "search", who: "You", icon: "🔍" }).querySelector(".pb-msg__body").textContent = q;
  const wait = newMsg({ role: "sup", who: "Supervisor", icon: "🛡️" });
  wait.classList.add("pb-msg--thinking");
  wait.querySelector(".pb-msg__body").textContent = "searching the day…";
  let d;
  try {
    const day0 = segEpochMs(dateSel.value, "000000");
    const res = await fetch("/api/reid/playback/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, camera: camSel.value, start_ms: day0, end_ms: day0 + DAY_MS }),
    });
    d = await res.json();
  } catch { d = { ok: false, error: "search service unreachable" }; }
  wait.remove();
  if (!d.ok) {
    const e2 = newMsg({ role: "sup", who: "Supervisor", icon: "🛡️", warn: true });
    e2.querySelector(".pb-msg__body").textContent = d.error || "search failed";
    return;
  }
  const sel = newMsg({ role: "sup", who: "Supervisor", icon: "🛡️" });
  sel.querySelector(".pb-msg__body").textContent = d.summary || "";
  if (d.results && d.results.length) {
    const box = document.createElement("div");
    box.className = "pb-results";
    for (const rr of d.results) {
      const btn = document.createElement("button");
      btn.className = "pb-result";
      btn.innerHTML = `<b>${clock(rr.ts_ms)}</b>`;
      btn.append(" " + (rr.label || ""));
      btn.addEventListener("click", () => seekToTime(rr.ts_ms));
      box.appendChild(btn);
    }
    sel.appendChild(box);
  }
  suggestLog.scrollTop = suggestLog.scrollHeight;
});

/* ---- cross-camera command: a team of agents finds the right camera ---- */
async function crossCameraSearch(q) {
  newMsg({ role: "search", who: "You", icon: "🔍" })
    .querySelector(".pb-msg__body").textContent = q;
  const wait = newMsg({ role: "analyst", who: "Dispatcher", icon: "🧭" });
  wait.classList.add("pb-msg--thinking");
  wait.querySelector(".pb-msg__body").textContent = "scanning every camera…";
  let d;
  try {
    const day0 = segEpochMs(dateSel.value, "000000");
    const res = await fetch("/api/reid/playback/find_camera", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, start_ms: day0, end_ms: day0 + DAY_MS,
                             exclude: camSel.value }),
    });
    d = await res.json();
  } catch { d = { ok: false, error: "search service unreachable" }; }
  wait.remove();
  if (!d.ok) {
    newMsg({ role: "sup", who: "Supervisor", icon: "🛡️", warn: true })
      .querySelector(".pb-msg__body").textContent = d.error || "search failed";
    return;
  }
  // play back each agent's reasoning as chat bubbles (skip the echoed query)
  for (const m of d.messages || []) {
    if (m.role === "search") continue;
    await typeInto(newMsg(m), m.text); await sleep(160);
  }
  // clickable results — each switches to that camera at that moment
  if (d.results && d.results.length) {
    const box = document.createElement("div");
    box.className = "pb-results";
    for (const rr of d.results) {
      const btn = document.createElement("button");
      btn.className = "pb-result";
      btn.innerHTML = `<b>${clock(rr.ts_ms)}</b>`;
      const tag = rr.place ? `${rr.camera} · ${rr.place}` : rr.camera;
      btn.append(` ${tag} — ${rr.label || ""}`);
      btn.addEventListener("click", () => jumpTo(rr.camera, rr.ts_ms));
      box.appendChild(btn);
    }
    suggestLog.lastElementChild?.appendChild(box);
  }
  suggestLog.scrollTop = suggestLog.scrollHeight;
  // auto-jump to the Supervisor's top pick
  if (d.jump && d.jump.camera) await jumpTo(d.jump.camera, d.jump.ts_ms);
}

/* switch camera (same day) and seek to a moment */
async function jumpTo(cam, tsMs) {
  const wantDate = dateSel.value;         // stay on the day being reviewed
  if (cam && cam !== camSel.value) {
    camSel.value = cam;
    await loadDates();                    // repopulates dates + loads segments
    if (wantDate && dateSel.value !== wantDate &&
        [...dateSel.options].some((o) => o.value === wantDate)) {
      dateSel.value = wantDate;
      await loadSegments();
    }
    announceCamera();
  }
  seekToTime(tsMs);
}

init();
