/* ccvt_stream_ai frontend
 * - Live video: go2rtc MSE via the VideoRTC element (vendored, MIT).
 * - AI annotation: detection records arrive over the bridge WebSocket and
 *   are drawn on a canvas over each tile (class, confidence, track id).
 */

import { VideoRTC } from "../video-rtc.js";
import { classColor, iouDedupe, latestBatch } from "./boxes.js";
import "./nav.js";   // shared ☰ menu (all pages)
import { startIdentityFeed, identityLabel, isNamed, faceBox, hasRealFace,
         loadViewer, viewerIsAdmin } from "./identity.js";

const STATUS_POLL_MS = 5000;
// box-hold auto-follows the detection interval (Settings): a skipped-frame
// gap is (interval+1)/fps; hold ~2x that + 500ms so old boxes persist through
// the skip but still drop when detection genuinely stops. Refreshed live.
let ANNO_TTL_MS = 900;
const DET_FPS = 8;                    // airocm --fps
async function refreshBoxHold() {
  try {
    const d = await fetch("/api/config/classes").then((r) => r.json());
    const gapMs = ((+d.interval || 0) + 1) / DET_FPS * 1000;
    ANNO_TTL_MS = Math.round(Math.max(700, gapMs * 2 + 500));
  } catch { /* keep the current value */ }
}

/* ---- <cam-stream> element ------------------------------------ */

class CamStream extends VideoRTC {
  constructor() {
    super();
    this.mode = "mse";
    this.media = "video";
    this.controls = false;
    // "scene": stretch anamorphic CCTV streams to the stage's aspect
    // (grid tiles). "contain": letterbox at the stream's own ratio —
    // used by the fullscreen viewer, whose main streams have square
    // pixels and whose stage is the window (arbitrary shape).
    this.fit = "scene";
  }

  oninit() {
    super.oninit();
    this.video.controls = this.controls;
    this.video.muted = true;
    this.video.autoplay = true;
    // fill the aspect-corrected box (16:9/4:3) so anamorphic cameras
    // (e.g. nvr2 main 960x1080) don't show skinny in the fullscreen
    // viewer — only trust the raw stream shape in explicit "native" mode
    this.video.style.objectFit =
      (state.aspect === "native") ? "contain" : "fill";
    this.video.addEventListener("playing", () => this._emit("live"));
    // routine buffer pauses on low-fps cameras must not flash the overlay
    this.video.addEventListener("waiting", () => this._emit("buffer"));
  }

  onconnect() {
    const started = super.onconnect();
    if (started) this._emit("connecting");
    return started;
  }

  ondisconnect() {
    super.ondisconnect();
    this._emit("offline");
  }

  _emit(state) {
    this.dispatchEvent(new CustomEvent("statechange", { detail: state }));
  }
}

customElements.define("cam-stream", CamStream);

/* ---- app state ------------------------------------------------ */

const MYCAMS_KEY = "ccvt.mycams";      // survives a reload; per browser, per phone

const state = {
  nvrs: [],
  view: "all",
  myCams: new Set(),       // camera keys the user chose to watch
  aspect: "16:9",   // "16:9" | "4:3" stretch to true scene shape; "native" trust stream
  annotate: true,
  tiles: new Map(),        // camera key -> {canvas, video getter}
  detections: new Map(),   // camera key -> {records, at}
  viewerKey: null,
};

const grid = document.getElementById("camera-grid");
const viewTabs = document.getElementById("view-tabs");
const statusText = document.getElementById("status-text");

const camKey = (nvrId, ch) => `${nvrId}_ch${String(ch).padStart(2, "0")}`;

/* ---- boot ------------------------------------------------------ */

async function init() {
  refreshBoxHold();
  setInterval(refreshBoxHold, 10000);
  try {
    const res = await fetch("config.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();
    state.nvrs = cfg.nvrs;
    state.aspect = cfg.aspect || "16:9";
    document.body.dataset.aspect = state.aspect;
    buildLegend(cfg.classes);
    {
      document.getElementById("statusbar-right").textContent = "Live · GPU accelerated";
    }
    setLed("gateway", true);
  } catch (err) {
    setLed("gateway", false);
    statusText.textContent = `Gateway unreachable: ${err.message}`;
    setTimeout(init, 5000);
    return;
  }
  loadMyCams();
  if (state.myCams.size) state.view = "mine";   // a phone reopens where it left off
  buildTabs();
  viewTabs.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("tab--active", t.dataset.view === state.view));
  renderView();
  pollStatus();
  connectBridge();
  loadViewer();
  startIdentityFeed();       // names and plates, polled and cached
  setInterval(sweepStale, 500);
}

/* ---- header widgets -------------------------------------------- */


function setLed(which, ok) {
  document.getElementById(`${which}-led`).className =
    `led ${ok ? "led--ok" : "led--err"}`;
  document.getElementById(`${which}-text`).textContent = ok ? "online" : "offline";
  const menuRow = document.getElementById(`menu-${which}`);
  if (menuRow) menuRow.textContent = ok ? "online" : "offline";
}

async function pollStatus() {
  try {
    const res = await fetch("api/streams");
    const streams = await res.json();
    const active = Object.values(streams)
      .filter((s) => Array.isArray(s?.consumers) && s.consumers.length > 0).length;
    document.getElementById("stream-count").textContent = active;
    setLed("gateway", true);
  } catch {
    setLed("gateway", false);
  }
  setTimeout(pollStatus, STATUS_POLL_MS);
}

/* ---- bridge WebSocket (detections) ------------------------------ */

let bridgeRetry = 3000;

function bridgeWsUrl() {
  // The reverse proxy routes /ws to the bridge on the same origin,
  // both on the LAN and through Cloudflare.
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws`;
}

function connectBridge() {
  const ws = new WebSocket(bridgeWsUrl());

  ws.onopen = () => {
    bridgeRetry = 3000;
    setLed("ai", true);
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type !== "detections") return;
    const now = performance.now();
    const byCam = new Map();
    for (const r of msg.records) {
      if (!byCam.has(r.camera_id)) byCam.set(r.camera_id, []);
      byCam.get(r.camera_id).push(r);
    }
    for (const [cam, records] of byCam) {
      state.detections.set(cam, { records: iouDedupe(latestBatch(records)), at: now });
      drawCam(cam);
    }
  };
  ws.onclose = () => {
    setLed("ai", false);
    setTimeout(connectBridge, bridgeRetry);
    bridgeRetry = Math.min(bridgeRetry * 2, 30000);
  };
  ws.onerror = () => ws.close();
}

/* ---- annotation drawing ------------------------------------------ */


function buildLegend(classes) {
  const legend = document.getElementById("legend");
  if (!legend || !classes?.length) return;
  legend.innerHTML = classes.map((c) =>
    `<span class="legend__item"><span class="legend__swatch" ` +
    `style="background:${classColor(c)}"></span>${c}</span>`).join("");
}

function sceneRatio(video) {
  // The true scene shape: configured for anamorphic CCTV streams,
  // or the stream's own pixel ratio in "native" mode.
  if (state.aspect === "4:3") return 4 / 3;
  if (state.aspect === "16:9") return 16 / 9;
  const vw = video?.videoWidth, vh = video?.videoHeight;
  return vw && vh ? vw / vh : null;
}

function contentRect(video, el, fit) {
  const ew = el.clientWidth, eh = el.clientHeight;
  if (!ew || !eh) return null;
  // Stretched tiles fill the whole stage, so boxes map to the full canvas.
  if (fit === "scene" && state.aspect !== "native") return { x: 0, y: 0, w: ew, h: eh };
  // Letterboxed: the largest scene-ratio box that fits inside el.
  const ar = sceneRatio(video);
  if (!ar) return null;
  const w = Math.min(ew, eh * ar);
  const h = w / ar;
  return { x: (ew - w) / 2, y: (eh - h) / 2, w, h };
}

function drawOverlay(canvas, video, records, fit, cameraKey) {
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.annotate || !records || !video) return;

  const rect = contentRect(video, canvas, fit);
  if (!rect) return;

  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.lineWidth = 2;

  for (const r of records) {
    // bbox pixels are in the AI's streammux frame; normalize, then map
    // into the displayed video rect (same scene fraction either way).
    const fw = r.frame?.width || 960, fh = r.frame?.height || 544;
    const b = r.bounding_box;
    const x = rect.x + (b.x / fw) * rect.w;
    const y = rect.y + (b.y / fh) * rect.h;
    const w = (b.w / fw) * rect.w;
    const h = (b.h / fh) * rect.h;

    // Every head gets a box: grey while the person is a stranger, green with a
    // name once the face module recognises somebody the operator enrolled. Same
    // primitive as the person box below, so if this one does not appear the fault
    // is in the drawing path and not in the idea.
    //
    // The head comes from COCO joints 0-4 (nose, eyes, ears) that the pose model
    // already emits every frame. The face detector itself runs only after a track
    // settles and publishes no box, so it cannot draw at frame rate.
    if (r.class_name === "person") {
      const [hx, hy, hw, hh] = faceBox(r);
      const px = rect.x + (hx / fw) * rect.w;
      const py = rect.y + (hy / fh) * rect.h;
      const pw = Math.max((hw / fw) * rect.w, 8);
      const ph = Math.max((hh / fh) * rect.h, 8);
      const known = isNamed(cameraKey, r);
      ctx.strokeStyle = known ? "#1f9d55" : "#9aa1a9";
      // solid = SCRFD actually detected a face; dashed = head inferred from pose
      ctx.setLineDash(hasRealFace(r) ? [] : [3, 3]);
      ctx.strokeRect(px, py, pw, ph);
      ctx.setLineDash([]);
      if (!known && !viewerIsAdmin()) {
        // a stranger's face is not for a public screen
        ctx.fillStyle = "rgba(0,0,0,0.92)";
        ctx.fillRect(px, py, pw, ph);
      }
    }

    const color = isNamed(cameraKey, r) ? "#1f9d55" : classColor(r.class_name);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    // prefer the cross-camera identity (stable across cameras and across
    // tracker drops); fall back to the per-camera tracker id until ReID has
    // bound this tracklet to a Global ID
    const label =
      `${r.class_name} ${(r.confidence * 100) | 0}%` +
      (r.global_id != null ? ` G${r.global_id}`
        : r.track_id != null ? ` #${r.track_id}` : "") +
      identityLabel(cameraKey, r);   // a recognised name, or a voted plate
    const tw = ctx.measureText(label).width + 8;
    const ly = y >= 15 ? y - 15 : y + 1;
    ctx.fillStyle = color;
    ctx.fillRect(x - 1, ly, tw, 15);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 3, ly + 2);
  }
}

function drawCam(cam) {
  const det = state.detections.get(cam);
  const tile = state.tiles.get(cam);
  if (tile) drawOverlay(tile.canvas, tile.video, det?.records, "scene", cam);
  if (state.viewerKey === cam && viewerCam) {
    drawOverlay(viewerCanvas, viewerCam.video, det?.records, "contain", cam);
  }
}

function sweepStale() {
  const now = performance.now();
  for (const [cam, det] of state.detections) {
    if (now - det.at > ANNO_TTL_MS) {
      state.detections.delete(cam);
      drawCam(cam);
    }
  }
}

/* ---- view tabs / toolbar ------------------------------------------ */

/* A tile is a live WebSocket. On a phone, "All cameras" opens 28 of them, which is
 * slow before it is useful. "My cameras" streams only the handful you picked, and
 * the choice is remembered on that device. */
function loadMyCams() {
  try {
    state.myCams = new Set(JSON.parse(localStorage.getItem(MYCAMS_KEY) || "[]"));
  } catch { state.myCams = new Set(); }
}

function saveMyCams() {
  localStorage.setItem(MYCAMS_KEY, JSON.stringify([...state.myCams]));
  const tab = viewTabs.querySelector('[data-view="mine"]');
  if (tab) tab.firstChild.textContent = `★ My cameras (${state.myCams.size}) `;
}

function buildTabs() {
  for (const nvr of state.nvrs) {
    const btn = document.createElement("button");
    btn.className = "tab";
    btn.dataset.view = nvr.id;
    btn.textContent = nvr.name;
    viewTabs.appendChild(btn);
  }
  const mine = document.createElement("button");
  mine.className = "tab";
  mine.dataset.view = "mine";
  mine.innerHTML = `★ My cameras (${state.myCams.size}) <span class="tab__edit"
    id="mycams-edit" title="choose which cameras">✎</span>`;
  viewTabs.appendChild(mine);

  viewTabs.addEventListener("click", (e) => {
    if (e.target.id === "mycams-edit") { openPicker(); return; }
    const btn = e.target.closest(".tab");
    if (!btn || btn.dataset.view === state.view) return;
    if (btn.dataset.view === "mine" && !state.myCams.size) { openPicker(); return; }
    state.view = btn.dataset.view;
    viewTabs.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("tab--active", t === btn));
    renderView();
  });

  document.getElementById("grid-cols").addEventListener("change", (e) => {
    grid.dataset.cols = e.target.value;
  });
  grid.dataset.cols = "auto";

  const annoBtn = document.getElementById("anno-toggle");
  annoBtn.addEventListener("click", () => {
    state.annotate = !state.annotate;
    annoBtn.classList.toggle("tab--active", state.annotate);
    annoBtn.textContent = `AI boxes: ${state.annotate ? "on" : "off"}`;
    for (const cam of state.tiles.keys()) drawCam(cam);
  });
}

/* ---- camera tiles ------------------------------------------------ */

function eachCamera(fn) {
  for (const nvr of state.nvrs)
    for (let ch = 1; ch <= nvr.channels; ch++)
      if (!(nvr.skip && nvr.skip.includes(ch))) fn(nvr, ch, camKey(nvr.id, ch));
}

function renderView() {
  grid.innerHTML = ""; // removing <cam-stream> closes its WebSocket
  state.tiles.clear();

  let count = 0;
  eachCamera((nvr, ch, key) => {
    const show = state.view === "all" ? true
      : state.view === "mine" ? state.myCams.has(key)
      : nvr.id === state.view;
    if (!show) return;
    grid.appendChild(createTile(nvr, ch));
    count++;
  });
  statusText.textContent = count
    ? `${count} camera${count === 1 ? "" : "s"} · view: ${state.view.toUpperCase()}`
    : "no cameras selected — press ✎ on the My cameras tab";
}

/* ---- camera picker ------------------------------------------------ */

function openPicker() {
  const box = document.getElementById("cam-picker");
  const list = document.getElementById("picker-list");
  const draft = new Set(state.myCams);
  list.innerHTML = "";
  let group = null, lastNvr = null;
  eachCamera((nvr, ch, key) => {
    if (nvr.id !== lastNvr) {
      lastNvr = nvr.id;
      group = document.createElement("div");
      group.className = "picker__group";
      group.innerHTML = `<div class="picker__nvr">${nvr.name}</div>`;
      list.appendChild(group);
    }
    const lab = document.createElement("label");
    lab.className = "picker__cam";
    lab.innerHTML = `<input type="checkbox"${draft.has(key) ? " checked" : ""}>
      <span>CH ${String(ch).padStart(2, "0")}</span>`;
    lab.querySelector("input").onchange = (e) => {
      e.target.checked ? draft.add(key) : draft.delete(key);
      count();
    };
    group.appendChild(lab);
  });
  const count = () => {
    document.getElementById("picker-count").textContent = `${draft.size} selected`;
  };
  count();
  document.getElementById("picker-none").onclick = () => {
    draft.clear();
    list.querySelectorAll("input").forEach((i) => (i.checked = false));
    count();
  };
  document.getElementById("picker-cancel").onclick = () => { box.hidden = true; };
  document.getElementById("picker-save").onclick = () => {
    state.myCams = draft;
    saveMyCams();
    box.hidden = true;
    state.view = "mine";
    viewTabs.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("tab--active", t.dataset.view === "mine"));
    renderView();
  };
  box.onclick = (e) => { if (e.target === box) box.hidden = true; };
  box.hidden = false;
}

function createTile(nvr, channel) {
  const key = camKey(nvr.id, channel);
  const el = document.createElement("section");
  el.className = "cam";
  const chLabel = `CH ${String(channel).padStart(2, "0")}`;
  el.innerHTML = `
    <div class="cam__head">
      <span class="cam__id">${nvr.name} · ${chLabel}</span>
      <span class="cam__state"><span class="led led--off"></span><span>connecting</span></span>
    </div>
    <div class="cam__stage">
      <canvas class="cam__anno"></canvas>
      <div class="cam__overlay"><span>No signal</span></div>
    </div>`;

  const stage = el.querySelector(".cam__stage");
  const canvas = el.querySelector(".cam__anno");
  const led = el.querySelector(".cam__state .led");
  const stateLabel = el.querySelector(".cam__state span:last-child");

  const cam = document.createElement("cam-stream");
  let everLive = false;
  cam.addEventListener("statechange", (e) => {
    const mode = e.detail;
    if (mode === "live") everLive = true;
    const showVideo = mode === "live" || (mode === "buffer" && everLive);
    el.classList.toggle("cam--online", showVideo);
    led.className = "led " +
      (mode === "live" ? "led--ok"
        : mode === "buffer" ? "led--warn"
        : mode === "offline" ? "led--err" : "led--off");
    stateLabel.textContent = mode === "buffer" ? "live" : mode;
  });
  stage.prepend(cam);
  cam.src = `api/ws?src=${key}_sub`;

  state.tiles.set(key, { canvas, get video() { return cam.video; } });

  stage.addEventListener("click", () => openViewer(nvr, channel));
  return el;
}

/* ---- fullscreen viewer -------------------------------------------- */

const viewer = document.getElementById("viewer");
const viewerStage = document.getElementById("viewer-stage");
const viewerTitle = document.getElementById("viewer-title");
const viewerCanvas = document.getElementById("viewer-anno");
let viewerCam = null;

function layoutViewer() {
  // Size the viewer's video box to the true scene ratio, centered and
  // letterboxed in the window — old anamorphic main streams (D1 etc.)
  // get stretched to shape inside it (object-fit: fill).
  if (!viewerCam) return;
  const rect = contentRect(viewerCam.video, viewerStage, "viewer-box");
  if (!rect) return;
  Object.assign(viewerCam.style, {
    position: "absolute",
    left: rect.x + "px", top: rect.y + "px",
    width: rect.w + "px", height: rect.h + "px",
  });
}

function openViewer(nvr, channel) {
  const key = camKey(nvr.id, channel);
  viewerTitle.textContent =
    `${nvr.name} · CH ${String(channel).padStart(2, "0")} · ${nvr.host}`;
  viewerCam = document.createElement("cam-stream");
  viewerCam.controls = true;
  viewerStage.insertBefore(viewerCam, viewerCanvas);
  viewerCam.src = `api/ws?src=${key}_main`;
  state.viewerKey = key;
  viewer.hidden = false;
  layoutViewer();
  // in native mode the ratio comes from the stream — relayout when known
  setTimeout(() => {
    viewerCam?.video?.addEventListener("loadedmetadata", layoutViewer);
    layoutViewer();
  }, 100);
}

window.addEventListener("resize", layoutViewer);

function closeViewer() {
  viewer.hidden = true;
  state.viewerKey = null;
  if (viewerCam) { viewerCam.remove(); viewerCam = null; }
  const ctx = viewerCanvas.getContext("2d");
  ctx.clearRect(0, 0, viewerCanvas.width, viewerCanvas.height);
}

document.getElementById("viewer-close").addEventListener("click", closeViewer);
viewerStage.addEventListener("click", (e) => {
  if (e.target === viewerStage) closeViewer();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !viewer.hidden) closeViewer();
});

init();
