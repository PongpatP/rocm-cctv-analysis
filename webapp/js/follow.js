/* Live Follow — lock onto one Global ID and track it across cameras in real time.
 *
 * The detection WebSocket already carries a cross-camera global_id on most person
 * detections, so "which camera is the target on right now" is just a filter. When
 * the id drops (a gap between cameras) we watch the reachable neighbours and ask
 * the backend to scan them with the VLM. */

import { VideoRTC } from "../video-rtc.js";
import { classColor } from "./boxes.js";
import "./nav.js";

const API = "/api/reid";
const gid = +new URLSearchParams(location.search).get("gid") || 0;
const LOST_MS = 4000;         // no sighting this long -> lost
const el = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clean = (s) => String(s ?? "").replace(/^\s*\[[a-z ]+\]\s*/i, "");
const hhmmss = (ms) => new Date(+ms).toLocaleTimeString([], { hour12: false });

class FollowCam extends VideoRTC {
  constructor() { super(); this.mode = "mse"; this.media = "video"; }
  oninit() { super.oninit(); this.video.muted = true; this.video.autoplay = true;
    this.video.style.objectFit = "fill"; }
}
customElements.define("follow-cam", FollowCam);

const state = {
  camera: null,              // camera the target is currently on
  box: null, frame: null,    // target's latest box + frame size
  at: 0,                     // last time we saw the target
  places: {},                // camera -> place label
  neighbours: [],            // reachable cameras of the last-seen camera
  route: [],                 // [{camera, place, at}]
  lost: false,
  camEl: null,               // the big live-view element
  neighEls: new Map(),       // camera -> small neighbour tile
  reacqAt: 0,
};

/* ---- target header + route ---- */
function paintTarget(t) {
  el("target").innerHTML = `
    ${t.crop ? `<img class="follow-target__crop" src="${API}/reid/crop?p=${encodeURIComponent(t.crop)}" alt="">`
             : `<div class="follow-target__crop follow-target__crop--none">no crop</div>`}
    <div class="follow-target__body">
      <div class="follow-target__id">${t.name ? esc(t.name) + " · " : ""}G${gid}</div>
      <div class="follow-target__desc">${esc(clean(t.description) || "no description")}</div>
    </div>`;
}
function paintRoute() {
  el("route").innerHTML = state.route.slice().reverse().map((r) => `
    <div class="follow-route__stop">
      <span class="follow-route__place">${esc(r.place || r.camera)}</span>
      <span class="follow-route__time">${hhmmss(r.at)}</span>
    </div>`).join("") || `<div class="follow-route__empty">no movement yet</div>`;
}

/* ---- big live view ---- */
function showCamera(cam) {
  if (state.camEl && state.cameraShown === cam) return;
  state.cameraShown = cam;
  if (state.camEl) state.camEl.remove();
  const c = document.createElement("follow-cam");
  c.src = `api/ws?src=${cam}_sub`;
  el("stage").prepend(c);
  state.camEl = c;
  el("cam-tag").textContent = (state.places[cam] ? state.places[cam] + " · " : "") + cam;
}

function drawBox() {
  const canvas = el("anno"), ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.lost || !state.box || performance.now() - state.drawAt > 1500) {
    requestAnimationFrame(drawBox); return;
  }
  const b = state.box, fw = state.frame?.width || 960, fh = state.frame?.height || 544;
  const x = (b.x / fw) * canvas.width, y = (b.y / fh) * canvas.height;
  const w = (b.w / fw) * canvas.width, h = (b.h / fh) * canvas.height;
  ctx.lineWidth = 3; ctx.strokeStyle = "#d92b2b"; ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = "#d92b2b";
  const label = `G${gid}`;
  ctx.font = "bold 13px ui-monospace, monospace";
  const tw = ctx.measureText(label).width + 10;
  ctx.fillRect(x - 1, y - 18, tw, 18);
  ctx.fillStyle = "#fff"; ctx.fillText(label, x + 4, y - 4);
  requestAnimationFrame(drawBox);
}

/* ---- lost state: neighbour grid + VLM re-acquire ---- */
function enterLost() {
  state.lost = true;
  el("lost").hidden = false;
  el("status").textContent = `Lost — last seen ${state.places[state.lastCam] || state.lastCam || "?"}`;
  el("status").className = "follow-status follow-status--lost";
  const box = el("neighbours");
  box.innerHTML = "";
  state.neighEls.clear();
  for (const n of state.neighbours) {
    const tile = document.createElement("div");
    tile.className = "follow-neigh";
    tile.dataset.cam = n.camera;
    const c = document.createElement("follow-cam");
    c.src = `api/ws?src=${n.camera}_sub`;
    tile.appendChild(c);
    tile.appendChild(Object.assign(document.createElement("span"),
      { className: "follow-neigh__tag", textContent: n.place || n.camera }));
    box.appendChild(tile);
    state.neighEls.set(n.camera, tile);
  }
  reacquire();
}
function exitLost() {
  if (!state.lost) return;
  state.lost = false;
  el("lost").hidden = true;
  el("status").className = "follow-status";
}
async function reacquire() {
  if (!state.lost) return;
  const now = performance.now();
  if (now - state.reacqAt > 5000) {
    state.reacqAt = now;
    el("lost-head").textContent = "Lost — scanning neighbours with AI…";
    try {
      const r = await fetch(`${API}/reid/reacquire/${gid}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ last_camera: state.lastCam }) }).then((x) => x.json());
      for (const [, tile] of state.neighEls) tile.classList.remove("follow-neigh--hit");
      if (r.candidates && r.candidates.length) {
        const c = r.candidates[0];
        state.neighEls.get(c.camera)?.classList.add("follow-neigh--hit");
        el("lost-head").textContent = `AI thinks they are on ${c.place || c.camera} — confirming…`;
        showCamera(c.camera);              // point the big view at the likely camera
      } else {
        el("lost-head").textContent = "Lost — scanning neighbours with AI…";
      }
    } catch { /* keep watching */ }
  }
  if (state.lost) setTimeout(reacquire, 1500);
}

/* ---- the live feed ---- */
function onRecords(records) {
  let hit = null;
  for (const r of records) {
    if (r.class_name === "person" && +r.global_id === gid) { hit = r; break; }
  }
  if (hit) {
    state.at = Date.now();
    state.drawAt = performance.now();
    state.box = hit.bounding_box; state.frame = hit.frame;
    const cam = hit.camera_id;
    if (cam !== state.camera) {           // moved to a new camera
      state.camera = cam; state.lastCam = cam;
      state.route.push({ camera: cam, place: state.places[cam] || "", at: Date.now() });
      paintRoute();
      refreshContext(cam);                // fresh neighbours + place for this camera
    }
    showCamera(cam);
    exitLost();
    el("status").textContent = `Following · ${state.places[cam] || cam}`;
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "detections") onRecords(m.records || []);
  };
  ws.onclose = () => setTimeout(connect, 3000);
  ws.onerror = () => ws.close();
}

/* lost-watchdog: if we have not seen the target for LOST_MS, go to lost state */
setInterval(() => {
  if (gid && state.at && !state.lost && Date.now() - state.at > LOST_MS) enterLost();
}, 1000);

/* ---- context (target, route places, neighbours) from the backend ---- */
async function refreshContext(cam) {
  try {
    const d = await fetch(`${API}/reid/locate/${gid}`).then((x) => x.json());
    if (d.description !== undefined) paintTarget(d);
    for (const r of d.route || []) state.places[r.camera] = r.place;
    for (const n of d.neighbours || []) state.places[n.camera] = n.place;
    if (d.last_camera) state.places[d.last_camera] = d.last_place;
    if (!cam) {
      // initial load: seed route + last camera
      state.lastCam = d.last_camera;
      state.route = (d.route || []).map((r) => ({ camera: r.camera, place: r.place, at: r.first_ts }));
      paintRoute();
    }
    state.neighbours = d.neighbours || [];
  } catch { /* backend may be restarting */ }
}

/* ---- boot ---- */
if (!gid) {
  el("status").textContent = "No target — open with ?gid=<person id>";
} else {
  el("status").textContent = "Locating…";
  refreshContext(null);
  connect();
  requestAnimationFrame(drawBox);
}
