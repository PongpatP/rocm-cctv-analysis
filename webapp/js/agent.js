/* Investigator — a chat that searches the activity record.
 *
 * Left rail: conversations (server-side, shared). Middle: the thread. Right:
 * evidence the agent pulled — suspect cards, behaviour snapshots, recording
 * clips — defaulting to a live camera. The server (siglip /agent/*) holds the
 * conversation memory; this file holds only which conversation is open. */

import { VideoRTC } from "../video-rtc.js";
import { classColor, iouDedupe, latestBatch } from "./boxes.js";
import "./nav.js";

const API = "/api/reid";
const j = (p, opt) => fetch(API + p, opt).then((r) => r.json());
const post = (p, b) => j(p, { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(b || {}) });

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
/* descriptions are stored as "[colour] man, grey t-shirt …" — the leading tag-mode
   marker is for the pipeline, not for a reader. */
const cleanDesc = (s) => String(s ?? "").replace(/^\s*\[[a-z ]+\]\s*/i, "");
const when = (ms) => new Date(+ms).toLocaleString([], {
  month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });

/* ---- live camera element (stretch-fill so detection boxes map linearly) --- */
class InvCam extends VideoRTC {
  constructor() { super(); this.mode = "mse"; this.media = "video";
    this.visibilityThreshold = 0.1; }
  oninit() { super.oninit(); this.video.muted = true; this.video.autoplay = true;
    this.video.style.objectFit = "fill"; }
}
customElements.define("inv-cam", InvCam);

/* ---- state --------------------------------------------------------------- */
const state = { conv: null, cameras: [], busy: false, showBoxes: true,
  evidence: [], evView: "cameras" };
const el = (id) => document.getElementById(id);

/* ---- detection overlay: the same boxes the CCTV page draws, over each live
 * evidence tile. Fed by the bridge WebSocket, keyed by camera. Drawing this on
 * the live evidence is the standard the owner asked for. */
const detections = new Map();   // camera -> {records, at}
function connectDetections() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type !== "detections") return;
    const now = performance.now();
    const byCam = new Map();
    for (const r of m.records) {
      if (!byCam.has(r.camera_id)) byCam.set(r.camera_id, []);
      byCam.get(r.camera_id).push(r);
    }
    for (const [cam, records] of byCam)
      detections.set(cam, { records: iouDedupe(latestBatch(records)), at: now });
  };
  ws.onclose = () => setTimeout(connectDetections, 3000);
  ws.onerror = () => ws.close();
}

function drawBoxes() {
  const now = performance.now();
  for (const wrap of document.querySelectorAll(".ev-live")) {
    const canvas = wrap.querySelector(".ev-anno");
    const cam = wrap.dataset.cam;
    if (!canvas) continue;
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const d = detections.get(cam);
    if (!state.showBoxes || !d || now - d.at > 2000) continue;
    ctx.lineWidth = 2; ctx.font = "11px ui-monospace, monospace"; ctx.textBaseline = "top";
    for (const r of d.records) {
      const b = r.bounding_box; if (!b) continue;
      const fw = r.frame?.width || 960, fh = r.frame?.height || 544;
      const x = (b.x / fw) * canvas.width, y = (b.y / fh) * canvas.height;
      const w = (b.w / fw) * canvas.width, h = (b.h / fh) * canvas.height;
      const col = classColor(r.class_name);
      ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
      const label = `${r.class_name} ${(r.confidence * 100) | 0}%`;
      const tw = ctx.measureText(label).width + 8, ly = y >= 15 ? y - 15 : y + 1;
      ctx.fillStyle = col; ctx.fillRect(x - 1, ly, tw, 15);
      ctx.fillStyle = "#111"; ctx.fillText(label, x + 3, ly + 2);
    }
  }
  requestAnimationFrame(drawBoxes);
}

/* ---- camera list (for the default evidence view) ------------------------- */
async function loadCameras() {
  try {
    const cfg = await (await fetch("config.json")).json();
    for (const nvr of cfg.nvrs || []) {
      const skip = new Set(nvr.skip || []);
      for (let ch = 1; ch <= (nvr.channels || 0); ch++)
        if (!skip.has(ch)) state.cameras.push(`${nvr.id}_ch${String(ch).padStart(2, "0")}`);
    }
  } catch { /* leave empty; evidence just won't show a default camera */ }
}

/* ---- evidence panel ------------------------------------------------------ */
function liveCamera(key, opts = {}) {
  const wrap = document.createElement("div");
  wrap.className = "ev-live";
  wrap.dataset.cam = key;
  const cam = document.createElement("inv-cam");
  cam.src = `api/ws?src=${key}_sub`;
  wrap.appendChild(cam);

  // (the scene-segmentation overlay lived here — it needed the calib/track23
  // service, which does not exist on the AMD server)

  // detection-box canvas (drawn by the rAF loop)
  const canvas = document.createElement("canvas");
  canvas.className = "ev-anno";
  wrap.appendChild(canvas);

  wrap.appendChild(Object.assign(document.createElement("span"),
    { className: "ev-live__tag", textContent: key }));
  return wrap;
}

/* Evidence is shown in switchable views (Cameras · People · Events · Video), so
 * a search that returns twenty snapshots is browsable instead of a wall of tiny
 * slivers. Cards are large and the panel scrolls; duplicates are dropped. */
const EV_VIEWS = [
  ["cameras", "Cameras"], ["people", "People"], ["events", "Events"], ["video", "Video"],
  ["charts", "Chart"],
];

function groupEvidence(list) {
  const g = { people: [], events: [], video: [], cameras: [], charts: [] };
  const seen = new Set();
  for (const e of list || []) {
    if (e.type === "verification") continue;   // rendered as a badge, not a card
    if (e.type === "chart") { g.charts.push(e); continue; }  // never dedup
    const k = JSON.stringify([e.type, e.gid, e.snapshot, e.url, e.camera]);
    if (seen.has(k)) continue;
    seen.add(k);
    ({ person: g.people, behavior_snapshot: g.events,
       recording: g.video, live_camera: g.cameras }[e.type] || []).push(e);
  }
  return g;
}

/* Resolve a camera + moment to its 60 s recording and PLAY IT IN THE PANEL — a
 * "Video" view, not a browser jump. Instant: the recording list is read
 * client-side. */
async function openClip(camera, tsMs) {
  const d = new Date(+tsMs);
  const p2 = (n) => String(n).padStart(2, "0");
  const date = `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}`;
  const want = `${date}_${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}`;
  let names = [];
  try {
    const list = await (await fetch(`/api/recordings/${camera}?date=${date}`)).json();
    names = (Array.isArray(list) ? list : []).map((f) => f.file).filter(Boolean).sort();
  } catch { /* handled below */ }
  let chosen = null;
  for (const n of names) if (n.slice(0, 15) <= want) chosen = n;
  if (!chosen && names.length) chosen = names[0];
  if (!chosen) { addBubble("assistant", "No recording was kept for that moment."); return; }
  const m = chosen.match(/(\d{4})(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)/);
  const startMs = m ? new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime() : +tsMs;
  const rec = { type: "recording", camera, startMs, seekMs: +tsMs,
                url: `/api/recordings/file/${camera}/${chosen}`,
                when: new Date(+tsMs).toLocaleString([], { hour: "2-digit",
                  minute: "2-digit", second: "2-digit" }) };
  state.evidence = [rec, ...state.evidence.filter(
    (e) => !(e.type === "recording" && e.url === rec.url))];
  state.evView = "video";
  renderEvidence();
}

/* Ask the investigator to profile a person — where they went, what they did,
 * when they arrived and left — as a real chat turn so it reads in context. */
function askAboutPerson(gid, desc) {
  const who = desc ? `${cleanDesc(desc)} (G${gid})` : `G${gid}`;
  send(`Tell me more about this person — ${who}. Where did they go, what did they `
     + `do, and when did they arrive and leave?`);
}

function personCard(e) {
  const d = document.createElement("div");
  d.className = "ev-person" + (e.flagged ? " ev-person--flag" : "");
  d.innerHTML = `
    <a class="ev-person__link" href="${e.href || "#"}">
      ${e.crop ? `<img loading="lazy" src="${API}/reid/crop?p=${encodeURIComponent(e.crop)}" alt="">`
               : `<div class="ev-person__noimg">no crop</div>`}
      <div class="ev-person__body">
        <div class="ev-person__id">${e.name ? esc(e.name) + " · " : ""}G${e.gid}${e.flagged ? ' <span class="ev-flag">flagged</span>' : ""}</div>
        <div class="ev-person__desc">${esc(cleanDesc(e.description) || "no description")}</div>
      </div>
    </a>
    <div class="ev-person__actions">
      <button class="ev-person__more">＋ more about this person</button>
      <button class="ev-person__follow">🎞 Where did they go</button>
    </div>`;
  d.querySelector(".ev-person__more").onclick = () => askAboutPerson(e.gid, e.description);
  // Retrospective journey (owner's call): walking the recorded clips of where
  // this person went beats chasing a live lock that mostly says "Locating…".
  d.querySelector(".ev-person__follow").onclick = () =>
    send(`Follow G${e.gid}'s journey through the building and show me the `
       + `video clips of where they walked, in order.`);
  return d;
}

/* Inline live-follow, right inside the person card — no new page. The shared
 * `detections` map (fed by the /ws feed) already carries a cross-camera
 * global_id, so we find the camera the target is on and show it, auto-switching
 * as they move. One rAF loop paints every open follow. */
const follows = new Map();   // gid -> {stage, canvas, status, camEl, cameraShown, at}
function toggleFollow(gid, card, btn) {
  const area = card.querySelector(".ev-follow");
  if (!area) return;               // person cards no longer embed a live stage
  if (follows.has(gid)) {
    const f = follows.get(gid);
    if (f.camEl) f.camEl.remove();
    follows.delete(gid);
    area.hidden = true; btn.classList.remove("on");
    return;
  }
  area.hidden = false; btn.classList.add("on");
  follows.set(gid, {
    stage: area.querySelector(".ev-follow__stage"),
    canvas: area.querySelector(".ev-follow__anno"),
    status: area.querySelector(".ev-follow__status"),
    camEl: null, cameraShown: null, at: 0 });
}

function drawFollows() {
  const now = performance.now();
  for (const [gid, f] of follows) {
    let cam = null, box = null, frame = null;
    for (const [c, d] of detections) {
      if (now - d.at > 2500) continue;
      for (const r of d.records) {
        if (r.class_name === "person" && +r.global_id === gid) {
          cam = c; box = r.bounding_box; frame = r.frame; break;
        }
      }
      if (cam) break;
    }
    if (cam) {
      f.at = now;
      if (f.cameraShown !== cam) {
        f.cameraShown = cam;
        if (f.camEl) f.camEl.remove();
        const c = document.createElement("inv-cam");
        c.src = `api/ws?src=${cam}_sub`;
        f.stage.prepend(c); f.camEl = c;
      }
      f.status.textContent = "🔴 Following · " + cam;
      const cv = f.canvas, ctx = cv.getContext("2d");
      cv.width = cv.clientWidth; cv.height = cv.clientHeight;
      ctx.clearRect(0, 0, cv.width, cv.height);
      if (box) {
        const fw = frame?.width || 960, fh = frame?.height || 544;
        const x = (box.x / fw) * cv.width, y = (box.y / fh) * cv.height;
        const w = (box.w / fw) * cv.width, h = (box.h / fh) * cv.height;
        ctx.lineWidth = 3; ctx.strokeStyle = "#d92b2b"; ctx.strokeRect(x, y, w, h);
      }
    } else {
      const cv = f.canvas;
      if (cv.width) cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
      if (now - f.at > 3000)
        f.status.textContent = "Not on any camera right now — waiting…";
    }
  }
  requestAnimationFrame(drawFollows);
}

function eventCard(e) {
  const d = document.createElement("div");
  d.className = "ev-event" + (e.suspicious ? " ev-event--flag" : "");
  d.innerHTML = `
    <div class="ev-event__imgwrap">
      <img src="${API}/behavior/snapshot?p=${encodeURIComponent(e.snapshot)}" alt="">
      <button class="ev-event__play" title="Play the video of this moment here">▶ video</button>
    </div>
    <div class="ev-event__meta">${esc(e.camera)} · ${when(e.ts_ms)}${e.gid ? ` · G${e.gid}` : ""}</div>
    <div class="ev-event__act">${esc(e.activity || "")}</div>
    ${e.gid ? '<button class="ev-event__more">＋ more about this person</button>' : ""}`;
  d.querySelector(".ev-event__play").onclick = () => openClip(e.camera, +e.ts_ms);
  const more = d.querySelector(".ev-event__more");
  if (more) more.onclick = () => askAboutPerson(e.gid, e.activity);
  return d;
}

/* Recorded clip WITH detection boxes: the stored boxes for the clip's window are
 * fetched once and drawn in sync with the video's current time — the same boxes
 * the live view shows, replayed. */
function videoCard(e) {
  const d = document.createElement("div");
  d.className = "ev-event";
  d.innerHTML = `
    <div class="ev-rec__wrap">
      <video controls preload="metadata" src="${e.url}"></video>
      <canvas class="ev-rec__anno"></canvas>
    </div>
    <div class="ev-event__meta">${e.place ? esc(e.place) + " · " : ""}${esc(e.camera)} · ${esc(e.when || "")}</div>`;
  const video = d.querySelector("video"), canvas = d.querySelector(".ev-rec__anno");
  if (e.seekMs && e.startMs) video.addEventListener("loadedmetadata", () => {
    try { video.currentTime = Math.max(0, (e.seekMs - e.startMs) / 1000); } catch {}
  }, { once: true });

  let boxes = [];
  if (e.startMs) fetch(`/api/detections?camera=${e.camera}&start_ms=${e.startMs}&end_ms=${e.startMs + 65000}`)
    .then((r) => r.json()).then((rows) => { boxes = Array.isArray(rows) ? rows : []; }).catch(() => {});

  function draw() {
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.showBoxes && boxes.length && e.startMs) {
      const wall = e.startMs + video.currentTime * 1000;
      let best = Infinity, bts = null;
      for (const b of boxes) { const dt = Math.abs(b.ts - wall); if (dt < best) { best = dt; bts = b.ts; } }
      if (bts !== null && best < 700) {
        ctx.lineWidth = 2; ctx.font = "11px ui-monospace, monospace"; ctx.textBaseline = "top";
        for (const b of boxes) {
          if (b.ts !== bts) continue;
          const bb = b.bounding_box, fw = b.frame?.width || 960, fh = b.frame?.height || 544;
          const x = (bb.x / fw) * canvas.width, y = (bb.y / fh) * canvas.height;
          const w = (bb.w / fw) * canvas.width, h = (bb.h / fh) * canvas.height;
          const col = classColor(b.class_name);
          ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
          const lb = `${b.class_name} ${(b.confidence * 100) | 0}%`;
          const tw = ctx.measureText(lb).width + 8, ly = y >= 15 ? y - 15 : y + 1;
          ctx.fillStyle = col; ctx.fillRect(x - 1, ly, tw, 15);
          ctx.fillStyle = "#111"; ctx.fillText(lb, x + 3, ly + 2);
        }
      }
    }
    if (!video.paused && !video.ended) requestAnimationFrame(draw);
  }
  video.addEventListener("play", () => requestAnimationFrame(draw));
  for (const evn of ["seeked", "loadeddata", "pause"]) video.addEventListener(evn, draw);
  return d;
}

/* Interactive chart card: inline SVG (no library) with recessive gridlines,
 * one hue, top-rounded bars / line+area, thinned x labels, and a hover
 * crosshair + per-point tooltip. */
function chartCard(e) {
  // Compact viewBox ≈ panel width so 1 SVG unit ≈ 1 CSS px — text renders at
  // its intended size instead of being scaled down. Click = fullscreen zoom.
  const HUE = "#2563eb", W = 460, H = 320, P = { t: 12, r: 12, b: 46, l: 46 };
  const xs = e.x || [], vs = (e.series?.[0]?.values) || [];
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const vmax = Math.max(1, ...vs);
  const X = (i) => P.l + (xs.length < 2 ? iw / 2 : (i / (xs.length - 1)) * iw);
  const BW = Math.max(3, Math.min(44, iw / Math.max(1, xs.length) - 4));
  const BX = (i) => P.l + (i + 0.5) * (iw / Math.max(1, xs.length)) - BW / 2;
  const Y = (v) => P.t + ih - (v / vmax) * ih;

  let inner = "";
  for (let t = 0; t <= 4; t++) {                       // gridlines + y labels
    const v = (vmax / 4) * t, y = Y(v);
    inner += `<line x1="${P.l}" y1="${y}" x2="${W - P.r}" y2="${y}"
      stroke="#e5e7eb" stroke-width="1"/>
      <text x="${P.l - 8}" y="${y + 4}" text-anchor="end" font-size="12"
      fill="#4b5563">${Math.round(v)}</text>`;
  }
  const step = Math.max(1, Math.ceil(xs.length / 6)); // thinned x labels
  xs.forEach((x, i) => {
    if (i % step) return;
    const cx = e.chart === "bar" ? BX(i) + BW / 2 : X(i);
    inner += `<text x="${cx}" y="${H - P.b + 20}" text-anchor="middle"
      font-size="12" fill="#4b5563">${esc(String(x))}</text>`;
  });
  if (e.chart === "bar") {
    vs.forEach((v, i) => {
      const y = Y(v), h = P.t + ih - y, r = Math.min(3, BW / 2, h);
      inner += `<path d="M${BX(i)},${y + r} a${r},${r} 0 0 1 ${r},-${r}
        h${BW - 2 * r} a${r},${r} 0 0 1 ${r},${r} v${h - r} h${-BW} z"
        fill="${HUE}"/>`;
    });
  } else {
    const pts = vs.map((v, i) => `${X(i)},${Y(v)}`).join(" ");
    inner += `<polygon points="${P.l},${P.t + ih} ${pts} ${X(vs.length - 1)},${P.t + ih}"
      fill="${HUE}" opacity="0.12"/>
      <polyline points="${pts}" fill="none" stroke="${HUE}" stroke-width="2.5"/>`;
  }
  inner += `<line class="ev-chart__cross" y1="${P.t}" y2="${P.t + ih}"
    stroke="#9ca3af" stroke-dasharray="3,3" visibility="hidden"/>`;

  const d = document.createElement("div");
  d.className = "ev-chart";
  d.innerHTML = `<div class="ev-chart__title">${esc(e.title || "")}</div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet"
      role="img" aria-label="${esc(e.title || "chart")}">${inner}</svg>
    <div class="ev-chart__tip" hidden></div>
    <div class="ev-chart__hint">click to enlarge</div>`;
  const svg = d.querySelector("svg"), tip = d.querySelector(".ev-chart__tip");
  const cross = d.querySelector(".ev-chart__cross");
  svg.addEventListener("mousemove", (ev) => {
    const box = svg.getBoundingClientRect();
    const mx = ((ev.clientX - box.left) / box.width) * W;
    let best = 0, dist = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const cx = e.chart === "bar" ? BX(i) + BW / 2 : X(i);
      const dd = Math.abs(cx - mx);
      if (dd < dist) { dist = dd; best = i; }
    }
    const cx = e.chart === "bar" ? BX(best) + BW / 2 : X(best);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx);
    cross.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.textContent = `${xs[best]} — ${vs[best]} ${e.ylabel || ""}`;
    tip.style.left = `${(cx / W) * 100}%`;
  });
  svg.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden"); tip.hidden = true;
  });
  // fullscreen zoom: same SVG, blown up — vector, so text scales up crisply
  svg.addEventListener("click", () => {
    const z = document.createElement("div");
    z.className = "ev-chart-zoom";
    z.innerHTML = `<div class="ev-chart-zoom__card">
        <div class="ev-chart__title">${esc(e.title || "")}</div></div>`;
    z.firstElementChild.appendChild(svg.cloneNode(true));
    z.onclick = () => z.remove();
    document.body.appendChild(z);
  });
  return d;
}

/* "time · place · camera" — how a clip is named in the journey dropdown */
function clipLabel(e) {
  return [e.when, e.place, e.camera].filter(Boolean).join(" · ");
}

function renderActiveView(g) {
  const box = el("evidence");
  box.innerHTML = "";
  if (state.evView === "cameras") {
    const cams = g.cameras.length ? g.cameras
      : state.cameras.map((c) => ({ camera: c }));
    if (!cams.length) { box.innerHTML = `<div class="ev-empty">Evidence appears here as I find it.</div>`; return; }
    for (const e of cams) box.appendChild(liveCamera(e.camera, { segment: !!e.segment }));
  } else if (state.evView === "people") {
    for (const e of g.people) box.appendChild(personCard(e));
  } else if (state.evView === "events") {
    for (const e of g.events) box.appendChild(eventCard(e));
  } else if (state.evView === "video") {
    if (g.video.length > 1) {                 // #4: jump-to-event dropdown
      const sel = document.createElement("select");
      sel.className = "ev-video-jump";
      sel.append(new Option(`${g.video.length} clips — jump to…`, ""));
      g.video.forEach((e, i) => sel.append(new Option(clipLabel(e), String(i))));
      sel.onchange = () => {
        const card = box.querySelectorAll(".ev-event")[+sel.value];
        if (!card) return;
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.querySelector("video")?.play().catch(() => {});
      };
      box.appendChild(sel);
    }
    for (const e of g.video) box.appendChild(videoCard(e));
  } else if (state.evView === "charts") {
    for (const e of g.charts) box.appendChild(chartCard(e));
  }
}

function renderEvidence(list) {
  if (list && list.length) {
    state.evidence = list;
    const g0 = groupEvidence(list);
    // jump to the view the newest answer is really about (a chart wins)
    state.evView = g0.charts.length ? "charts"
      : g0.people.length ? "people" : g0.events.length ? "events"
      : g0.video.length ? "video" : "cameras";
  }
  const g = groupEvidence(state.evidence);
  const counts = { cameras: g.cameras.length, people: g.people.length,
                   events: g.events.length, video: g.video.length,
                   charts: g.charts.length };
  const tabs = el("ev-tabs");
  tabs.innerHTML = "";
  for (const [id, label] of EV_VIEWS) {
    if (id !== "cameras" && !counts[id]) continue;   // hide empty views
    const b = document.createElement("button");
    b.className = "ev-tab" + (id === state.evView ? " on" : "");
    b.textContent = label + (counts[id] ? ` ${counts[id]}` : "");
    b.onclick = () => { state.evView = id; renderEvidence(); };
    tabs.appendChild(b);
  }
  renderActiveView(g);
}

function showDefaultEvidence() {
  state.evidence = [];
  state.evView = "cameras";
  renderEvidence();
}
el("evidence-reset").onclick = showDefaultEvidence;

/* ---- thread rendering ---------------------------------------------------- */
let UI_TZ = "Asia/Bangkok";
(async () => {
  try { UI_TZ = (await (await fetch("/api/settings")).json()).timezone || UI_TZ; }
  catch { /* browser zone problems only affect display */ }
})();

function fmtTime(ms) {
  if (!ms) return "";
  try {
    const d = new Date(+ms);
    const day = new Intl.DateTimeFormat("en-GB",
      { timeZone: UI_TZ, day: "numeric", month: "short" }).format(d);
    const today = new Intl.DateTimeFormat("en-GB",
      { timeZone: UI_TZ, day: "numeric", month: "short" }).format(new Date());
    const hm = new Intl.DateTimeFormat("en-GB",
      { timeZone: UI_TZ, hour: "2-digit", minute: "2-digit" }).format(d);
    return day === today ? hm : `${day} ${hm}`;
  } catch { return ""; }
}

function verifyBadge(v) {
  if (!v || !v.checked) return "";
  const issues = esc((v.issues || []).join(" · "));
  if (v.corrected)
    return `<span class="vbadge vbadge--fix" title="${issues}">✓ Corrected by verifier</span>`;
  if (v.ok) return `<span class="vbadge vbadge--ok">✓ Verified</span>`;
  return `<span class="vbadge vbadge--bad" title="${issues}">⚠ Unverified — ${
    esc((v.issues || ["unchecked"])[0])}</span>`;
}

function bubble(role, content, verification, ts) {
  const d = document.createElement("div");
  d.className = "msg msg--" + role;
  const t = fmtTime(ts);
  d.innerHTML = `<div class="msg__body">${esc(content).replace(/\n/g, "<br>")}${
    role === "assistant" ? verifyBadge(verification) : ""}</div>${
    t ? `<span class="msg__time">${t}</span>` : ""}`;
  return d;
}

function clearThread() {
  el("thread").innerHTML = "";
}

function addBubble(role, content, verification, ts) {
  el("welcome")?.remove();
  const b = bubble(role, content, verification, ts ?? Date.now());
  el("thread").appendChild(b);
  el("thread").scrollTop = el("thread").scrollHeight;
  return b;
}

/* ---- sessions ------------------------------------------------------------ */
async function loadSessions() {
  const d = await j("/agent/conversations");
  const box = el("sessions");
  box.innerHTML = "";
  for (const c of d.conversations || []) {
    const row = document.createElement("div");
    row.className = "sess" + (c.id === state.conv ? " sess--on" : "");
    row.innerHTML = `<span class="sess__title">${esc(c.title)}</span>
      <button class="sess__x" title="Delete">✕</button>`;
    row.querySelector(".sess__title").onclick = () => openConversation(c.id);
    row.querySelector(".sess__x").onclick = async (e) => {
      e.stopPropagation();
      await fetch(`${API}/agent/conversations/${c.id}`, { method: "DELETE" });
      if (c.id === state.conv) { state.conv = null; clearThread(); showWelcome(); }
      loadSessions();
    };
    row.querySelector(".sess__title").ondblclick = async () => {
      const t = prompt("Rename conversation", c.title);
      if (t) { await fetch(`${API}/agent/conversations/${c.id}`,
        { method: "PATCH", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: t }) }); loadSessions(); }
    };
    box.appendChild(row);
  }
}

function showWelcome() {
  clearThread();
  el("thread").innerHTML = `<div class="inv-welcome" id="welcome">
      <h2>What would you like to look into?</h2>
      <p>Describe an incident and I’ll search what the building recorded.</p></div>`;
  showDefaultEvidence();
}

async function openConversation(id) {
  state.conv = id;
  const d = await j(`/agent/conversations/${id}`);
  clearThread();
  let lastEvidence = [];
  for (const m of d.messages || []) {
    const v = (m.evidence || []).find((e) => e.type === "verification");
    if (m.role === "user" || m.role === "assistant") addBubble(m.role, m.content, v, m.ts);
    if (m.evidence && m.evidence.length) lastEvidence = m.evidence;
  }
  if (lastEvidence.length) renderEvidence(lastEvidence); else showDefaultEvidence();
  loadSessions();
}

async function newConversation() {
  const d = await post("/agent/conversations", {});
  state.conv = d.id;
  showWelcome();
  await loadSessions();
  el("input").focus();
}
el("new-chat").onclick = newConversation;

/* ---- sending ------------------------------------------------------------- */
async function send(text) {
  text = (text || "").trim();
  if (!text || state.busy) return;
  if (!state.conv) { const d = await post("/agent/conversations", {}); state.conv = d.id; }
  state.busy = true;
  el("send").disabled = true;
  el("input").value = "";
  autosize();
  // show and process the message in English so reviewers can follow along
  try { const tr = await post("/translate", { text }); if (tr && tr.text) text = tr.text; } catch { /* keep original */ }
  addBubble("user", text);
  const thinking = addBubble("assistant", "…", undefined, 0);
  thinking.classList.add("msg--thinking");
  try {
    const d = await post("/agent/chat", {
      conversation_id: state.conv, message: text,
      hours: +el("hours").value,
      provider: el("model")?.value || undefined });
    thinking.remove();
    if (!d.ok) { addBubble("assistant", "Sorry — " + (d.error || "the search failed.")); }
    else {
      addBubble("assistant", d.reply, d.verification);
      renderEvidence(d.evidence);
    }
    loadSessions();
  } catch (e) {
    thinking.remove();
    addBubble("assistant", "The investigator is unreachable right now.");
  } finally {
    state.busy = false;
    el("send").disabled = false;
    el("input").focus();
  }
}

/* ---- image upload: VLM describes it, offers next-step actions ----------- */
async function handleImage(file) {
  if (!file || state.busy) return;
  const dataUrl = await new Promise((res) => {
    const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(file);
  });
  el("welcome")?.remove();
  // show the user's image as a bubble
  const ub = document.createElement("div");
  ub.className = "msg msg--user";
  ub.innerHTML = `<div class="msg__body"><img class="msg__img" src="${dataUrl}" alt=""></div>`;
  el("thread").appendChild(ub);
  const thinking = addBubble("assistant", "Looking at the image…");
  thinking.classList.add("msg--thinking");
  el("thread").scrollTop = el("thread").scrollHeight;
  state.busy = true; el("send").disabled = true;
  try {
    const d = await post("/agent/image", { image: dataUrl,
      conversation_id: state.conv || undefined });
    thinking.remove();
    if (!d.ok) { addBubble("assistant", "Sorry — " + (d.error || "couldn't read that image.")); return; }
    const b = document.createElement("div");
    b.className = "msg msg--assistant";
    const acts = (d.actions || []).map((a, i) =>
      `<button class="img-action" data-i="${i}">${esc(a.label)}</button>`).join("");
    b.innerHTML = `<div class="msg__body">` +
      `<div class="img-subject">${esc(d.subject || "")}</div>` +
      `${esc(d.caption || "")}` +
      `<div style="margin-top:8px;font-size:12px;color:var(--text-muted)">What would you like to do?</div>` +
      `<div class="img-actions">${acts}</div></div>`;
    el("thread").appendChild(b);
    b.querySelectorAll(".img-action").forEach((btn) => {
      btn.onclick = () => send(d.actions[+btn.dataset.i].prompt);
    });
    el("thread").scrollTop = el("thread").scrollHeight;
  } catch {
    thinking.remove();
    addBubble("assistant", "The image service is unreachable right now.");
  } finally { state.busy = false; el("send").disabled = false; }
}
const imgBtn = document.getElementById("img-btn");
const imgInput = document.getElementById("img-input");
if (imgBtn && imgInput) {
  imgBtn.onclick = () => imgInput.click();
  imgInput.onchange = () => { if (imgInput.files[0]) handleImage(imgInput.files[0]); imgInput.value = ""; };
}
// paste an image straight into the chat
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith("image/"));
  if (item) handleImage(item.getAsFile());
});

/* ---- compose box --------------------------------------------------------- */
const input = el("input");
function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input.value); }
});
el("compose").addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
document.addEventListener("click", (e) => {
  const b = e.target.closest(".inv-suggest button");
  if (b) send(b.dataset.q);
});

/* ---- boot ---------------------------------------------------------------- */
/* header toggle: detection boxes on/off (standard-on) */
const boxesBtn = document.getElementById("boxes-toggle");
if (boxesBtn) boxesBtn.onclick = () => {
  state.showBoxes = !state.showBoxes;
  boxesBtn.classList.toggle("on", state.showBoxes);
};

/* ---- tools discovery panel + model picker (from /agent/meta) ---------- */
async function loadMeta() {
  let meta = { tools: [], models: [] };
  try { meta = await j("/agent/meta"); } catch { /* panel just stays empty */ }
  const sel = el("model");
  if (sel) {
    sel.innerHTML = "";
    for (const m of meta.models || []) sel.append(new Option(m.name, m.id));
    if (sel.options.length <= 1) sel.classList.add("inv-model--single");
  }
  const panel = el("tools-panel");
  if (panel) {
    panel.innerHTML = `<div class="inv-tools-panel__head">What I can do</div>` +
      (meta.tools || []).map((t) => `
        <div class="inv-tool">
          <span class="inv-tool__icon">${t.icon}</span>
          <span><b>${esc(t.name)}</b><br><span class="inv-tool__desc">${esc(t.desc)}</span></span>
        </div>`).join("");
  }
  const fab = el("tools-fab");
  if (fab) fab.onclick = () => { panel.hidden = !panel.hidden; };
  document.addEventListener("click", (e) => {
    if (panel && !panel.hidden && !panel.contains(e.target) && e.target !== fab)
      panel.hidden = true;
  });
}

(async function () {
  await loadMeta();
  await loadCameras();
  showDefaultEvidence();
  await loadSessions();
  connectDetections();
  requestAnimationFrame(drawBoxes);
  requestAnimationFrame(drawFollows);
})();
