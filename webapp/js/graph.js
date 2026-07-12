/* Camera graph editor. Each camera is a CARD: live thumbnail (framed) + three
 * stacked boxes — (1) VLM place caption, (2) LLM connectivity summary, (3) a
 * free human note. Cards live in a FIXED GRID of blocks inside their floor
 * lane: dragging snaps a card into a block, a card can never leave its own
 * floor (use the floor selector for that), and dropping onto an occupied block
 * swaps the two. Edges are manual reachability links (click 🔗 on two cards)
 * and feed the cross-camera ReID reachability gate. No auto/statistical links. */
const SVGNS = "http://www.w3.org/2000/svg";
const stage = document.getElementById("stage");
const edges = document.getElementById("edges");
const _hint = document.getElementById("hint");
if (_hint) _hint.dataset.def = _hint.textContent;

const CARD_W = 210, CARD_H = 330, HGAP = 20, VGAP = 26, LABEL_H = 34, PAD = 16;
const MIN_COLS = 4, MIN_ROWS = 1;     // the grid never shrinks below this
const CELL_W = CARD_W + HGAP;

const state = { nodes: new Map(), edges: new Set(), pending: null, cfg: {},
  calibFloors: {}, nFloors: 8, bands: new Map(), cellH: CARD_H + VGAP };

// The grid is ELASTIC and PER FLOOR: a floor's columns/rows exist only because
// one of ITS cards occupies them. Dropping a card one block past that floor's
// right/bottom edge grows it; vacating the last column/row shrinks it back
// (never below MIN_COLS/MIN_ROWS). One busy floor never pins the others open.
function colsOf(floor) {
  let m = 0;
  for (const n of state.nodes.values())
    if (flr(n) === floor && n.col != null) m = Math.max(m, n.col + 1);
  return Math.max(MIN_COLS, m);
}
function rowsOf(floor) {
  let m = 0;
  for (const n of state.nodes.values())
    if (flr(n) === floor && n.row != null) m = Math.max(m, n.row + 1);
  return Math.max(MIN_ROWS, m);
}

// Floors have ONE source of truth: calib's floors.json (the same store the 3D
// Editor writes). This page reads AND writes it, so a building with no floor
// plan can be organised entirely from here.
const flr = (n) => state.calibFloors[n.cam] || "1F";
const setFloorRemote = (cam, label) =>
  fetch("/api/calib/floors", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ floors_patch: { [cam]: label } }) });
const byFloor = (a, b) => (parseInt(a) || 99) - (parseInt(b) || 99) || String(a).localeCompare(b);
function floorLabels() {
  const set = new Set();
  for (let i = 1; i <= (state.nFloors || 8); i++) set.add(i + "F");
  for (const n of state.nodes.values()) set.add(flr(n));
  return [...set].sort(byFloor);
}

const api = async (p, opts) => (await fetch("/api/reid" + p, opts)).json();
const post = (p, body) => api(p, { method: "POST",
  headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const ekey = (a, b) => [a, b].sort().join("|");
const frameURL = (cam) => `/api/frame.jpeg?src=${cam}_sub&t=${Date.now()}`;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

async function deriveCameras() {
  try {
    const d = await (await fetch("/config.json")).json();
    const ids = [];
    for (const nvr of (d.nvrs || [])) {
      const skip = new Set(nvr.skip || []);
      for (let ch = 1; ch <= nvr.channels; ch++)
        if (!skip.has(ch)) ids.push(`${nvr.id}_ch${String(ch).padStart(2, "0")}`);
    }
    return ids;
  } catch { return []; }
}

async function load() {
  const warn = [];
  const cams = await deriveCameras();
  if (!cams.length) warn.push("config.json gave 0 cameras");
  let graph = { nodes: [], edges: [] }, prov = { cfg: {} };
  try { graph = await api("/graph"); } catch { warn.push("graph fetch failed"); }
  try { prov = await api("/providers"); } catch { warn.push("providers fetch failed"); }
  try {
    const fl = await (await fetch("/api/calib/floors")).json();
    state.calibFloors = fl.floors || {};
    if (fl.n_floors) state.nFloors = fl.n_floors;
  } catch { /* per-node default */ }
  initFloorCount();
  state.cfg = prov.cfg || {};
  const byCam = new Map((graph.nodes || []).map((n) => [n.camera, n]));
  const mk = (cam, g) => ({ cam,
    col: Number.isInteger(g.slot_col) ? g.slot_col : null,
    row: Number.isInteger(g.slot_row) ? g.slot_row : null,
    vlm_caption: g.vlm_caption, confirmed_label: g.confirmed_label,
    human_comment: g.human_comment, connectivity_summary: g.connectivity_summary });
  for (const c of cams) state.nodes.set(c, mk(c, byCam.get(c) || {}));
  for (const n of (graph.nodes || []))
    if (!state.nodes.has(n.camera)) state.nodes.set(n.camera, mk(n.camera, n));
  state.edges = new Set((graph.edges || []).map(([a, b]) => ekey(a, b)));
  assignSlots(true);
  fillProviders();
  fullRender();
  if (warn.length) {
    _hint.textContent = "⚠ " + warn.join("; ") + " — check sign-in / ReID service.";
    _hint.style.color = "#d6336c";
  }
}

// ---- slots ------------------------------------------------------------------
// Every card owns a (col,row) block inside its floor. Slots — not pixels — are
// the stored truth, so a floor above gaining a row never shifts anyone else.
function assignSlots(persistNew) {
  const groups = new Map();
  for (const n of state.nodes.values()) {
    const f = flr(n);
    if (!groups.has(f)) groups.set(f, []);
    groups.get(f).push(n);
  }
  const fresh = [];
  for (const arr of groups.values()) {
    arr.sort((a, b) => a.cam.localeCompare(b.cam));
    const taken = new Set();
    for (const n of arr) {                       // keep valid, non-clashing slots
      if (Number.isInteger(n.col) && Number.isInteger(n.row)
          && n.col >= 0 && n.row >= 0
          && !taken.has(n.col + "," + n.row)) { taken.add(n.col + "," + n.row); continue; }
      n.col = null; n.row = null;
    }
    const cols = colsOf(flr(arr[0]));
    for (const n of arr) {                       // fill the rest row-major
      if (n.col != null) continue;
      let c = 0, r = 0;
      while (taken.has(c + "," + r)) { if (++c >= cols) { c = 0; r++; } }
      n.col = c; n.row = r; taken.add(c + "," + r);
      fresh.push(n);
    }
  }
  if (persistNew) for (const n of fresh)
    post("/graph/slot", { camera: n.cam, col: n.col, row: n.row }).catch(() => {});
}

function freeSlot(floor) {
  const taken = new Set([...state.nodes.values()].filter((n) => flr(n) === floor)
    .map((n) => n.col + "," + n.row));
  const cols = colsOf(floor);
  let c = 0, r = 0;
  while (taken.has(c + "," + r)) { if (++c >= cols) { c = 0; r++; } }
  return { col: c, row: r };
}

// ---- building floor count (shared with the 3D Editor) -----------------------
function initFloorCount() {
  const el = document.getElementById("n-floors");
  if (!el) return;
  el.value = state.nFloors;
  el.onchange = async () => {
    const v = parseInt(el.value, 10);
    if (!(v >= 1 && v <= 60)) { el.value = state.nFloors; return; }
    try {
      const r = await fetch("/api/calib/floors", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_floors: v }) });
      if (!r.ok) throw new Error("calib rejected it");
      state.nFloors = v;
      fullRender();                    // floor dropdowns now offer 1F..vF
      flash(`✓ building set to ${v} floors — synced with 3D Editor`);
    } catch (e) { el.value = state.nFloors; flash(`⚠ ${e.message || e}`); }
  };
}

// ---- providers --------------------------------------------------------------
function fillProviders() {
  const vlm = document.getElementById("vlm-provider");
  const llm = document.getElementById("llm-provider");
  vlm.innerHTML = ""; llm.innerHTML = "";
  // must match the providers that actually exist in siglip/providers.py
  // (Spark box is gone; local-gemma is the on-box default)
  for (const [v, t] of [["local-gemma", "On this machine"],
                        ["anthropic", "Anthropic (cloud)"]])
    vlm.add(new Option(t, v, false, v === state.cfg.vlm_provider));
  // "Google (cloud)" (gemma) is intentionally NOT offered: no API key is set for
  // it, so picking it made ✨ auto-label fail with "no gemma_api_key set". The
  // on-box model and Anthropic both work. If cfg still points at gemma, fall the
  // selection back to the on-box model so the buttons keep working.
  const llmSel = ["gemma", ""].includes(state.cfg.llm_provider)
    ? "local-gemma" : state.cfg.llm_provider;
  for (const [v, t] of [["local-gemma", "On this machine"],
                        ["anthropic", "Anthropic (cloud)"]])
    llm.add(new Option(t, v, false, v === llmSel));
  vlm.onchange = () => saveProvider("vlm_provider", vlm.value);
  llm.onchange = () => saveProvider("llm_provider", llm.value);
}
async function saveProvider(field, value) {
  state.cfg[field] = value;
  try {
    const r = await post("/providers", { [field]: value });
    if (r && r.cfg) state.cfg = r.cfg;
    flash(`✓ ${field === "vlm_provider" ? "VLM" : "LLM"} set to ${value} (saved)`);
  } catch (e) { flash(`⚠ could not save provider: ${e.message || e}`); }
}

// ---- render -----------------------------------------------------------------
function fullRender() {
  [...stage.querySelectorAll(".gnode, .floor-band, .slot-ghost")].forEach((e) => e.remove());
  for (const n of state.nodes.values()) { n.el = buildCard(n); stage.appendChild(n.el); }
  const ghost = document.createElement("div");
  ghost.className = "slot-ghost"; ghost.hidden = true; ghost.id = "ghost";
  stage.appendChild(ghost);
  stat();
  requestAnimationFrame(placeAll);
}

function placeAll() {
  let maxH = CARD_H;                       // uniform block height = tallest card
  for (const n of state.nodes.values()) if (n.el) maxH = Math.max(maxH, n.el.offsetHeight);
  const cellH = state.cellH = maxH + VGAP;
  const floors = [...new Set([...state.nodes.values()].map(flr))].sort(byFloor);
  state.bands = new Map();
  let y = PAD, widest = MIN_COLS;
  for (const f of floors) {                // exactly as many rows/cols as are occupied
    const rows = rowsOf(f), cols = colsOf(f);
    widest = Math.max(widest, cols);
    state.bands.set(f, { top: y, rows, cols, height: LABEL_H + rows * cellH,
      width: HGAP * 2 + cols * CELL_W });
    y += LABEL_H + rows * cellH + PAD;
  }
  for (const n of state.nodes.values()) {
    const b = state.bands.get(flr(n));
    n.el.style.left = (HGAP + n.col * CELL_W) + "px";
    n.el.style.top = (b.top + LABEL_H + n.row * cellH) + "px";
  }
  const W = HGAP * 2 + (widest + 1) * CELL_W, H = y + cellH;   // one cell of slack
  stage.style.width = W + "px"; stage.style.height = H + "px";
  edges.setAttribute("width", W); edges.setAttribute("height", H);
  for (const [f, b] of state.bands) {      // floor lanes only — the grid is invisible
    const band = document.createElement("div");
    band.className = "floor-band";
    band.style.top = b.top + "px"; band.style.height = b.height + "px";
    band.style.width = b.width + "px";
    band.innerHTML = `<span class="floor-band__label">Floor ${f}</span>`;
    stage.appendChild(band);
  }
  drawEdges();
}

// ---- one card ---------------------------------------------------------------
const boxText = (v) => (v ? v : "—");
function buildCard(n) {
  const el = document.createElement("div");
  n.el = el;
  el.className = "gnode" + (n.confirmed_label ? " gnode--labeled" : "")
    + (n.cam === state.pending ? " gnode--pending" : "");
  const cur = flr(n);
  const floorOpts = floorLabels().map((f) =>
    `<option value="${f}" ${f === cur ? "selected" : ""}>${f}</option>`).join("");
  el.innerHTML = `
    <div class="gnode__bar" data-role="drag">
      <span class="gnode__cam">${n.cam}</span>
      <select class="gnode__floor" title="floor">${floorOpts}</select>
      <button class="gnode__link" title="link to another camera">🔗</button>
    </div>
    <img class="gnode__img" src="${frameURL(n.cam)}" alt="${n.cam}" loading="lazy">
    <div class="gnode__labelrow">
      <input class="gnode__label" placeholder="short label (you)" value="${(n.confirmed_label || "").replace(/"/g, "&quot;")}">
      <button class="gnode__mini" data-lbl="gen" title="auto-generate from the Place caption + Connectivity">✨</button>
      <button class="gnode__mini" data-lbl="fix" title="fix spelling / grammar">Aa</button>
    </div>
    <div class="gnode__box">
      <div class="gnode__boxhead"><span>Place · <b>VLM</b></span><button class="gbtn gbtn--sm" data-gen="vlm">Generate</button></div>
      <div class="gnode__text ${n.vlm_caption ? "" : "gnode__text--empty"}" data-f="vlm">${boxText(n.vlm_caption)}</div>
    </div>
    <div class="gnode__box">
      <div class="gnode__boxhead"><span>Connectivity · <b>LLM</b></span><button class="gbtn gbtn--sm" data-gen="llm">Generate</button></div>
      <div class="gnode__text ${n.connectivity_summary ? "" : "gnode__text--empty"}" data-f="conn">${boxText(n.connectivity_summary)}</div>
    </div>
    <div class="gnode__box">
      <div class="gnode__boxhead"><span>Note · you</span></div>
      <textarea class="gnode__comment" placeholder="extra description">${n.human_comment || ""}</textarea>
    </div>
    <div class="gnode__note" data-role="note"></div>`;

  const linkBtn = el.querySelector(".gnode__link");
  if (n.cam === state.pending) linkBtn.classList.add("on");
  linkBtn.onclick = (e) => { e.stopPropagation(); toggleConnect(n.cam); };
  el.querySelector(".gnode__floor").onchange = async (e) => {
    e.stopPropagation();
    const label = e.target.value;
    try {
      const r = await setFloorRemote(n.cam, label);   // -> calib floors.json
      if (!r.ok) throw new Error("calib rejected the floor");
      state.calibFloors[n.cam] = label;
      const s = freeSlot(label);                      // first free block up there
      n.col = s.col; n.row = s.row;
      await post("/graph/slot", { camera: n.cam, col: n.col, row: n.row });
      fullRender();
      flash(`✓ ${n.cam} moved to ${label} — synced with 3D Editor`);
    } catch (err) {
      e.target.value = flr(n);
      flash(`⚠ could not save floor: ${err.message || err}`);
    }
  };
  const img = el.querySelector(".gnode__img");
  img.onclick = (e) => { e.stopPropagation(); img.src = frameURL(n.cam); };
  const labelIn = el.querySelector(".gnode__label");
  labelIn.onchange = () => {
    n.confirmed_label = labelIn.value;
    post("/graph/node", { camera: n.cam, field: "confirmed_label", value: labelIn.value });
    el.classList.toggle("gnode--labeled", !!labelIn.value);
  };
  const applyLabel = (v) => {
    labelIn.value = v; n.confirmed_label = v;
    el.classList.toggle("gnode--labeled", !!v);
  };
  el.querySelector('[data-lbl="gen"]').onclick = async (e) => {
    e.stopPropagation(); note(el, "writing label…");
    // try the chosen provider; if it fails (e.g. a cloud model with no key),
    // fall back to the on-box model so the button always produces a label.
    const chosen = document.getElementById("llm-provider").value || "local-gemma";
    try {
      let r = await post("/label/generate", { camera: n.cam, provider: chosen });
      if (!r.ok && chosen !== "local-gemma")
        r = await post("/label/generate", { camera: n.cam, provider: "local-gemma" });
      if (r.ok) { applyLabel(r.label); note(el, "label written"); }
      else note(el, "failed: " + (r.error || "?"));
    } catch (err) { note(el, "failed: " + (err.message || err)); }
  };
  el.querySelector('[data-lbl="fix"]').onclick = async (e) => {
    e.stopPropagation();
    if (!labelIn.value.trim()) { note(el, "type a label first"); return; }
    note(el, "correcting…");
    const r = await post("/label/fix", { camera: n.cam, text: labelIn.value,
      provider: document.getElementById("llm-provider").value });
    if (r.ok) { applyLabel(r.label); note(el, r.changed ? "corrected" : "already correct"); }
    else note(el, "failed: " + (r.error || "?"));
  };

  const cmt = el.querySelector(".gnode__comment");
  cmt.onchange = () => {
    n.human_comment = cmt.value;
    post("/graph/node", { camera: n.cam, field: "human_comment", value: cmt.value });
    note(el, "note saved");
  };
  el.querySelector('[data-gen="vlm"]').onclick = (e) => { e.stopPropagation(); genVLM(n, el); };
  el.querySelector('[data-gen="llm"]').onclick = (e) => { e.stopPropagation(); genLLM(n, el); };
  attachDrag(el, n);
  return el;
}
const note = (el, msg) => { const nn = el.querySelector('[data-role="note"]'); if (nn) nn.textContent = msg; };

// ---- drag: free move, snap into a block of the SAME floor -------------------
function attachDrag(el, n) {
  const bar = el.querySelector('[data-role="drag"]');
  let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0, target = null;
  const ghost = () => document.getElementById("ghost");

  bar.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".gnode__floor, .gnode__link")) return;
    dragging = true; sx = e.clientX; sy = e.clientY;
    ox = el.offsetLeft; oy = el.offsetTop;
    el.classList.add("gnode--dragging"); el.style.zIndex = 6;
    bar.setPointerCapture(e.pointerId);
  });
  bar.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const left = ox + (e.clientX - sx), top = oy + (e.clientY - sy);
    el.style.left = left + "px"; el.style.top = top + "px";
    // clamped to this card's OWN floor; one block past the right/bottom edge is
    // allowed so a drop there grows the grid by a column / a row
    const b = state.bands.get(flr(n));
    const col = clamp(Math.round((left - HGAP) / CELL_W), 0, b.cols);
    const row = clamp(Math.round((top - b.top - LABEL_H) / state.cellH), 0, b.rows);
    target = { col, row };
    const g = ghost();
    g.hidden = false;
    g.style.left = (HGAP + col * CELL_W) + "px";
    g.style.top = (b.top + LABEL_H + row * state.cellH) + "px";
    g.style.width = CARD_W + "px";
    g.style.height = (state.cellH - VGAP) + "px";
    drawEdges();
  });
  bar.addEventListener("pointerup", async (e) => {
    if (!dragging) return;
    dragging = false;
    el.classList.remove("gnode--dragging"); el.style.zIndex = "";
    try { bar.releasePointerCapture(e.pointerId); } catch {}
    ghost().hidden = true;
    if (!target) { placeAll(); return; }
    const { col, row } = target; target = null;
    if (col === n.col && row === n.row) { placeAll(); return; }
    const occupant = [...state.nodes.values()].find((m) =>
      m !== n && flr(m) === flr(n) && m.col === col && m.row === row);
    const oldCol = n.col, oldRow = n.row;
    n.col = col; n.row = row;
    const writes = [post("/graph/slot", { camera: n.cam, col, row })];
    if (occupant) {                              // swap blocks
      occupant.col = oldCol; occupant.row = oldRow;
      writes.push(post("/graph/slot", { camera: occupant.cam, col: oldCol, row: oldRow }));
    }
    placeAll();
    try { await Promise.all(writes); flash(`✓ ${n.cam} → block ${col + 1},${row + 1}${occupant ? ` (swapped with ${occupant.cam})` : ""} (saved)`); }
    catch (err) { flash(`⚠ save failed: ${err.message || err}`); }
  });
}

// ---- edges ------------------------------------------------------------------
function center(n) {
  const el = n.el;
  return { x: el.offsetLeft + el.offsetWidth / 2, y: el.offsetTop + el.offsetHeight / 2 };
}
function drawEdges() {
  while (edges.firstChild) edges.removeChild(edges.firstChild);
  for (const e of state.edges) {
    const [a, b] = e.split("|");
    const na = state.nodes.get(a), nb = state.nodes.get(b);
    if (!na || !nb || !na.el || !nb.el) continue;
    const ca = center(na), cb = center(nb);
    const line = document.createElementNS(SVGNS, "line");
    line.setAttribute("x1", ca.x); line.setAttribute("y1", ca.y);
    line.setAttribute("x2", cb.x); line.setAttribute("y2", cb.y);
    edges.appendChild(line);
  }
}

// ---- connect ----------------------------------------------------------------
const stat = () => { document.getElementById("gstat").textContent =
  `${state.nodes.size} cameras · ${state.edges.size} edges`; };
async function toggleConnect(cam) {
  if (!state.pending) { state.pending = cam; mark(); flash(`linking from ${cam}…`); return; }
  if (state.pending === cam) { state.pending = null; mark(); flash(""); return; }
  const a = state.pending, b = cam, key = ekey(a, b);
  const adding = !state.edges.has(key);
  if (adding) state.edges.add(key); else state.edges.delete(key);
  state.pending = null; mark(); drawEdges(); stat();
  try {
    const r = await post(adding ? "/graph/edge" : "/graph/edge/delete", { a, b });
    if (r && r.ok === false) throw new Error(r.error || "server said no");
    flash(`✓ ${adding ? "linked" : "unlinked"} ${a} ↔ ${b} (saved)`);
  } catch (e) {
    if (adding) state.edges.delete(key); else state.edges.add(key);
    drawEdges(); stat(); flash(`⚠ save failed: ${e.message || e}`);
  }
}
function mark() {
  for (const n of state.nodes.values()) {
    if (!n.el) continue;
    const on = n.cam === state.pending;
    n.el.classList.toggle("gnode--pending", on);
    n.el.querySelector(".gnode__link")?.classList.toggle("on", on);
  }
}
function flash(msg) {
  if (!_hint) return;
  _hint.textContent = msg || _hint.dataset.def || "";
  _hint.style.color = (msg || "").startsWith("⚠") ? "#d6336c" : "";
}

// ---- generate ---------------------------------------------------------------
async function genVLM(n, el) {
  note(el, "generating caption…");
  const r = await post("/vlm/caption", { camera: n.cam,
    provider: document.getElementById("vlm-provider").value });
  if (r.ok) {
    n.vlm_caption = r.caption;
    const t = el.querySelector('[data-f="vlm"]');
    t.textContent = r.caption; t.classList.remove("gnode__text--empty");
    note(el, ""); requestAnimationFrame(placeAll);   // card grew -> re-fit blocks
  } else note(el, "failed: " + (r.error || "?"));
}
async function genLLM(n, el) {
  note(el, "generating connectivity…");
  const r = await post("/llm/summary", { camera: n.cam,
    provider: document.getElementById("llm-provider").value });
  if (r.ok) {
    n.connectivity_summary = r.summary;
    const t = el.querySelector('[data-f="conn"]');
    t.textContent = r.summary; t.classList.remove("gnode__text--empty");
    note(el, ""); requestAnimationFrame(placeAll);
  } else note(el, "failed: " + (r.error || "?"));
}

// ---- toolbar ----------------------------------------------------------------
document.getElementById("refresh-thumbs").onclick = () => {
  for (const n of state.nodes.values()) {
    const img = n.el?.querySelector(".gnode__img");
    if (img) img.src = frameURL(n.cam);
  }
};
window.addEventListener("resize", () => drawEdges());
load();
