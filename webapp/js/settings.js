/* AI settings page: live threshold + class toggles (applied by the bridge
 * instantly), plus read-only detector info. */

function esc(x) {
  return String(x ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const CHART_COLORS = {
  person: "#d45a00", car: "#0b7fc2", truck: "#7a4fd8",
  motorcycle: "#d1216f", bicycle: "#178a4a",
};

const conf = document.getElementById("conf");
const confValue = document.getElementById("conf-value");

let classes = [];

function renderConf() {
  confValue.textContent = Number(conf.value).toFixed(2);
}


async function init() {
  const note = (id, msg) => { const el = document.getElementById(id); if (el) el.textContent = msg; };
  const ZONES = ["Asia/Bangkok", "UTC", "Asia/Singapore", "Asia/Tokyo",
    "Asia/Hong_Kong", "Asia/Dubai", "Europe/London", "Europe/Berlin",
    "America/New_York", "America/Los_Angeles", "Australia/Sydney"];
  const tzOffset = (z) => {
    try {
      return new Intl.DateTimeFormat("en", { timeZone: z, timeZoneName: "shortOffset" })
        .formatToParts(new Date()).find((p) => p.type === "timeZoneName").value
        .replace("GMT", "UTC");
    } catch { return ""; }
  };
  const tzSel = document.getElementById("tz");
  try {
    const s = await (await fetch("/api/settings")).json();
    conf.value = s.min_confidence;
    if (tzSel) {
      const cur = s.timezone || "Asia/Bangkok";
      const cities = document.createElement("optgroup");
      cities.label = "Cities";
      for (const z of ZONES.includes(cur) || /^Etc\//.test(cur) ? ZONES : [cur, ...ZONES])
        cities.append(new Option(`${z.replace("_", " ")} (${tzOffset(z)})`, z, false, z === cur));
      tzSel.append(cities);
      const offsets = document.createElement("optgroup");
      offsets.label = "UTC offsets";
      for (let n = -12; n <= 14; n++) {
        const zone = n === 0 ? "UTC" : `Etc/GMT${n > 0 ? "-" : "+"}${Math.abs(n)}`;
        if (n === 0 && ZONES.includes("UTC")) continue;
        offsets.append(new Option(`UTC${n >= 0 ? "+" : "−"}${Math.abs(n)}`, zone,
                                  false, zone === cur));
      }
      tzSel.append(offsets);
    }
  } catch {
    note("conf-note", "bridge unreachable");
    note("tz-note", "bridge unreachable");
  }
  renderConf();
  conf.addEventListener("input", () => { renderConf(); note("conf-note", "unsaved"); });

  // each card owns its save: partial POSTs — the bridge patches only the
  // fields present, so the two cards never clobber each other
  const save = async (body, noteId) => {
    note(noteId, "saving…");
    try {
      const r = await fetch("/api/settings", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) });
      if (!r.ok) throw 0;
      note(noteId, "applied " + new Date().toLocaleTimeString(undefined, { hour12: false }));
    } catch { note(noteId, "save failed"); }
  };
  document.getElementById("conf-save").onclick = () =>
    save({ min_confidence: Number(conf.value), disabled_classes: [] }, "conf-note");
  document.getElementById("tz-save").onclick = () =>
    save({ timezone: tzSel ? tzSel.value : undefined }, "tz-note");

  const clearBtn = document.getElementById("chat-clear");
  if (clearBtn) clearBtn.onclick = async () => {
    if (!confirm("Delete ALL Investigator chat history? This cannot be undone."))
      return;
    clearBtn.disabled = true;
    note("chat-clear-note", "clearing…");
    try {
      const r = await fetch("/api/reid/agent/conversations/clear",
        { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const d = await r.json();
      if (!r.ok || !d.ok) throw 0;
      note("chat-clear-note",
        `cleared ${d.cleared} conversation${d.cleared === 1 ? "" : "s"}`);
    } catch { note("chat-clear-note", "clear failed"); }
    clearBtn.disabled = false;
  };
}

/* section nav highlights while scrolling (industrial settings staple) */
function initScrollspy() {
  const links = [...document.querySelectorAll("#set-nav a")];
  const cards = links.map((a) => document.querySelector(a.getAttribute("href")));
  const spy = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      links.forEach((a) => a.classList.toggle(
        "on", a.getAttribute("href") === "#" + e.target.id));
      break;
    }
  }, { root: document.querySelector(".ai-set"), rootMargin: "-10% 0px -70% 0px" });
  cards.forEach((c) => c && spy.observe(c));
}

/* ---- Detector classes: dual-list shuttle, applies live ----------------- */
async function initClasses() {
  const selEl = document.getElementById("cls-sel");
  const availEl = document.getElementById("cls-avail");
  const note = document.getElementById("cls-note");
  if (!selEl) return;
  let data;
  try { data = await (await fetch("/api/config/classes")).json(); }
  catch { note.textContent = "bridge unreachable"; return; }
  const selected = new Set(data.selected || []);
  const intEl = document.getElementById("det-interval");
  const intHint = document.getElementById("det-interval-hint");
  const showInt = () => { const v = +intEl.value || 0;
    intHint.textContent = v === 0 ? "every frame"
      : `1 of every ${v + 1} frames`; };
  const intNote = document.getElementById("det-interval-note");
  let intTimer;
  const saveInterval = async () => {
    if (intNote) intNote.textContent = "saving…";
    try {
      const r = await (await fetch("/api/config/classes", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected: [...selected],
          interval: +intEl.value || 0 }) })).json();
      if (intNote) intNote.textContent = r.ok ? "✓ applied live" : "failed";
    } catch { if (intNote) intNote.textContent = "save failed"; }
  };
  if (intEl) {
    intEl.value = data.interval || 0; showInt();
    intEl.oninput = () => { showInt();
      clearTimeout(intTimer); intTimer = setTimeout(saveInterval, 600); };
  }
  const render = () => {
    selEl.innerHTML = ""; availEl.innerHTML = "";
    for (const c of data.available || []) {
      (selected.has(c) ? selEl : availEl).append(new Option(c, c));
    }
    document.getElementById("cls-sel-n").textContent = selEl.options.length;
    document.getElementById("cls-avail-n").textContent = availEl.options.length;
  };
  const move = (from, add) => {
    for (const o of [...from.selectedOptions]) {
      add ? selected.add(o.value) : selected.delete(o.value);
    }
    render();
    note.textContent = "unsaved changes";
  };
  document.getElementById("cls-add").onclick = () => move(availEl, true);
  document.getElementById("cls-rem").onclick = () => move(selEl, false);
  availEl.ondblclick = () => move(availEl, true);
  selEl.ondblclick = () => move(selEl, false);
  render();
  document.getElementById("cls-save").onclick = async () => {
    note.textContent = "saving…";
    try {
      const r = await (await fetch("/api/config/classes", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected: [...selected],
          interval: intEl ? (+intEl.value || 0) : 0 }) })).json();
      note.textContent = r.ok
        ? `saved — detector applies it live (~2 s), ${r.selected.length} classes`
        : `failed: ${r.error}`;
    } catch { note.textContent = "save failed"; }
  };
}

/* ---- Camera boxes: one square per camera — rtsp link + floor ----------- */
async function initCameras() {
  const grid = document.getElementById("cam-boxes");
  const note = document.getElementById("nvr-note");
  if (!grid) return;
  let nFloors = 8;
  try {
    const d = await (await fetch("/api/config/cameras")).json();
    nFloors = d.n_floors || 8;
    grid.innerHTML = `<div class="cam-head">
      <span>Name</span><span>RTSP source</span><span>Floor</span><span></span></div>`;
    for (const b of d.boxes || []) grid.appendChild(box(b));
  } catch { note.textContent = "bridge unreachable"; return; }

  function box(b) {
    const el = document.createElement("div");
    el.className = "cam-box";
    el.innerHTML = `
      <input class="cam-box__name" spellcheck="false" placeholder="name"
        value="${esc(b.id || "")}" title="Camera name — a renamed camera keeps this name everywhere">
      <input class="cam-box__url" spellcheck="false"
        placeholder="rtsp://user:pass@host:554/…" value="${esc(b.url || "")}">
      <button type="button" class="cam-box__x" title="Remove">✕</button>`;
    const sel = document.createElement("select");
    sel.className = "cam-box__floor";
    sel.append(new Option("— floor —", ""));
    for (let f = 1; f <= nFloors; f++)
      sel.append(new Option(`Floor ${f}F`, `${f}F`, false, b.floor === `${f}F`));
    el.insertBefore(sel, el.querySelector(".cam-box__x"));
    el.querySelector(".cam-box__name").addEventListener("input", (e) => {
      e.target.value = e.target.value.toLowerCase()
        .replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "").slice(0, 24);
    });
    el.querySelector(".cam-box__x").onclick = () => {
      el.remove(); note.textContent = "unsaved changes";
    };
    el.addEventListener("input", () => { note.textContent = "unsaved changes"; });
    return el;
  }

  document.getElementById("cam-add").onclick = () => {
    const el = box({ id: "", url: "", floor: "" });
    grid.appendChild(el);
    el.querySelector(".cam-box__name").focus();
  };

  document.getElementById("nvr-save").onclick = async () => {
    const boxes = [...grid.querySelectorAll(".cam-box")].map((el) => ({
      id: el.querySelector(".cam-box__name").value.trim(),
      url: el.querySelector(".cam-box__url").value.trim(),
      floor: el.querySelector(".cam-box__floor").value,
    }));
    note.textContent = "saving…";
    try {
      const r = await (await fetch("/api/config/cameras", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ boxes }) })).json();
      note.textContent = r.ok
        ? `saved — ${r.cameras} cameras, ${r.streams_live} streams live, ` +
          `${r.floors_set} floors set; restart: ${r.restart_needed[0]}`
        : r.error;
      if (r.ok) initCameras();
    } catch { note.textContent = "save failed"; }
  };
}

/* Every knob the behaviour watcher exposes. The server owns the defaults and
 * the units; this only renders them. */
async function initBehavior() {
  const box = document.getElementById("behavior-cfg");
  const note = document.getElementById("bhv-note");
  const meta = document.getElementById("bhv-stats");
  if (!box) return;
  let d;
  try { d = await (await fetch("/api/reid/behavior/config")).json(); }
  catch { meta.textContent = "behaviour service unreachable"; return; }
  const c = d.cfg || {};
  const on = (v) => (v ? "1" : "0");
  box.classList.add("bhv");

  // master switch
  const en = _sel(box, "Watch behaviour", on(c.enabled), [["1", "on"], ["0", "off"]]);

  // ---- the four rules: each is a toggle + plain description; its numeric
  //      fine-tuning hides in an <details> so the cockpit isn't overwhelming ----
  function rule(title, desc, enabled, buildAdvanced) {
    const wrap = document.createElement("div");
    wrap.className = "bhv-rule";
    const head = document.createElement("label");
    head.className = "bhv-rule__head";
    head.innerHTML = `<div><b>${title}</b><span class="bhv-rule__desc">${desc}</span></div>`;
    const sel = document.createElement("select");
    for (const [v, t] of [["1", "On"], ["0", "Off"]]) {
      const o = document.createElement("option");
      o.value = v; o.textContent = t; if (v === enabled) o.selected = true;
      sel.append(o);
    }
    head.append(sel); wrap.append(head);
    const det = document.createElement("details");
    det.className = "bhv-adv";
    det.innerHTML = "<summary>Fine-tuning</summary>";
    const grid = document.createElement("div");
    grid.className = "bhv-adv__grid";
    det.append(grid); wrap.append(det);
    const fields = buildAdvanced(grid);
    box.append(wrap);
    return { sel, fields };
  }

  const crowd = rule("Crowd", "Fires when several people are in view at once.",
    on(c.crowd_enabled), (g) => ({
      n: _num(g, "How many people counts as a crowd", c.crowd_min_persons, 1) }));
  const dwell = rule("Loitering (dwell)", "Fires when someone stands still for a while.",
    on(c.dwell_enabled), (g) => ({
      s: _num(g, "Still for at least (seconds)", c.dwell_s, 1),
      m: _num(g, "Counts as “still” if they move less than (× their height)",
              c.dwell_move_frac, 0.01) }));
  const vanish = rule("Disappearance (vanish)",
    "Fires when someone vanishes away from the frame edge — a door, a lift.",
    on(c.vanish_enabled), (g) => ({
      lv: _num(g, "Ignore tracks shorter than (seconds)", c.vanish_min_s, 1),
      gp: _num(g, "Count as gone after unseen for (seconds)", c.vanish_gap_s, 0.5),
      ed: _num(g, "Ignore if within this fraction of the frame border",
               c.vanish_edge_frac, 0.01) }));
  const parcel = rule("Parcel watch",
    "Checks named cameras on a timer and when a new person appears.",
    on(c.parcel_new_id) === "1" || (c.parcel_cameras || []).length ? "1" : "0",
    (g) => ({
      cams: _txt(g, "Cameras to watch (comma separated)",
                 (c.parcel_cameras || []).join(", "), "nvr1_ch10, nvr2_ch02", true),
      every: _num(g, "Look every (seconds)", c.parcel_period_s, 10),
      newid: _sel(g, "Also look when a new person appears",
                  on(c.parcel_new_id), [["1", "yes"], ["0", "no"]]) }));

  // ---- performance budget: one advanced block, plainly described ----
  const perf = document.createElement("details");
  perf.className = "bhv-adv bhv-adv--wide";
  perf.innerHTML = "<summary>Performance &amp; cost (advanced)</summary>";
  const pg = document.createElement("div");
  pg.className = "bhv-adv__grid";
  perf.append(pg); box.append(perf);
  const bg = _num(pg, "Max VLM checks per minute (caps GPU cost)", c.max_calls_per_min, 10);
  const cc = _num(pg, "Checks running at once", c.concurrency, 1);
  const cd = _num(pg, "Wait before re-checking the same person+rule (s)", c.cooldown_s, 5);
  const ff = _num(pg, "Frame grabs per camera per second", c.frame_fps, 0.5);
  const fb = _num(pg, "Frame buffer depth (seconds)", c.frame_buffer_s, 1);

  const paint = (s) => {
    meta.textContent = ` · fired ${s.fired || 0} · skipped ${s.skipped_cooldown || 0}`
      + ` (cooldown) / ${s.skipped_budget || 0} (budget) · errors ${s.errors || 0}`;
  };
  paint(d.stats || {});

  document.getElementById("bhv-save").onclick = async () => {
    note.textContent = "saving…";
    try {
      const r = await fetch("/api/reid/behavior/config", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled: en.value === "1", provider: c.provider || "local-gemma",
          max_calls_per_min: +bg.value, concurrency: +cc.value,
          cooldown_s: +cd.value, frame_fps: +ff.value, frame_buffer_s: +fb.value,
          crowd_enabled: crowd.sel.value === "1",
          crowd_min_persons: +crowd.fields.n.value,
          dwell_enabled: dwell.sel.value === "1", dwell_s: +dwell.fields.s.value,
          dwell_move_frac: +dwell.fields.m.value,
          vanish_enabled: vanish.sel.value === "1",
          vanish_min_s: +vanish.fields.lv.value,
          vanish_gap_s: +vanish.fields.gp.value,
          vanish_edge_frac: +vanish.fields.ed.value,
          parcel_cameras: parcel.fields.cams.value.split(",")
            .map((x) => x.trim()).filter(Boolean),
          parcel_period_s: +parcel.fields.every.value,
          parcel_new_id: parcel.fields.newid.value === "1",
        }) });
      const j = await r.json();
      note.textContent = j.ok ? "✓ applied live" : "failed";
    } catch (e) { note.textContent = `failed: ${e.message || e}`; }
  };
}

function _num(parent, label, val, step) {
  const l = document.createElement("label");
  l.innerHTML = `<span>${label}</span>`;
  const i = document.createElement("input");
  i.type = "number"; i.step = step; i.value = val;
  l.appendChild(i); parent.appendChild(l);
  return i;
}
function _sel(parent, label, val, opts, full) {
  const l = document.createElement("label");
  if (full) l.className = "reid-full";
  l.innerHTML = `<span>${label}</span>`;
  const s = document.createElement("select");
  for (const [v, t] of opts) { const o = document.createElement("option");
    o.value = v; o.textContent = t; if (v === val) o.selected = true; s.appendChild(o); }
  l.appendChild(s); parent.appendChild(l); return s;
}
function _area(parent, label, val, ph) {
  const l = document.createElement("label");
  l.className = "reid-full";
  l.innerHTML = `<span>${label}</span>`;
  const t = document.createElement("textarea");
  t.rows = 8; t.value = val || ""; if (ph) t.placeholder = ph;
  t.style.cssText = "font:12px/1.45 ui-monospace,monospace;resize:vertical";
  l.appendChild(t); parent.appendChild(l); return t;
}
function _txt(parent, label, val, ph, full, pw) {
  const l = document.createElement("label");
  if (full) l.className = "reid-full";
  l.innerHTML = `<span>${label}</span>`;
  const i = document.createElement("input");
  i.type = pw ? "password" : "text"; i.value = val || ""; if (ph) i.placeholder = ph;
  l.appendChild(i); parent.appendChild(l); return i;
}

async function initReid() {
  const box = document.getElementById("reid-cfg");
  const note = document.getElementById("reid-note");
  if (!box) return;
  let cfg;
  try { cfg = await (await fetch("/api/reid/config")).json(); }
  catch { note.textContent = "ReID service unreachable"; return; }
  const g = cfg.gate || {}, m = cfg.match || {};

  const en = _sel(box, "Cross-camera ReID", g.enabled ? "1" : "0",
    [["1", "on"], ["0", "off"]]);

  // strictness = the match cosine. Presets are quick-sets; the exact field is
  // the source of truth (so the owner can pick >0.8 precisely). Presets fill
  // the number; save sends the number — no conflict between the two controls.
  const cur = +m.match_thresh || 0.67;
  const PRESETS = { loose: 0.60, balanced: 0.67, strict: 0.75, very: 0.82 };
  const ms = _sel(box, "Matching strictness", "",
    [["", "— choose a preset —"],
     ["loose", "Loose 0.60 — merge more (fewer duplicates, more mistakes)"],
     ["balanced", "Balanced 0.67 (recommended)"],
     ["strict", "Strict 0.75 — split more (fewer wrong merges)"],
     ["very", "Very strict 0.82 — near-identical only"]]);
  const exact = _num(box, "Exact match cosine (0.40–0.95)", cur.toFixed(2), "0.01");
  exact.min = "0.40"; exact.max = "0.95";
  ms.onchange = () => { if (PRESETS[ms.value] != null) exact.value = PRESETS[ms.value].toFixed(2); };

  document.getElementById("reid-save").onclick = async () => {
    note.textContent = "saving…";
    let thr = parseFloat(exact.value);
    if (!(thr >= 0.4 && thr <= 0.95)) { note.textContent = "cosine must be 0.40–0.95"; return; }
    try {
      const r = await (await fetch("/api/reid/config", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gate: { enabled: en.value === "1" },
          match: { match_thresh: thr } }) })).json();
      note.textContent = r.ok ? `✓ applied live · match ${thr.toFixed(2)}` : "failed";
    } catch (e) { note.textContent = "failed: " + (e.message || e); }
  };
}

async function initProviders() {
  const box = document.getElementById("reid-providers");
  const note = document.getElementById("prov-note");
  if (!box) return;
  let p;
  try { p = await (await fetch("/api/reid/providers")).json(); }
  catch { note.textContent = "provider service unreachable"; return; }
  const c = p.cfg || {}, ks = p.keys_set || {};

  // grouped select: "On this machine" vs "Cloud" so local/cloud is obvious.
  // Only provider CHOICE + API keys are editable — URLs and model names stay
  // fixed so a user can't break the on-box AI beyond repair.
  function grouped(label, value, cloud) {
    const l = document.createElement("label");
    l.innerHTML = `<span>${label}</span>`;
    const sel = document.createElement("select");
    const local = document.createElement("optgroup");
    local.label = "On this machine (free)";
    local.append(new Option("Gemma 4 31B", "local-gemma",
                            false, value === "local-gemma"));
    sel.append(local);
    const cg = document.createElement("optgroup");
    cg.label = "Cloud (needs API key)";
    for (const [v, t] of cloud)
      cg.append(new Option(t, v, false, value === v));
    sel.append(cg);
    l.append(sel); box.append(l);
    return sel;
  }
  const vlm = grouped("VLM — reads camera images", c.vlm_provider,
    [["anthropic", "Anthropic Claude"]]);
  const llm = grouped("LLM — writes summaries", c.llm_provider,
    [["gemma", "Google Gemini"], ["anthropic", "Anthropic Claude"]]);
  const ak = _txt(box, `Anthropic API key ${ks.anthropic ? "(saved — blank keeps it)" : "(not set)"}`,
    "", "sk-ant-…", true, true);
  const gk = _txt(box, `Google API key ${ks.gemma ? "(saved — blank keeps it)" : "(not set)"}`,
    "", "AIza…", true, true);

  document.getElementById("prov-save").onclick = async () => {
    note.textContent = "saving…";
    try {
      await fetch("/api/reid/providers", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vlm_provider: vlm.value,
                               llm_provider: llm.value }) });
      if (ak.value || gk.value) {
        await fetch("/api/reid/providers/keys", { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ anthropic_key: ak.value, gemma_key: gk.value }) });
        ak.value = ""; gk.value = "";
      }
      note.textContent = "✓ saved";
    } catch (e) { note.textContent = "failed: " + (e.message || e); }
  };
}


/* ---- AI information: cards for the models that are ACTIVE right now ----
 * Data-driven on purpose: to add/replace an AI, add one entry below. `resolve`
 * returns null when that model is switched off, and the card is not rendered. */
const AI_STACK = [
  { role: "Object detection", where: "MI300X · MIGraphX",
    blurb: "Finds people, vehicles and objects in every frame. Everything else starts from these boxes.",
    change: "Object detection card",
    resolve: () => ({ name: "RT-DETR R50",
      meta: "Apache-2.0 · 640×640 · one batch across all cameras" }) },

  { role: "In-camera tracking", where: "MI300X · MIGraphX",
    blurb: "Keeps one track id on a person across frames, and holds it when two people cross.",
    change: "airocm (fixed)",
    resolve: () => ({ name: "BoT-SORT + OSNet appearance",
      meta: "clean-room (no AGPL) · appearance breaks ID swaps" }) },

  { role: "Cross-camera ReID", where: "siglip · MIGraphX",
    blurb: "Recognises the same person again on a different camera and gives them one Global ID.",
    change: "Person ReID card",
    resolve: (s) => (s.reid && s.reid.ready)
      ? { name: s.reid.embedder === "youtureid" ? "YoutuReID (Apache-2.0)" : "SigLIP2",
          meta: `${s.reid.dim || "?"}-d vector · gated by the camera graph` } : null },

  { role: "Visual grounding", where: "ground · CPU",
    blurb: "Draws a box around anything you ask about — the “point at it” tool.",
    change: "fixed",
    resolve: () => ({ name: "Grounding DINO",
      meta: "Apache-2.0 · open-vocabulary text → boxes" }) },

  { role: "VLM — vision", where: "this machine / cloud",
    blurb: "Reads a camera image — place captions, behaviour, and the “look now” answers.",
    change: "VLM & LLM card",
    resolve: (s) => {
      if (!s.prov) return null;
      const p = s.prov.cfg.vlm_provider;
      const local = p !== "anthropic";
      const ok = local || s.prov.keys_set.anthropic;
      return { name: local ? "Gemma 4 31B — this machine" : "Anthropic Claude (cloud)",
               meta: ok ? "ready" : "⚠ no API key", warn: !ok };
    } },

  { role: "LLM — text", where: "this machine / cloud",
    blurb: "Writes summaries, connectivity notes and the investigator's answers.",
    change: "VLM & LLM card",
    resolve: (s) => {
      if (!s.prov) return null;
      const p = s.prov.cfg.llm_provider;
      const local = p !== "gemma" && p !== "anthropic";
      const need = local || (p === "gemma" ? s.prov.keys_set.gemma : s.prov.keys_set.anthropic);
      const name = local ? "Gemma 4 31B — this machine"
        : p === "gemma" ? "Google Gemini (cloud)" : "Anthropic Claude (cloud)";
      return { name, meta: need ? "ready" : "⚠ no API key", warn: !need };
    } },

  { role: "Video captioning", where: "vlmscan · MI300X",
    blurb: "Narrates every recorded minute that had people, so the investigator can search what happened.",
    change: "fixed",
    resolve: () => ({ name: "Gemma 4 31B (vLLM)",
      meta: "activity-gated · writes searchable captions" }) },
];

async function initAiCards() {
  const box = document.getElementById("ai-cards");
  const note = document.getElementById("ai-cards-note");
  if (!box) return;
  const get = async (u) => { try { return await (await fetch(u)).json(); } catch { return null; } };
  const [reid, prov] = await Promise.all([
    get("/api/reid/healthz"), get("/api/reid/providers"),
  ]);
  const s = { reid, prov };

  box.innerHTML = "";
  let n = 0;
  for (const item of AI_STACK) {
    let r = null;
    try { r = item.resolve(s); } catch { r = null; }
    if (!r) continue;
    n++;
    const el = document.createElement("div");
    el.className = "ai-card" + (r.warn ? " ai-card--warn" : "");
    el.innerHTML = `
      <div class="ai-card__role">${item.role}</div>
      <div class="ai-card__name">${r.name}</div>
      <div class="ai-card__blurb">${item.blurb}</div>
      <div class="ai-card__meta">${r.meta}</div>
      <div class="ai-card__foot">${item.where} · change in “${item.change}”</div>`;
    box.appendChild(el);
  }
  note.textContent = `${n} active`;
  if (!n) box.innerHTML = '<span class="ins-note">no AI service reachable</span>';
}

init();
initScrollspy();
initClasses();
initCameras();
initBehavior();
initReid();
initProviders();
initAiCards();
