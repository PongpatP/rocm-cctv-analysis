/* Person tracking: every cross-camera Global ID, its crops, its route, and a
 * one-line VLM description you can search. Retrospective look-up: find the
 * person, see where they came from and where they went. */
const api = async (p) => (await fetch("/api/reid" + p)).json();
const post = (p, b) => fetch("/api/reid" + p, { method: "POST",
  headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) })
  .then((r) => r.json());
const cropURL = (p) => `/api/reid/reid/crop?p=${encodeURIComponent(p)}`;
const snapURL = (p) => `/api/reid/behavior/snapshot?p=${encodeURIComponent(p)}`;
const hhmm = (ts) => new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const state = { persons: [], filter: new Set(), cameras: [] };

/* Tags come from the VLM as six fixed fields. `mode` is measured from the pixels,
 * not asked for: an infrared night crop carries no colour at all. */
const TAG_ORDER = ["mode", "visibility", "sex", "upper", "lower", "carry", "head"];
const SKIP_VALUES = new Set(["none", "unknown", "person", "not visible", "n/a",
  "na", "not applicable", "obscured", "hidden"]);
const tagKey = (f, v) => `${f}:${v}`;
const tagText = (f, v) => (f === "mode" ? (v === "ir" ? "IR night" : "colour") : v);

/* Identity is described by the on-board vision AI; the specific model name is
 * not surfaced in the product UI. */
async function showModel() {
  const el = document.getElementById("model");
  if (el) el.hidden = true;                 // no model name in the product UI
  const chip = document.getElementById("reid-thr");
  if (chip) {
    chip.textContent = "identity by clothing description";
    chip.title = "Global IDs are linked by the AI's clothing description (sex, garments, carried items, headwear).";
    chip.hidden = false;
    window.__reidThr = 999;   // identity is by description; nothing is 'low score'
  }
}

async function load() {
  const h = document.getElementById("hours").value;
  const mc = document.getElementById("mincam").value;
  const ms = document.getElementById("minsee").value;
  const cam = document.getElementById("cam").value;
  document.getElementById("stat").textContent = "loading…";
  try {
    const r = await api(`/reid/persons?hours=${h}&min_cameras=${mc}&min_sightings=${ms}`
      + `&camera=${encodeURIComponent(cam)}&limit=300`);
    state.persons = r.persons || [];
    state.cameras = r.cameras || [];
  } catch (e) { document.getElementById("stat").textContent = "ReID service unreachable"; return; }
  renderCameras();
  render();
}

/* Cameras are named "<recorder>_<channel>", so the recorder is the natural group.
 * Picking a recorder means "anyone this NVR ever saw". Rebuilt on every load, but
 * only when the list actually changed — otherwise the open dropdown would reset. */
function renderCameras() {
  const sel = document.getElementById("cam");
  const key = state.cameras.join(",");
  if (sel.dataset.key === key) return;
  sel.dataset.key = key;
  const keep = sel.value;
  const groups = new Map();
  for (const c of state.cameras) {
    const [nvr, ...rest] = c.split("_");
    if (!groups.has(nvr)) groups.set(nvr, []);
    groups.get(nvr).push([c, rest.join("_") || c]);
  }
  sel.innerHTML = '<option value="">All cameras</option>';
  for (const [nvr, cams] of groups) {
    const g = document.createElement("optgroup");
    g.label = nvr;
    g.appendChild(new Option(`${nvr} — all ${cams.length} cameras`, nvr));
    for (const [val, label] of cams) g.appendChild(new Option(label, val));
    sel.appendChild(g);
  }
  sel.value = keep;                       // survive a reload; "" if it vanished
}

/* Tags are typed, not clicked: there can be hundreds of distinct garment values,
 * so a chip wall is unusable. The box autocompletes over the tags that actually
 * exist. A token is `field:value` (exact) or a bare `value` (matches any field,
 * substring). Tokens on DIFFERENT fields are ANDed; tokens on the SAME field are
 * ORed; bare tokens are ANDed. */
function tagIndex() {
  const counts = new Map();
  for (const p of state.persons)
    for (const f of TAG_ORDER) {
      const v = (p.tags || {})[f];
      if (v && !SKIP_VALUES.has(v))
        counts.set(tagKey(f, v), (counts.get(tagKey(f, v)) || 0) + 1);
    }
  return counts;
}

function renderTagOptions() {
  const dl = document.getElementById("tagopts");
  const names = new Map();
  for (const p of state.persons)
    if (p.person_name) names.set(p.person_name, (names.get(p.person_name) || 0) + 1);
  dl.innerHTML = [...names].map(([n, k]) => `<option value="${n}">${k} identities · recognised face</option>`)
    .concat([...tagIndex()].sort((a, b) => b[1] - a[1])
      .map(([k, n]) => `<option value="${k}">${n} people</option>`)).join("");
}

/* One term = one bubble. `field:value` matches that tag exactly; anything else is
 * a free word matched against the tags, the description, the route and the G-id —
 * so "backpack", "ch07" and "3538" all work without knowing where they live. */
function matchesTerm(p, t) {
  const tags = p.tags || {};
  const i = t.indexOf(":");
  if (i > 0 && TAG_ORDER.includes(t.slice(0, i)))
    return tags[t.slice(0, i)] === t.slice(i + 1);
  // a G-id may be typed WITH or WITHOUT the "G" — "g398", "G398" and "398" all
  // match Global ID 398 (the "G" prefix used to make the search miss).
  const gd = String(p.gid);
  if (/^g?\d+$/.test(t) && gd.includes(t.replace(/^g/, ""))) return true;
  return gd.includes(t)
    || (p.person_name || "").toLowerCase().includes(t)
    || (p.description || "").toLowerCase().includes(t)
    || Object.values(tags).some((v) => v.toLowerCase().includes(t))
    || p.path.some((s) => s.camera.toLowerCase().includes(t));
}

/* Terms on the SAME tag field are ORed (black OR blue shirt); everything else is
 * ANDed (a man AND carrying a backpack). Free words are always ANDed. */
function passesTerms(p, terms) {
  if (!terms.length) return true;
  const byField = new Map();
  for (const t of terms) {
    const i = t.indexOf(":");
    const f = i > 0 && TAG_ORDER.includes(t.slice(0, i)) ? t.slice(0, i) : t;
    if (!byField.has(f)) byField.set(f, []);
    byField.get(f).push(t);
  }
  return [...byField.values()].every((ts) => ts.some((t) => matchesTerm(p, t)));
}

function renderBubbles() {
  const box = document.getElementById("bubbles");
  box.innerHTML = "";
  for (const t of state.filter) {
    const b = document.createElement("span");
    b.className = "ppl-bubble";
    b.innerHTML = `${t}<button title="remove">✕</button>`;
    b.querySelector("button").onclick = () => dropTerm(t);
    box.appendChild(b);
  }
}

function addTerm(t) {
  t = t.trim().toLowerCase();
  if (t) state.filter.add(t);
  document.getElementById("q").value = "";
  render();
}

function dropTerm(t) {
  state.filter.delete(t);
  render();
}

function toggleTag(k) {          // a chip on a card is a shortcut into the box
  state.filter.has(k) ? state.filter.delete(k) : state.filter.add(k);
  render();
}

/* One "page" is 5 rows of cards. The grid is responsive (auto-fill), so the
 * number of columns — and therefore how many cards make five rows — is read
 * from the live layout instead of guessed. */
const PAGE_ROWS = 5;
function pageSize() {
  const list = document.getElementById("list");
  const cols = getComputedStyle(list).gridTemplateColumns
    .split(" ").filter(Boolean).length || 1;
  return PAGE_ROWS * cols;
}

/* `resetPage` (default) shows the first page — used whenever the result set
 * changes (load, filter, search). The "Show more" button calls render(false)
 * to keep the pages already revealed and append the next one. */
function render(resetPage = true) {
  const typing = document.getElementById("q").value.trim().toLowerCase();
  const terms = [...state.filter, ...(typing ? [typing] : [])];
  const list = document.getElementById("list");
  list.innerHTML = "";
  const shown = state.persons.filter((p) => passesTerms(p, terms));
  const ps = pageSize();
  if (resetPage || !state.pageLimit) state.pageLimit = ps;
  const visible = shown.slice(0, state.pageLimit);
  for (const p of visible) list.appendChild(card(p));
  if (shown.length > state.pageLimit) {
    const more = document.createElement("button");
    more.className = "ppl-more";
    more.textContent = `Show more results — ${shown.length - state.pageLimit} left`;
    more.onclick = () => { state.pageLimit += ps; render(false); };
    list.appendChild(more);
  }
  renderTagOptions();
  renderBubbles();
  const tagged = state.persons.filter((p) => Object.keys(p.tags || {}).length).length;
  document.getElementById("stat").textContent =
    `showing ${visible.length} of ${shown.length} people · ${tagged} tagged`;
  if (!shown.length) list.innerHTML = '<div class="ppl-empty">no one matched — remove a bubble, widen the time range, or people simply have not walked yet</div>';
}

function tagChips(p) {
  const el = document.createElement("div");
  el.className = "ppl-tags";
  for (const f of TAG_ORDER) {
    const v = (p.tags || {})[f];
    if (!v || SKIP_VALUES.has(v)) continue;
    const b = document.createElement("button");
    b.className = "ppl-tag ppl-tag--sm ppl-tag--" + f
      + (state.filter.has(tagKey(f, v)) ? " ppl-tag--on" : "");
    b.title = `${f}: ${v} — click to filter`;
    b.textContent = tagText(f, v);
    b.onclick = (e) => { e.stopPropagation(); toggleTag(tagKey(f, v)); };
    el.appendChild(b);
  }
  return el;
}

function card(p) {
  const el = document.createElement("div");
  el.className = "ppl-card";
  // route was a huge "nvrX_chYY → …" wall (a clothing bucket can cross every
  // camera). Show the DISTINCT cameras as short chips instead — scannable, and
  // the card keeps a uniform height. Full path stays on hover + in the detail.
  const cams = [...new Set(p.path.map((s) => s.camera))];
  const shortCam = (c) => c.replace(/^nvr/, "n").replace(/_ch/, "·");
  const CAP = 8;
  const route = `<div class="ppl-route" title="${p.path.map((s) => s.camera).join(" → ")}">`
    + cams.slice(0, CAP).map((c) => `<span class="ppl-cam">${shortCam(c)}</span>`).join("")
    + (cams.length > CAP ? `<span class="ppl-cam ppl-cam--more">+${cams.length - CAP}</span>` : "")
    + "</div>";
  const named = p.person_name
    ? `<span class="ppl-name" title="recognised by face, cosine ${(+p.face_score).toFixed(3)}">👤 ${p.person_name}</span>`
    : "";
  el.innerHTML = `
    <div class="ppl-card__head">
      ${named}<span class="ppl-gid">G${p.gid}</span>
      <span class="ppl-when">${hhmm(p.first_ts)} → ${hhmm(p.last_ts)}</span>
    </div>
    <div class="ppl-thumbs">${
      (p.thumbs || []).map((t) => `<img loading="lazy" src="${cropURL(t)}" alt="">`).join("")
      || '<span class="ppl-nocrop">no crop stored</span>'}</div>
    ${route}
    <div class="ppl-desc ${p.description ? "" : "ppl-desc--empty"}">${p.description || "— no description —"}</div>
    <div class="ppl-tagslot"></div>
    <div class="ppl-card__foot">
      <span>${p.sightings} sightings · ${p.cameras} cameras</span>
      <span>
        <button class="pbtn pbtn--sm" data-a="describe">✨ Describe</button>
      </span>
    </div>
    <div class="ppl-note"></div>`;
  const note = el.querySelector(".ppl-note");
  el.querySelector(".ppl-tagslot").replaceWith(tagChips(p));
  el.dataset.gid = p.gid;
  el.onclick = () => {
    document.querySelectorAll(".ppl-card--active")
      .forEach((e) => e.classList.remove("ppl-card--active"));
    el.classList.add("ppl-card--active");
    openPerson(p.gid);
  };
  el.querySelector('[data-a="describe"]').onclick = async (e) => {
    e.stopPropagation();
    e.target.disabled = true; note.textContent = "asking the VLM…";
    const r = await post(`/reid/person/${p.gid}/describe`, {});
    if (r.ok) {
      p.description = r.description; p.tags = r.tags || {}; p.mode = r.mode;
      p.visibility = r.visibility;
      const d = el.querySelector(".ppl-desc");
      d.textContent = r.description; d.classList.remove("ppl-desc--empty");
      el.querySelector(".ppl-tags").replaceWith(tagChips(p));
      renderTagOptions();
      note.textContent = "";
    } else note.textContent = "failed: " + (r.error || "?");
    e.target.disabled = false;
  };
  return el;
}

/* The detail panel is rebuilt from a fresh skeleton on every open, so its m-*
 * containers always exist even after a vehicle (which replaces the content). */
const PERSON_SKELETON = `
  <div class="ppl-desc" id="m-desc">loading…</div>
  <details class="ppl-sec">
    <summary>Snapshots<span class="ppl-sec__n" id="n-crops"></span></summary>
    <div class="ppl-grid" id="m-crops"></div>
  </details>
  <details class="ppl-sec">
    <summary>Movement record</summary>
    <div class="ppl-article" id="m-article"></div>
  </details>
  <details class="ppl-sec">
    <summary>Route · every camera &amp; time<span class="ppl-sec__n" id="n-route"></span></summary>
    <div class="ppl-route" id="m-route"></div>
  </details>
  <details class="ppl-sec">
    <summary>Behaviour<span class="ppl-sec__n" id="n-behav"></span></summary>
    <div class="ppl-behav" id="m-behav"></div>
  </details>
  <details class="ppl-sec">
    <summary>Local tracks<span class="ppl-sec__n" id="n-tracks"></span></summary>
    <div class="ppl-tracks" id="m-tracks"></div>
  </details>`;
function resetModal() {
  document.getElementById("detail-content").innerHTML = PERSON_SKELETON;
  wireHoverSections();
}
/* Open a section when the mouse enters it, collapse it when the mouse leaves —
 * no clicking needed. The scrollable body is INSIDE the <details>, so hovering
 * to read/scroll keeps it open; the native click still works for touch (iPad). */
function wireHoverSections() {
  document.querySelectorAll("#detail-content .ppl-sec").forEach((sec) => {
    sec.addEventListener("mouseenter", () => { sec.open = true; });
    sec.addEventListener("mouseleave", () => { sec.open = false; });
  });
}
function showDetail() {
  document.getElementById("detail-idle").hidden = true;
  document.getElementById("detail-view").hidden = false;
}
function clearDetail() {
  document.getElementById("detail-view").hidden = true;
  document.getElementById("detail-idle").hidden = false;
  document.querySelectorAll(".ppl-card--active, .veh--active")
    .forEach((e) => e.classList.remove("ppl-card--active", "veh--active"));
}

/* What the VLM said this person was DOING, and — just as usefully — how often it
 * said there was no person there at all. A detector false positive (a traffic
 * cone, a rolled mat) is tracked, given a Global ID and even tagged; the only
 * component that ever disagrees is the one that looks at the whole frame with an
 * open question. */
/* One block per LOCAL TRACK. A local track is one camera, therefore one place —
 * so the room's name, what it is, and what it connects to (all from the Camera
 * graph) can be shown and handed to the LLM. The vision model that judged each
 * frame was never told any of it, which is why it cannot have invented a room. */
async function renderBehaviours(gid) {
  const box = document.getElementById("m-behav");
  box.innerHTML = '<div class="ppl-hint">reading the movement record…</div>';
  let d;
  try { d = await api(`/reid/person/${gid}/story`); }
  catch { box.innerHTML = '<div class="ppl-hint">behaviour service unreachable</div>'; return; }
  if (!d.episodes || !d.episodes.length) {
    box.innerHTML = '<div class="ppl-hint">no behaviour recorded for this person yet</div>';
    return;
  }
  box.innerHTML = d.episodes.map((ep) => `
    <div class="ppl-ep">
      <div class="ppl-ep__head">
        <b>${ep.place || ep.camera}</b>
        <span class="ppl-ep__cam">${ep.camera} · track ${ep.track}</span>
        <span class="ppl-ep__time">${hhmm(ep.first_ms)} → ${hhmm(ep.last_ms)}</span>
      </div>
      ${ep.caption ? `<div class="ppl-ep__place">${ep.caption}</div>` : ""}
      ${ep.neighbours && ep.neighbours.length
        ? `<div class="ppl-ep__nbr">connects to: ${ep.neighbours.join(" · ")}</div>` : ""}
      <div class="ppl-ep__sum">${ep.summary || ""}</div>
      <div class="ppl-ep__evs">${ep.events.map((e) => `
        <div class="ppl-bev${+e.suspicious ? " ppl-bev--susp" : ""}">
          ${e.snapshot ? `<img loading="lazy" src="${snapURL(e.snapshot)}" alt="">`
                       : '<span class="ppl-nocrop">frame not kept</span>'}
          <div>
            <div class="ppl-bev__head"><b>${e.trigger}</b> · ${hhmm(+e.ts_ms)}
              ${+e.suspicious ? '<span class="ppl-flag">flagged</span>' : ""}</div>
            <div class="ppl-bev__act">${e.activity || "—"}</div>
          </div>
        </div>`).join("")}</div>
    </div>`).join("");
  const nb = document.getElementById("n-behav");
  if (nb) nb.textContent = ` ${d.episodes.length}`;
}

/* A Global ID belongs to one day, so its summary is written ONCE after that day
 * closes — not rewritten on every pause, which wasted the LLM. The button forces
 * a fresh write on demand (the emergency case, or to see today's story early). */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* Typewriter for a freshly-written movement summary: the operator asked for it,
 * so let them watch it appear. A blinking caret (CSS) runs while it types. */
async function typeArticle(el, text) {
  el.textContent = "";
  el.classList.add("ppl-article--typing");
  const speed = Math.max(4, Math.min(18, 3000 / Math.max(text.length, 1)));
  for (let i = 1; i <= text.length; i++) {
    el.textContent = text.slice(0, i);
    if (i % 2 === 0) await sleep(speed);
  }
  el.classList.remove("ppl-article--typing");
}

async function renderArticle(gid, force = false) {
  const art = document.getElementById("m-article");
  art.className = "ppl-article";
  art.textContent = force ? "writing…" : "…";
  const btn = document.createElement("button");
  btn.className = "pbtn pbtn--sm ppl-article__btn";
  btn.textContent = force ? "writing…" : "✎ Summarise now";
  btn.disabled = force;
  btn.onclick = () => renderArticle(gid, true);
  let r;
  try { r = await api(`/reid/person/${gid}/article${force ? "?force=true" : ""}`); }
  catch { art.textContent = "article service unreachable"; return; }
  if (r.article) {
    const t = document.createElement("div");
    t.className = "ppl-article__meta";
    t.textContent = (r.made_ms ? `written ${hhmm(r.made_ms)} from ${r.episodes} `
      + `block${r.episodes === 1 ? "" : "s"}`
      + (r.places && r.places.length ? ` · ${r.places.join(" → ")}` : "") : "");
    // Animate ONLY when it was just written on demand (force). Opening a person
    // whose summary already exists shows it instantly — no re-typing.
    if (force) await typeArticle(art, r.article);
    else art.textContent = r.article;
    art.appendChild(t);
  } else {
    art.className = "ppl-article ppl-article--wait";
    art.textContent = r.error
      ? `could not write it: ${r.error}`
      : "not written yet — the whole-day summary is written automatically after "
        + "this person's day ends. Press below to write it now.";
  }
  art.appendChild(btn);
}

async function renderTracks(gid) {
  const box = document.getElementById("m-tracks");
  box.innerHTML = '<div class="ppl-hint">loading tracks…</div>';
  let d;
  try { d = await api(`/reid/person/${gid}/tracks`); }
  catch { box.innerHTML = '<div class="ppl-hint">track evidence unreachable</div>'; return; }
  if (!d.n) { box.innerHTML = '<div class="ppl-hint">no tracks recorded</div>'; return; }
  const warn = (d.rejected_tracks
    ? `<div class="ppl-ghost">⚠ ${d.rejected_tracks} of ${d.n} tracks here were rejected
        by the gate — they predate it, or the identity was built before the check existed.
        New tracks cannot enter an identity until the VLM confirms a person.</div>`
    : "")
    + (d.names && d.names.length
      ? `<div class="ppl-named">🙂 recognised as <b>${d.names.join(", ")}</b> from the
          enrolled gallery.</div>` : "")
    + (d.faces_seen
      ? `<div class="ppl-hint">A face was visible on ${d.faces_seen} of ${d.n} tracks.
          On these overhead cameras that is about one track in ten.</div>` : "");
  const nt = document.getElementById("n-tracks");
  if (nt) nt.textContent = ` ${d.n}`;
  box.innerHTML = warn + `
    <table class="ppl-trk">
      <tr><th>camera</th><th>track</th><th>crops pooled</th>
          <th>what the VLM saw</th><th>gate verdict</th><th>face</th></tr>
      ${d.tracks.map((t) => {
        const bad = t.verdict === "reject";
        return `<tr class="${bad ? "ppl-trk--bad" : ""}">
          <td>${t.camera}</td><td>${t.track}</td><td>${t.observations}</td>
          <td>${t.visibility || "—"}</td>
          <td>${bad ? '<b class="ppl-bad">rejected — not a person</b>'
                    : (t.verdict === "person" ? "person" : "not checked")}</td>
          <td>${t.face_name ? `<b class="ppl-face">${t.face_name}</b> ${t.face_score}`
                            : (t.face_seen ? `seen · ${t.face_score}` : "—")}</td></tr>`;
      }).join("")}
    </table>`;
}

async function openPerson(gid) {
  showDetail();
  resetModal();
  document.getElementById("m-title").textContent = `Global ID G${gid}`;
  document.getElementById("m-desc").textContent = "loading…";
  const r = await api(`/reid/person/${gid}`);
  document.getElementById("m-desc").textContent = r.description || "— no description yet —";
  // identity is decided by the VLM clothing tags, not a cosine — so no score is
  // shown. Each hop just states the camera, time, local track id and crop count.
  document.getElementById("m-route").innerHTML = r.journey.map((j) =>
    `<div class="ppl-hop"><b>${j.camera}</b> ${hhmm(j.ts)}
      ${j.track != null ? `<span class="ppl-lid" title="local track id on this camera">#${j.track}</span>` : ""}
      <span class="${j.matched ? "ok" : "new"}">${j.matched ? "matched" : "new id"}</span>
      <span class="ppl-nobs">${j.n_obs} crop${j.n_obs === 1 ? "" : "s"}</span></div>`).join("");
  const crops = r.crops || [];
  // hover a crop -> its track and what the VLM read there (no score).
  const imgHtml = (o) => {
    const tip = [
      o.track != null ? `track #${o.track}` : "",
      o.vlm ? `VLM: ${o.vlm}` : "VLM: not judged (no tags)",
    ].filter(Boolean).join("  ·  ").replace(/"/g, "'");
    return `<img loading="lazy" class="ppl-crop${o.vlm ? "" : " ppl-crop--novlm"}"`
         + ` src="${cropURL(o.path)}" title="${tip}">`;
  };
  const box = document.getElementById("m-crops");
  box.innerHTML = crops.length
    ? crops.map(imgHtml).join("")
    : '<span class="ppl-nocrop">no crops stored for this person</span>';
  const nc = document.getElementById("n-crops"); if (nc) nc.textContent = crops.length ? ` ${crops.length}` : "";
  const nr = document.getElementById("n-route"); if (nr) nr.textContent = r.journey.length ? ` ${r.journey.length}` : "";
  renderArticle(gid);
  renderBehaviours(gid);
  renderTracks(gid);
}

document.getElementById("m-close").onclick = clearDetail;
const qbox = document.getElementById("q");
qbox.oninput = render;                       // filter live, before Enter
qbox.onchange = (e) => addTerm(e.target.value);       // a datalist pick commits it
qbox.onkeydown = (e) => {
  if (e.key === "Enter") { e.preventDefault(); addTerm(e.target.value); }
  // backspace on an empty box eats the last bubble, as in Gmail
  else if (e.key === "Backspace" && !e.target.value && state.filter.size)
    dropTerm([...state.filter].pop());
};
document.getElementById("search").onclick = () => qbox.focus();
document.getElementById("reload").onclick = load;
showModel();
for (const id of ["hours", "mincam", "minsee", "cam"])
  document.getElementById(id).onchange = load;
load();


const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---- Vehicles ------------------------------------------------------------
 * The same building, seen from the other side: nvr1_ch03 looks at the drive.
 * A vehicle is not a Global ID — nothing re-identifies a car across cameras,
 * because only one camera can see one. So a card here is one visit by one
 * tracker track, and the plate is the majority vote of every readable frame
 * of that visit (usually none: the plate is ~34 px across at this camera).
 * The appearance line comes from the VLM, which is shown the vehicle crop and
 * is never told the plate, so it cannot echo one back to us.
 */
const vfmt = (ms) => new Date(ms).toLocaleString([], {
  month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  second: "2-digit",
});

let _vehicles = [];

function vehicleCard(v) {
  const el = document.createElement("article");
  el.className = "veh";
  const plate = `<span class="veh__plate">${esc(v.plate)}</span>`;
  const snap = v.snapshot
    ? `<img class="veh__img" loading="lazy" src="/api/plate/snapshot?p=${encodeURIComponent(v.snapshot)}" alt="">`
    : `<div class="veh__img veh__img--none">no snapshot</div>`;
  el.innerHTML = `
    ${snap}
    <div class="veh__body">
      <div class="veh__top">${plate}<span class="veh__cls">${esc(v.class)}</span></div>
      <p class="veh__desc">${esc(v.description || "—")}</p>
      <div class="veh__meta">
        <span>${esc(v.camera)}</span><span>trk ${v.track}</span>
        <span>${v.frames} frames</span><span>${vfmt(+v.first_ms)}</span>
      </div>
    </div>`;
  el.onclick = () => {
    document.querySelectorAll(".veh--active")
      .forEach((e) => e.classList.remove("veh--active"));
    el.classList.add("veh--active");
    showVehicle(v);
  };
  return el;
}

function showVehicle(v) {
  showDetail();
  document.getElementById("m-title").textContent =
    v.plate ? v.plate : (v.description || v.class || "Vehicle");
  const rows = [
    ["Colour", v.colour], ["Body", v.body], ["Markings", v.markings],
    ["Camera", v.camera], ["Local track", v.track], ["Class", v.class],
    ["Frames seen", v.frames],
    ["First seen", vfmt(+v.first_ms)], ["Last seen", vfmt(+v.ts_ms)],
    ["Plate", v.plate],
    ["OCR confidence", (+v.plate_conf).toFixed(3)],
    ["Plate size", `${v.plate_px} px wide`],
  ];
  document.getElementById("detail-content").innerHTML = `
    ${v.snapshot ? `<img class="veh__big" src="/api/plate/snapshot?p=${encodeURIComponent(v.snapshot)}" alt="">` : ""}
    <p class="veh__desc veh__desc--big">${esc(v.description || "")}</p>
    <table class="veh__table">${rows.map(([k, val]) =>
      `<tr><th>${k}</th><td>${esc(String(val ?? "") || "—")}</td></tr>`).join("")}</table>
    <p class="ppl-hint">Appearance is written by the VLM from the vehicle crop
      alone. It never sees the plate text, so a colour or body style here is an
      independent observation, not a restatement of the OCR.</p>`;
}

function renderVehicles() {
  const q = document.getElementById("vq").value.trim().toLowerCase();
  const list = document.getElementById("vlist");
  const shown = _vehicles.filter((v) => !q || [
    v.plate, v.colour, v.body, v.markings, v.description, v.camera, v.class,
  ].some((f) => (f || "").toLowerCase().includes(q)));
  list.replaceChildren(...shown.map(vehicleCard));
  document.getElementById("vstat").textContent =
    `${shown.length} vehicle${shown.length === 1 ? "" : "s"}`;
  if (!shown.length) list.innerHTML = `<p class="ppl-hint">No vehicles in this window.</p>`;
}

async function loadVehicles() {
  const hours = document.getElementById("vhours").value;
  document.getElementById("vstat").textContent = "loading…";
  try {
    const r = await fetch(`/api/plate/vehicles?hours=${hours}&limit=300`);
    const j = await r.json();
    _vehicles = j.vehicles || [];
    if (!j.ok) throw new Error(j.error || "plate service unavailable");
  } catch (e) {
    _vehicles = [];
    document.getElementById("vstat").textContent = String(e.message || e);
    document.getElementById("vlist").replaceChildren();
    return;
  }
  renderVehicles();
}

let _vloaded = false;
for (const b of document.querySelectorAll(".trk-tab")) {
  b.onclick = () => {
    const veh = b.dataset.tab === "vehicles";
    for (const o of document.querySelectorAll(".trk-tab")) o.classList.toggle("trk-tab--on", o === b);
    document.getElementById("filters-people").hidden = veh;
    document.getElementById("filters-vehicles").hidden = !veh;
    document.getElementById("list").hidden = veh;
    document.getElementById("vlist").hidden = !veh;
    clearDetail();
    if (veh && !_vloaded) { _vloaded = true; loadVehicles(); }
  };
}
document.getElementById("vreload").onclick = loadVehicles;
document.getElementById("vhours").onchange = loadVehicles;
document.getElementById("vq").oninput = renderVehicles;
