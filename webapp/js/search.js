/* Retrospective search: ask in words, get an answer that can only be built from
 * stored behaviour records, each one showing the exact frame it was judged on.
 *
 * The model is explicitly told to say "the records do not answer that" rather
 * than guess — a CCTV search that invents an event is worse than no search. */
const post = (p, b) => fetch("/api/reid" + p, { method: "POST",
  headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) })
  .then((r) => r.json());
const snapURL = (p) => `/api/reid/behavior/snapshot?p=${encodeURIComponent(p)}`;
const when = (ms) => new Date(+ms).toLocaleString([], {
  month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });

const qEl = document.getElementById("q");
const noteEl = document.getElementById("note");
const ansEl = document.getElementById("answer");
const evEl = document.getElementById("events");
const hintEl = document.getElementById("hint");

/* The model cites lines as [3]; turn those into links that scroll to the event. */
function linkCitations(text) {
  return text.replace(/\[(\d+)\]/g, (_, n) => `<a class="srch-cite" href="#ev${n}">[${n}]</a>`);
}

async function ask(question) {
  if (!question.trim()) return;
  qEl.value = question;
  noteEl.textContent = "asking…";
  ansEl.hidden = false;
  ansEl.textContent = "…";
  evEl.innerHTML = "";
  hintEl.hidden = true;
  const hours = +document.getElementById("hours").value;
  let d;
  try { d = await post("/behavior/ask", { question, hours, limit: 60 }); }
  catch (e) { noteEl.textContent = "search unreachable"; ansEl.textContent = ""; return; }
  if (!d.ok) {
    noteEl.textContent = "";
    ansEl.textContent = `failed: ${d.error || "?"}`;
    return;
  }
  noteEl.textContent = `${d.n_considered || 0} records read`
    + (d.terms && d.terms.length ? ` · matched on ${d.terms.join(", ")}` : "");
  ansEl.innerHTML = linkCitations(d.answer);
  evEl.innerHTML = (d.events || []).map((e, i) => `
    <div class="srch-ev${+e.suspicious ? " srch-ev--susp" : ""}" id="ev${i + 1}">
      <span class="srch-ev__n">[${i + 1}]</span>
      ${e.snapshot ? `<img loading="lazy" src="${snapURL(e.snapshot)}" alt="">`
                   : '<span class="srch-nosnap">frame not kept</span>'}
      <div class="srch-ev__body">
        <div class="srch-ev__meta">
          ${when(e.ts_ms)} · <b>${e.camera}</b> · ${e.trigger}
          ${+e.global_id ? ` · <a href="/people.html?gid=${e.global_id}">G${e.global_id}</a>`
                         : ` · track ${e.track}`}
          ${+e.suspicious ? '<span class="srch-flag">flagged</span>' : ""}
        </div>
        <div class="srch-ev__act">${e.activity}</div>
      </div>
    </div>`).join("")
    || '<div class="srch-hint">No stored behaviour matched. Widen the time range.</div>';
}

document.getElementById("ask").onclick = () => ask(qEl.value);
qEl.onkeydown = (e) => { if (e.key === "Enter") ask(qEl.value); };
document.getElementById("hours").onchange = () => { if (qEl.value) ask(qEl.value); };
for (const b of document.querySelectorAll("#examples button"))
  b.onclick = () => ask(b.dataset.q);
