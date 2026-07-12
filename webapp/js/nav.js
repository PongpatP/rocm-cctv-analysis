/* Shared navigation: injects the ☰ menu into every page's topbar.
 * Pages, account (Google sign-in), admin + sign-out entries.
 * Import from any page module or load directly with <script type="module">. */

const LINK_GROUPS = [
  ["Monitor", [
    ["/cameras.html", "Live cameras"],
    ["/dashboard.html", "Dashboard"],
  ]],
  ["Investigate", [
    ["/", "Investigator"],
    ["/playback.html", "Playback"],
    ["/people.html", "Tracking"],
  ]],
  ["Manage", [
    ["/graph.html", "Camera graph"],
    ["/faces.html", "Face registration"],
  ]],
  ["System", [
    ["/system.html", "System monitor"],
    ["/settings.html", "Settings"],
  ]],
  // hidden per owner request (pages still exist, just unlisted):
  // ["/follow.html", "Follow live"], ["/search.html", "Retrospective search"]
]

function h(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}

function currentPath() {
  const p = location.pathname;
  return p === "/index.html" ? "/" : p;
}

/* Live clock, right side of the topbar, in the timezone chosen on the
 * Settings page (falls back to building time). */
function buildClock(meta) {
  if (document.getElementById("topbar-clock")) return;
  const clock = h(`<span class="topbar__clock" id="topbar-clock" title="Current time">
    <span class="topbar__clock-time">--:--:--</span>
    <span class="topbar__clock-sub">…</span></span>`);
  meta.insertBefore(clock, meta.firstChild);
  let tz = "Asia/Bangkok";
  fetch("/api/settings").then((r) => r.json())
    .then((s) => { if (s.timezone) tz = s.timezone; }).catch(() => {});
  const timeEl = clock.querySelector(".topbar__clock-time");
  const subEl = clock.querySelector(".topbar__clock-sub");
  function tick() {
    try {
      const now = new Date();
      timeEl.textContent = new Intl.DateTimeFormat("en-GB", { timeZone: tz,
        hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(now);
      subEl.textContent = new Intl.DateTimeFormat("en-GB", { timeZone: tz,
        weekday: "short", day: "numeric", month: "short" }).format(now)
        + " · " + tz.split("/").pop().replace("_", " ");
    } catch { /* bad zone: keep last shown */ }
  }
  tick();
  setInterval(tick, 1000);
}

function buildMenu() {
  const meta = document.querySelector(".topbar__meta");
  if (!meta || document.getElementById("menu-btn")) return;
  buildClock(meta);

  const btn = h(`<button class="menu-btn" id="menu-btn" title="Menu"
    aria-label="Menu" aria-expanded="false"><span></span><span></span><span></span></button>`);
  meta.appendChild(btn);

  const menu = h(`<div class="menu" id="menu" hidden>
    <div class="menu__section">
      <span class="menu__label">Pages</span>
      <nav class="menu__nav" id="menu-nav"></nav>
    </div>
    <div class="menu__section" id="menu-account">
      <span class="menu__label">Account</span>
      <div class="menu__user">
        <img class="menu__avatar" id="menu-avatar" alt="" hidden referrerpolicy="no-referrer">
        <div>
          <div class="menu__name" id="menu-name">—</div>
          <div class="menu__email" id="menu-email">—</div>
        </div>
      </div>
    </div>
    <div class="menu__section">
      <span class="menu__label">System</span>
      <div class="menu__row"><span>Gateway</span><span id="menu-gateway">—</span></div>
      <div class="menu__row"><span>AI detector</span><span id="menu-ai">—</span></div>
    </div>
    <a class="menu__action menu__action--link" id="menu-admin" href="/access-requests" hidden>People <span class="menu__count" id="menu-new" hidden></span></a>
    <a class="menu__action" id="menu-logout" href="/logout" hidden>Sign out</a>
  </div>`);
  document.body.appendChild(menu);

  const nav = menu.querySelector("#menu-nav");
  const cur = currentPath();
  for (const [group, items] of LINK_GROUPS) {
    const gl = document.createElement("div");
    gl.className = "menu__group";
    gl.textContent = group;
    nav.appendChild(gl);
    for (const [href, label] of items) {
      const a = document.createElement("a");
      a.className = "menu__navlink" + (href === cur ? " menu__navlink--current" : "");
      a.href = href;
      a.textContent = label;
      if (href === cur) a.setAttribute("aria-current", "page");
      nav.appendChild(a);
    }
  }

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
    btn.setAttribute("aria-expanded", String(!menu.hidden));
    if (!menu.hidden) refreshStatus();
  });
  document.addEventListener("click", (e) => {
    if (!menu.hidden && !menu.contains(e.target)) menu.hidden = true;
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") menu.hidden = true;
  });

  loadAccount();
}

async function refreshStatus() {
  const set = (id, ok) => {
    const el = document.getElementById(id);
    if (el && el.textContent === "—") el.textContent = ok ? "online" : "offline";
  };
  try {
    const r = await fetch("/api/streams");
    set("menu-gateway", r.ok);
  } catch { set("menu-gateway", false); }
  try {
    const r = await fetch("/api/settings");   // served by the bridge
    set("menu-ai", r.ok);
  } catch { set("menu-ai", false); }
}

async function loadAccount() {
  const name = document.getElementById("menu-name");
  const email = document.getElementById("menu-email");
  const avatar = document.getElementById("menu-avatar");
  const logout = document.getElementById("menu-logout");
  try {
    const res = await fetch("/auth/me");
    const me = await res.json();
    if (!me.enabled) {
      name.textContent = "Open access";
      email.textContent = "login disabled";
      return;
    }
    if (me.authenticated) {
      name.textContent = me.name || "—";
      email.textContent = me.email || "";
      if (me.picture) { avatar.src = me.picture; avatar.hidden = false; }
      logout.hidden = false;
      if (me.admin) {
        document.getElementById("menu-admin").hidden = false;
        try {
          const n = (await (await fetch("/auth/new-people")).json()).new || 0;
          const b = document.getElementById("menu-new");
          if (n > 0) { b.textContent = `${n} new`; b.hidden = false; }
        } catch { /* count is best-effort */ }
      }
    } else {
      name.textContent = "LAN access";
      email.textContent = "no login required";
    }
  } catch {
    name.textContent = "—";
    email.textContent = "auth unavailable";
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", buildMenu);
} else {
  buildMenu();
}
