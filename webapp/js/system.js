/* System monitor: polls /api/system every 5s.
 * Monochrome industrial scheme: solid black = primary series / this app,
 * dashed or hatched gray = secondary / other, white = free.
 * (Pattern-coded, so it stays readable for colorblind users and print.) */

const POLL_MS = 5000;

const GB = 1024 ** 3;
const fmtGB = (b) => (b / GB).toFixed(b >= 100 * GB ? 0 : 1) + " GB";
const fmtTB = (b) => b >= 1024 ** 4 ? (b / 1024 ** 4).toFixed(2) + " TB" : fmtGB(b);

async function poll() {
  try {
    const res = await fetch("/api/system");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    render(await res.json());
    document.getElementById("status-text").textContent =
      "updated " + new Date().toLocaleTimeString(undefined, { hour12: false });
  } catch (e) {
    document.getElementById("status-text").textContent = "sysmon unreachable: " + e.message;
  }
  setTimeout(poll, POLL_MS);
}

let specsDone = false;

function renderSpecs(s) {
  if (specsDone || !s.specs) return;
  specsDone = true;
  const sp = s.specs;
  const rows = [
    ["Hostname", sp.hostname],
    ["OS", sp.os],
    ["Kernel", sp.kernel],
    ["Architecture", sp.arch],
    ["CPU", sp.cpu_model],
    ["Cores / threads", `${sp.cores_physical} / ${sp.cores_logical}`],
    ["RAM", fmtGB(sp.ram_total)],
    ...sp.gpus.map((g) => [`GPU ${g.index}`,
      `${shortName(g.name)} · ${(g.vram_total / 1024).toFixed(0)} GB VRAM`]),
    ["GPU driver", sp.gpu_driver || sp.nvidia_driver],
  ];
  const holder = document.getElementById("specs");
  holder.innerHTML = "";
  for (const [k, v] of rows) {
    if (!v) continue;
    const row = document.createElement("div");
    row.className = "spec";
    const key = document.createElement("span");
    key.className = "spec__key";
    key.textContent = k;
    const val = document.createElement("span");
    val.className = "spec__val";
    val.textContent = v;
    row.append(key, val);
    holder.appendChild(row);
  }
}

function fmtUptime(sec) {
  const d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600),
    m = Math.floor(sec % 3600 / 60);
  return (d ? d + "d " : "") + h + "h " + m + "m";
}

function render(s) {
  renderSpecs(s);
  document.getElementById("spec-uptime").textContent =
    "uptime " + fmtUptime(s.uptime || 0);
  document.getElementById("t-cpu").textContent = s.cpu.percent.toFixed(0) + "%";
  document.getElementById("t-ram").textContent =
    s.ram.percent.toFixed(0) + "%";
  document.getElementById("t-cpu").title = `${s.cpu.cores} cores · load ${s.cpu.load.map(x => x.toFixed(1)).join(" ")}`;
  document.getElementById("t-ram").title = `${fmtGB(s.ram.used)} / ${fmtGB(s.ram.total)}`;

  const solo = s.gpus.length === 1;
  s.gpus.slice(0, 2).forEach((g, i) => {
    const tile = document.getElementById(`tile-gpu${i}`);
    tile.hidden = false;
    document.getElementById(`t-gpu${i}-label`).textContent =
      `${solo ? "GPU" : "GPU " + g.index} · ${shortName(g.name)}`;
    document.getElementById(`t-gpu${i}`).textContent = g.util.toFixed(0) + "%";
    document.getElementById(`t-gpu${i}`).title =
      `VRAM ${(g.vram_used / 1024).toFixed(1)} / ${(g.vram_total / 1024).toFixed(0)} GB · ${g.temp}°C · ${g.power.toFixed(0)} W`;
  });

  drawChart("chart-host", s.history.t, [
    { vals: s.history.cpu, dash: false },
    { vals: s.history.ram, dash: true },
  ]);

  renderGpuPanels(s);
  renderTemps(s);
  renderDisks(s);
}

function shortName(n) {
  return (n || "GPU")
    .replace(/NVIDIA |GeForce |Workstation Edition|Max-Q /g, "")
    .replace(/AMD |Radeon | VF$/g, "")   // "AMD Instinct MI300X VF" -> "Instinct MI300X"
    .trim();
}

/* ---- per-device temperatures --------------------------------------- */

function tempClass(c, crit) {
  const hot = crit || 90;
  if (c >= hot - 5) return "temp--hot";
  if (c >= hot - 20) return "temp--warm";
  return "temp--ok";
}

function renderTemps(s) {
  const holder = document.getElementById("temps");
  if (!holder) return;
  let list = s.temps || [];
  if ((s.gpus || []).length === 1)          // single GPU: "GPU 0" -> "GPU"
    list = list.map((t) => t.label === "GPU 0" ? { ...t, label: "GPU" } : t);
  if (!list.length) { holder.textContent = "no sensors available"; return; }
  holder.replaceChildren(...list.map((t) => {
    const crit = t.critical || 90;
    const pct = Math.max(4, Math.min(100, (t.current / crit) * 100));
    const cls = tempClass(t.current, crit);
    const el = document.createElement("div");
    el.className = "temp " + cls;
    el.innerHTML =
      `<div class="temp__head">` +
        `<span class="temp__label">${t.label}` +
          `<span class="temp__sub">${shortName(t.sub || t.kind || "")}</span>` +
        `</span>` +
        `<span class="temp__val">${t.current.toFixed(0)}°C</span>` +
      `</div>` +
      `<div class="temp__bar"><span style="width:${pct}%"></span></div>` +
      `<div class="temp__crit">${t.critical ? "crit " + Math.round(t.critical) + "°C" : ""}</div>`;
    return el;
  }));
}

/* ---- GPU panels (created once per GPU) ------------------------------ */

function renderGpuPanels(s) {
  const holder = document.getElementById("gpu-panels");
  for (const [idx, h] of Object.entries(s.history.gpus)) {
    let panel = document.getElementById(`gpu-panel-${idx}`);
    if (!panel) {
      const g = s.gpus.find((x) => String(x.index) === idx);
      panel = document.createElement("section");
      panel.className = "panel";
      panel.id = `gpu-panel-${idx}`;
      panel.innerHTML = `
        <div class="panel__head">
          <h2 class="panel__title">${Object.keys(s.history.gpus).length === 1 ? "GPU" : "GPU " + idx} — ${g ? shortName(g.name) : ""}</h2>
          <div class="chart-legend">
            <span><span class="legend__swatch" style="background:#16181b"></span>util %</span>
            <span><span class="legend__swatch sw-dash"></span>VRAM %</span>
          </div>
        </div>
        <div class="chart" id="gpu-chart-${idx}"></div>`;
      holder.appendChild(panel);
    }
    drawChart(`gpu-chart-${idx}`, s.history.t, [
      { vals: h.util, dash: false },
      { vals: h.vram, dash: true },
    ]);
  }
}

/* ---- % time chart: solid black + dashed gray, fixed 0-100 scale ---- */

function drawChart(id, times, series) {
  const holder = document.getElementById(id);
  if (!holder || !times.length) return;
  const W = holder.clientWidth || 600, H = 130;
  const M = { l: 34, r: 8, t: 8, b: 18 };
  const n = times.length;
  const x = (i) => M.l + (i / Math.max(1, n - 1)) * (W - M.l - M.r);
  const y = (v) => M.t + (1 - v / 100) * (H - M.t - M.b);

  let g = "";
  for (const v of [0, 50, 100]) {
    g += `<line x1="${M.l}" y1="${y(v)}" x2="${W - M.r}" y2="${y(v)}" stroke="#e9ebee"/>`
      + `<text x="${M.l - 5}" y="${y(v) + 3}" text-anchor="end" class="ax">${v}</text>`;
  }
  const fmt = (t) => new Date(t * 1000).toLocaleTimeString(undefined,
    { hour: "2-digit", minute: "2-digit", hour12: false });
  for (const i of [0, n - 1]) {
    g += `<text x="${x(i)}" y="${H - 4}" text-anchor="${i ? "end" : "start"}" class="ax">${fmt(times[i])}</text>`;
  }

  let lines = "";
  for (const srs of series) {
    const off = srs.vals.length < n ? n - srs.vals.length : 0;
    const d = srs.vals.map((v, i) =>
      `${i ? "L" : "M"}${x(i + off).toFixed(1)},${y(Math.min(100, v)).toFixed(1)}`).join("");
    lines += `<path d="${d}" fill="none" stroke="${srs.dash ? "#676e76" : "#16181b"}"
      stroke-width="2" ${srs.dash ? 'stroke-dasharray="5 4"' : ""} stroke-linejoin="round"/>`;
  }

  holder.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:${H}px;width:100%">`
    + `<style>.ax{font:9px ui-monospace,monospace;fill:#676e76}</style>`
    + g + lines + `</svg>`;
}

/* ---- storage bars ----------------------------------------------------- */

function renderDisks(s) {
  const holder = document.getElementById("disks");
  holder.innerHTML = "";
  for (const d of s.disks) {
    const other = Math.max(0, d.used - (d.app_bytes || 0));
    const row = document.createElement("div");
    row.className = "disk";

    const head = document.createElement("div");
    head.className = "disk__head";
    const dev = document.createElement("span");
    dev.className = "disk__dev";
    dev.textContent = `${d.device}  →  ${d.mount}`;
    const nums = document.createElement("span");
    nums.className = "disk__nums";
    nums.textContent =
      (d.app_bytes ? `app ${fmtTB(d.app_bytes)} · ` : "") +
      `used ${fmtTB(d.used)} · free ${fmtTB(d.free)} · total ${fmtTB(d.total)}`;
    head.append(dev, nums);

    const bar = document.createElement("div");
    bar.className = "disk__bar";
    const segApp = document.createElement("div");
    segApp.className = "seg seg--app";
    segApp.style.width = (100 * (d.app_bytes || 0) / d.total) + "%";
    segApp.title = `this app: ${fmtTB(d.app_bytes || 0)}`;
    const segOther = document.createElement("div");
    segOther.className = "seg seg--other";
    segOther.style.width = (100 * other / d.total) + "%";
    segOther.title = `other: ${fmtTB(other)}`;
    const segFree = document.createElement("div");
    segFree.className = "seg seg--free";
    segFree.title = `free: ${fmtTB(d.free)}`;
    bar.append(segApp, segOther, segFree);

    row.append(head, bar);
    holder.appendChild(row);
  }

  const a = s.app;
  document.getElementById("app-note").textContent =
    `this app's data: recordings ${fmtTB(a.dirs["recordings"] || 0)} · ` +
    `detections ${fmtTB(a.dirs["output"] || 0)} · models ${fmtTB(a.dirs["ai/models"] || 0)} ` +
    `— total ${fmtTB(a.total || 0)}`;
}

poll();
