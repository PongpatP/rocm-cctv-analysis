/* Face registration page: upload photos + name + role, manage the
 * registry that the (future) recognition service consumes. */

const form = document.getElementById("reg-form");
const nameEl = document.getElementById("name");
const roleEl = document.getElementById("role");
const filesEl = document.getElementById("files");
const drop = document.getElementById("drop");
const previews = document.getElementById("previews");
const note = document.getElementById("note");
const saveBtn = document.getElementById("save");
const peopleEl = document.getElementById("people");

let picked = [];   // File objects queued for upload

/* ---- image picking + previews ------------------------------------ */

drop.addEventListener("click", () => filesEl.click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drop--over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("drop--over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("drop--over");
  addFiles(e.dataTransfer.files);
});
filesEl.addEventListener("change", () => addFiles(filesEl.files));

function addFiles(list) {
  for (const f of list) {
    if (!/^image\/(jpeg|png|webp)$/.test(f.type)) continue;
    if (f.size > 10 * 1024 * 1024) { note.textContent = `${f.name}: over 10MB, skipped`; continue; }
    if (picked.length >= 10) break;
    picked.push(f);
  }
  renderPreviews();
}

function renderPreviews() {
  previews.innerHTML = "";
  picked.forEach((f, i) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    img.title = `${f.name} — click to remove`;
    img.style.cursor = "pointer";
    img.addEventListener("click", () => { picked.splice(i, 1); renderPreviews(); });
    previews.appendChild(img);
  });
  drop.textContent = picked.length
    ? `${picked.length} photo${picked.length > 1 ? "s" : ""} selected — click to add more`
    : "click or drop images here";
}

/* ---- register ------------------------------------------------------ */

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!picked.length) { note.textContent = "add at least one photo"; return; }
  saveBtn.disabled = true;
  note.textContent = "uploading…";
  try {
    const fd = new FormData();
    fd.append("name", nameEl.value);
    fd.append("role", roleEl.value);
    for (const f of picked) fd.append("images", f);
    const res = await fetch("/api/faces", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
    note.textContent = `registered ${body.name} (${body.images.length} photos)`;
    form.reset();
    picked = [];
    renderPreviews();
    await loadPeople();
  } catch (err) {
    note.textContent = "failed: " + err.message;
  }
  saveBtn.disabled = false;
});

/* ---- people list ---------------------------------------------------- */

async function loadPeople() {
  const res = await fetch("/api/faces");
  const items = await res.json();
  document.getElementById("count").textContent =
    `${items.length} ${items.length === 1 ? "person" : "people"}`;
  peopleEl.innerHTML = "";
  if (!items.length) {
    peopleEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:14px 0;">No one registered yet.</div>';
    return;
  }
  for (const p of items.slice().reverse()) {
    peopleEl.appendChild(personRow(p));
  }
}

function imgUrl(p, f) {
  return `/api/faces/image/${p.id}/${f}`;
}

function personRow(p) {
  const row = document.createElement("div");
  row.className = "person";

  const photo = document.createElement("img");
  photo.className = "person__photo";
  if (p.images?.length) photo.src = imgUrl(p, p.images[0]);

  const mid = document.createElement("div");
  const nameLine = document.createElement("div");
  const nm = document.createElement("span");
  nm.className = "person__name";
  nm.textContent = p.name;
  nameLine.appendChild(nm);
  if (p.role) {
    const role = document.createElement("span");
    role.className = "person__role";
    role.textContent = p.role;
    nameLine.appendChild(role);
  }
  const meta = document.createElement("div");
  meta.className = "person__meta";
  meta.textContent = `${p.images?.length || 0} photos · added ` +
    new Date((p.created_at || 0) * 1000).toLocaleDateString();
  const thumbs = document.createElement("div");
  thumbs.className = "person__thumbs";
  for (const f of (p.images || []).slice(1, 8)) {
    const t = document.createElement("img");
    t.src = imgUrl(p, f);
    thumbs.appendChild(t);
  }
  mid.append(nameLine, meta, thumbs);

  const acts = document.createElement("div");
  acts.className = "person__acts";
  const addBtn = document.createElement("button");
  addBtn.className = "mini";
  addBtn.textContent = "Add photos";
  addBtn.addEventListener("click", () => addPhotos(p));
  const delBtn = document.createElement("button");
  delBtn.className = "mini mini--danger";
  delBtn.textContent = "Delete";
  delBtn.addEventListener("click", async () => {
    if (!confirm(`Delete ${p.name} and all photos?`)) return;
    await fetch(`/api/faces/${p.id}`, { method: "DELETE" });
    loadPeople();
  });
  acts.append(addBtn, delBtn);

  row.append(photo, mid, acts);
  return row;
}

function addPhotos(p) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/jpeg,image/png,image/webp";
  input.multiple = true;
  input.addEventListener("change", async () => {
    const fd = new FormData();
    for (const f of input.files) fd.append("images", f);
    const res = await fetch(`/api/faces/${p.id}/images`, { method: "POST", body: fd });
    if (!res.ok) alert("upload failed");
    loadPeople();
  });
  input.click();
}

loadPeople();
