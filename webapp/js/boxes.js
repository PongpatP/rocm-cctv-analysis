/* Shared detection-box helpers for the overlay renderers. */

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

/* Merge near-duplicate detections of the same target (the detector can
 * emit several overlapping boxes for one object, especially at low
 * thresholds — sometimes with different classes). Keeps the most
 * confident box of each overlapping group. */
export function iouDedupe(records, thr = 0.55) {
  const sorted = records.slice().sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  const kept = [];
  for (const r of sorted) {
    if (!r.bounding_box) continue;
    if (kept.some((k) => iou(k.bounding_box, r.bounding_box) > thr)) continue;
    kept.push(r);
  }
  return kept;
}

/* Keep only the newest inference batch from a mixed list (live WS
 * messages can span 2-3 batches ~125ms apart -> ghost multi-boxes). */
export function latestBatch(records) {
  if (records.length < 2) return records;
  let latest = "";
  for (const r of records) {
    if (r.timestamp > latest) latest = r.timestamp;
  }
  return records.filter((r) => r.timestamp === latest);
}

/* Box colour per object class — ONE source of truth for every page
 * (front page tiles/viewer + 2D->3D tracking). Unknown classes get a
 * stable fallback hue. */
const CLASS_COLORS = {
  person: "#ff6a00",      // signal orange
  car: "#2ec5ff",         // light blue
  truck: "#b085ff",       // violet
  bus: "#ffd23c",         // amber
  motorcycle: "#ff4fa3",  // magenta
  bicycle: "#3ddc84",     // green
};
const FALLBACK_COLORS = ["#ffd23c", "#7fdbca", "#f97b7b", "#9bb8ff", "#e4c1f9"];

export function classColor(name) {
  if (CLASS_COLORS[name]) return CLASS_COLORS[name];
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return FALLBACK_COLORS[h % FALLBACK_COLORS.length];
}
