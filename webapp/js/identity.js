/* Names and licence plates for the live overlays.
 *
 * Both services answer per LOCAL TRACK — the face reader because a face is
 * recognised from several crops of one track, the plate reader because a plate is
 * voted on across the frames one vehicle is in view. So the overlay looks up
 * `camera:track`, not a box.
 *
 * Polled once every few seconds and cached. The detection WebSocket runs at frame
 * rate; a fetch per box would melt the browser and the services.
 *
 * Both pages import this. It exists so the label logic lives in exactly one file:
 * the live grid (app.js) and the AI Lab (track23.js) draw the same truth. */
const faces = new Map();     // "camera:track" -> {name, score}
const plates = new Map();    // "camera:track" -> {plate, confidence}
const key = (cam, trk) => `${cam}:${trk}`;

let started = false;

async function pullFaces() {
  try {
    const r = await (await fetch("/api/face/recent?hours=2&limit=200")).json();
    if (r.ok) for (const f of r.faces) {
      if (f.person_name) faces.set(key(f.camera, f.track),
        { name: f.person_name, score: +f.score });
    }
  } catch { /* the face service may be restarting; keep the last names */ }
}

/* A car crosses the gate in about three seconds, so the plate cache is rebuilt
 * every second, not every four. `/live` is the running vote on vehicles still in
 * frame; `/recent` covers the seconds between a car leaving and its row landing.
 *
 * The map is REBUILT rather than merged: the tracker reuses track ids, and a stale
 * entry would paint the previous car's plate onto a new one. Five minutes of
 * history is the most that can be justified for a live overlay. */
async function pullPlates() {
  const next = new Map();
  for (const url of ["/api/plate/live", "/api/plate/recent?hours=0.083&limit=100"]) {
    try {
      const r = await (await fetch(url)).json();
      if (r.ok) for (const p of r.plates) {
        if (!next.has(key(p.camera, p.track)))
          next.set(key(p.camera, p.track), { plate: p.plate, confidence: +p.confidence });
      }
    } catch { return; }        // service restarting: keep what we had
  }
  plates.clear();
  for (const [k, v] of next) plates.set(k, v);
}

export function startIdentityFeed(periodMs = 4000) {
  if (started) return;
  started = true;
  pullFaces();
  pullPlates();
  setInterval(pullFaces, periodMs);
  setInterval(pullPlates, 1000);
}

/** The extra text to append to a detection's box label, or "". */
export function identityLabel(camera, record) {
  const trk = record.track_id;
  if (trk == null) return "";
  const f = faces.get(key(camera, trk));
  if (f) return `  ${f.name}`;
  const p = plates.get(key(camera, trk));
  if (p) return `  ${p.plate}`;
  return "";
}

/** True when this box carries a recognised name — the overlay paints it green. */
export function isNamed(camera, record) {
  return record.track_id != null && faces.has(key(camera, record.track_id));
}

/* ---- privacy ------------------------------------------------------------
 * The mirror image of writing a name over someone's head: when we do NOT know
 * who they are, we cover the head instead. Admins see through it — they are the
 * people the system exists to serve — everyone else sees a public view.
 *
 * `/auth/me` already reports `admin`. Until it answers, assume NOT admin: a page
 * that shows faces for the first two seconds after load has not protected anyone.
 */
let isAdmin = false;
let adminKnown = false;

export async function loadViewer() {
  try {
    const r = await fetch("/auth/me");
    const j = await r.json();
    isAdmin = !!j.admin;
  } catch { isAdmin = false; }
  adminKnown = true;
  return isAdmin;
}

/** Should this person's head be covered on screen? */
export function shouldCensor(camera, record) {
  if (isAdmin) return false;
  if (record.class_name !== "person") return false;
  return !isNamed(camera, record);
}

export const viewerIsAdmin = () => isAdmin;
export const viewerKnown = () => adminKnown;

/* The box to draw on someone's face.
 *
 * First choice is SCRFD's own face box, decoded in the detector probe and
 * carried on the record — a real face, on the same frame as the person box.
 * When SCRFD sees no face (turned away, too small, too dark) we fall back to the
 * head implied by COCO joints 0-4, and past that to the top slice of the person
 * box. The last two are heads, not faces; `record.face` is the only one that
 * means "a face was detected here". */
export function faceBox(record) {
  const f = record.face;
  if (f) return [f.x, f.y, f.w, f.h];
  return headBox(record);
}

export const hasRealFace = (record) => !!record.face;

/* COCO joints 0 nose, 1-2 eyes, 3-4 ears. When the pose model saw the head we
 * cover exactly the head; when it did not (facing away, occluded, low light) we
 * fall back to the top slice of the person box. Never fail open. */
export function headBox(record) {
  const { x, y, w, h } = record.bounding_box;
  const kp = record.keypoints;
  if (kp && kp.length >= 5) {
    const pts = kp.slice(0, 5).filter((k) => k[2] > 0.3);
    if (pts.length) {
      const xs = pts.map((k) => k[0]), ys = pts.map((k) => k[1]);
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
      const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      // a head is about a fifth of a standing body; pad generously around the joints
      const r = Math.max(Math.max(...xs) - Math.min(...xs),
                         Math.max(...ys) - Math.min(...ys), h * 0.10) * 1.5;
      return [cx - r, cy - r * 1.15, r * 2, r * 2.3];
    }
  }
  return [x + w * 0.18, y, w * 0.64, h * 0.26];
}
