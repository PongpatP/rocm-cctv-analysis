# ROCm-ccvt-monitoring

CCTV monitoring with live AI object detection, in one system, run as one
Docker Compose stack behind a single port. The whole AI stack runs on a
single **AMD Instinct MI300X** (~191 GB VRAM, ROCm) — detection, tracking,
cross-camera ReID, face, plate, and the vision-language model all share
that one GPU.

- **Proxy** (Caddy) — the only published port (:8080); routes `/ws` to the
  bridge and everything else to the gateway.
- **Gateway** (go2rtc) — pulls every NVR camera once; serves the web UI and
  video (H.264 passthrough, MSE) and an internal RTSP re-stream for the AI.
- **AI detector** (`airocm`) — **RT-DETR R50** (Apache-2.0) as ONNX, run via
  **onnxruntime-migraphx** on the MI300X, plus a clean-room **BoT-SORT**
  tracker (Kalman + IoU two-stage) with **OSNet** appearance ReID — all in
  one Python service (`ai_rocm/airocm.py`).
- **Bridge** — receives detections, archives to JSONL, broadcasts to
  browsers over WebSocket.
- **Web UI** — white industrial theme; detection boxes are a canvas
  overlay drawn in the browser (the video itself stays clean).

```
                          single public port
browser ──────────────► proxy :8080 ─┬─ /ws ──► bridge ◄── detections ─┐
                                     └─ (rest) ► gateway               │
                                                  │  ▲                 │
camera source (see modes) ── one conn/camera ─────┘  │                 │
                                                     └ RTSP re-stream ▼
                                       AI detector (airocm, RT-DETR on ROCm)

internal compose network: gateway:1984/8554, bridge:8081 — no host ports
detections archive: output/detections.jsonl
```

## Quick start

You need an **AMD Instinct GPU host with ROCm installed** (`/opt/rocm`) and
**Docker + Docker Compose**. Nothing in `docker-compose*.yml` is machine-specific —
the ROCm device mounts (`/dev/kfd`, `/dev/dri`, `/opt/rocm`) are the same on every
AMD host, so you do **not** edit any compose or Dockerfile. Only two things are
per-deployment: your config and the model weights.

```bash
# 1. get the code
git clone <this-repo> && cd <repo>

# 2. your settings (cameras, credentials, OAuth) — never committed
cp config.example.yaml config.local.yaml
$EDITOR config.local.yaml

# 3. model weights (not in git — too large; see "Models")
bash download_models.sh

# 4. generate the runtime configs from config.local.yaml
./configure.py

# 5. build the images from source and start everything
docker compose up -d --build
```

Open **http://\<host\>:8080** — the only port. First build compiles the RT-DETR
MIGraphX cache (~9 min once). That's it — every container is rebuilt from this
repo, so you never need anyone else's images.

## The AI services (all on the MI300X)

| Service | What it does | How it runs |
|---|---|---|
| `airocm` | detection + in-camera tracking | RT-DETR R50 ONNX via onnxruntime-migraphx; BoT-SORT + OSNet ReID in Python |
| `siglip` | cross-camera ReID (Global IDs, sightings) | YoutuReID embedder (Apache-2.0) via onnxruntime-migraphx |
| `face` | face detection + recognition | cvlface ONNX on MIGraphX + MediaPipe alignment (CPU) |
| `plate` | licence-plate read | plate detector on MIGraphX + PaddleOCR (CPU) |
| `vllm` | vision-language + LLM backend | vLLM (ROCm) serving Google Gemma 4 31B at bf16 |
| `vlmscan` | video captioning | drives the VLM over the vLLM API |

GPU access is via the `/dev/kfd` + `/dev/dri` device mounts and the host's
`/opt/rocm` bind-mounted read-only — no vendor container runtime is needed.
The detector pushes person crops to `siglip` for Global-ID assignment; each
GPU service persists its compiled MIGraphX cache (`.mxr`) so restarts skip
the first-boot compile.

## AI Investigator, Tracking & Playback

Three operator apps sit on top of the pipeline — all served from the same
single port, all powered by the on-box models:

- **Tracking** (`/people.html`) — every person the system linked across cameras
  into a **Global ID**, with their snapshots, cross-camera route, a per-minute
  behaviour story, and a search box ("green shirt", "G398", a camera name).
  Identities are decided by the **VLM's clothing description** (colour, garment,
  silhouette) — a link a human can read and vouch for, not a black-box vector.

- **Investigator** (`/`) — a chat where you ask in plain English (or Thai) and
  the assistant *investigates*: it calls tools over the recorded data (who was
  where and when, find a person by looks, count objects, pull a clip, draw a
  chart, read the camera map) and answers with evidence. Built to be reliable on
  a **small 31B model**: the intelligence lives in **"fat tools"** (the logic is
  in code, not the prompt), a **deterministic grounding gate** rejects any
  invented id/time, and a **verifier agent** re-checks every answer. It never
  dead-ends — an empty result comes back with the nearest times that *do* have
  data and how far back the records go.

- **Playback** (`/playback.html`) — recorded footage with a scrub timeline and
  event markers, plus a plain-language search that jumps to the moment ("a group
  of people on bicycles") and a "move to the storage room" command that switches
  cameras for you.

```
                    ┌─────────────────────────────────────────────┐
  operator ───────► │  Investigator chat   Tracking   Playback     │  web UI
                    └───────────────┬─────────────────────────────-┘
                                    │ tools (search people / objects / clips…)
                                    ▼
        detections · sightings · behaviours · captions   (ClickHouse + files)
                                    ▲
        RT-DETR detection · BoT-SORT tracking · VLM tags · Global-ID ReID
                                    ▲
                        one AMD Instinct MI300X (ROCm)
```

## Camera source: two modes

The gateway can pull cameras from two places, chosen per machine with
`gateway.upstream` in `config.local.yaml`:

| | `direct` | `forwarder` |
|---|---|---|
| When | machine can reach the CCTV subnet (`ping 192.168.0.163` works) | plain LAN client (ping fails) |
| Streams from | the NVRs themselves | an upstream go2rtc relay at `192.168.1.151:8554` |
| Credentials | in this machine's `config.local.yaml` | only on the relay |

Everything downstream (AI, UI, bridge, stream names, output) is identical
in both modes. To switch: edit `upstream:`, then
`./configure.py && docker compose up -d --force-recreate gateway`.
Migration details: [AI_SERVER_CHANGES.md](AI_SERVER_CHANGES.md).

## Setup (once)

```bash
cp config.example.yaml config.local.yaml   # credentials + upstream mode
./configure.py                             # generate runtime configs
```

Requires: Docker + a working ROCm install on the host (`/opt/rocm`).

### Models

Model weights are **not committed** (they are large — the RT-DETR detector alone
is ~160 MB, over GitHub's per-file limit). Fetch them with:

```bash
MODELS_URL="https://github.com/<owner>/<repo>/releases/download/models-v1/models.tar.gz" \
  bash download_models.sh
```

The maintainer publishes the weights as a **GitHub Release** asset
(`models.tar.gz`, ~200 MB — build it with `tar -czf models.tar.gz ai/models
face/models plate/models` and attach it to a release). The files it contains,
each a standard public checkpoint:

| Path | Model |
|---|---|
| `ai/models/rtdetr_r50_uint8.onnx` | RT-DETR R50 detector (Apache-2.0) |
| `ai/models/face_detection_yunet_2023mar.onnx` | YuNet face detector |
| `face/models/recognition.onnx`, `face/models/face_landmarker.task` | face recognition + MediaPipe landmarker |
| `plate/models/…` | plate detector (YOLOv9-t) + PaddleOCR |
| `output/siglip/cache/reid/youtureid.onnx` | YoutuReID embedder (auto-downloaded on first boot) |

The RT-DETR model is compiled to a MIGraphX cache on the detector's first boot
(~9 min once; ~5 s from cache after).

## Run

All commands from the repo root:

```bash
docker compose up -d        # start everything
docker compose down         # stop everything
docker compose restart      # restart everything
docker compose ps           # status of the services
docker compose logs -f airocm   # follow detector logs (proxy|gateway|bridge|airocm|siglip|...)
rocm-smi                    # MI300X utilization / VRAM / power
```

Web UI: http://\<host\>:8080 — that's the only port; open it in ufw /
Cloudflare and you're done. Services restart automatically (crash or
reboot) thanks to `restart: unless-stopped`.

Health checks: `tail -f output/detections.jsonl` (records should keep
appearing), and in the UI header the Gateway and AI detector LEDs
should both be green.

## Configuration & secrets

All machine-specific settings live in **`config.local.yaml`** (NVRs, camera
credentials, Google OAuth, upstream mode, detector classes/thresholds). It is
**git-ignored** — copy the template and fill it in:

```bash
cp config.example.yaml config.local.yaml   # then edit: cameras + credentials
./configure.py                             # regenerate the runtime configs
docker compose up -d --force-recreate gateway airocm
```

`./configure.py` regenerates `gateway/go2rtc.yaml`, `webapp/config.json` and
`ai/configs/cameras.yaml` from it. Those generated files are **git-ignored too**
(they embed camera RTSP credentials), so every deployment produces its own —
never edit a generated file by hand. Webapp files are served straight from
disk: UI tweaks only need a browser refresh.

AI model providers/keys live in **`output/ai_settings.json`** (also git-ignored);
see **`ai_settings.example.json`** for the shape. The default provider is the
**on-box Gemma** (no key needed) — a cloud key (Anthropic/Google) is only
required if you deliberately switch to a cloud model.

> **Secrets never enter this repo.** `config.local.yaml`, `gateway/go2rtc.yaml`,
> `auth/.state/` and `output/ai_*.json` are all git-ignored. If you fork it,
> keep them that way — and rotate any key that was ever committed.

## Login (Google sign-in)

With `auth.enabled: true` in `config.local.yaml`, the proxy requires a
Google sign-in for every page and stream; only `allowed_emails` /
`allowed_domains` get in. Sessions are signed cookies (7 days); the ☰
menu in the header shows the signed-in account and sign-out.

- Register `https://<public-host>/auth/google/callback` as an Authorized
  redirect URI on the Google OAuth client.
- `skip_lan: true` (default) lets LAN/localhost in without login — Google
  forbids http redirect URIs on private IPs; the gate protects the
  public (Cloudflare) hostname, ufw protects the LAN.

## Recording & playback

The `recorder` service records **every camera continuously** (H.264
stream copy, no transcode, no audio) into 60s MP4 segments under
`recordings/<camera>/`, keeping **24 hours** (sub-stream quality ≈
60–65 GB/day for 28 cameras). Browse it at **/playback.html**: camera +
date, hour grid, minute segments, auto-advancing player. Tuning knobs
in docker-compose.yml: `RETENTION_HOURS`, `RECORDER_PROFILE`
(`sub`/`main` — main is ~10× the storage).

## Detection output

`output/detections.jsonl`, one record per tracked object per frame:

```json
{"camera_id":"nvr1_ch03","timestamp":"2026-07-03T09:00:00.123+07:00",
 "class_name":"person","confidence":0.87,
 "bounding_box":{"x":412.0,"y":233.5,"w":58.2,"h":141.0},
 "frame":{"width":960,"height":544,"number":1234},"track_id":17}
```

Boxes are pixels in the AI processing frame; normalize by
`frame.width/height` to map onto any view. The same records stream live on
`ws(s)://<host>/ws` (via the proxy). `track_id` is the in-camera BoT-SORT
track; cross-camera Global IDs come from the `siglip` service.

## 3D Map — camera calibration with MapAnything (step toward a digital twin)

A scan captures one sharp **main-stream** frame per camera and reconstructs
all cameras jointly in **metric 3D** with
[MapAnything](https://github.com/facebookresearch/map-anything)
(Meta's feed-forward multi-view reconstruction transformer), on the
**MI300X**. We load the **`facebook/map-anything-apache`** checkpoint —
Apache-2.0, commercially clean (the default checkpoint is CC-BY-NC; do not
switch to it).

This replaced the earlier SuperPoint+LightGlue+COLMAP SfM path (`hloc`),
which registered only 2/27 cameras on this building's wide-baseline,
low-texture, repetitive indoor views. MapAnything is feed-forward (no
matching/triangulation), predicts a pose + per-pixel metric 3D points for
*every* view, and outputs real-world scale in metres.

Start a scan with the **▶ Run scan** button on **/map3d.html** (live
progress bar, page refreshes itself when done) or from a terminal:

```bash
bash scripts/run_calib.sh       # same trigger via the API, with progress
```

The `calib` container idles as a tiny control server (`/api/calib/run`,
`/api/calib/status`, `/api/calib/floors`); GPU memory is only used while a
scan runs. First run downloads the model weights into `output/calib/cache`.

Results at **/map3d.html**: dense textured **mesh** (demo-style, per-view
triangulated depth maps in `result/scene.glb`) plus the metric point
cloud and camera poses. Viewer controls: mesh/points display mode,
**alpha** (opacity), **per-floor filter**, point size, per-point
confidence threshold, colour modes (true colour / confidence / height /
floor). Also a per-pair **mutual frustum overlap** matrix, a table with
per-view validity, and thumbnails with model-invalid pixels dimmed.

**/demo3d.html — Demo 3D map**: upload your own images or a video of any
scene (one frame sampled every N seconds, 0.02–10 s adjustable like the
official demo; no fixed view limit — hundreds of views auto-switch to
memory-efficient inference), press Reconstruct, and explore the dense
mesh in the same viewer. Save/download buttons archive the video and the
resulting .glb on the server (`output/calib/saved/`) or download them.
Dropping a **.glb** file just displays it (browser-side only).

**Walkthrough aid videos:** film a floor with a phone (walk slowly, include
what the CCTV cameras see) and upload it on the page assigned to that
floor — the next scan reconstructs those frames together with the CCTV
views, giving the model the moving-camera structure the fixed views lack.
Detail level per scan: standard / fine / extremely fine (518/1036/1554 px
inference). After a scan, any view (CCTV or aid) can be manually adjusted
in the viewer: pick it in "adjust view" and use W/A/S/D + Q/E to move,
arrow keys to rotate, R to reset — changes render live and persist on the
server (`output/calib/adjust.json`).

**Per-floor layout (defeats perceptual aliasing):** floors 2–8 have
identical-looking corridors, and a joint reconstruction merges those
clones into one volume. So the scan reconstructs **each floor's cameras
independently** (grouping from `calib/floors.json`, editable on the page),
levels every group to its RANSAC-detected ground plane, and stacks the
groups at `floor_height_m` (default 3 m) per storey. In-floor geometry is
model-measured; vertical placement is ground truth. Overlap is computed
within a floor only. A walkthrough capture pass can later replace the
assumed stacking with measured stairwell geometry for the full twin.

## Repo layout

| Path | What |
|---|---|
| `docker-compose.yml` + `docker-compose.override.yml` | the whole stack; the override carries the AMD/ROCm device mounts and MIGraphX settings |
| `configure.py`, `config.example.yaml` | single-source-of-truth config |
| `proxy/Caddyfile` | single-port routing |
| `gateway/` | generated go2rtc stream config |
| `webapp/` | UI (plain HTML/CSS/JS, canvas overlay for AI boxes) |
| `bridge/` | FastAPI detections bridge + playback/faces APIs |
| `recorder/` | continuous segment recorder (ffmpeg, 24h retention) |
| `ai_rocm/` | `airocm.py` — RT-DETR detector + BoT-SORT/OSNet tracker |
| `ai/` | detector models (`models/`), camera configs, detector docs |
| `siglip/`, `face/`, `plate/`, `vlmscan/` | ReID, face, plate, video-captioning services |
| `calib/` | MapAnything metric calibration job (3D Map) |
| `scripts/` | ops helpers (`run_calib.sh`, benchmarks, backfills) |

## Notes

- **Cloudflare Tunnel**: point one route at `http://localhost:8080` —
  nothing else. (A separate `/ws` route is no longer needed; delete it
  if present, since port 8081 is not published anymore.)
- **GPU access**: every GPU service mounts `/dev/kfd` + `/dev/dri` and the
  host's `/opt/rocm` (read-only) — the image itself carries no ROCm, so the
  userspace always matches the host amdgpu driver. No vendor container
  runtime is involved.
- The AI reads cameras via the gateway's internal RTSP re-stream —
  one NVR connection per camera total; credentials only in the gateway
  config. To expose RTSP to other tools, add `- "8554:8554"` under the
  gateway service.
- First `docker compose up` builds the service images and pulls
  caddy/go2rtc. The detector compiles its RT-DETR MIGraphX cache on first
  boot (~9 min) and loads it in a few seconds on every boot after.

## License

**PolyForm Noncommercial License 1.0.0** — see [LICENSE.md](LICENSE.md).

You may use, modify, share and build on this project for any **non‑commercial**
purpose (study, research, personal projects, non‑profit / educational / public‑
safety organizations). **Commercial use is not granted** by this license. For a
commercial license, contact the copyright holder.
