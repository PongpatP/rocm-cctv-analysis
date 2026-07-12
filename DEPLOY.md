# Deploying the CCTV-AI stack on the AMD MI300X server

This is the running deployment: the whole stack runs on **one AMD Instinct
MI300X** (~191 GB VRAM, ROCm). Read this before bringing it up on a fresh
box — a few services need their GPU device mounts and their MIGraphX cache,
and there is a security switch you must flip before exposing it.

## What the deploy carries

- All source, configs, and the web app.
- Every model weight the services need: the RT-DETR `.onnx` detector under
  `ai/models/`, plus the ONNX weights for `siglip` (YoutuReID), `face`
  (cvlface), and `plate`. These are portable — each GPU service compiles
  its own MIGraphX cache (`.mxr`) on first boot and reuses it after.
- Secrets: `config.local.yaml`, `output/ai_settings.json`, and
  `auth/.state/session_secret.key`. Keep these out of git.
- A data slice under `data/clickhouse/`: the semantic tables whole (persons,
  articles, behaviours, episodes, sightings, faces, plates, vehicles,
  minute_stats) and **2 hours** of the raw `detections`/`poses`.
  `output/siglip/reid.db` whole — the cross-camera identities, their tags,
  and the camera graph.

Optional bulk data (copy separately only if you need it):
- `recordings/` (CCTV video) and `output/calib/` (3D scan output).
- `output/clickhouse/` (the live DB data dir — replaced by the slice above).
- `output/siglip/crops`, `cache`, `embeddings.db` — regenerated at runtime.

## Host prerequisites

- Docker + docker compose. `docker-compose.override.yml` is picked up
  automatically and carries the AMD/ROCm settings.
- A working ROCm install on the host at `/opt/rocm` (matching the loaded
  `amdgpu` driver). Every GPU service bind-mounts it read-only, so the
  container userspace always matches the host driver.
- The `/dev/kfd` and `/dev/dri` devices present (they are the GPU access
  path for all GPU services). Confirm the GPU is visible with `rocm-smi`.

## The services

**The spine — bring these up first** (video ingest, recording, database,
web UI and login):
`proxy`, `auth`, `gateway` (go2rtc), `clickhouse`, `bridge`, `recorder`.

**The AI services** (all on the MI300X via `/dev/kfd` + `/dev/dri`, ONNX
through onnxruntime-migraphx):
- `airocm` — detection + in-camera tracking. RT-DETR R50 ONNX for
  detection, a clean-room BoT-SORT tracker (Kalman + IoU) with OSNet
  appearance ReID, all in Python. Pushes person crops to `siglip`.
- `siglip` — cross-camera ReID (Global IDs, sightings); YoutuReID embedder.
- `face` — face detection/recognition (cvlface ONNX on MIGraphX +
  MediaPipe alignment on CPU).
- `plate` — plate detector on MIGraphX + PaddleOCR (CPU).
- `vllm` — vLLM (ROCm) serving Google Gemma 4 31B at bf16; the VLM/LLM
  backend other services call.
- `vlmscan` — video captioning; drives the VLM over the vLLM API.

First boot of each GPU service compiles its MIGraphX cache (the detector
~9 min; face ~47 s; plate a few seconds); every boot after loads from the
persisted `.mxr` cache in a few seconds.

## Security — do this before the tunnel is pointed at it

1. **`skip_lan`.** On a server there is no LAN to trust. In
   `config.local.yaml` set `skip_lan: false`, so every visitor must log in.
   (The LAN check uses `CF-Connecting-IP` / the real client IP, which the
   client cannot forge via a `Host` header. Keep the `auth/` and
   `proxy/Caddyfile` as shipped.)
2. **`admin_emails`.** Add your Google address in `config.local.yaml`, or
   nobody can ever see an uncensored face — non-admins get every face
   blurred.
3. Confirm only `proxy` publishes a port. Everything else must stay on the
   internal Docker network.

## Bring-up order

```
docker compose up -d clickhouse
# wait for healthy, then:
bash data/clickhouse/load.sh          # schema + carried tables
docker compose up -d gateway auth proxy bridge recorder
# then the AI services (they compile their MIGraphX cache on first boot):
docker compose up -d airocm siglip face plate vllm vlmscan
```

Verify: `rocm-smi` shows the services using the GPU, and
`tail -f output/detections.jsonl` shows records appearing.

## Point the cameras at the new host

`ai/configs/cameras.yaml` and `gateway`'s config hold the RTSP sources of
the NVRs. If the server cannot reach the CCTV subnet directly, switch
`gateway.upstream` to `forwarder` in `config.local.yaml` and re-run
`./configure.py` (see [AI_SERVER_CHANGES.md](AI_SERVER_CHANGES.md)).
