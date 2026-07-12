# Camera upstream modes — `direct` vs `forwarder`

The stack can pull RTSP **directly from the NVR IPs**
(`192.168.0.163` / `192.168.0.158`) or via an **upstream go2rtc relay**
(`rtsp://192.168.1.151:8554/<stream>`). The AMD MI300X server is a plain
room-LAN client and cannot reach the CCTV subnet, so it uses the relay.
This is the `gateway.upstream` switch in `config.local.yaml`.

## Which mode to use — decision rule

- **Machine can reach the CCTV subnet** (`192.168.0.x` is routable) →
  `upstream: direct`. Streams come straight from the NVRs; the relay is not
  involved.
- **Machine is a plain room-LAN client** (no route to the CCTV subnet, e.g.
  the AMD MI300X server) → `upstream: forwarder`. Streams come via the
  relay at `192.168.1.151:8554`.

Quick test for which case you're in:
`ping -c1 192.168.0.163` succeeds → `direct`; fails → `forwarder`.

## What a source URL actually changes to

Same camera (NVR-01, channel 1, sub-stream), both modes are full RTSP URLs:

```
direct    : rtsp://<user>:<password>@192.168.0.163:554/cam/realmonitor?channel=1&subtype=1
forwarder : rtsp://192.168.1.151:8554/nvr1_ch01_sub
```

| Part | direct (NVR) | forwarder (relay) |
|---|---|---|
| Host:port | `192.168.0.163:554` — CCTV subnet | `192.168.1.151:8554` — room LAN |
| Path | `/cam/realmonitor?channel=1&subtype=1` (Dahua format) | `/nvr1_ch01_sub` (stream name) |
| Credentials | `<user>:<password>@` required | none — the relay holds them |

The stream name is mechanical: channel 1 + subtype 1 on NVR-01 →
`nvr1_ch01_sub`; subtype 0 → `nvr1_ch01_main`. The relay serves all
64 names (`nvr1_ch01_sub` … `nvr2_ch16_main`).

You never hand-edit these URLs — `configure.py` generates them all into
`gateway/go2rtc.yaml` and builds either form from the switch below.

Only the gateway's *source URLs* change. Everything downstream is
compose-internal and stays exactly as it is:

```
BEFORE   NVRs (192.168.0.x) ──────────────────────► gateway ─► airocm / webapp / bridge
AFTER    NVRs ─► relay hop ─► go2rtc relay ─────────► gateway ─► airocm / webapp / bridge
                              192.168.1.151:8554       (unchanged internals)
```

The relay already serves every stream under the **same names** the
stack uses (`nvr1_ch01_sub`, `nvr1_ch01_main`, … `nvr2_ch16_main`), so the
mapping is 1:1. Chaining go2rtc → go2rtc is a supported, normal setup:
still one NVR connection per camera, on demand.

---

## 1. `config.local.yaml` — the upstream switch

Under the `gateway:` section:

```yaml
gateway:
  web_port: 8080
  rtsp_port: 8554
  webrtc_port: 8555
  log_level: info
  # where the gateway pulls camera streams from:
  #   direct    — NVR IPs above (needs a route to 192.168.0.0/24)
  #   forwarder — go2rtc relay on the room LAN
  upstream: forwarder
  forwarder: {host: 192.168.1.151, port: 8554}
```

Keep the `nvrs:` section as-is — ids, names, `channels`, `skip` still
drive stream naming, the web UI, and the AI camera list. (`user` /
`password` / `host` / `port` are simply unused in forwarder mode;
credentials live only on the relay.)

## 2. How `configure.py` builds the URL

`write_go2rtc()` picks the URL per stream from the switch:

```python
def write_go2rtc(cfg):
    gw = cfg["gateway"]
    upstream = gw.get("upstream", "direct")
    fw = gw.get("forwarder") or {}
    ...
    for nvr in cfg["nvrs"]:
        for ch in active_channels(nvr):
            for profile, subtype in (("sub", 1), ("main", 0)):
                name = f"{cam_key(nvr, ch)}_{profile}"
                if upstream == "forwarder":
                    url = f"rtsp://{fw['host']}:{fw.get('port', 8554)}/{name}"
                else:
                    url = (
                        f"rtsp://{nvr['user']}:{nvr['password']}@{nvr['host']}:{nvr['port']}"
                        f"/cam/realmonitor?channel={ch}&subtype={subtype}"
                    )
                lines.append(f"  {name}: {url}")
```

Nothing in `write_webapp()` or `write_ai()` changes — the AI keeps
reading `rtsp://gateway:8554/<key>_sub` inside the compose network.

## 3. Apply on the server

```bash
./configure.py
docker compose up -d --force-recreate gateway
docker compose logs -f gateway     # sources should show 192.168.1.151
tail -f output/detections.jsonl    # records keep appearing = done
```

## 4. Verify reachability first (30 seconds)

From the server, before touching anything:

```bash
ffprobe -rtsp_transport tcp -v error \
  -show_entries stream=codec_name,width,height -of default=nw=1 \
  rtsp://192.168.1.151:8554/nvr1_ch01_sub
```

Expected: `codec_name=h264`, a resolution, exit 0.

---

## Notes

- **Bandwidth**: the AI watches ~29 sub-streams continuously; each is
  pulled once across the relay to the LAN (a few Mbps total). Main
  streams cross the network only while someone has a camera full-screened
  in the UI.
- **Skip lists**: the relay defines all 32 channels; the stack's `skip:`
  lists still control what the AI/UI request. Unrequested streams cost
  nothing (go2rtc pulls on demand).
- **Stable IP**: `192.168.1.151` is the relay's DHCP address — give it a
  reservation on the router, or streams die when the lease changes.
- **Failure mode**: if the relay or its uplink goes down, streams stall and
  both go2rtc instances auto-reconnect when the path returns; the stack
  itself needs no restart.
