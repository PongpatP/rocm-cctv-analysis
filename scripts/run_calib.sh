#!/usr/bin/env bash
# Trigger a 3D-map calibration scan (MapAnything) through the running stack and follow its
# progress. Same thing as pressing "Run scan" on /map3d.html.
# Usage: scripts/run_calib.sh [base-url]   (default http://localhost:8080)
set -euo pipefail
BASE="${1:-http://localhost:8080}"

code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/calib/run")
case "$code" in
  200) echo "scan started" ;;
  409) echo "a scan is already running — attaching" ;;
  *)   echo "could not start scan (HTTP $code) — is the stack up?" >&2; exit 1 ;;
esac

while true; do
  s=$(curl -sf "$BASE/api/calib/status") || { echo; echo "status unavailable" >&2; exit 1; }
  read -r state pct label < <(python3 - "$s" <<'EOF'
import json, sys
d = json.loads(sys.argv[1])
print(d["state"], d["pct"], (d.get("stage_label") or "") .replace(" ", "_"))
EOF
)
  printf "\r  %-28s %3s%%   " "${label//_/ }" "$pct"
  if [ "$state" = "done" ]; then echo; echo "done — results on /map3d.html"; exit 0; fi
  if [ "$state" = "error" ]; then echo; echo "FAILED — docker logs ccvt-calib" >&2; exit 1; fi
  sleep 3
done
