"""Push the SAM 3 scene scans into ClickHouse.

`calib/seg_scan.py` writes one JSON per camera under output/calib/sam3/. Each
holds the prompted keywords and how many instances of each were found — counts
and overlay colours, not masks. Re-running is safe: the table is a
ReplacingMergeTree keyed on (camera, label).

    docker compose exec siglip python3 /app/export_segments.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/app")
import ch                                    # noqa: E402

SRC = "/output/calib/sam3"


def main():
    rows = []
    files = sorted(glob.glob(f"{SRC}/*_sam3.json"))
    for f in files:
        d = json.load(open(f))
        ts = int(os.path.getmtime(f) * 1000)   # when the scan was written
        for label, r in d.get("results", {}).items():
            rows.append({
                "scanned_ts": ts, "camera": d["cam"], "label": label,
                "instances": int(r.get("instances", 0)),
                "colour_bgr": list(r.get("colour_bgr", [0, 0, 0])),
            })
    ch._insert_now("segments", rows)
    found = sum(r["instances"] for r in rows)
    print(f"{len(files)} cameras, {len(rows)} labels, {found} instances -> clickhouse")


if __name__ == "__main__":
    main()
