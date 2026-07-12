"""VLM preprocessing of recorded video: caption every person-active minute.

The investigator can only search what is written down. Detections say THAT a
person was there; behaviours only fire on rules. This walks the recorded sub
segments and has the local VLM narrate the scene for every camera-minute where
the detector saw a person — so "what happened" is searchable text afterwards.

Activity-gated on purpose: an empty corridor at 03:00 costs nothing. Work is
newest-first, so live minutes are captioned within ~2 min and the backfill
(LOOKBACK_H) fills quiet GPU time. Already-captioned minutes are excluded by
an anti-join against clip_captions, which makes the whole loop idempotent —
restarts, crashes and re-runs are all safe.
"""
import base64
import json
import logging
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("vlmscan")

CH_URL = os.environ.get("CLICKHOUSE_URL", "http://clickhouse:8123")
CH_DB = os.environ.get("CLICKHOUSE_DB", "ccvt")
CH_AUTH = (os.environ.get("CLICKHOUSE_USER", "ccvt"),
           os.environ.get("CLICKHOUSE_PASSWORD", "ccvt"))
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "gemma-4-31b-it")
RECORDINGS = Path(os.environ.get("RECORDINGS_DIR", "/recordings"))

LOOKBACK_H = float(os.environ.get("LOOKBACK_H", "24"))
MAX_PER_CYCLE = int(os.environ.get("MAX_PER_CYCLE", "30"))   # captions per minute-cycle
CONCURRENCY = int(os.environ.get("CONCURRENCY", "2"))        # parallel VLM calls
CYCLE_S = float(os.environ.get("CYCLE_S", "60"))

PROMPT = (
    "This is one frame from a fixed CCTV camera. In 1-3 short English "
    "sentences, state how many people are visible and what each is doing — "
    "posture, movement, what they carry or interact with, notable clothing. "
    "Mention vehicles only if a person is involved with one. No preamble."
)

SEG_RE = re.compile(r"^(\d{8})_(\d{6})\.mp4$")


def ch(sql):
    r = requests.post(CH_URL, params={"database": CH_DB, "query": sql + " FORMAT JSON"},
                      auth=CH_AUTH, timeout=30)
    r.raise_for_status()
    return r.json()["data"]


def ch_insert(rows):
    body = "\n".join(json.dumps(x, separators=(",", ":")) for x in rows)
    r = requests.post(CH_URL, params={
        "database": CH_DB,
        "query": "INSERT INTO clip_captions FORMAT JSONEachRow"},
        data=body.encode(), auth=CH_AUTH, timeout=30)
    r.raise_for_status()


def pending():
    """(camera, minute, persons) still lacking a caption, newest first."""
    # m_epoch, not the DateTime string: ClickHouse renders strings in ITS
    # timezone (Asia/Bangkok) while this container and the recorder's
    # filenames follow the host clock — epoch seconds are unambiguous.
    return ch(f"""
        SELECT camera, toUnixTimestamp(m) AS m_epoch, m, persons FROM (
            SELECT camera, toStartOfMinute(ts) AS m, uniq(track) AS persons
            FROM detections
            WHERE class = 'person' AND ts > now() - INTERVAL {LOOKBACK_H} HOUR
              AND ts < now() - INTERVAL 90 SECOND
            GROUP BY camera, m
        ) AS act LEFT ANTI JOIN (
            SELECT camera, toStartOfMinute(ts) AS m FROM clip_captions
            WHERE ts > now() - INTERVAL {LOOKBACK_H + 1} HOUR
        ) AS done USING (camera, m)
        ORDER BY m DESC LIMIT {MAX_PER_CYCLE}""")


def find_segment(camera, minute_epoch):
    """The sub segment whose 60s window covers the middle of this minute."""
    d = RECORDINGS / camera
    if not d.is_dir():
        return None, 0.0
    mid = minute_epoch + 30
    best, best_off = None, None
    for f in d.iterdir():
        m = SEG_RE.match(f.name)
        if not m:
            continue
        try:
            start = datetime.strptime(m.group(1) + m.group(2),
                                      "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            continue
        off = mid - start
        if -5 <= off < 65 and (best is None or abs(off - 30) < abs(best_off - 30)):
            best, best_off = f, off
    return best, max(0.0, min(best_off or 0.0, 58.0))


def grab_frame(path, offset):
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-ss", f"{offset:.1f}", "-i", str(path),
         "-frames:v", "1", "-q:v", "3", "-f", "image2", "-"],
        capture_output=True, timeout=30)
    if p.returncode != 0 or not p.stdout:      # frame may sit before a keyframe
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-frames:v", "1", "-q:v", "3", "-f", "image2", "-"],
            capture_output=True, timeout=30)
    return p.stdout if p.returncode == 0 else None


def caption(jpeg):
    r = requests.post(VLLM_URL.rstrip("/") + "/chat/completions", json={
        "model": MODEL, "max_tokens": 160,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url":
             "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
            {"type": "text", "text": PROMPT},
        ]}]}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def process(row):
    cam, persons = row["camera"], int(row["persons"])
    m = row["m"]                                   # display only (server TZ)
    minute_epoch = int(row["m_epoch"])
    ts_ms = minute_epoch * 1000
    seg, off = find_segment(cam, minute_epoch)
    if seg is None:
        # no recording for this minute (recorder down / pruned): write an empty
        # marker so the anti-join never offers this minute again
        ch_insert([{"ts": ts_ms, "camera": cam, "segment": "", "persons": persons,
                    "caption": "", "model": "none", "latency_ms": 0}])
        return f"{cam} {m}: no segment"
    jpeg = grab_frame(seg, off)
    if not jpeg:
        ch_insert([{"ts": ts_ms, "camera": cam, "segment": seg.name,
                    "persons": persons, "caption": "", "model": "none",
                    "latency_ms": 0}])
        return f"{cam} {m}: frame extract failed"
    t0 = time.time()
    try:
        text = caption(jpeg)
    except Exception as e:
        return f"{cam} {m}: vlm error {str(e)[:80]}"     # no marker -> retried
    ch_insert([{"ts": ts_ms, "camera": cam, "segment": seg.name,
                "persons": persons, "caption": text, "model": MODEL,
                "latency_ms": int((time.time() - t0) * 1000)}])
    return f"{cam} {m}: {persons}p, {len(text)}ch"


def main():
    log.info("captioning person-active minutes: lookback %.0fh, "
             "%d/cycle, %d concurrent, model %s", LOOKBACK_H, MAX_PER_CYCLE,
             CONCURRENCY, MODEL)
    while True:
        t0 = time.time()
        try:
            work = pending()
        except Exception as e:
            log.warning("clickhouse: %s", str(e)[:120])
            time.sleep(30)
            continue
        if work:
            with ThreadPoolExecutor(CONCURRENCY) as ex:
                done = list(ex.map(process, work))
            ok = sum(1 for x in done if ": vlm error" not in x)
            log.info("cycle: %d captioned (%d issues) in %.0fs — e.g. %s",
                     ok, len(done) - ok, time.time() - t0, done[0])
        time.sleep(max(5.0, CYCLE_S - (time.time() - t0)))


if __name__ == "__main__":
    main()
