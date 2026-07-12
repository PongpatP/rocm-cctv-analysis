"""Vision throughput of the two OpenAI-compatible VLM endpoints.

ONE frame, captured once from a camera's sub stream (352x288 — what the CCTV
actually gives at low res), sent to every request. Same prompt, same forced
output length, so the only variable is how many requests are in flight.

Prompt tokens are reported too: with an image, most of the prefill is the image,
and prefill is where a VLM spends its time at low concurrency.

    docker compose exec siglip python3 /app/bench_vlm.py --camera nvr1_ch04
"""
import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

ENDPOINTS = [
    ("gemma-4-31b (GPU 0, local)", "http://vllm:8000/v1", "gemma-4-31b-it"),
]
GATEWAY = "http://gateway:1984"
PROMPT = ("Describe this CCTV frame: the place, what is in it, and any people. "
          "Be specific.")


def grab(camera, stream):
    url = f"{GATEWAY}/api/frame.jpeg?src={camera}_{stream}"
    raw = urllib.request.urlopen(url, timeout=15).read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return im.size, base64.b64encode(buf.getvalue()).decode()


def one(url, model, b64, tokens, ignore_eos):
    body = {"model": model, "max_tokens": tokens, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT}]}]}
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(url + "/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except Exception as e:
        return None, None, str(e)[:80], time.time() - t0
    u = r["usage"]
    return u["completion_tokens"], u["prompt_tokens"], None, time.time() - t0


def run(url, model, b64, conc, tokens, ie):
    n_req = max(6, conc * 2)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda _: one(url, model, b64, tokens, ie), range(n_req)))
    wall = time.time() - t0
    ok = [(c, p, dt) for c, p, err, dt in res if err is None]
    if not ok:
        return None
    tot = sum(c for c, _, _ in ok)
    lat = sorted(dt for _, _, dt in ok)
    return {"conc": conc, "n": len(ok), "failed": len(res) - len(ok),
            "wall": wall, "avg": statistics.mean(dt for _, _, dt in ok),
            "total_tps": tot / wall,
            "per_req_tps": statistics.mean(c / dt for c, _, dt in ok),
            "p50": statistics.median(lat), "p95": lat[max(0, int(len(lat) * .95) - 1)],
            "img_prompt_tokens": ok[0][1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="nvr1_ch04")
    ap.add_argument("--stream", default="sub")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--conc", default="1,2,4,8")
    a = ap.parse_args()

    size, b64 = grab(a.camera, a.stream)
    print(f"frame: {a.camera}_{a.stream}  {size[0]}x{size[1]}  "
          f"{len(b64)*3//4//1024} KB jpeg — the SAME frame for every request",
          flush=True)

    for name, url, model in ENDPOINTS:
        print(f"\n=== {name}  ({model}, {a.tokens} output tokens/request)", flush=True)
        c, p, err, _ = one(url, model, b64, 4, False)
        if err:
            print(f"  unreachable: {err[:70]}", flush=True)
            continue
        ie = one(url, model, b64, 4, True)[2] is None
        print(f"  ignore_eos: {ie} · prompt tokens with this image: {p}", flush=True)
        one(url, model, b64, 8, ie)                      # warm up
        print(f"  {'conc':>4} {'avg s/img':>10} {'p95 s':>7} {'img/min':>8} "
              f"{'tok/s':>8} {'fail':>5}", flush=True)
        for conc in (int(x) for x in a.conc.split(",")):
            r = run(url, model, b64, conc, a.tokens, ie)
            if not r:
                print(f"  {conc:>4}  all requests failed", flush=True)
                continue
            img_per_min = r["n"] / r["wall"] * 60
            print(f"  {r['conc']:>4} {r['avg']:>10.2f} {r['p95']:>7.2f} "
                  f"{img_per_min:>8.0f} {r['total_tps']:>8.1f} {r['failed']:>5}", flush=True)


if __name__ == "__main__":
    main()
