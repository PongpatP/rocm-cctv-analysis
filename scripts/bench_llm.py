"""Text-generation throughput of the two OpenAI-compatible LLM endpoints.

Same prompt, same output length, same client, one endpoint at a time — the only
variable is how many requests are in flight.

`ignore_eos` + `max_tokens` pins every request to exactly N generated tokens, so
"tokens/sec" compares work done, not how chatty a model happened to be. If a
server rejects `ignore_eos` the run falls back to natural stopping and the
per-request token count is read from `usage`, which is still exact.

    docker compose exec siglip python3 /app/bench_llm.py --tokens 200
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINTS = [
    ("gemma-4-31b (GPU 0, local)", "http://vllm:8000/v1", "gemma-4-31b-it"),
    ("qwen3-vl-32b (secondary VLM, LAN)", "http://192.168.1.151:8080/v1", "qwen3-vl-32b"),
]
PROMPT = ("Write a detailed operational summary of a CCTV control room shift: "
          "cameras monitored, incidents logged, handover notes.")


def one(url, model, tokens, ignore_eos):
    body = {"model": model, "max_tokens": tokens, "temperature": 0.0,
            "messages": [{"role": "user", "content": PROMPT}]}
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(url + "/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
    except Exception as e:
        return None, str(e)[:80], time.time() - t0
    dt = time.time() - t0
    return r["usage"]["completion_tokens"], None, dt


def supports_ignore_eos(url, model):
    n, err, _ = one(url, model, 8, True)
    return err is None


def run(url, model, conc, tokens, ignore_eos):
    n_req = max(8, conc * 2)                 # amortise ramp-up and drain
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda _: one(url, model, tokens, ignore_eos), range(n_req)))
    wall = time.time() - t0
    ok = [(n, dt) for n, err, dt in res if err is None]
    fail = len(res) - len(ok)
    if not ok:
        return None
    total_tok = sum(n for n, _ in ok)
    lat = [dt for _, dt in ok]
    return {
        "conc": conc, "requests": len(ok), "failed": fail, "wall": wall,
        "total_tps": total_tok / wall,                    # system throughput
        "per_req_tps": statistics.mean(n / dt for n, dt in ok),
        "p50": statistics.median(lat),
        "p95": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--conc", default="1,2,4,8,16,32")
    a = ap.parse_args()
    levels = [int(x) for x in a.conc.split(",")]

    for name, url, model in ENDPOINTS:
        print(f"\n=== {name}  ({model}, {a.tokens} tokens/request)", flush=True)
        # /v1/models is not universal — the Spark proxy returns 404 for it.
        # The only probe that proves an endpoint works is a real completion.
        n, err, _ = one(url, model, 4, False)
        if err is not None:
            print(f"  unreachable: {err[:70]}", flush=True)
            continue
        ie = supports_ignore_eos(url, model)
        print(f"  ignore_eos supported: {ie}", flush=True)
        one(url, model, 16, ie)                            # warm up
        print(f"  {'conc':>4} {'total tok/s':>12} {'tok/s/req':>10} "
              f"{'p50 s':>7} {'p95 s':>7} {'fail':>5}", flush=True)
        for c in levels:
            r = run(url, model, c, a.tokens, ie)
            if r is None:
                print(f"  {c:>4} {'all requests failed':>12}", flush=True)
                continue
            print(f"  {r['conc']:>4} {r['total_tps']:>12.1f} {r['per_req_tps']:>10.1f} "
                  f"{r['p50']:>7.2f} {r['p95']:>7.2f} {r['failed']:>5}", flush=True)


if __name__ == "__main__":
    main()
