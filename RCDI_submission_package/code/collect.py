"""Generate the raw records for the RCDI study.

For every query and every model tier (S, M, L) it stores: whether the greedy answer is correct, three risk signals
(mean token log-probability, self-consistency agreement over 5 samples, verbalised confidence) and the token costs.
Everything the paper reports is computed from the resulting records by analyze.py (no further model calls).

  python collect.py --n 800 --out results           # real run (vLLM servers must be up, see runpod.sh)
  python collect.py --n 40 --out results_mock --mock  # code test with synthetic behaviour (never reported)
"""
import argparse
import asyncio
import json
import os
import re
import time

import numpy as np

import rcdi as R


def parse_conf(text):
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    if not m:
        return None
    v = float(m.group(0))
    return v if 0 <= v <= 100 else None


async def one(llm, task, tier):
    msgs = R.build_messages(task)
    mt = R.MAXTOK[task["workload"]]
    g = await llm.call(tier, msgs, mt, 0.0, 1, True, (task, "gen"))
    s = await llm.call(tier, msgs, mt, R.TEMP, R.N_SAMPLES, False, (task, "gen"))
    go = g["outs"][0]
    ans = R.extract(go["text"], task, go["finish"] != "length")
    norm = R.normalize(task, ans)
    sc = float(np.mean([R.normalize(task, R.extract(o["text"], task, o["finish"] != "length")) == norm for o in s["outs"]])) if norm is not None else 0.0
    v = await llm.call(tier, R.verbal_messages(msgs, go["text"]), 8, 0.0, 1, False, (task, "verbal"))
    return {
        "id": task["id"], "workload": task["workload"], "tier": tier,
        "correct": int(R.is_correct(task, ans)), "parsed": int(norm is not None), "truncated": int(go["finish"] == "length"),
        "lp_mean": go.get("lp_mean"), "lp_tail": go.get("lp_tail"), "sc": sc, "vb": parse_conf(v["outs"][0]["text"]),
        "pred_stakes": int(task["workload"] == "banking77" and norm in R.HIGH_STAKES),
        "c_gen": R.gen_cost(tier, g["n_in"], g["n_out"]),
        "c_sc": R.gen_cost(tier, s["n_in"], s["n_out"]),
        "c_vb": R.gen_cost(tier, v["n_in"], v["n_out"]),
        "n_in": g["n_in"], "n_out": g["n_out"],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="queries per workload")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workloads", nargs="+", default=R.WORKLOADS)
    ap.add_argument("--concurrency", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = R.load_tasks(a.n, a.seed, a.mock, a.workloads)
    with open(f"{a.out}/tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps({"id": t["id"], "workload": t["workload"], "qlen": len(t["question"]),
                                "true_stakes": int(t["workload"] == "banking77" and t["answer"] in R.HIGH_STAKES)}) + "\n")
    rec_path = f"{a.out}/records.jsonl"
    done = set()
    if os.path.exists(rec_path):
        for line in open(rec_path):
            r = json.loads(line)
            done.add((r["id"], r["tier"]))
    llm = R.LLM(f"{a.out}/cache.jsonl", mock=a.mock, concurrency=a.concurrency)
    fh = open(rec_path, "a")
    jobs = [(t, tier) for tier in R.TIERS for t in tasks if (t["id"], tier) not in done]
    print(f"{len(tasks)} queries x 3 tiers; {len(jobs)} (query, tier) jobs to run", flush=True)
    t0, finished = time.time(), 0

    async def run(t, tier):
        nonlocal finished
        r = await one(llm, t, tier)
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        finished += 1
        if finished % 100 == 0 or finished == len(jobs):
            print(f"  {finished}/{len(jobs)} done, {time.time() - t0:.0f}s, cache hits {llm.n_hits}/{llm.n_calls}", flush=True)
    await asyncio.gather(*[run(t, tier) for t, tier in jobs])
    json.dump({"models": R.MODELS, "n_per_workload": a.n, "seed": a.seed, "mock": a.mock, "w_in": R.W_IN,
               "n_samples": R.N_SAMPLES, "temperature": R.TEMP, "workloads": a.workloads,
               "high_stakes": sorted(R.HIGH_STAKES), "wall_seconds": time.time() - t0,
               "llm_calls": llm.n_calls, "cache_hits": llm.n_hits}, open(f"{a.out}/meta.json", "w"), indent=2)
    print("collect done")


if __name__ == "__main__":
    asyncio.run(main())
