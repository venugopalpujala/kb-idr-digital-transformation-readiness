"""Re-score the GSM8K records from the saved model outputs (no GPU, no new model calls).

Why: in the real run about a third of the 7B and a quarter of the 1.5B outputs ended with the result in a \\boxed{} or in
the last sentence instead of the requested 'Answer:' line, so the strict extractor counted correct work as unparsable.
extract() in rcdi.py now falls back to the boxed value and then to the last number for finished GSM8K outputs.
Outputs are read from cache.jsonl (keyed exactly as collect.py keys them); correct, parsed and sc are recomputed,
every other field (log-probabilities, verbalised confidence, costs, truncation) is unchanged.

  python rescore.py --res results --out results_rescored [--gsm8k-jsonl path]   # jsonl: openai/grade-school-math test.jsonl
"""
import argparse
import json
import os
import random
import shutil

import numpy as np

import rcdi as R


def gsm8k_tasks(res, path):
    ids = [json.loads(l)["id"] for l in open(f"{res}/tasks.jsonl") if json.loads(l)["workload"] == "gsm8k"]
    if path:
        rows = [json.loads(l) for l in open(path)]
        rows = [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for r in rows]
    else:
        from datasets import load_dataset
        g = load_dataset("openai/gsm8k", "main", split="test")
        rows = [{"question": g[i]["question"], "answer": g[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in range(len(g))]
    out = {}
    for i in ids:
        j = int(i.split("-")[1])
        out[i] = {"id": i, "workload": "gsm8k", "question": rows[j]["question"], "answer": rows[j]["answer"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="results_rescored")
    ap.add_argument("--gsm8k-jsonl", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in ("tasks.jsonl", "meta.json"):
        shutil.copy(f"{a.res}/{f}", f"{a.out}/{f}")
    cache = {}
    for l in open(f"{a.res}/cache.jsonl"):
        r = json.loads(l)
        cache[r["key"]] = r["val"]
    tasks = gsm8k_tasks(a.res, a.gsm8k_jsonl)
    mt = R.MAXTOK["gsm8k"]
    before = {t: [0, 0, 0] for t in R.TIERS}
    after = {t: [0, 0, 0] for t in R.TIERS}
    n_fix = 0
    with open(f"{a.out}/records.jsonl", "w") as fo:
        for line in open(f"{a.res}/records.jsonl"):
            r = json.loads(line)
            if r["workload"] != "gsm8k":
                fo.write(line)
                continue
            task, tier = tasks[r["id"]], r["tier"]
            msgs = R.build_messages(task)
            g = cache[R.LLM.key(tier, msgs, mt, 0.0, 1, True)]
            s = cache[R.LLM.key(tier, msgs, mt, R.TEMP, R.N_SAMPLES, False)]
            go = g["outs"][0]
            assert (go["finish"] == "length") == bool(r["truncated"]), "cache does not match the record"
            ans = R.extract(go["text"], task, go["finish"] != "length")
            norm = R.normalize(task, ans)
            sc = float(np.mean([R.normalize(task, R.extract(o["text"], task, o["finish"] != "length")) == norm for o in s["outs"]])) if norm is not None else 0.0
            new = dict(r, correct=int(R.is_correct(task, ans)), parsed=int(norm is not None), sc=sc)
            before[tier][0] += r["correct"]; before[tier][1] += 1 - r["parsed"]; before[tier][2] += 1
            after[tier][0] += new["correct"]; after[tier][1] += 1 - new["parsed"]; after[tier][2] += 1
            n_fix += int(new["correct"] != r["correct"])
            fo.write(json.dumps(new) + "\n")
    print("GSM8K records whose correctness changed:", n_fix)
    for t in R.TIERS:
        print(f"  tier {t}: accuracy {before[t][0] / before[t][2]:.3f} -> {after[t][0] / after[t][2]:.3f}; "
              f"unparsed {before[t][1] / before[t][2]:.3f} -> {after[t][1] / after[t][2]:.3f}")


if __name__ == "__main__":
    main()
