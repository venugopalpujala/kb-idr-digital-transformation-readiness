"""RCDI core: workloads, LLM client, risk signals, calibration, cost-aware decision policies, conformal risk control.

Used by collect.py (generation on a GPU, or --mock for code tests) and analyze.py (analysis from the saved records).
"""
import asyncio
import hashlib
import json
import math
import os
import random
import re
import string
import time
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. Models, costs, constants
# ----------------------------------------------------------------------------------------------
# tier -> (HF model id, nominal parameters in billions, local vLLM port)
MODELS = {
    "S": ("Qwen/Qwen2.5-1.5B-Instruct", 1.5, 8011),
    "M": ("Qwen/Qwen2.5-7B-Instruct", 7.0, 8002),
    "L": ("Qwen/Qwen2.5-32B-Instruct-AWQ", 32.0, 8003),
}
TIERS = ["S", "M", "L"]
W_IN = 0.25          # relative price of an input token vs an output token
N_SAMPLES = 5        # samples for the self-consistency signal
TEMP = 0.7
MAXTOK = {"gsm8k": 512, "hotpotqa": 256, "banking77": 96}
WORKLOADS = ["gsm8k", "hotpotqa", "banking77"]
# Knowledge rule R1: intents whose error is treated as high-stakes (fixed before any experiment)
HIGH_STAKES = {"compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone", "card_payment_not_recognised",
               "cash_withdrawal_not_recognised", "direct_debit_payment_not_recognised"}
STAKES_WEIGHT = 5.0


def gen_cost(tier, n_in, n_out, w_in=W_IN):
    """Compute-proportional cost: parameters (B) x weighted tokens / 1000."""
    return MODELS[tier][1] * (w_in * n_in + n_out) / 1000.0


# ----------------------------------------------------------------------------------------------
# 2. Workloads: prompts, answer extraction, scoring
# ----------------------------------------------------------------------------------------------
def build_messages(task):
    w = task["workload"]
    if w == "gsm8k":
        u = f"{task['question']}\n\nSolve the problem step by step. End with a final line of the form 'Answer: <number>'."
    elif w == "hotpotqa":
        ctx = "\n".join(f"[{t}] {' '.join(s)}" for t, s in task["context"])
        u = (f"Paragraphs:\n{ctx}\n\nQuestion: {task['question']}\n\nThink briefly, then end with a final line of the "
             f"form 'Answer: <short answer>'.")
    else:
        labels = ", ".join(task["labels"])
        u = (f"Classify the customer message into exactly one intent from this list:\n{labels}\n\n"
             f"Message: {task['question']}\n\nReply with only one line of the form 'Answer: <intent>', using an intent from the list exactly as written.")
    return [{"role": "system", "content": "You are a careful assistant."}, {"role": "user", "content": u}]


def verbal_messages(messages, answer_text):
    return messages + [{"role": "assistant", "content": answer_text},
                       {"role": "user", "content": "How confident are you that your answer is correct? "
                                                   "Reply with only a number from 0 to 100."}]


def extract(text, task=None, complete=True):
    """Final answer: the last 'Answer: ...' line. For GSM8K only, when a finished output has no such line (the model
    wrote its result in a \\boxed{} or in the last sentence), fall back to the boxed value, then to the last number."""
    m = re.findall(r"answer\s*[:：]\s*(.+)", text or "", flags=re.I)
    if m:
        return m[-1].strip()
    if task is not None and task.get("workload") == "gsm8k" and complete and text:
        b = re.findall(r"\\boxed\{([^{}]*)\}", text)
        if b:
            return b[-1].strip()
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
        if nums:
            return nums[-1]
    return None


def _norm_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def normalize(task, ans):
    if ans is None:
        return None
    w = task["workload"]
    if w == "gsm8k":
        m = re.findall(r"-?\d[\d,]*\.?\d*", ans.replace("$", ""))
        if not m:
            return None
        try:
            v = float(m[-1].replace(",", "").rstrip("."))
        except ValueError:
            return None
        return str(int(v)) if abs(v - round(v)) < 1e-6 else f"{v:.4f}"
    if w == "banking77":
        return re.sub(r"[\s\-]+", "_", ans.strip().strip(".'\"`*").lower())
    return _norm_text(ans)


def _f1(a, b):
    ta, tb = a.split(), b.split()
    c = sum((Counter(ta) & Counter(tb)).values())
    if not ta or not tb or c == 0:
        return 0.0
    p, r = c / len(ta), c / len(tb)
    return 2 * p * r / (p + r)


def is_correct(task, ans):
    n = normalize(task, ans)
    if n is None:
        return False
    g = normalize(task, task["answer"])
    if task["workload"] == "hotpotqa":
        return n == g or _f1(n, g) >= 0.8
    return n == g


# ----------------------------------------------------------------------------------------------
# 3. Data loading
# ----------------------------------------------------------------------------------------------
def load_tasks(n_per_workload, seed, mock=False, workloads=WORKLOADS):
    rng = random.Random(seed)
    out = []
    if mock:
        labels = [f"intent_{i}" for i in range(20)] + sorted(HIGH_STAKES)
        for w in workloads:
            for i in range(n_per_workload):
                d = rng.random()
                t = {"id": f"{w}-{i}", "workload": w, "question": "Q" * rng.randint(60, 400), "difficulty": d}
                if w == "gsm8k":
                    t["answer"] = str(rng.randint(1, 500))
                elif w == "hotpotqa":
                    t["answer"] = f"entity {i}"
                    t["context"] = [(f"T{j}", ["sentence " * 30]) for j in range(10)]
                else:
                    t["labels"], t["answer"] = labels, rng.choice(labels)
                out.append(t)
        return out
    from datasets import load_dataset
    if "gsm8k" in workloads:
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        for i in rng.sample(range(len(gsm)), n_per_workload):
            out.append({"id": f"gsm8k-{i}", "workload": "gsm8k", "question": gsm[i]["question"],
                        "answer": gsm[i]["answer"].split("####")[-1].strip().replace(",", "")})
    if "hotpotqa" in workloads:
        hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        for i in rng.sample(range(len(hp)), n_per_workload):
            r = hp[i]
            out.append({"id": f"hotpotqa-{r['id']}", "workload": "hotpotqa", "question": r["question"],
                        "answer": r["answer"], "context": list(zip(r["context"]["title"], r["context"]["sentences"]))})
    if "banking77" in workloads:
        ds = None
        for name in ("mteb/banking77", "PolyAI/banking77"):
            try:
                ds = load_dataset(name, split="test", trust_remote_code=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"!! could not load {name}: {e}")
        if ds is None:
            raise RuntimeError("Banking77 could not be loaded; run with --workloads gsm8k hotpotqa or fix the loader")
        if "label_text" in ds.column_names:
            names = sorted(set(ds["label_text"]))
            gold = lambda r: r["label_text"]  # noqa: E731
        else:
            names = ds.features["label"].names
            gold = lambda r: names[r["label"]]  # noqa: E731
        names = [re.sub(r"[\s\-]+", "_", n.strip().lower()) for n in names]
        for i in rng.sample(range(len(ds)), n_per_workload):
            out.append({"id": f"banking77-{i}", "workload": "banking77", "question": ds[i]["text"],
                        "answer": re.sub(r"[\s\-]+", "_", gold(ds[i]).strip().lower()), "labels": names})
    return out


# ----------------------------------------------------------------------------------------------
# 4. LLM client with persistent cache (vLLM OpenAI-compatible servers, or MOCK)
# ----------------------------------------------------------------------------------------------
class LLM:
    def __init__(self, cache_path, mock=False, concurrency=48, host="http://127.0.0.1"):
        self.mock, self.host = mock, host
        self.cache, self.n_calls, self.n_hits = {}, 0, 0
        self.sem = {t: asyncio.Semaphore(concurrency) for t in TIERS}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r["val"]
                except Exception:  # noqa: BLE001
                    pass
        self._fh = open(cache_path, "a")
        self._client = None

    @staticmethod
    def key(tier, messages, max_tokens, temperature, n, logprobs):
        return hashlib.sha1(json.dumps([MODELS[tier][0], messages, max_tokens, temperature, n, logprobs],
                                       sort_keys=True).encode()).hexdigest()

    async def call(self, tier, messages, max_tokens, temperature=0.0, n=1, logprobs=False, mock_ctx=None):
        k = self.key(tier, messages, max_tokens, temperature, n, logprobs)
        self.n_calls += 1
        if k in self.cache:
            self.n_hits += 1
            return self.cache[k]
        val = self._mock(tier, messages, max_tokens, temperature, n, logprobs, mock_ctx) if self.mock \
            else await self._http(tier, messages, max_tokens, temperature, n, logprobs)
        self.cache[k] = val
        self._fh.write(json.dumps({"key": k, "val": val}) + "\n")
        self._fh.flush()
        return val

    async def _http(self, tier, messages, max_tokens, temperature, n, logprobs):
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=900)
        name, _, port = MODELS[tier]
        body = {"model": name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "n": n, "seed": 0}
        if logprobs:
            body["logprobs"] = True
        async with self.sem[tier]:
            for attempt in range(6):
                try:
                    r = await self._client.post(f"{self.host}:{port}/v1/chat/completions", json=body)
                    r.raise_for_status()
                    j = r.json()
                    outs = []
                    for ch in j["choices"]:
                        o = {"text": ch["message"]["content"] or "", "finish": ch.get("finish_reason") or "stop"}
                        if logprobs and ch.get("logprobs") and ch["logprobs"].get("content"):
                            lps = [t["logprob"] for t in ch["logprobs"]["content"]]
                            o["lp_mean"] = float(np.mean(lps)) if lps else None
                            tail = lps[-10:]
                            o["lp_tail"] = float(np.mean(tail)) if tail else None
                        outs.append(o)
                    return {"outs": outs, "n_in": j["usage"]["prompt_tokens"], "n_out": j["usage"]["completion_tokens"]}
                except Exception:  # noqa: BLE001
                    if attempt == 5:
                        raise
                    await asyncio.sleep(2 ** attempt)

    # --- MOCK: synthetic behaviour for code testing only (never used for reported results) ----
    def _mock(self, tier, messages, max_tokens, temperature, n, logprobs, ctx):
        task, kind = ctx
        rnd = random.Random(self.key(tier, messages, max_tokens, temperature, n, logprobs))
        r0 = random.Random(f"{task['id']}-{tier}")
        size = {"S": 0, "M": 1, "L": 2}[tier]
        p_ok = 1 / (1 + math.exp(-(1.2 + 1.3 * size - 4.0 * task["difficulty"])))
        ok_g = r0.random() < p_ok

        def ans(correct):
            if correct:
                return task["answer"]
            return f"wrong{rnd.randint(0, 4)}" if task["workload"] != "gsm8k" else str(rnd.randint(1000, 2000))
        n_in = len(json.dumps(messages)) // 4
        if kind == "verbal":
            v = int(min(100, max(0, 100 * (0.6 + 0.35 * ok_g - 0.25 * rnd.random() + 0.2 * rnd.random()))))
            return {"outs": [{"text": str(v), "finish": "stop"}], "n_in": n_in + 30, "n_out": 3}
        outs = []
        for i in range(n):
            ok = ok_g if (n == 1) else (rnd.random() < (0.85 if ok_g else 0.2))
            o = {"text": f"reasoning... Answer: {ans(ok)}", "finish": "stop"}
            if logprobs:
                base = -0.06 - 0.35 * (1 - p_ok) * rnd.random() - (0.25 if not ok else 0.0) * rnd.random()
                o["lp_mean"], o["lp_tail"] = base, base - 0.3 * rnd.random() * (not ok)
            outs.append(o)
        n_out = {"gsm8k": 220, "hotpotqa": 80, "banking77": 8}[task["workload"]] * n
        return {"outs": outs, "n_in": n_in, "n_out": n_out + rnd.randint(0, 20)}


# ----------------------------------------------------------------------------------------------
# 5. Metrics and calibrators
# ----------------------------------------------------------------------------------------------
def ece(p, y, bins=15):
    """Expected calibration error with equal-mass bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p)
    e = 0.0
    for idx in np.array_split(order, min(bins, len(p))):
        if len(idx):
            e += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    return float(e)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def auroc(p, y):
    from scipy.stats import rankdata
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, raw, y):
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y, int)
        self.const = float(y.mean()) if len(set(y)) < 2 else None
        if self.const is None:
            self.m = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(raw).reshape(-1, 1), y)
        return self

    def predict(self, raw):
        if self.const is not None:
            return np.full(len(raw), self.const)
        return self.m.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


class Iso:
    def fit(self, raw, y):
        from sklearn.isotonic import IsotonicRegression
        self.m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(raw, float), np.asarray(y, float))
        return self

    def predict(self, raw):
        return self.m.predict(np.asarray(raw, float))


class Combined:
    """Multi-signal risk model: standardised logistic regression, then isotonic recalibration on a separate split."""

    def fit(self, X_fit, y_fit, X_cal, y_cal):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        y_fit = np.asarray(y_fit, int)
        self.sc = StandardScaler().fit(X_fit)
        self.const = float(y_fit.mean()) if len(set(y_fit)) < 2 else None
        if self.const is None:
            self.lr = LogisticRegression(C=1.0, max_iter=2000).fit(self.sc.transform(X_fit), y_fit)
        self.iso = Iso().fit(self._raw(X_cal), y_cal)
        return self

    def _raw(self, X):
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.lr.predict_proba(self.sc.transform(X))[:, 1]

    def predict(self, X):
        return self.iso.predict(self._raw(X))


# ----------------------------------------------------------------------------------------------
# 6. Cascade evaluation, cost-aware (Bayes) policy, tuned thresholds, conformal risk control
# ----------------------------------------------------------------------------------------------
ACCEPT, ESCALATE, ABSTAIN = 0, 1, 2


def evaluate(A, Y, G, ce, ch, W_eval=None):
    """Run the cascade S->M->L with stage actions A (Q x 3). At the last stage ESCALATE is treated as ABSTAIN.

    Y: correctness of each tier's answer (Q x 3); G: generation + signal cost paid when a stage is run (Q x 3);
    ce: cost of an uncaught wrong answer; ch: cost of sending the query to a human (who is assumed correct);
    W_eval: per-query error-cost multiplier for evaluation (default 1).
    Returns dict of per-query arrays: cost, accepted_wrong (uncaught error), abstained, stage.
    """
    Q = len(Y)
    A = A.copy()
    A[:, 2] = np.where(A[:, 2] == ESCALATE, ABSTAIN, A[:, 2])
    stage = np.full(Q, 2)
    act = A[:, 2].copy()
    done = np.zeros(Q, bool)
    for s in range(3):
        stop = (~done) & (A[:, s] != ESCALATE)
        stage[stop], act[stop] = s, A[stop, s]
        done |= stop
    cum = np.cumsum(G, axis=1)
    gen = cum[np.arange(Q), stage]
    ok = Y[np.arange(Q), stage].astype(bool)
    acc = act == ACCEPT
    w = np.ones(Q) if W_eval is None else W_eval
    wrong = acc & ~ok
    cost = gen + np.where(wrong, ce * w, 0.0) + np.where(act == ABSTAIN, ch, 0.0)
    return {"cost": cost, "gen": gen, "uncaught": wrong.astype(float), "human": (act == ABSTAIN).astype(float), "stage": stage}


def bayes_actions(P, G, ce, ch, cal, W=None):
    """Expected-cost decision rule with cost-to-go estimated on the calibration split `cal` (index array).

    Stage s: accept if its expected error cost is the smallest of {accept, hand to human, escalate};
    the cost of escalating is the next stage's generation cost plus its expected decision cost (averaged over `cal`).
    """
    Q = len(P)
    W = np.ones_like(P) if W is None else W
    acc = (1 - P) * ce * W
    term2 = np.minimum(acc[:, 2], ch)
    E1 = G[cal, 2].mean() + term2[cal].mean()               # cost-to-go when escalating from M to L
    term1 = np.minimum(np.minimum(acc[:, 1], ch), E1)
    E0 = G[cal, 1].mean() + term1[cal].mean()               # cost-to-go when escalating from S to M
    A = np.zeros((Q, 3), int)
    for s, E in ((0, E0), (1, E1), (2, np.inf)):
        opts = np.stack([acc[:, s], np.full(Q, ch), np.full(Q, E)], axis=1)   # ACCEPT, ABSTAIN, ESCALATE
        pick = np.argmin(opts, axis=1)
        A[:, s] = np.where(pick == 0, ACCEPT, np.where(pick == 1, ABSTAIN, ESCALATE))
    return A


def threshold_actions(P, a, b):
    """Accept if score >= a[s]; hand to human if score < b[s]; otherwise escalate (b <= a)."""
    A = np.full(P.shape, ESCALATE, int)
    for s in range(3):
        A[:, s] = np.where(P[:, s] >= a[s], ACCEPT, np.where(P[:, s] < b[s], ABSTAIN, ESCALATE))
    return A


def tune_thresholds(P, Y, G, ce, ch, idx, grid=21, sweeps=2):
    """Decision-aware tuning of (a, b) per stage: coordinate descent minimising realised mean cost on `idx`.

    Three starting points avoid local optima: L accepts everything; everything goes to a human at S; S accepts everything.
    """
    qs = np.unique(np.quantile(P[idx], np.linspace(0, 1, grid)))
    cands = np.concatenate([[-np.inf], qs, [np.inf]])
    inf = np.inf

    def cost(a_, b_):
        return evaluate(threshold_actions(P[idx], a_, b_), Y[idx], G[idx], ce, ch)["cost"].mean()
    starts = [(np.array([inf, inf, -inf]), np.array([-inf, -inf, -inf])),
              (np.array([inf, inf, inf]), np.array([inf, inf, inf])),
              (np.array([-inf, -inf, -inf]), np.array([-inf, -inf, -inf]))]
    best_all = None
    for a, b in starts:
        best = cost(a, b)
        for _ in range(sweeps):
            for s in range(3):
                for kind in ("a", "b"):
                    for c in cands:
                        a2, b2 = a.copy(), b.copy()
                        if kind == "a":
                            a2[s] = c
                            b2[s] = min(b2[s], c)
                        else:
                            b2[s] = min(c, a2[s])
                        v = cost(a2, b2)
                        if v < best - 1e-12:
                            best, a, b = v, a2, b2
        if best_all is None or best < best_all[0]:
            best_all = (best, a, b)
    return best_all[1], best_all[2]


def crc_threshold(loss_fn, lam_grid, alpha):
    """Conformal risk control (Angelopoulos et al.): smallest lambda whose corrected empirical risk is <= alpha.

    loss_fn(lam) returns the per-calibration-query losses in [0,1], non-increasing in lam.
    Requires the condition at every larger grid value (conservative against small non-monotonicity).
    """
    risks = []
    for lam in lam_grid:
        l = loss_fn(lam)
        n = len(l)
        risks.append(n / (n + 1) * l.mean() + 1.0 / (n + 1))
    ok = np.array(risks) <= alpha
    for i in range(len(lam_grid)):
        if ok[i:].all():
            return lam_grid[i]
    return lam_grid[-1]
