"""Analysis for the RCDI study: all tables, figures and statistics from the saved records (no model calls).

  python analyze.py --res results --out results/analysis --reps 200
Each replicate: random disjoint split of the queries into fit (40%), calibration (30%) and test (30%); every risk model,
calibrator and threshold is fitted without the test split. Per-query test outcomes are averaged over replicates
(repeated cross-fitting) and confidence intervals bootstrap over queries.
"""
import argparse
import json
import os
import time

import numpy as np

import rcdi as R

CE, CH = 25.0, 5.0           # primary scenario (fixed before the real run), in units of the mean cost of one L answer
ALPHA = 0.05                 # target rate of uncaught errors per query
NCALS = [50, 100, 200, 400, None]
LAMS = np.round(np.arange(0.0, 1.0001, 0.005), 4)
CEG = CE * np.geomspace(1.0, 400.0, 61)   # error-cost grid for the risk-controlled expected-cost rule
GRID_CE, GRID_CH = [5.0, 25.0, 100.0, 400.0], [1.0, 5.0, 20.0]
FR_CE = np.geomspace(1.0, 80, 12)
FR_RATIO = CH / CE


# ----------------------------------------------------------------------------------------------
def load(res):
    tasks = [json.loads(l) for l in open(f"{res}/tasks.jsonl")]
    by = {}
    for l in open(f"{res}/records.jsonl"):
        r = json.loads(l)
        by[(r["id"], r["tier"])] = r
    T = [t for t in tasks if all((t["id"], tr) in by for tr in R.TIERS)]
    Q = len(T)
    g = lambda k, d=np.nan: np.array([[(by[(t["id"], tr)][k] if by[(t["id"], tr)][k] is not None else d) for tr in R.TIERS] for t in T], float)  # noqa: E731
    D = {"Y": g("correct").astype(int), "parsed": g("parsed"), "trunc": g("truncated"), "pred_stakes": g("pred_stakes"),
         "lpm": g("lp_mean"), "lpt": g("lp_tail"), "sc": g("sc"), "vb": g("vb"),
         "c_gen": g("c_gen"), "c_sc": g("c_sc"), "c_vb": g("c_vb")}
    for k in ("lpm", "lpt", "vb"):
        n = int(np.isnan(D[k]).sum())
        if n:
            print(f"warning: {n} missing values in {k}; imputed with the median")
            D[k] = np.where(np.isnan(D[k]), np.nanmedian(D[k]), D[k])
    D["wl"] = np.array([R.WORKLOADS.index(t["workload"]) for t in T])
    D["qlen"] = np.array([t["qlen"] for t in T], float)
    D["true_stakes"] = np.array([t["true_stakes"] for t in T], float)
    D["ids"] = [t["id"] for t in T]
    cbar = D["c_gen"][:, 2].mean()
    D["cbar"] = cbar
    for k in ("c_gen", "c_sc", "c_vb"):
        D[k] = D[k] / cbar
    D["LP"] = np.clip(np.exp(D["lpm"]), 1e-3, 1 - 1e-3)
    D["VB"] = np.clip(D["vb"] / 100.0, 1e-3, 1 - 1e-3)
    D["SC"] = np.clip(D["sc"], 1e-3, 1 - 1e-3)
    wl = D["wl"]
    X = np.stack([D["lpm"], D["lpt"], D["vb"] / 100.0, D["sc"], D["parsed"], D["trunc"],
                  np.repeat(np.log1p(D["qlen"])[:, None], 3, 1),
                  np.repeat((wl == 1)[:, None].astype(float), 3, 1), np.repeat((wl == 2)[:, None].astype(float), 3, 1)], axis=2)
    D["X"] = X
    D["G"] = {"lp": D["c_gen"], "vb": D["c_gen"] + D["c_vb"], "sc": D["c_gen"] + D["c_sc"],
              "comb": D["c_gen"] + D["c_vb"] + D["c_sc"]}
    D["Q"] = Q
    return D


def make_P(D, fit, cal):
    Y, P = D["Y"], {}
    for sig in ("lp", "vb", "sc"):
        raw = D[sig.upper()]
        P[f"raw_{sig}"] = raw.copy()
        pl, iso = np.zeros_like(raw), np.zeros_like(raw)
        for s in range(3):
            pl[:, s] = R.Platt().fit(raw[cal, s], Y[cal, s]).predict(raw[:, s])
            iso[:, s] = R.Iso().fit(raw[cal, s], Y[cal, s]).predict(raw[:, s])
        P[f"platt_{sig}"], P[f"iso_{sig}"] = pl, iso
    lr, cb = np.zeros_like(D["LP"]), np.zeros_like(D["LP"])
    for s in range(3):
        m = R.Combined().fit(D["X"][fit, s], Y[fit, s], D["X"][cal, s], Y[cal, s])
        cb[:, s], lr[:, s] = m.predict(D["X"][:, s]), m._raw(D["X"][:, s])
    P["lr_comb"], P["iso_comb"] = lr, cb
    return P


def sig_of(name):
    return name.split("_")[1]


def rule_r2(P, D):
    """R2: a structurally failed output (no parsable answer, or truncated) is never accepted."""
    return np.where((D["parsed"] == 0) | (D["trunc"] == 1), 0.0, P)


def stage_alone(P, lam):
    A = np.full(P.shape, R.ESCALATE, int)
    A[:, 2] = np.where(P[:, 2] >= lam, R.ACCEPT, R.ABSTAIN)
    return A


def alone_G(G):
    G = G.copy()
    G[:, :2] = 0.0
    return G


def casc_actions(P, lam):
    return R.threshold_actions(P, [lam] * 3, [-np.inf, -np.inf, lam])


def pick_lambda(P, Y, G, idx, alpha, mode, kind):
    """Threshold or error-cost parameter meeting the risk target on the calibration queries `idx`.

    kind: 'casc' (common threshold on the cascade), 'alone' (L only) or 'bayes' (expected-cost rule whose error cost is
    raised until the target holds); mode: 'crc' (conformal risk control) or 'naive' (empirical risk only).
    """
    grid = CEG if kind == "bayes" else LAMS

    def loss(lam):
        if kind == "bayes":
            A = R.bayes_actions(P, G, lam, CH, idx)[idx]
            return R.evaluate(A, Y[idx], G[idx], CE, CH)["uncaught"]
        A = casc_actions(P[idx], lam) if kind == "casc" else stage_alone(P[idx], lam)
        GG = G[idx] if kind == "casc" else alone_G(G[idx])
        return R.evaluate(A, Y[idx], GG, CE, CH)["uncaught"]
    if mode == "crc":
        return R.crc_threshold(loss, grid, alpha)
    risks = np.array([loss(l).mean() for l in grid])
    ok = risks <= alpha
    for i in range(len(grid)):
        if ok[i:].all():
            return grid[i]
    return grid[-1]


def run_policy(P, D, test, lam=None, kind="casc", G=None, cal=None):
    G = D["G"]["comb"] if G is None else G
    if kind == "bayes":
        A = R.bayes_actions(P, G, lam, CH, cal)[test]
        return R.evaluate(A, D["Y"][test], G[test], CE, CH)
    A = casc_actions(P[test], lam) if kind == "casc" else stage_alone(P[test], lam)
    GG = G[test] if kind == "casc" else alone_G(G[test])
    return R.evaluate(A, D["Y"][test], GG, CE, CH)


# ----------------------------------------------------------------------------------------------
class Acc:
    """Per-query accumulators of test outcomes over replicates."""
    def __init__(self, Q):
        self.Q = Q
        self.s = {}
        self.n = np.zeros(Q)

    def add(self, name, test, res):
        d = self.s.setdefault(name, {k: np.zeros(self.Q) for k in ("cost", "gen", "uncaught", "human")})
        for k in d:
            d[k][test] += res[k]

    def count(self, test):
        self.n[test] += 1

    def mean(self, name, k):
        ok = self.n > 0
        return np.where(ok, self.s[name][k] / np.maximum(self.n, 1), np.nan)


def boot_ci(v, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = v[~np.isnan(v)]
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = a.out or f"{a.res}/analysis"
    os.makedirs(out, exist_ok=True)
    D = load(a.res)
    Q, Y = D["Q"], D["Y"]
    print(f"{Q} queries; mean L-answer cost = {D['cbar']:.4f} model units (all costs below are in L-answer units)")
    acc_rows = []
    for w, name in enumerate(R.WORKLOADS):
        for s, t in enumerate(R.TIERS):
            m = D["wl"] == w
            if m.any():
                acc_rows.append((name, t, float(Y[m, s].mean()), int(m.sum())))
    with open(f"{out}/accuracy_by_tier.csv", "w") as f:
        f.write("workload,tier,accuracy,n\n" + "\n".join(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]}" for r in acc_rows) + "\n")
    print("accuracy by workload and tier:", [(r[0], r[1], round(r[2], 3)) for r in acc_rows])

    rng = np.random.default_rng(a.seed)
    acc = Acc(Q)
    metrics = {}      # (signal+calibrator) -> list of (ece, brier, auroc) per replicate
    refl = {k: ([], []) for k in ("raw_lp", "platt_lp", "iso_lp", "iso_comb")}
    guar = {}         # (method, kind, ncal) -> list of (uncaught, cost)
    frontier = {k: [] for k in ("raw_lp", "iso_lp", "iso_comb")}
    sens = {}         # (ce, ch, method) -> list of mean costs
    stakes = {k: [] for k in ("rcdi", "rcdi_rules")}
    t0 = time.time()
    for rep in range(a.reps):
        perm = rng.permutation(Q)
        nf, nc = int(0.4 * Q), int(0.3 * Q)
        fit, cal, test = perm[:nf], perm[nf:nf + nc], perm[nf + nc:]
        acc.count(test)
        P = make_P(D, fit, cal)
        G = D["G"]
        # calibration quality (pooled over the three tiers)
        yy = Y[test].ravel()
        for k, p in P.items():
            metrics.setdefault(k, []).append((R.ece(p[test].ravel(), yy), R.brier(p[test].ravel(), yy), R.auroc(p[test].ravel(), yy)))
        for k in refl:
            refl[k][0].append(P[k][test].ravel())
            refl[k][1].append(yy)
        # Bayes policies for every signal/calibrator (net of the signal's own cost)
        for k, p in P.items():
            sg = "comb" if "comb" in k else sig_of(k)
            A = R.bayes_actions(p, G[sg], CE, CH, cal)
            acc.add(f"bayes::{k}", test, R.evaluate(A[test], Y[test], G[sg][test], CE, CH))
        # baselines
        z = np.zeros((len(test), 3), int)
        z[:, 2] = R.ACCEPT
        for s, tier in enumerate(R.TIERS):
            A = np.full((len(test), 3), R.ESCALATE, int)
            A[:, 0:s] = R.ESCALATE
            A[:, s] = R.ACCEPT
            GG = np.zeros((len(test), 3))
            GG[:, s] = D["c_gen"][test, s]
            acc.add(f"always_{tier}", test, R.evaluate(A, Y[test], GG, CE, CH))
        ah = np.full((len(test), 3), R.ABSTAIN, int)
        acc.add("always_human", test, R.evaluate(ah, Y[test], np.zeros((len(test), 3)), CE, CH))
        # decision-aware tuned thresholds on the raw log-probability (no calibration needed)
        a_, b_ = R.tune_thresholds(P["raw_lp"], Y, G["lp"], CE, CH, cal, grid=11, sweeps=2)
        acc.add("tuned_raw_lp", test, R.evaluate(R.threshold_actions(P["raw_lp"][test], a_, b_), Y[test], G["lp"][test], CE, CH))
        # full RCDI with knowledge rule R2
        Pr = rule_r2(P["iso_comb"], D)
        A = R.bayes_actions(Pr, G["comb"], CE, CH, cal)
        acc.add("rcdi_rules", test, R.evaluate(A[test], Y[test], G["comb"][test], CE, CH))
        # stakes scenario: R1 raises the error cost of predicted high-stakes intents; evaluation uses true stakes
        Wd = np.where(D["pred_stakes"] == 1, R.STAKES_WEIGHT, 1.0)
        We = np.where(D["true_stakes"] == 1, R.STAKES_WEIGHT, 1.0)
        for nm, (PP, WW) in {"rcdi": (P["iso_comb"], None), "rcdi_rules": (Pr, Wd)}.items():
            A = R.bayes_actions(PP, G["comb"], CE, CH, cal, WW)
            r = R.evaluate(A[test], Y[test], G["comb"][test], CE, CH, We[test])
            ts = D["true_stakes"][test] == 1
            stakes[nm].append((r["cost"].mean(), r["uncaught"][ts].mean() if ts.any() else np.nan, r["human"].mean(), r["human"][ts].mean() if ts.any() else np.nan))
        # conformal risk control and naive empirical-risk thresholds, calibration-size sweep
        Pc = P["iso_comb"]
        for nc_ in NCALS:
            idx = cal if nc_ is None else cal[:nc_]
            nlab = "all" if nc_ is None else nc_
            for kind in ("casc", "alone", "bayes"):
                for mode in ("crc", "naive"):
                    lam = pick_lambda(Pc, Y, G["comb"], idx, ALPHA, mode, kind)
                    r = run_policy(Pc, D, test, lam, kind, cal=idx)
                    guar.setdefault((mode, kind, nlab), []).append((r["uncaught"].mean(), r["cost"].mean(), r["human"].mean()))
                    if nc_ is None:
                        acc.add(f"{mode}_{kind}", test, r)
        # risk-spend frontier by sweeping the error cost (human cost tied to it)
        for k in frontier:
            sg = "comb" if "comb" in k else sig_of(k)
            row = []
            for ce in FR_CE:
                ch = FR_RATIO * ce
                A = R.bayes_actions(P[k], G[sg], ce, ch, cal)
                r = R.evaluate(A[test], Y[test], G[sg][test], ce, ch)
                row.append((r["uncaught"].mean(), (r["gen"] + ch * r["human"]).mean()))
            frontier[k].append(row)
        # cost-sensitivity grid: calibrated Bayes (re-derived for free) vs thresholds tuned at the primary costs (stale)
        Pk = P["iso_comb"]
        a0, b0 = R.tune_thresholds(Pk, Y, G["comb"], CE, CH, cal, grid=11, sweeps=2)
        for ce in GRID_CE:
            for ch in GRID_CH:
                A = R.bayes_actions(Pk, G["comb"], ce, ch, cal)
                sens.setdefault((ce, ch, "bayes"), []).append(R.evaluate(A[test], Y[test], G["comb"][test], ce, ch)["cost"].mean())
                sens.setdefault((ce, ch, "stale"), []).append(R.evaluate(R.threshold_actions(Pk[test], a0, b0), Y[test], G["comb"][test], ce, ch)["cost"].mean())
                af, bf = R.tune_thresholds(Pk, Y, G["comb"], ce, ch, cal, grid=11, sweeps=2)
                sens.setdefault((ce, ch, "fresh"), []).append(R.evaluate(R.threshold_actions(Pk[test], af, bf), Y[test], G["comb"][test], ce, ch)["cost"].mean())
        if (rep + 1) % 10 == 0:
            print(f"  replicate {rep + 1}/{a.reps} ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ Table 1
    names1 = [("always_human", "Always send to a human"), ("always_L", "Always accept L (32B)"), ("always_M", "Always accept M (7B)"),
              ("always_S", "Always accept S (1.5B)"), ("tuned_raw_lp", "Cascade, tuned thresholds, raw log-prob"),
              ("bayes::raw_lp", "Cascade, expected-cost rule, raw log-prob"), ("bayes::platt_lp", "Cascade, expected-cost rule, Platt log-prob"),
              ("bayes::iso_lp", "Cascade, expected-cost rule, isotonic log-prob"), ("bayes::iso_comb", "RCDI: multi-signal risk model + expected-cost rule"),
              ("rcdi_rules", "RCDI + knowledge rule R2"), ("crc_bayes", f"RCDI + conformal risk control, error cost raised (alpha = {ALPHA})"),
              ("crc_casc", f"Cascade, common threshold + conformal risk control (alpha = {ALPHA})"),
              ("crc_alone", f"L alone + conformal risk control (alpha = {ALPHA})")]
    rows1, ci1 = [], {}
    for key, lab in names1:
        c, u, h = (boot_ci(acc.mean(key, k)) for k in ("cost", "uncaught", "human"))
        ci1[key] = (c, u, h)
        rows1.append([lab, fmt(c), f"{100 * u[0]:.1f} [{100 * u[1]:.1f}, {100 * u[2]:.1f}]", f"{100 * h[0]:.1f} [{100 * h[1]:.1f}, {100 * h[2]:.1f}]"])
    hdr1 = ["Policy", "Mean cost (L-answer units) [95% CI]", "Uncaught errors (% of queries) [95% CI]", "Sent to human (% of queries) [95% CI]"]
    write_table(out, "table1", hdr1, rows1)

    # paired differences
    def pdiff(a_, b_, k="cost"):
        d = acc.mean(a_, k) - acc.mean(b_, k)
        return boot_ci(d)
    paired = {f"{x} minus {y}": pdiff(x, y) for x, y in [
        ("bayes::raw_lp", "bayes::iso_lp"), ("bayes::raw_lp", "bayes::platt_lp"), ("tuned_raw_lp", "bayes::iso_lp"),
        ("bayes::iso_lp", "bayes::iso_comb"), ("always_L", "bayes::iso_comb"), ("always_L", "bayes::iso_lp"),
        ("crc_casc", "bayes::iso_comb"), ("crc_bayes", "bayes::iso_comb"), ("crc_casc", "crc_bayes"), ("bayes::iso_comb", "rcdi_rules")]}

    # ------------------------------------------------------------------ Table 2
    order = [("raw_lp", "Log-probability, raw"), ("platt_lp", "Log-probability, Platt"), ("iso_lp", "Log-probability, isotonic"),
             ("raw_vb", "Verbalised confidence, raw"), ("platt_vb", "Verbalised confidence, Platt"), ("iso_vb", "Verbalised confidence, isotonic"),
             ("raw_sc", "Self-consistency, raw"), ("platt_sc", "Self-consistency, Platt"), ("iso_sc", "Self-consistency, isotonic"),
             ("lr_comb", "Combined model, logistic"), ("iso_comb", "Combined model, logistic + isotonic")]
    rows2, t2num = [], []
    for k, lab in order:
        m = np.array(metrics[k])
        c = boot_ci(acc.mean(f"bayes::{k}", "cost"))
        sg = "comb" if "comb" in k else sig_of(k)
        extra_s = float(np.mean(D["G"][sg][:, 0] - D["c_gen"][:, 0]))
        extra_l = float(np.mean(D["G"][sg][:, 2] - D["c_gen"][:, 2]))
        t2num.append((m[:, 0].mean(), c[0]))
        rows2.append([lab, f"{m[:, 0].mean():.3f}", f"{m[:, 1].mean():.3f}", f"{m[:, 2].mean():.3f}", fmt(c), f"{extra_s:.2f} / {extra_l:.2f}"])
    hdr2 = ["Risk signal and calibrator", "ECE", "Brier", "AUROC", "Cascade cost, expected-cost rule [95% CI]", "Extra cost of the signal, S tier / L tier"]
    write_table(out, "table2", hdr2, rows2)
    from scipy.stats import spearmanr
    rho = spearmanr([x[0] for x in t2num], [x[1] for x in t2num])
    ece_gain = {}
    for sg in ("lp", "vb", "sc"):
        e0 = np.array(metrics[f"raw_{sg}"])[:, 0]
        for c_ in ("platt", "iso"):
            e1 = np.array(metrics[f"{c_}_{sg}"])[:, 0]
            ece_gain[f"{sg}_{c_}"] = (float(e0.mean()), float(e1.mean()))

    # ------------------------------------------------------------------ guarantee, sensitivity, stakes
    with open(f"{out}/guarantee.csv", "w") as f:
        f.write("mode,kind,n_cal,mean_uncaught,share_over_alpha,p05,p95,mean_cost,mean_human\n")
        for (mode, kind, nl), v in sorted(guar.items(), key=lambda kv: (kv[0][1], kv[0][0], 10 ** 9 if kv[0][2] == "all" else kv[0][2])):
            v = np.array(v)
            f.write(f"{mode},{kind},{nl},{v[:, 0].mean():.4f},{(v[:, 0] > ALPHA).mean():.3f},{np.percentile(v[:, 0], 5):.4f},{np.percentile(v[:, 0], 95):.4f},{v[:, 1].mean():.3f},{v[:, 2].mean():.3f}\n")
    with open(f"{out}/sensitivity.csv", "w") as f:
        f.write("ce,ch,bayes_cost,stale_tuned_cost,fresh_tuned_cost,stale_vs_bayes_pct,fresh_vs_bayes_pct,"
                "always_human_cost,always_L_cost,saving_vs_best_simple_pct\n")
        for ce in GRID_CE:
            for ch in GRID_CH:
                b, s_, fr = (float(np.mean(sens[(ce, ch, k)])) for k in ("bayes", "stale", "fresh"))
                aL = float(D["c_gen"][:, 2].mean() + ce * (1 - Y[:, 2].mean()))
                best = min(ch, aL)
                f.write(f"{ce},{ch},{b:.3f},{s_:.3f},{fr:.3f},{100 * (s_ - b) / b:.2f},{100 * (fr - b) / b:.2f},"
                        f"{ch:.3f},{aL:.3f},{100 * (best - b) / best:.2f}\n")
    with open(f"{out}/stakes.csv", "w") as f:
        f.write("method,mean_cost,uncaught_among_high_stakes,human_rate,human_rate_high_stakes\n")
        for k, v in stakes.items():
            v = np.array(v, float)
            f.write(f"{k},{np.nanmean(v[:, 0]):.3f},{np.nanmean(v[:, 1]):.4f},{np.nanmean(v[:, 2]):.4f},{np.nanmean(v[:, 3]):.4f}\n")

    # ------------------------------------------------------------------ distribution shift (leave one workload out)
    shift = []
    present = sorted(set(D["wl"]))
    for w in present:
        tw = np.where(D["wl"] == w)[0]
        rest = np.where(D["wl"] != w)[0]
        for mode_lab in ("shifted", "in_workload"):
            outs = []
            for rep in range(min(50, a.reps)):
                if mode_lab == "shifted":
                    pr = rng.permutation(rest)
                    fit, cal = pr[:int(.55 * len(pr))], pr[int(.55 * len(pr)):]
                    test = tw
                else:
                    pr = rng.permutation(tw)
                    fit, cal, test = pr[:int(.4 * len(pr))], pr[int(.4 * len(pr)):int(.7 * len(pr))], pr[int(.7 * len(pr)):]
                Pw = make_P(D, fit, cal)["iso_comb"]
                A = R.bayes_actions(Pw, D["G"]["comb"], CE, CH, cal)
                rb = R.evaluate(A[test], Y[test], D["G"]["comb"][test], CE, CH)
                lam = pick_lambda(Pw, Y, D["G"]["comb"], cal, ALPHA, "crc", "bayes")
                rc = run_policy(Pw, D, test, lam, "bayes", cal=cal)
                outs.append((rb["cost"].mean(), rb["uncaught"].mean(), rc["uncaught"].mean(), rc["cost"].mean(), R.ece(Pw[test].ravel(), Y[test].ravel())))
            o = np.array(outs)
            shift.append((R.WORKLOADS[w], mode_lab, *o.mean(0), float((o[:, 2] > ALPHA).mean())))
    with open(f"{out}/shift.csv", "w") as f:
        f.write("held_out_workload,setting,bayes_cost,bayes_uncaught,crc_uncaught,crc_cost,ece,share_crc_over_alpha\n")
        for r in shift:
            f.write(",".join([r[0], r[1]] + [f"{x:.4f}" for x in r[2:]]) + "\n")

    verdicts = {
        "n_queries": Q, "replicates": a.reps, "alpha": ALPHA, "primary_costs": {"error": CE, "human": CH},
        "paired_cost_differences": {k: list(v) for k, v in paired.items()},
        "ece_raw_vs_calibrated": ece_gain, "spearman_ece_vs_cost": [float(rho.statistic), float(rho.pvalue)],
        "mean_cost_unit_model_units": float(D["cbar"]),
        "table1": {k: [list(x) for x in v] for k, v in ci1.items()},
    }
    json.dump(verdicts, open(f"{out}/verdicts.json", "w"), indent=2)

    # ------------------------------------------------------------------ figures
    make_figures(out, refl, frontier, guar, sens, ci1, acc, D)
    print("analysis done ->", out)


def write_table(out, name, hdr, rows):
    import csv
    with open(f"{out}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    with open(f"{out}/{name}.md", "w") as f:
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            f.write("| " + " | ".join(map(str, r)) + " |\n")


def make_figures(out, refl, frontier, guar, sens, ci1, acc, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    C = {"raw_lp": "#D55E00", "platt_lp": "#E69F00", "iso_lp": "#0072B2", "iso_comb": "#009E73"}
    L = {"raw_lp": "Raw log-probability", "platt_lp": "Platt", "iso_lp": "Isotonic", "iso_comb": "Combined + isotonic"}
    # Figure 1: reliability and risk-spend frontier
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.28})
    ax[0].plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
    for k in C:
        p, y = np.concatenate(refl[k][0]), np.concatenate(refl[k][1])
        order = np.argsort(p)
        xs, ys = [], []
        for idx in np.array_split(order, 10):
            xs.append(p[idx].mean())
            ys.append(y[idx].mean())
        ax[0].plot(xs, ys, marker="o", ms=3, lw=1.2, color=C[k], label=L[k])
    ax[0].set_xlabel("Predicted probability the answer is correct")
    ax[0].set_ylabel("Observed accuracy")
    ax[0].set_title("(a) Reliability (test splits, all tiers)", fontsize=8, loc="left")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    for k in ("raw_lp", "iso_lp", "iso_comb"):
        arr = np.array(frontier[k]).mean(0)
        ax[1].plot(100 * arr[:, 0], arr[:, 1], marker="o", ms=3, lw=1.2, color=C[k], label=f"Expected-cost rule, {L[k].lower()}")
    for tier, mk in zip(R.TIERS, ("s", "^", "D")):
        key = f"always_{tier}"
        ax[1].scatter(100 * ci1[key][1][0], ci1[key][0][0] - CE * ci1[key][1][0], marker=mk, color="#555", s=22, zorder=5)
        ax[1].annotate(tier, (100 * ci1[key][1][0], ci1[key][0][0] - CE * ci1[key][1][0]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax[1].set_xlabel("Uncaught errors (% of queries)")
    ax[1].set_ylabel("Spend: compute + human review (L-answer units)")
    ax[1].set_title("(b) Risk-spend frontier (error cost swept)", fontsize=8, loc="left")
    ax[1].legend(frameon=False, fontsize=6.5)
    fig.savefig(f"{out}/fig1_calibration_frontier.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}/fig1_calibration_frontier.pdf", bbox_inches="tight")
    plt.close(fig)
    # Figure 2: guarantee vs calibration size; cost of stale thresholds
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.3})
    xlab = [str(n) if n is not None else f"all ({int(0.3 * D['Q'])})" for n in NCALS]
    for (mode, kind, col, ls) in (("crc", "bayes", "#009E73", "-"), ("naive", "bayes", "#D55E00", "-"), ("crc", "alone", "#009E73", ":"), ("naive", "alone", "#D55E00", ":")):
        m, lo, hi = [], [], []
        for nl in NCALS:
            v = np.array(guar[(mode, kind, "all" if nl is None else nl)])[:, 0]
            m.append(100 * v.mean())
            lo.append(100 * np.percentile(v, 5))
            hi.append(100 * np.percentile(v, 95))
        ax[0].plot(range(len(NCALS)), m, color=col, ls=ls, marker="o", ms=3, lw=1.2,
                   label=f"{'CRC' if mode == 'crc' else 'Empirical risk only'}, {'expected-cost rule' if kind == 'bayes' else 'L alone'}")
        if mode == "crc" and kind == "bayes":
            ax[0].fill_between(range(len(NCALS)), lo, hi, color=col, alpha=0.15, lw=0)
    ax[0].axhline(100 * ALPHA, color="#333", lw=0.8, ls="--")
    ax[0].text(len(NCALS) - 1, 100 * ALPHA, "target", va="bottom", ha="right", fontsize=7)
    ax[0].set_xticks(range(len(NCALS)))
    ax[0].set_xticklabels(xlab)
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel("Calibration queries")
    ax[0].set_ylabel("Uncaught errors on test (% of queries)")
    ax[0].set_title(f"(a) Risk control vs calibration size (alpha = {int(100 * ALPHA)}%)", fontsize=8, loc="left")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ces, chs = GRID_CE, GRID_CH
    H = np.array([[100 * (np.mean(sens[(ce, ch, "stale")]) - np.mean(sens[(ce, ch, "bayes")])) / np.mean(sens[(ce, ch, "bayes")]) for ce in ces] for ch in chs])
    im = ax[1].imshow(H, cmap="Blues", vmin=0, aspect="auto", origin="lower")
    ax[1].set_xticks(range(len(ces)))
    ax[1].set_xticklabels([f"{c:g}" for c in ces])
    ax[1].set_yticks(range(len(chs)))
    ax[1].set_yticklabels([f"{c:g}" for c in chs])
    for i in range(len(chs)):
        for j in range(len(ces)):
            ax[1].text(j, i, f"{round(H[i, j]) + 0:d}%", ha="center", va="center", fontsize=7, color="white" if H[i, j] > H.max() * 0.6 else "black")
    ax[1].set_xlabel("Cost of an uncaught error")
    ax[1].set_ylabel("Cost of human review")
    ax[1].set_title("(b) Extra cost of thresholds tuned at the old costs", fontsize=8, loc="left")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    fig.savefig(f"{out}/fig2_guarantee_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}/fig2_guarantee_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
