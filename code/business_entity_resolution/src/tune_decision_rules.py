"""
Decision-rule experiment on clean splits (see score_lr_clean_split.py).

  FIT  (8K training-side S1): LR fitting (done upstream); also used to fit the
        S1-level singleton gate in variant B3.
  TUNE (2K training-side S1): ALL threshold / margin / gate parameters selected
        here, by per-S1 entity-level macro F0.5 (never pairwise/pooled).
  VAL  (10K val_s1_ids sample): untouched; each frozen variant evaluated once.

Every variant keeps the multiple-match decision: within an accepted S1, every
candidate with score >= its threshold is predicted (never top-1).

Variants
  A   global threshold t on the LR logit z
  B1  two-level threshold: t_high for S1 whose above-t_low candidates contain no
      exact (suffix-stripped) name match, else t_low
  B2  two-level threshold: t_high for S1 whose own address is missing, else t_low
  B3  two-level threshold: t_high for S1 with p_singleton >= s, where
      p_singleton comes from an S1-level LR trained on FIT S1 using only
      observable candidate-score aggregates (GT is the training target only)
  C1  S1 gate: accept S1 only if top z >= T  (then predict all z >= t)
  C2  S1 gate: top z >= T AND (top z - second z) >= M
"Singleton-like" in B* is always an inference-time observable; ground-truth
match counts are used only as a training target (B3) or for scoring.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
SCORE_DIR = os.path.join(ROOT, "data_cache", "dev_scores")
REPORT_DIR = os.path.join(ROOT, "reports")

Z_FLOOR = 2.0                                   # rows below this can never be predicted (grid min 3.0)
T_GRID = np.round(np.arange(3.0, 16.01, 0.1), 2)
M_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
S_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
PRIOR_THRESHOLDS_P = {"D3": 0.9995, "C": 0.9999}   # val-tuned thresholds from train_evaluate_matcher.py
N_BOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}]  {msg}", file=sys.stderr, flush=True)


def logit(p):
    return float(np.log(p / (1 - p)))


def sigmoid(z):
    return float(1 / (1 + np.exp(-z)))


# ----------------------------------------------------------------------------- data

class Part:
    """Per-S1 arrays + pruned, S1-sorted candidate rows for one part (fit/tune/val)."""

    def __init__(self, s1_df, rows):
        self.s1_ids = s1_df["entity_id"].values
        self.n = len(s1_df)
        self.n_true = s1_df["n_true"].values
        self.country = s1_df["country"].values
        idx_of = pd.Series(np.arange(self.n), index=self.s1_ids)
        s = idx_of.reindex(rows["s1_id"].astype(str).values).values
        assert not np.isnan(s).any(), "pair rows for S1 outside the sample"
        s = s.astype(np.int64)
        z = rows["z"].values
        order = np.lexsort((-z, s))
        rows = rows.iloc[order].reset_index(drop=True)
        s, z = s[order], z[order]

        # per-S1 aggregates over ALL candidates (before pruning)
        self.n_cand = np.bincount(s, minlength=self.n)
        start = np.zeros(self.n + 1, dtype=np.int64)
        start[1:] = np.cumsum(self.n_cand)
        has1 = self.n_cand >= 1
        has2 = self.n_cand >= 2
        self.z1 = np.full(self.n, -np.inf)
        self.z2 = np.full(self.n, -np.inf)
        self.z1[has1] = z[start[:-1][has1]]
        self.z2[has2] = z[start[:-1][has2] + 1]
        top = start[:-1][has1]
        self.top_name_exact = np.zeros(self.n)
        self.top_name_exact[has1] = rows["name_nosuffix_exact"].values[top]
        self.top_addr_miss_cand = np.zeros(self.n)
        self.top_addr_miss_cand[has1] = rows["address_missing_cand"].values[top]
        self.top_nch = np.zeros(self.n)
        self.top_nch[has1] = rows["n_channels_hit"].values[top]
        self.top_postal = np.zeros(self.n)
        self.top_postal[has1] = rows["postal_equal"].values[top]
        self.top_label = np.zeros(self.n)
        self.top_label[has1] = rows["label"].values[top]   # diagnostics only
        am = np.zeros(self.n)
        np.maximum.at(am, s, rows["address_missing_s1"].values.astype(float))
        self.addr_missing_s1 = am
        self.n_ge = {k: np.bincount(s[z >= k], minlength=self.n) for k in (4, 6, 8, 10, 12)}

        keep = z >= Z_FLOOR
        self.r_s = s[keep]
        self.r_z = z[keep]
        self.r_label = rows["label"].values[keep].astype(np.int64)
        self.r_name_exact = rows["name_nosuffix_exact"].values[keep].astype(np.int64)
        self.r_cand = rows["cand_id"].astype(str).values[keep]

    def s1_features(self):
        z1 = np.where(np.isfinite(self.z1), self.z1, -10.0)
        z2 = np.where(np.isfinite(self.z2), self.z2, -10.0)
        cols = {
            "z1": z1, "z2": z2, "margin": z1 - z2,
            "log_n_cand": np.log1p(self.n_cand),
            "top_name_exact": self.top_name_exact, "top_addr_miss_cand": self.top_addr_miss_cand,
            "top_nch": self.top_nch, "top_postal": self.top_postal,
            "addr_missing_s1": self.addr_missing_s1,
            "is_india": (self.country == "India").astype(float),
        }
        for k, v in self.n_ge.items():
            cols[f"log_n_ge{k}"] = np.log1p(v)
        return pd.DataFrame(cols)


def load(config):
    s1 = pd.read_parquet(os.path.join(SCORE_DIR, f"{config}_s1_table.parquet"))
    tr = pd.read_parquet(os.path.join(SCORE_DIR, f"{config}_train_scores.parquet"))
    for c in ("s1_id", "cand_id"):
        tr[c] = tr[c].astype(str)
    parts = {}
    for p in ("fit", "tune"):
        parts[p] = Part(s1[s1["part"] == p].reset_index(drop=True), tr[tr["part"] == p])
        log(f"  {p}: {parts[p].n} S1, {len(parts[p].r_z)} rows with z>={Z_FLOOR}")
    del tr
    va = pd.read_parquet(os.path.join(SCORE_DIR, f"{config}_val_scores.parquet"))
    for c in ("s1_id", "cand_id"):
        va[c] = va[c].astype(str)
    parts["val"] = Part(s1[s1["part"] == "val"].reset_index(drop=True), va)
    log(f"  val: {parts['val'].n} S1, {len(parts['val'].r_z)} rows with z>={Z_FLOOR}")
    return parts


# ----------------------------------------------------------------------------- metric

def f05_vec(n_true, n_pred, tp):
    """Per-S1 (P, R, F0.5) with the same conventions as train_evaluate_matcher.entity_f05."""
    n_true = n_true.astype(float)
    n_pred = n_pred.astype(float)
    tp = tp.astype(float)
    P = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 1.0)
    R = np.where(n_true > 0, tp / np.maximum(n_true, 1), np.where(n_pred > 0, 1.0, 1.0))
    denom = 0.25 * P + R
    F = np.where(denom > 0, 1.25 * P * R / np.where(denom > 0, denom, 1), 0.0)
    both_empty = (n_true == 0) & (n_pred == 0)
    pred_empty = (n_pred == 0) & (n_true > 0)
    spurious = (n_true == 0) & (n_pred > 0)
    P = np.where(spurious, 0.0, P)
    R = np.where(pred_empty, 0.0, R)
    F = np.where(both_empty, 1.0, np.where(pred_empty | spurious, 0.0, F))
    return P, R, F


def counts_at(part, row_thresh):
    """row_thresh: per-row threshold array (or scalar). Returns per-S1 n_pred, tp, row mask."""
    m = part.r_z >= row_thresh
    n_pred = np.bincount(part.r_s[m], minlength=part.n)
    tp = np.bincount(part.r_s[m], weights=part.r_label[m], minlength=part.n)
    return n_pred, tp, m


class Cache:
    """Per-S1 F for each grid threshold on one part (computed once, reused by all rules)."""

    def __init__(self, part):
        self.part = part
        self.F = {}
        self.n_pred = {}
        for t in T_GRID:
            n_pred, tp, _ = counts_at(part, t)
            self.F[t] = f05_vec(part.n_true, n_pred, tp)[2]
            self.n_pred[t] = n_pred
        self.F_empty = f05_vec(part.n_true, np.zeros(part.n), np.zeros(part.n))[2]


# ----------------------------------------------------------------------------- rules
# Each rule is (name, params) and maps to a per-row threshold on a part; an S1
# rejected by a gate gets threshold +inf (predicted empty).

def row_threshold(part, rule, params, s1_gate_prob=None):
    t = params["t"]
    if rule == "A":
        s1_t = np.full(part.n, t)
    elif rule == "B1":
        # singleton-like := no exact-name candidate among candidates with z >= t_low
        m = part.r_z >= t
        has_exact = np.bincount(part.r_s[m], weights=part.r_name_exact[m], minlength=part.n) > 0
        s1_t = np.where(has_exact, t, params["t_high"])
    elif rule == "B2":
        s1_t = np.where(part.addr_missing_s1 > 0, params["t_high"], t)
    elif rule == "B3":
        s1_t = np.where(s1_gate_prob >= params["s"], params["t_high"], t)
    elif rule == "C1":
        s1_t = np.where(part.z1 >= params["T"], t, np.inf)
    elif rule == "C2":
        ok = (part.z1 >= params["T"]) & ((part.z1 - part.z2) >= params["M"])
        s1_t = np.where(ok, t, np.inf)
    else:
        raise ValueError(rule)
    return s1_t[part.r_s]


def tune_rules(cache, tune_gate_prob):
    """Grid search every rule on the TUNE part. Returns {rule: (params, tune_macro_f05)}."""
    part = cache.part
    best = {}

    # A
    scores = {t: cache.F[t].mean() for t in T_GRID}
    t_a = max(scores, key=scores.get)
    best["A"] = ({"t": float(t_a)}, float(scores[t_a]))

    # B1: per-S1 has_exact depends on t_low -> compute per t_low
    b = (None, -1)
    for t in T_GRID:
        m = part.r_z >= t
        has_exact = np.bincount(part.r_s[m], weights=part.r_name_exact[m], minlength=part.n) > 0
        for th in T_GRID[T_GRID >= t]:
            v = np.where(has_exact, cache.F[t], cache.F[th]).mean()
            if v > b[1]:
                b = ({"t": float(t), "t_high": float(th)}, float(v))
    best["B1"] = b

    # B2
    sl = part.addr_missing_s1 > 0
    b = (None, -1)
    for t in T_GRID:
        for th in T_GRID[T_GRID >= t]:
            v = np.where(sl, cache.F[th], cache.F[t]).mean()
            if v > b[1]:
                b = ({"t": float(t), "t_high": float(th)}, float(v))
    best["B2"] = b

    # B3 (t_high = inf allowed: singleton-like S1 predicted empty)
    b = (None, -1)
    Fs = dict(cache.F)
    Fs[np.inf] = cache.F_empty
    for s in S_GRID:
        sl = tune_gate_prob >= s
        for t in T_GRID:
            for th in list(T_GRID[T_GRID >= t]) + [np.inf]:
                v = np.where(sl, Fs[th], cache.F[t]).mean()
                if v > b[1]:
                    b = ({"t": float(t), "t_high": float(th), "s": float(s)}, float(v))
    best["B3"] = b

    # C1 / C2
    for rule, mgrid in (("C1", [0.0]), ("C2", M_GRID)):
        b = (None, -1)
        margin = part.z1 - part.z2
        for T in T_GRID:
            for M in mgrid:
                ok = (part.z1 >= T) & (margin >= M)
                for t in T_GRID[T_GRID <= T]:
                    v = np.where(ok, cache.F[t], cache.F_empty).mean()
                    if v > b[1]:
                        p = {"t": float(t), "T": float(T)}
                        if rule == "C2":
                            p["M"] = float(M)
                        b = (p, float(v))
        best[rule] = b
    return best


# ----------------------------------------------------------------------------- eval

def evaluate(part, rule, params, gate_prob=None):
    rt = row_threshold(part, rule, params, gate_prob)
    n_pred, tp, m = counts_at(part, rt)
    P, R, F = f05_vec(part.n_true, n_pred, tp)
    single = part.n_true == 0
    matched = ~single
    multi = part.n_true >= 2
    claims = pd.Series(part.r_cand[m]).value_counts()
    res = {
        "macro_f05": round(float(F.mean()), 4),
        "macro_precision": round(float(P.mean()), 4),
        "macro_recall": round(float(R.mean()), 4),
        "n_singleton_total": int(single.sum()),
        "singleton_correctly_empty": int((single & (n_pred == 0)).sum()),
        "singleton_fp_rate_pct": round(100 * float((n_pred[single] > 0).mean()), 2),
        "matched_s1_zero_predictions": int((matched & (n_pred == 0)).sum()),
        "avg_pred_per_s1": round(float(n_pred.mean()), 3),
        "duplicate_candidate_claims": int((claims > 1).sum()),
        "n_distinct_predicted_candidates": int(len(claims)),
        "macro_f05_US": round(float(F[part.country == "US"].mean()), 4),
        "macro_f05_India": round(float(F[part.country == "India"].mean()), 4),
        "macro_f05_matched_only": round(float(F[matched].mean()), 4),
        "multi_match_s1": int(multi.sum()),
        "multi_match_macro_recall": round(float(R[multi].mean()), 4),
        "multi_match_avg_pred": round(float(n_pred[multi].mean()), 3),
        "multi_match_s1_with_2plus_pred": int((multi & (n_pred >= 2)).sum()),
    }
    return res, F


def paired_bootstrap(Fa, Fb, n_boot=N_BOOT, seed=42):
    """95% CI of mean(Fb - Fa) resampling S1 with replacement (same S1 set)."""
    rng = np.random.default_rng(seed)
    d = Fb - Fa
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"delta": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "p_delta_le_0": round(float((boots <= 0).mean()), 4)}


def params_with_p(params):
    out = {}
    for k, v in params.items():
        out[k] = v
        if k in ("t", "t_high", "T") and np.isfinite(v):
            out[f"{k}_prob"] = sigmoid(v)
    return out


# ----------------------------------------------------------------------------- main

def explore(parts):
    """Training-side only (FIT+TUNE): how observable S1 signals relate to singleton status."""
    rows = []
    for p in ("fit", "tune"):
        part = parts[p]
        df = part.s1_features()
        df["single"] = part.n_true == 0
        df["top_label"] = part.top_label
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    out = {}
    out["means_by_singleton"] = df.groupby("single").mean().round(3).T.to_dict()
    df["z1_bin"] = pd.cut(df["z1"], [-99, 0, 4, 6, 7, 8, 9, 10, 11, 12, 14, 99]).astype(str)
    out["by_z1_bin"] = df.groupby("z1_bin").agg(n=("single", "size"), singleton_rate=("single", "mean"),
                                                 top_is_true=("top_label", "mean")).round(3).to_dict("index")
    df["margin_bin"] = pd.cut(df["margin"], [-1, 0.25, 0.5, 1, 2, 3, 5, 99]).astype(str)
    out["by_margin_bin"] = df.groupby("margin_bin").agg(n=("single", "size"), singleton_rate=("single", "mean"),
                                                        top_is_true=("top_label", "mean")).round(3).to_dict("index")
    out["by_addr_missing_s1"] = df.groupby("addr_missing_s1").agg(n=("single", "size"), singleton_rate=("single", "mean")).round(3).to_dict("index")
    out["by_top_name_exact"] = df.groupby("top_name_exact").agg(n=("single", "size"), singleton_rate=("single", "mean")).round(3).to_dict("index")
    out["by_is_india"] = df.groupby("is_india").agg(n=("single", "size"), singleton_rate=("single", "mean")).round(3).to_dict("index")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["D3", "C"], required=True)
    args = ap.parse_args()

    log(f"Loading scored parts for {args.config}...")
    parts = load(args.config)
    report = {"config": args.config,
              "split": {p: {"n_s1": parts[p].n, "n_singleton": int((parts[p].n_true == 0).sum())} for p in parts},
              "exploration_training_side": explore(parts)}

    # B3 gate: S1-level LR on FIT S1, observable aggregates only
    Xf = parts["fit"].s1_features()
    gsc = StandardScaler().fit(Xf.values)
    gate = LogisticRegression(max_iter=2000, C=1.0)
    gate.fit(gsc.transform(Xf.values), (parts["fit"].n_true == 0).astype(int))
    gate_prob = {p: gate.predict_proba(gsc.transform(parts[p].s1_features().values))[:, 1] for p in parts}
    report["B3_gate_coefficients"] = dict(zip(Xf.columns, gate.coef_[0].round(3).tolist()))
    from sklearn.metrics import roc_auc_score
    report["B3_gate_auc"] = {p: round(float(roc_auc_score(parts[p].n_true == 0, gate_prob[p])), 4) for p in ("fit", "tune", "val")}
    log(f"B3 gate AUC (singleton vs matched): {report['B3_gate_auc']}")

    log("Tuning all rules on TUNE (2K S1)...")
    tune_cache = Cache(parts["tune"])
    best = tune_rules(tune_cache, gate_prob["tune"])

    variants = {}
    F_val = {}
    for rule, (params, tune_f) in best.items():
        tune_res, _ = evaluate(parts["tune"], rule, params, gate_prob["tune"])
        val_res, F = evaluate(parts["val"], rule, params, gate_prob["val"])
        F_val[rule] = F
        variants[rule] = {"params": params_with_p(params), "tune": tune_res, "val": val_res}
        log(f"  {rule}: params={params} tune_f05={tune_f:.4f} -> VAL f05={val_res['macro_f05']} "
            f"P={val_res['macro_precision']} R={val_res['macro_recall']} "
            f"singletonFP={val_res['singleton_fp_rate_pct']}%")

    for rule in variants:
        if rule != "A":
            variants[rule]["val_delta_vs_A"] = paired_bootstrap(F_val["A"], F_val[rule])
    report["variants"] = variants

    # reference only: previously selected (val-tuned) global threshold, 8K-fit model
    p_prior = PRIOR_THRESHOLDS_P[args.config]
    ref_res, F_ref = evaluate(parts["val"], "A", {"t": logit(p_prior)})
    report["reference_prior_val_tuned_threshold"] = {
        "note": "threshold previously chosen on this same val sample; NOT a clean validation score",
        "threshold_prob": p_prior, "threshold_z": round(logit(p_prior), 4), "val": ref_res}

    # oracle-ish diagnostic: best achievable global threshold on val (for measuring tune->val gap only)
    val_cache_scores = {t: f05_vec(parts["val"].n_true, *counts_at(parts["val"], t)[:2])[2].mean() for t in T_GRID}
    t_or = max(val_cache_scores, key=val_cache_scores.get)
    report["diagnostic_val_optimal_global_threshold"] = {
        "note": "val-optimized, optimistic; reported only to size the tuning gap",
        "t": float(t_or), "t_prob": sigmoid(t_or), "macro_f05": round(float(val_cache_scores[t_or]), 4)}

    np.save(os.path.join(SCORE_DIR, f"{args.config}_val_perS1_F_A.npy"), F_val["A"])
    pd.DataFrame({"s1_id": parts["val"].s1_ids, **{f"F_{k}": v for k, v in F_val.items()}}).to_parquet(
        os.path.join(SCORE_DIR, f"{args.config}_val_perS1_F.parquet"), index=False)

    out = os.path.join(REPORT_DIR, f"decision_rules_{args.config}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
