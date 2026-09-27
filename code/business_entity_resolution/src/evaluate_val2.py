"""
Independent replication of the C+B3 vs D3+B3 comparison on a SECOND untouched
validation sample (val2).

val2 = 10K S1 drawn (seed 2027) from val_s1_ids MINUS the first 10K val sample
(seed 42) -- disjoint from FIT, TUNE and the first VAL sample. Candidates and
features were produced by the unchanged pipeline
(generate_dev_candidates.py --val2-seed 2027, build_pairwise_features.py --split val2).

Everything is FROZEN before val2 is touched:
  * LR: refit on the same FIT rows with the same hyperparameters as
    score_lr_clean_split.py (lbfgs is deterministic); coefficients are
    asserted equal to the saved {config}_lr_coef.json.
  * B3 singleton gate: refit on FIT S1 exactly as tune_decision_rules.py.
  * Decision parameters: read from reports/decision_rules_{config}.json
    (TUNE-selected); no parameter is chosen on val2.
Sanity check: the frozen rules are re-evaluated on the first VAL sample and must
reproduce the published numbers before val2 is scored.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from score_lr_clean_split import (FEAT_DIR, CACHE_DIR, SPLIT_DIR, GT_PATH, OUT_DIR, NON_FEATURE_COLS,
                                  KEEP_OBS, read_features)
from tune_decision_rules import Part, evaluate, paired_bootstrap, f05_vec, counts_at, row_threshold

HERE = os.path.dirname(__file__)
REPORT_DIR = os.path.join(HERE, "..", "..", "..", "reports")
VAL2_SEED = 2027
N = 10000
CONFIGS = ("D3", "C")
EXPECTED = {("C", "B3"): {"t": 8.7, "t_high": 13.4, "s": 0.3}}   # user-stated frozen C+B3 params
PUBLISHED_VAL = {("D3", "A"): 0.7602, ("D3", "B3"): 0.7601, ("C", "A"): 0.7620, ("C", "B3"): 0.7644}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}]  {msg}", file=sys.stderr, flush=True)


def val2_s1_table():
    """Replicates generate_dev_candidates.py sample_split() for val and then val2."""
    ids_all = set(pd.read_csv(os.path.join(SPLIT_DIR, "val_s1_ids.txt"), header=None)[0].astype(str))
    s1_full = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"), columns=["entity_id", "country"])

    def sample(ids, seed):
        s1 = s1_full[s1_full["entity_id"].isin(ids)].reset_index(drop=True)
        idx = np.random.default_rng(seed).choice(len(s1), size=min(N, len(s1)), replace=False)
        return s1.iloc[idx].reset_index(drop=True)

    v1 = sample(ids_all, 42)
    v2 = sample(ids_all - set(v1["entity_id"]), VAL2_SEED)
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    gt = gt[gt["source1_entity_id"].isin(set(v2["entity_id"]))]
    n_true = {r.source1_entity_id: len(r.matched_entity_ids.split(","))
              for r in gt.itertuples() if isinstance(r.matched_entity_ids, str) and r.matched_entity_ids}
    v2["n_true"] = v2["entity_id"].map(n_true).fillna(0).astype(int)
    v2["part"] = "val2"
    # disjointness from everything used so far
    used = pd.read_parquet(os.path.join(OUT_DIR, "D3_s1_table.parquet"))["entity_id"]
    assert not set(v2["entity_id"]) & set(used), "val2 overlaps fit/tune/val"
    assert set(v1["entity_id"]) == set(used[pd.read_parquet(os.path.join(OUT_DIR, "D3_s1_table.parquet"))["part"] == "val"])
    return v2


def score_val2(config):
    out_path = os.path.join(OUT_DIR, f"{config}_val2_scores.parquet")
    if os.path.exists(out_path):
        log(f"{config}: val2 scores exist, reusing")
        return
    s1 = pd.read_parquet(os.path.join(OUT_DIR, f"{config}_s1_table.parquet"))
    tune_ids = set(s1.loc[s1["part"] == "tune", "entity_id"])
    tr = read_features(os.path.join(FEAT_DIR, f"{config}_train_features.parquet"))
    feat_cols = [c for c in tr.columns if c not in NON_FEATURE_COLS]
    is_fit = ~tr["s1_id"].astype(str).isin(tune_ids).values
    X = tr[feat_cols].values.astype(np.float32)
    y = tr["label"].values
    del tr
    scaler = StandardScaler()
    Xf = scaler.fit_transform(X[is_fit])
    del X
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    lr.fit(Xf, y[is_fit])
    del Xf
    saved = pd.read_json(os.path.join(OUT_DIR, f"{config}_lr_coef.json"), typ="series")
    diff = np.abs(saved[feat_cols].values - lr.coef_[0]).max()
    log(f"{config}: refit LR, max |coef - saved| = {diff:.2e}")
    assert diff < 1e-6, "refit LR differs from the frozen model"

    va = read_features(os.path.join(FEAT_DIR, f"{config}_val2_features.parquet"))
    assert [c for c in va.columns if c not in NON_FEATURE_COLS] == feat_cols
    Xv = va[feat_cols].values.astype(np.float32)
    z = np.empty(len(va))
    for i in range(0, len(va), 2_000_000):
        z[i:i + 2_000_000] = lr.decision_function(scaler.transform(Xv[i:i + 2_000_000]))
    out = va[["s1_id", "cand_id", "label", "country"] + KEEP_OBS].copy()
    out["z"] = z
    out["part"] = "val2"
    out.to_parquet(out_path, index=False)
    log(f"{config}: wrote {len(out)} val2 scores")


READ_COLS = ["s1_id", "cand_id", "label", "z", "name_nosuffix_exact", "address_missing_cand",
             "n_channels_hit", "postal_equal", "address_missing_s1"]


def load_part(config, s1_df, path, filters=None):
    rows = pd.read_parquet(path, columns=READ_COLS, filters=filters)
    for c in ("s1_id", "cand_id"):
        rows[c] = rows[c].astype(str)
    return Part(s1_df.reset_index(drop=True), rows)


def country_breakdown(part, rule, params, gate_prob):
    rt = row_threshold(part, rule, params, gate_prob)
    n_pred, tp, _ = counts_at(part, rt)
    P, R, F = f05_vec(part.n_true, n_pred, tp)
    out = {}
    for c in ("US", "India"):
        m = part.country == c
        single = m & (part.n_true == 0)
        out[c] = {"n_s1": int(m.sum()), "n_singleton": int(single.sum()),
                  "macro_f05": round(float(F[m].mean()), 4), "macro_precision": round(float(P[m].mean()), 4),
                  "macro_recall": round(float(R[m].mean()), 4),
                  "singleton_fp_rate_pct": round(100 * float((n_pred[single] > 0).mean()), 2),
                  "matched_s1_zero_predictions": int((m & (part.n_true > 0) & (n_pred == 0)).sum())}
    return out


def main():
    v2 = val2_s1_table()
    log(f"val2: {len(v2)} S1, {int((v2['n_true'] == 0).sum())} singletons, "
        f"US={int((v2['country'] == 'US').sum())} India={int((v2['country'] == 'India').sum())}")
    report = {"val2": {"seed": VAL2_SEED, "n_s1": len(v2), "n_singleton": int((v2["n_true"] == 0).sum()),
                       "disjoint_from": ["fit", "tune", "val"]}, "configs": {}}
    F2 = {}
    for config in CONFIGS:
        score_val2(config)
        rules = json.load(open(os.path.join(REPORT_DIR, f"decision_rules_{config}.json")))["variants"]
        frozen = {r: {k: v for k, v in rules[r]["params"].items() if not k.endswith("_prob")} for r in ("A", "B3")}
        for key, exp in EXPECTED.items():
            if key[0] == config:
                assert frozen[key[1]] == exp, f"frozen params {frozen[key[1]]} != stated {exp}"
        s1 = pd.read_parquet(os.path.join(OUT_DIR, f"{config}_s1_table.parquet"))

        fit = load_part(config, s1[s1["part"] == "fit"], os.path.join(OUT_DIR, f"{config}_train_scores.parquet"),
                        filters=[("part", "==", "fit")])
        Xf = fit.s1_features()
        gsc = StandardScaler().fit(Xf.values)
        gate = LogisticRegression(max_iter=2000, C=1.0).fit(gsc.transform(Xf.values), (fit.n_true == 0).astype(int))
        del fit

        # sanity: reproduce published first-VAL numbers with the refit gate
        val = load_part(config, s1[s1["part"] == "val"], os.path.join(OUT_DIR, f"{config}_val_scores.parquet"))
        gp = gate.predict_proba(gsc.transform(val.s1_features().values))[:, 1]
        for r in ("A", "B3"):
            got = evaluate(val, r, frozen[r], gp)[0]["macro_f05"]
            log(f"{config}+{r} on first VAL: {got} (published {PUBLISHED_VAL[(config, r)]})")
            assert abs(got - PUBLISHED_VAL[(config, r)]) < 5e-5, "cannot reproduce published VAL result"
        del val

        p2 = load_part(config, v2, os.path.join(OUT_DIR, f"{config}_val2_scores.parquet"))
        gp2 = gate.predict_proba(gsc.transform(p2.s1_features().values))[:, 1]
        cres = {"frozen_params": frozen}
        for r in ("A", "B3"):
            res, F = evaluate(p2, r, frozen[r], gp2)
            res["by_country"] = country_breakdown(p2, r, frozen[r], gp2)
            res["s1_flagged_singleton_like"] = int((gp2 >= frozen["B3"]["s"]).sum()) if r == "B3" else None
            cres[r] = res
            F2[(config, r)] = F
            log(f"{config}+{r} VAL2: F0.5={res['macro_f05']} P={res['macro_precision']} R={res['macro_recall']} "
                f"singletonFP={res['singleton_fp_rate_pct']}% matched0={res['matched_s1_zero_predictions']}")
        assert (p2.s1_ids == v2["entity_id"].values).all()
        report["configs"][config] = cres

    def boot(a, b):
        r = paired_bootstrap(F2[a], F2[b])
        for c in ("US", "India"):
            m = v2["country"].values == c
            r[f"{c}"] = paired_bootstrap(F2[a][m], F2[b][m])
        return r

    report["paired_bootstrap_val2"] = {
        "PRIMARY: C+B3 vs D3+B3": boot(("D3", "B3"), ("C", "B3")),
        "C+B3 vs D3+A (baseline)": boot(("D3", "A"), ("C", "B3")),
        "C+A vs D3+A": boot(("D3", "A"), ("C", "A")),
        "C+B3 vs C+A": boot(("C", "A"), ("C", "B3")),
        "D3+B3 vs D3+A": boot(("D3", "A"), ("D3", "B3")),
    }
    for k, v in report["paired_bootstrap_val2"].items():
        log(f"{k}: delta={v['delta']} CI={v['ci95']} P(<=0)={v['p_delta_le_0']}")
    pd.DataFrame({"s1_id": v2["entity_id"], "country": v2["country"], "n_true": v2["n_true"],
                  **{f"F_{c}_{r}": F for (c, r), F in F2.items()}}).to_parquet(
        os.path.join(OUT_DIR, "val2_perS1_F.parquet"), index=False)
    with open(os.path.join(REPORT_DIR, "val2_evaluation.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log("Wrote reports/val2_evaluation.json")


if __name__ == "__main__":
    main()
