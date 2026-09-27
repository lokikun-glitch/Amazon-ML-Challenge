"""
Final replication gate on VAL2: frozen B (V2-A) vs B + H2 (V2-B).

Nothing is tuned here. Fixed parameters: B threshold z >= 8.3; H2 t_low = 3.0 with the
number-agreement condition (tune_decision_rules_B.RulePart). The B model is the frozen
FIT-only LR, rebuilt by its deterministic refit and accepted only if it reproduces the
stored original-VAL scores (data_cache/dev_scores/D3B_num_val_z.parquet) -- an identity
check on scores, not a re-evaluation; original-VAL metrics are not recomputed or written.

VAL2 = 10K S1 from val_s1_ids minus the original VAL sample (seed 2027), disjoint from
FIT/TUNE/VAL. It is loaded only after the model is fixed.

Writes: data_cache/models/B_frozen.npz (scaler + LR + feature order + threshold),
        data_cache/dev_scores/D3B_num_val2_z.parquet, D3B_val2_perS1_F.parquet,
        reports/val2_B_H2_evaluation.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from blocking import rss_mb
from build_number_features import AGREEMENT, CONTRADICTION
from evaluate_val2 import val2_s1_table
from number_feature_experiment import load_split
from score_lr_clean_split import NON_FEATURE_COLS
from tune_decision_rules import f05_vec, paired_bootstrap
from tune_decision_rules_B import PART_COLS, T_FROZEN, RulePart

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
FEAT_DIR = os.path.join(ROOT, "data_cache", "dev_features")
SCORE_DIR = os.path.join(ROOT, "data_cache", "dev_scores")
MODEL_DIR = os.path.join(ROOT, "data_cache", "models")
REPORT_DIR = os.path.join(ROOT, "reports")
H2_T_LOW = 3.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr, flush=True)


def score(scaler, lr, X):
    return np.concatenate([lr.decision_function(scaler.transform(X[i:i + 2_000_000]))
                           for i in range(0, len(X), 2_000_000)])


def frozen_model():
    s1 = pd.read_parquet(os.path.join(SCORE_DIR, "D3_s1_table.parquet"))
    tune_ids = set(s1.loc[s1["part"] == "tune", "entity_id"])
    cols = [c for c in pq.read_schema(os.path.join(FEAT_DIR, "D3_train_features.parquet")).names
            if c not in NON_FEATURE_COLS] + AGREEMENT + CONTRADICTION
    tr = load_split("train")
    is_fit = ~tr["s1_id"].isin(tune_ids).values
    fit_ids = set(tr.loc[is_fit, "s1_id"].unique())
    X = tr[cols].values.astype(np.float32)
    scaler = StandardScaler()
    Xf = scaler.fit_transform(X[is_fit])
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    lr.fit(Xf, tr["label"].values[is_fit])
    del Xf, X, tr
    stored = json.load(open(os.path.join(REPORT_DIR, "number_features_experiment.json")))["coefficients"]["B"]
    coef_diff = max(abs(round(float(w), 4) - stored[c]) for c, w in zip(cols, lr.coef_[0]))
    va = load_split("val")
    zv = score(scaler, lr, va[cols].values.astype(np.float32))
    del va
    z_stored = pd.read_parquet(os.path.join(SCORE_DIR, "D3B_num_val_z.parquet"))["z_val"].values
    z_diff = float(np.abs(zv - z_stored).max())
    log(f"frozen B rebuilt: max|coef-stored(4dp)|={coef_diff:.1e}, max|z_val - stored z_val|={z_diff:.2e}")
    assert coef_diff == 0 and z_diff < 1e-9, "rebuilt model is not the frozen B"
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, "B_frozen.npz"), feature_order=np.array(cols), scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_, coef=lr.coef_[0], intercept=lr.intercept_, threshold_z=T_FROZEN,
             h2_t_low=H2_T_LOW)
    checks = {"model_fit_rows": "FIT S1 only (training-side features; TUNE excluded)",
              "n_fit_s1_in_model": len(fit_ids),
              "coef_max_abs_diff_vs_stored_4dp": coef_diff,
              "original_val_z_max_abs_diff_vs_stored": z_diff}
    return scaler, lr, cols, fit_ids, checks


def metrics(rp, acc, v2):
    p = rp.p
    n_pred = np.bincount(p.r_s[acc], minlength=p.n)
    tp = np.bincount(p.r_s[acc], weights=p.r_label[acc], minlength=p.n)
    P, R, F = f05_vec(p.n_true, n_pred, tp)
    single, matched, multi = p.n_true == 0, p.n_true > 0, p.n_true >= 2
    claims = pd.Series(p.r_cand[acc]).value_counts()
    res = {"macro_f05": round(float(F.mean()), 4), "macro_precision": round(float(P.mean()), 4),
           "macro_recall": round(float(R.mean()), 4),
           "singleton_fp_pct": round(100 * float((n_pred[single] > 0).mean()), 2),
           "matched_zero_pred_pct": round(100 * float((n_pred[matched] == 0).mean()), 2),
           "multi_match_recall": round(float(R[multi].mean()), 4),
           "avg_pred_per_s1": round(float(n_pred.mean()), 3),
           "duplicate_claims": int((claims > 1).sum()),
           "s1_zero_pred": int((n_pred == 0).sum()), "s1_one_pred": int((n_pred == 1).sum()),
           "s1_multi_pred": int((n_pred >= 2).sum())}
    for c in ("US", "India", "France"):
        m = p.country == c
        res[f"f05_{c}"] = round(float(F[m].mean()), 4) if m.any() else None
    return res, F


def main():
    scaler, lr, cols, fit_ids, checks = frozen_model()

    v2 = val2_s1_table()   # asserts disjointness from fit/tune/val
    s1 = pd.read_parquet(os.path.join(SCORE_DIR, "D3_s1_table.parquet"))
    v2_ids = set(v2["entity_id"])
    for part in ("fit", "tune", "val"):
        ov = len(v2_ids & set(s1.loc[s1["part"] == part, "entity_id"]))
        checks[f"val2_overlap_with_{part}"] = ov
        assert ov == 0
    checks["val2_overlap_with_model_fit_rows"] = len(v2_ids & fit_ids)
    assert checks["val2_overlap_with_model_fit_rows"] == 0

    va2 = load_split("val2")
    assert set(va2["s1_id"].unique()) <= v2_ids, "val2 pairs contain S1 outside the val2 sample"
    z2 = score(scaler, lr, va2[cols].values.astype(np.float32))
    pd.DataFrame({"z_val2": z2}).to_parquet(os.path.join(SCORE_DIR, "D3B_num_val2_z.parquet"), index=False)
    rows = va2[["s1_id", "cand_id", "label"] + PART_COLS].copy()
    rows["z"] = z2
    rows["num_ok"] = ((va2["num_hn_head_equal"] == 1) & (va2["num_hn_head_mismatch"] == 0)
                      & (va2["num_compound_mismatch"] == 0) & (va2["num_token_contradiction"] == 0)).values
    del va2
    rp = RulePart(v2.reset_index(drop=True), rows)
    assert (rp.p.s1_ids == v2["entity_id"].values).all()
    checks["val2_n_s1"] = int(rp.p.n)
    checks["val2_n_singleton"] = int((rp.p.n_true == 0).sum())
    checks["val2_countries"] = pd.Series(rp.p.country).value_counts().to_dict()
    checks["thresholds_selected_on_val2"] = "none (8.3 from TUNE; H2 t_low=3.0 from TUNE)"

    res_a, F_a = metrics(rp, rp.accepted("B"), v2)
    res_b, F_b = metrics(rp, rp.accepted("H2", H2_T_LOW), v2)
    boot = paired_bootstrap(F_a, F_b)
    for c in ("US", "India"):
        m = rp.p.country == c
        boot[c] = paired_bootstrap(F_a[m], F_b[m])
    # supplementary: frozen B vs A on VAL2 (A's per-S1 F from evaluate_val2.py, same S1 order)
    fa = pd.read_parquet(os.path.join(SCORE_DIR, "val2_perS1_F.parquet"))
    assert (fa["s1_id"].values == rp.p.s1_ids).all()
    supp = paired_bootstrap(fa["F_D3_A"].values, F_a)
    delta = {k: round(res_b[k] - res_a[k], 4) for k in res_a if isinstance(res_a[k], (int, float)) and res_a[k] is not None}
    out = {"integrity_checks": checks, "V2-A_frozen_B": res_a, "V2-B_B_plus_H2": res_b,
           "delta_H2_minus_B": delta, "bootstrap_H2_minus_B": boot,
           "supplementary_B_minus_A_on_val2": supp,
           "fixed_params": {"threshold_z": T_FROZEN, "h2_t_low": H2_T_LOW,
                            "h2_condition": "top candidate only, for S1 with no z>=8.3; num_hn_head_equal==1 & "
                                            "num_hn_head_mismatch==0 & num_compound_mismatch==0 & num_token_contradiction==0"}}
    pd.DataFrame({"s1_id": rp.p.s1_ids, "F_B": F_a, "F_B_H2": F_b}).to_parquet(
        os.path.join(SCORE_DIR, "D3B_val2_perS1_F.parquet"), index=False)
    with open(os.path.join(REPORT_DIR, "val2_B_H2_evaluation.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
