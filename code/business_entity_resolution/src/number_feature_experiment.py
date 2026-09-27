"""
Controlled ablation: do corrected address-number features improve D3 + LR?

  A  existing 28 D3 features                      (stored clean-protocol scores; NOT retrained)
  B  A + number agreement + number contradiction  (28 + 14)
  C  A + number agreement only                    (28 + 7)

Frozen for every variant: D3 candidate pairs, labels, 8K FIT / 2K TUNE / 10K VAL S1
split, StandardScaler + LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
fit on FIT, global-threshold grid (tune_decision_rules.T_GRID) selected on TUNE by
per-S1 macro F0.5, multi-match prediction (every candidate with z >= t), metric code.
VAL is evaluated once per variant; the failure-mode diagnostics are computed after
all three VAL results are fixed and never feed back into any choice.
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
from score_lr_clean_split import NON_FEATURE_COLS
from tune_decision_rules import T_GRID, Cache, Part, counts_at, evaluate, paired_bootstrap

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
FEAT_DIR = os.path.join(ROOT, "data_cache", "dev_features")
SCORE_DIR = os.path.join(ROOT, "data_cache", "dev_scores")
REPORT_DIR = os.path.join(ROOT, "reports")
PART_COLS = ["name_nosuffix_exact", "address_missing_cand", "n_channels_hit", "postal_equal", "address_missing_s1"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr, flush=True)


def load_split(split):
    base = pq.read_table(os.path.join(FEAT_DIR, f"D3_{split}_features.parquet"),
                         read_dictionary=["s1_id", "cand_id", "country"]).to_pandas()
    num = pd.read_parquet(os.path.join(FEAT_DIR, f"D3_{split}_numfeat.parquet"))
    assert len(num) == len(base)
    assert (num["cand_id"].astype(str).values == base["cand_id"].astype(str).values).all(), "numfeat misaligned"
    assert (num["s1_id"].astype(str).values == base["s1_id"].astype(str).values).all()
    for c in AGREEMENT + CONTRADICTION:
        base[c] = num[c].values
    base["s1_id"] = base["s1_id"].astype(str)
    base["cand_id"] = base["cand_id"].astype(str)
    return base


def rows_for_part(df, z, mask):
    r = df.loc[mask, ["s1_id", "cand_id", "label"] + PART_COLS].copy()
    r["z"] = z[mask]
    return r


def tune_global(part):
    cache = Cache(part)
    scores = {t: cache.F[t].mean() for t in T_GRID}
    t = max(scores, key=scores.get)
    return float(t), float(scores[t])


def pct(x):
    return round(100 * float(x), 2)


def main():
    s1 = pd.read_parquet(os.path.join(SCORE_DIR, "D3_s1_table.parquet"))
    s1p = {p: s1[s1["part"] == p].reset_index(drop=True) for p in ("fit", "tune", "val")}
    tune_ids = set(s1p["tune"]["entity_id"])
    prior = json.load(open(os.path.join(REPORT_DIR, "decision_rules_D3.json")))["variants"]["A"]

    log("loading D3 train/val features + number features")
    tr = load_split("train")
    va = load_split("val")
    base_cols = [c for c in pq.read_schema(os.path.join(FEAT_DIR, "D3_train_features.parquet")).names
                 if c not in NON_FEATURE_COLS]
    feature_sets = {"A": base_cols, "B": base_cols + AGREEMENT + CONTRADICTION, "C": base_cols + AGREEMENT}
    is_fit = ~tr["s1_id"].isin(tune_ids).values
    is_tune = ~is_fit

    # --- A: stored clean-protocol scores (authoritative baseline)
    st = pd.read_parquet(os.path.join(SCORE_DIR, "D3_train_scores.parquet"), columns=["cand_id", "z", "part"])
    sv = pd.read_parquet(os.path.join(SCORE_DIR, "D3_val_scores.parquet"), columns=["cand_id", "z"])
    assert (st["cand_id"].astype(str).values == tr["cand_id"].values).all()
    assert (sv["cand_id"].astype(str).values == va["cand_id"].values).all()
    assert ((st["part"].values == "tune") == is_tune).all()
    z = {"A": (st["z"].values, sv["z"].values)}
    del st, sv

    report = {"protocol": {"fit_s1": len(s1p["fit"]), "tune_s1": len(s1p["tune"]), "val_s1": len(s1p["val"]),
                           "lr": "StandardScaler + LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)",
                           "threshold": "global, T_GRID 3.0..16.0 step 0.1, selected on TUNE by macro F0.5"},
              "feature_counts": {k: len(v) for k, v in feature_sets.items()},
              "new_features": {"agreement": AGREEMENT, "contradiction": CONTRADICTION},
              "variants": {}, "coefficients": {}, "timing": {}}

    for v in ("B", "C"):
        cols = feature_sets[v]
        t0 = time.time()
        X = tr[cols].values.astype(np.float32)
        scaler = StandardScaler()
        Xf = scaler.fit_transform(X[is_fit])
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        lr.fit(Xf, tr["label"].values[is_fit])
        fit_s = time.time() - t0
        del Xf
        zt = np.concatenate([lr.decision_function(scaler.transform(X[i:i + 2_000_000]))
                             for i in range(0, len(X), 2_000_000)])
        del X
        Xv = va[cols].values.astype(np.float32)
        zv = np.concatenate([lr.decision_function(scaler.transform(Xv[i:i + 2_000_000]))
                             for i in range(0, len(Xv), 2_000_000)])
        del Xv
        z[v] = (zt, zv)
        report["coefficients"][v] = {c: round(float(w), 4) for c, w in zip(cols, lr.coef_[0])}
        report["coefficients"][v]["_intercept"] = round(float(lr.intercept_[0]), 4)
        report["timing"][v] = {"lr_fit_seconds": round(fit_s, 1), "n_iter": int(lr.n_iter_[0])}
        log(f"{v}: LR fit {fit_s:.0f}s, n_iter={lr.n_iter_[0]}")

    # --- tune threshold on TUNE, evaluate once on VAL
    parts_val, F_val, thr = {}, {}, {}
    for v in ("A", "B", "C"):
        zt, zv = z[v]
        p_tune = Part(s1p["tune"], rows_for_part(tr, zt, is_tune))
        t, tune_f = tune_global(p_tune)
        thr[v] = t
        p_val = Part(s1p["val"], rows_for_part(va, zv, np.ones(len(va), bool)))
        res, F = evaluate(p_val, "A", {"t": t})
        rt = t
        n_pred = np.bincount(p_val.r_s[p_val.r_z >= rt], minlength=p_val.n)
        single = p_val.n_true == 0
        matched = ~single
        res["threshold_z"] = t
        res["tune_macro_f05"] = round(tune_f, 4)
        res["singleton_s1_with_prediction"] = int((single & (n_pred > 0)).sum())
        res["matched_s1_zero_pred_pct"] = pct((matched & (n_pred == 0)).sum() / matched.sum())
        report["variants"][v] = {"features": len(feature_sets[v]), "val": res}
        parts_val[v], F_val[v] = p_val, F
        log(f"{v}: t={t} tuneF={tune_f:.4f} VAL F0.5={res['macro_f05']} P={res['macro_precision']} "
            f"R={res['macro_recall']} singFP={res['singleton_fp_rate_pct']}% matched0={res['matched_s1_zero_predictions']}")

    a = report["variants"]["A"]["val"]
    assert thr["A"] == prior["params"]["t"], "stored baseline threshold not reproduced"
    for k in ("macro_f05", "macro_precision", "macro_recall", "singleton_fp_rate_pct", "matched_s1_zero_predictions"):
        assert a[k] == prior["val"][k], f"baseline {k} {a[k]} != stored {prior['val'][k]}"
    report["baseline_check"] = "A reproduces reports/decision_rules_D3.json variant A exactly"

    ctry = parts_val["A"].country
    report["bootstrap"] = {}
    for x, y in (("A", "B"), ("C", "B"), ("A", "C")):
        r = paired_bootstrap(F_val[x], F_val[y])
        for c in ("US", "India"):
            r[c] = paired_bootstrap(F_val[x][ctry == c], F_val[y][ctry == c])
        report["bootstrap"][f"{y}-{x}"] = r
        log(f"{y}-{x}: {r['delta']} CI {r['ci95']}")

    # ================= post-freeze diagnostics (VAL results above are final) =================
    # (1) forensic subset: training-side true singletons whose D3/A top candidate had z >= 7
    cases = pd.read_csv(os.path.join(REPORT_DIR, "singleton_forensics_cases.tsv"), sep="\t", dtype=str,
                        keep_default_na=False)
    fz = {}
    for v in ("A", "B", "C"):
        zt = z[v][0]
        d = pd.DataFrame({"s1_id": tr["s1_id"].values, "z": zt, "label": tr["label"].values})
        fz[v] = d.groupby("s1_id")["z"].max()
    fm = {}
    for part_name in ("tune", "fit"):
        sub = cases[cases["part"] == part_name]
        ent = {}
        for v in ("A", "B", "C"):
            pred = fz[v].reindex(sub["s1_id"]).values >= thr[v]
            ent[v] = pred
        by_cat = {}
        for cat, g in sub.groupby("category"):
            m = sub.index.get_indexer(g.index)
            by_cat[cat] = {v: int(ent[v][m].sum()) for v in ent}
            by_cat[cat]["n"] = len(g)
        fm[part_name] = {"n_cases": len(sub),
                         "still_predicted_nonempty": {v: int(ent[v].sum()) for v in ent},
                         "fp_removed_vs_A": {v: int((ent["A"] & ~ent[v]).sum()) for v in ("B", "C")},
                         "fp_added_vs_A": {v: int((~ent["A"] & ent[v]).sum()) for v in ("B", "C")},
                         "by_category": by_cat}
    fm["note"] = ("TUNE cases are out-of-sample for all LRs; FIT cases are in-sample (the LR saw their "
                  "labels), so FIT numbers are optimistic. Singletons have no true pairs, so TP loss is "
                  "measured on VAL below.")
    report["forensic_subset"] = fm

    # (2) VAL pair-level change A -> B, split by whether the pair has a number contradiction
    contra = ((va["num_hn_head_mismatch"] == 1) | (va["num_compound_mismatch"] == 1)
              | (va["num_token_contradiction"] == 1)).values
    lab = va["label"].values.astype(bool)
    pa_, pb_, pc_ = (z["A"][1] >= thr["A"]), (z["B"][1] >= thr["B"]), (z["C"][1] >= thr["C"])
    diag = {}
    for name, m in (("number_contradiction", contra), ("no_contradiction", ~contra), ("all", np.ones_like(contra))):
        diag[name] = {
            "FP_pairs": {"A": int((pa_ & ~lab & m).sum()), "B": int((pb_ & ~lab & m).sum()), "C": int((pc_ & ~lab & m).sum())},
            "TP_pairs": {"A": int((pa_ & lab & m).sum()), "B": int((pb_ & lab & m).sum()), "C": int((pc_ & lab & m).sum())},
            "B_vs_A_FP_removed": int((pa_ & ~pb_ & ~lab & m).sum()), "B_vs_A_FP_added": int((~pa_ & pb_ & ~lab & m).sum()),
            "B_vs_A_TP_lost": int((pa_ & ~pb_ & lab & m).sum()), "B_vs_A_TP_gained": int((~pa_ & pb_ & lab & m).sum()),
        }
    # singleton S1 on VAL whose A-top false match had a number contradiction
    top = pd.DataFrame({"s1": va["s1_id"].values, "zA": z["A"][1], "zB": z["B"][1], "contra": contra})
    idx = top.groupby("s1")["zA"].idxmax()
    topA = top.loc[idx].set_index("s1")
    single_ids = set(s1p["val"].loc[s1p["val"]["n_true"] == 0, "entity_id"])
    tA = topA[topA.index.isin(single_ids) & (topA["zA"] >= thr["A"])]
    maxB = top.groupby("s1")["zB"].max()
    diag["val_singletons_predicted_by_A"] = {
        "n": len(tA), "top_pair_has_number_contradiction": int(tA["contra"].sum()),
        "cleared_by_B_among_contradiction": int(((maxB.reindex(tA.index) < thr["B"]) & tA["contra"]).sum()),
        "cleared_by_B_among_no_contradiction": int(((maxB.reindex(tA.index) < thr["B"]) & ~tA["contra"]).sum())}
    report["val_failure_mode_diagnostics"] = diag
    log(f"diagnostics: {json.dumps(diag)}")

    pd.DataFrame({"s1_id": parts_val["A"].s1_ids, **{f"F_{v}": F_val[v] for v in F_val}}).to_parquet(
        os.path.join(SCORE_DIR, "D3_numfeat_val_perS1_F.parquet"), index=False)
    for v in ("B", "C"):
        pd.DataFrame({"z_train": z[v][0]}).to_parquet(os.path.join(SCORE_DIR, f"D3{v}_num_train_z.parquet"), index=False)
        pd.DataFrame({"z_val": z[v][1]}).to_parquet(os.path.join(SCORE_DIR, f"D3{v}_num_val_z.parquet"), index=False)
    report["peak_rss_mb"] = round(rss_mb())
    with open(os.path.join(REPORT_DIR, "number_features_experiment.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log("wrote reports/number_features_experiment.json")


if __name__ == "__main__":
    main()
