"""
Investigate unexpected coefficient signs in the number-aware LR (variant B).

Uses FIT rows only (no TUNE/VAL). For each variant, the "number block" is the sum of
standardized contributions coef_j * (x_j - mu_j) / sd_j over all number-related
features (new num_* plus the old street_number_equal, address_numeric_token_overlap,
postal_equal). The block's NET effect per pair profile shows whether the model, taken
as a whole, treats number agreement as positive and contradiction as negative evidence
even when individual collinear coefficients have counter-intuitive signs.
Also reports univariate label rates per new binary feature on FIT.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_number_features import AGREEMENT, CONTRADICTION

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
FEAT_DIR = os.path.join(ROOT, "data_cache", "dev_features")
SCORE_DIR = os.path.join(ROOT, "data_cache", "dev_scores")
REPORT_DIR = os.path.join(ROOT, "reports")
OLD_NUM = ["street_number_equal", "address_numeric_token_overlap", "postal_equal"]


def main():
    rep = json.load(open(os.path.join(REPORT_DIR, "number_features_experiment.json")))
    s1 = pd.read_parquet(os.path.join(SCORE_DIR, "D3_s1_table.parquet"))
    fit_ids = set(s1.loc[s1["part"] == "fit", "entity_id"])
    base = pq.read_table(os.path.join(FEAT_DIR, "D3_train_features.parquet"),
                         columns=["s1_id", "label", "name_fuzz_token_set_ratio"] + OLD_NUM).to_pandas()
    num = pd.read_parquet(os.path.join(FEAT_DIR, "D3_train_numfeat.parquet"), columns=AGREEMENT + CONTRADICTION)
    df = pd.concat([base, num], axis=1)
    df = df[df["s1_id"].astype(str).isin(fit_ids)].drop(columns="s1_id")
    out = {"univariate_fit": {}, "net_number_block": {}}

    for c in AGREEMENT + CONTRADICTION:
        if df[c].max() <= 1 and df[c].dtype != np.float32:
            g = df.groupby(df[c] > 0)["label"].agg(["mean", "size"])
            out["univariate_fit"][c] = {"pos_rate_if_1": round(float(g.loc[True, "mean"]) * 100, 3) if True in g.index else None,
                                        "pos_rate_if_0": round(float(g.loc[False, "mean"]) * 100, 3),
                                        "n_if_1": int(g.loc[True, "size"]) if True in g.index else 0}

    strong = df["name_fuzz_token_set_ratio"] >= 0.9
    contra = (df["num_hn_head_mismatch"] == 1) | (df["num_compound_mismatch"] == 1) | (df["num_token_contradiction"] == 1)
    profiles = {
        "strong_name & numbers_agree (head equal, no contradiction)": strong & (df["num_hn_head_equal"] == 1) & ~contra,
        "strong_name & number_contradiction": strong & contra,
        "strong_name & no numbers on one side": strong & (df["num_token_jaccard"] == 0) & (df["num_token_contradiction"] == 0),
        "weak_name & numbers_agree": ~strong & (df["num_hn_head_equal"] == 1) & ~contra,
        "weak_name & number_contradiction": ~strong & contra,
    }
    for v in ("A", "B", "C"):
        coefs = rep["coefficients"].get(v)
        if coefs is None:   # A: stored model
            coefs = json.load(open(os.path.join(SCORE_DIR, "D3_lr_coef.json")))
        cols = [c for c in OLD_NUM + AGREEMENT + CONTRADICTION if c in coefs]
        X = df[cols].astype(np.float64)
        mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
        block = ((X - mu) / sd * pd.Series({c: coefs[c] for c in cols})).sum(axis=1)
        res = {}
        for name, m in profiles.items():
            for lab in (1, 0):
                mm = m & (df["label"] == lab)
                if mm.sum():
                    res[f"{name} | label={lab}"] = {"n": int(mm.sum()), "mean_block_logit": round(float(block[mm].mean()), 3)}
        out["net_number_block"][v] = res
    # the flagged coefficients in context: same pair with head_mismatch 0->1 while the (collinear) token
    # contradiction also flips, as it does in practice
    b = rep["coefficients"]["B"]
    X = df[["num_hn_head_mismatch", "num_token_contradiction", "num_strongname_contra", "num_hn_head_equal"]]
    sd = X.std(ddof=0)
    out["joint_flip_strongname_equal_to_contradicted_B"] = round(float(
        b["num_hn_head_mismatch"] / sd["num_hn_head_mismatch"] + b["num_token_contradiction"] / sd["num_token_contradiction"]
        + b["num_strongname_contra"] / sd["num_strongname_contra"] - b["num_hn_head_equal"] / sd["num_hn_head_equal"]), 3)
    out["co_occurrence_fit"] = {
        "P(token_contradiction | head_mismatch)": round(float(df.loc[df["num_hn_head_mismatch"] == 1, "num_token_contradiction"].mean()), 4),
        "P(head_mismatch | strongname_contra)": round(float(df.loc[df["num_strongname_contra"] == 1, "num_hn_head_mismatch"].mean()), 4),
        "corr(street_number_equal, num_hn_head_equal)": round(float(df[["street_number_equal", "num_hn_head_equal"]].corr().iloc[0, 1]), 4),
        "corr(address_numeric_token_overlap, num_token_overlap)": round(float(df[["address_numeric_token_overlap", "num_token_overlap"]].corr().iloc[0, 1]), 4),
    }
    with open(os.path.join(REPORT_DIR, "number_features_coef_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
