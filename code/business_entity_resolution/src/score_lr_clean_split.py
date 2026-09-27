"""
Clean-split LR scoring for the decision-rule experiment.

Fixes the optimistic evaluation in train_evaluate_matcher.py (threshold was
chosen on the same 10K val sample it reported). Here:

  training-side 10K S1 sample  --(S1-level 80/20, seed=42)-->  8K FIT / 2K TUNE
  val-side 10K S1 sample (val_s1_ids)                       -->  untouched, final eval only

LR (same hyperparameters as train_evaluate_matcher.py) is fit on the FIT pairs
only. Logit scores (decision_function; avoids float saturation of p near 1) are
written for FIT / TUNE / VAL rows together with a few observable pair features
used by the decision-rule variants. Ground truth is written separately as a
per-S1 table (n_true), used only for tuning/evaluation, never as a rule input.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
FEAT_DIR = os.path.join(ROOT, "data_cache", "dev_features")
CACHE_DIR = os.path.join(ROOT, "data_cache", "normalized")
SPLIT_DIR = os.path.join(ROOT, "data_cache", "splits")
OUT_DIR = os.path.join(ROOT, "data_cache", "dev_scores")
GT_PATH = os.path.join(ROOT, "student_resource", "dataset", "train", "train_ground_truth.tsv")

NON_FEATURE_COLS = {"s1_id", "cand_id", "label", "country"}
# observable pair features carried along for inference-time decision rules
KEEP_OBS = ["name_exact", "name_nosuffix_exact", "address_missing_s1", "address_missing_cand",
            "n_channels_hit", "name_fuzz_token_set_ratio", "address_fuzz_ratio", "postal_equal"]
SEED = 42
TUNE_FRACTION = 0.2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}]  {msg}", file=sys.stderr, flush=True)


def reconstruct_sample(ids_file, n=10000, seed=SEED):
    """Replicates generate_dev_candidates.py sample_split() exactly, so S1 with
    zero candidates (absent from the pairs file) are still part of the sample."""
    ids_all = set(pd.read_csv(os.path.join(SPLIT_DIR, ids_file), header=None)[0].astype(str))
    s1_full = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"), columns=["entity_id", "country"])
    s1 = s1_full[s1_full["entity_id"].isin(ids_all)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s1), size=min(n, len(s1)), replace=False)
    return s1.iloc[idx].reset_index(drop=True)


def read_features(path):
    t = pq.read_table(path, read_dictionary=["s1_id", "cand_id", "country"])
    df = t.to_pandas()
    del t
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["D3", "C"], required=True)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    log("Reconstructing training-side and val-side 10K S1 samples...")
    tr_sample = reconstruct_sample("train_s1_ids.txt")
    va_sample = reconstruct_sample("val_s1_ids.txt")

    # S1-level 80/20 split of the training-side sample
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr_sample))
    n_tune = int(round(TUNE_FRACTION * len(tr_sample)))
    tune_ids = set(tr_sample["entity_id"].iloc[perm[:n_tune]])
    tr_sample["part"] = np.where(tr_sample["entity_id"].isin(tune_ids), "tune", "fit")
    va_sample["part"] = "val"
    s1_table = pd.concat([tr_sample, va_sample], ignore_index=True)
    assert s1_table["entity_id"].is_unique, "train/val S1 samples overlap"

    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    gt = gt[gt["source1_entity_id"].isin(set(s1_table["entity_id"]))]
    n_true = {r.source1_entity_id: len(r.matched_entity_ids.split(","))
              for r in gt.itertuples() if isinstance(r.matched_entity_ids, str) and r.matched_entity_ids}
    s1_table["n_true"] = s1_table["entity_id"].map(n_true).fillna(0).astype(int)
    s1_table.to_parquet(os.path.join(OUT_DIR, f"{args.config}_s1_table.parquet"), index=False)
    for p, g in s1_table.groupby("part"):
        log(f"  part={p}: {len(g)} S1, {int((g['n_true'] == 0).sum())} singletons")

    log(f"Loading {args.config} training-side features...")
    tr = read_features(os.path.join(FEAT_DIR, f"{args.config}_train_features.parquet"))
    feat_cols = [c for c in tr.columns if c not in NON_FEATURE_COLS]
    is_fit = ~tr["s1_id"].astype(str).isin(tune_ids).values
    log(f"  {len(tr)} rows: fit={int(is_fit.sum())}, tune={int((~is_fit).sum())}")

    X = tr[feat_cols].values.astype(np.float32)
    y = tr["label"].values
    t0 = time.time()
    scaler = StandardScaler()
    Xf = scaler.fit_transform(X[is_fit])
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    lr.fit(Xf, y[is_fit])
    del Xf
    log(f"LR fit on FIT part in {time.time() - t0:.1f}s")

    def score_frame(df, Xm, part_col):
        z = np.empty(len(df), dtype=np.float64)
        step = 2_000_000
        for i in range(0, len(df), step):
            z[i:i + step] = lr.decision_function(scaler.transform(Xm[i:i + step]))
        out = df[["s1_id", "cand_id", "label", "country"] + KEEP_OBS].copy()
        out["z"] = z
        out["part"] = part_col
        return out

    tr_out = score_frame(tr, X, np.where(is_fit, "fit", "tune"))
    del tr, X
    tr_out.to_parquet(os.path.join(OUT_DIR, f"{args.config}_train_scores.parquet"), index=False)
    del tr_out

    log(f"Loading {args.config} val features...")
    va = read_features(os.path.join(FEAT_DIR, f"{args.config}_val_features.parquet"))
    Xv = va[feat_cols].values.astype(np.float32)
    va_out = score_frame(va, Xv, "val")
    del va, Xv
    va_out.to_parquet(os.path.join(OUT_DIR, f"{args.config}_val_scores.parquet"), index=False)
    pd.Series(dict(zip(feat_cols, lr.coef_[0]))).to_json(os.path.join(OUT_DIR, f"{args.config}_lr_coef.json"))
    log("Done.")


if __name__ == "__main__":
    main()
