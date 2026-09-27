"""
Train LogisticRegression + LightGBM on the D3/C development candidate-pair
features, then evaluate at the ACTUAL competition metric: entity-level F0.5
per S1, macro-averaged -- never pooled/pairwise F0.5.

Supports multiple matches per S1 by construction: the decision is an
independent per-candidate threshold, not a top-1 argmax, so an S1 can end up
with 0, 1, or many predicted matches.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

HERE = os.path.dirname(__file__)
FEAT_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "dev_features")
REPORT_DIR = os.path.join(HERE, "..", "..", "..", "reports")
CACHE_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "normalized")
SPLIT_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "splits")

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99,
              0.995, 0.998, 0.999, 0.9995, 0.9999]

NON_FEATURE_COLS = {"s1_id", "cand_id", "label", "country"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}]  {msg}", file=sys.stderr)


def entity_f05(true_set, pred_set):
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 1.0, 0.0, 0.0
    if not true_set:
        return 0.0, 1.0, 0.0
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall / denom) if denom > 0 else 0.0
    return precision, recall, f05


def load_features(config, split):
    path = os.path.join(FEAT_DIR, f"{config}_{split}_features.parquet")
    return pd.read_parquet(path)


def reconstruct_val_sample_ids(n_val=10000, seed=42):
    """
    Exactly replicates generate_dev_candidates.py's sample_split() for the val
    split (deterministic: same source parquet, same id set, same seed/choice)
    so that S1 entities with ZERO candidates -- entirely absent from the pairs
    parquet -- are still included in entity-level evaluation. Without this,
    such an S1 (correctly predicted empty if singleton, or a guaranteed-miss
    if matched) would silently vanish from the macro F0.5 denominator.
    """
    val_ids_all = set(pd.read_csv(os.path.join(SPLIT_DIR, "val_s1_ids.txt"), header=None)[0].astype(str))
    s1_full = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"), columns=["entity_id", "country"])
    s1 = s1_full[s1_full["entity_id"].isin(val_ids_all)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s1), size=min(n_val, len(s1)), replace=False)
    return s1.iloc[idx].reset_index(drop=True)


def get_feature_matrix(df):
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df[cols].values.astype(np.float32), cols


def entity_level_eval(val_df, probs, threshold, all_s1_ids, gt_true_sets):
    """all_s1_ids: every S1 in the val sample (including those with zero candidates
    at all, who never appear in val_df but must still count as a row -- predicted empty)."""
    pred_by_s1 = {}
    mask = probs >= threshold
    sub = val_df.loc[mask, ["s1_id", "cand_id"]]
    for s1, cand in zip(sub["s1_id"].values, sub["cand_id"].values):
        pred_by_s1.setdefault(s1, set()).add(cand)

    per_entity = []
    cand_claim_counter = Counter()
    for s1 in all_s1_ids:
        true_set = gt_true_sets.get(s1, frozenset())
        pred_set = pred_by_s1.get(s1, set())
        p, r, f = entity_f05(true_set, pred_set)
        per_entity.append((s1, len(true_set), len(pred_set), p, r, f))
        for c in pred_set:
            cand_claim_counter[c] += 1

    per_entity_df = pd.DataFrame(per_entity, columns=["s1_id", "n_true", "n_pred", "precision", "recall", "f05"])
    n_dup_claimed = sum(1 for c, n in cand_claim_counter.items() if n > 1)
    n_distinct_predicted = len(cand_claim_counter)
    return per_entity_df, n_dup_claimed, n_distinct_predicted


def summarize_threshold(per_entity_df, s1_country, n_dup_claimed, n_distinct_predicted):
    macro_f05 = per_entity_df["f05"].mean()
    macro_precision = per_entity_df["precision"].mean()
    macro_recall = per_entity_df["recall"].mean()

    singleton = per_entity_df[per_entity_df["n_true"] == 0]
    matched = per_entity_df[per_entity_df["n_true"] > 0]

    n_singleton_pred_empty = int((singleton["n_pred"] == 0).sum())
    n_singleton_total = len(singleton)
    singleton_fp_rate = round(100 * (singleton["n_pred"] > 0).mean(), 3) if n_singleton_total else None

    n_matched_zero_pred = int((matched["n_pred"] == 0).sum())

    result = {
        "macro_f05": round(float(macro_f05), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "n_singleton_total": n_singleton_total,
        "n_singleton_predicted_empty": n_singleton_pred_empty,
        "singleton_false_positive_rate_pct": singleton_fp_rate,
        "n_matched_s1_zero_predicted_despite_gt": n_matched_zero_pred,
        "avg_predicted_matches_per_s1": round(float(per_entity_df["n_pred"].mean()), 3),
        "n_distinct_predicted_candidate_ids": n_distinct_predicted,
        "n_candidate_ids_claimed_by_multiple_s1": n_dup_claimed,
        "duplicate_claim_rate_pct": round(100 * n_dup_claimed / n_distinct_predicted, 3) if n_distinct_predicted else None,
    }

    # country breakdown (matched S1 only, since recall is meaningful there)
    country_map = s1_country
    matched_c = matched.copy()
    matched_c["country"] = matched_c["s1_id"].map(country_map)
    country_result = {}
    for c, g in matched_c.groupby("country"):
        country_result[c] = {
            "n_s1": int(len(g)),
            "macro_f05": round(float(g["f05"].mean()), 4),
            "macro_recall": round(float(g["recall"].mean()), 4),
            "macro_precision": round(float(g["precision"].mean()), 4),
        }
    result["by_country"] = country_result
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["D3", "C"], required=True)
    args = ap.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    log(f"Loading features for config {args.config}...")
    train_df = load_features(args.config, "train")
    val_df = load_features(args.config, "val")
    log(f"train={len(train_df)} rows ({train_df['label'].mean()*100:.4f}% positive), "
        f"val={len(val_df)} rows ({val_df['label'].mean()*100:.4f}% positive)")

    X_train, feat_cols = get_feature_matrix(train_df)
    y_train = train_df["label"].values
    X_val, _ = get_feature_matrix(val_df)
    y_val = val_df["label"].values
    log(f"Feature columns ({len(feat_cols)}): {feat_cols}")

    # feature distributions: positive vs negative (train)
    dist = {}
    for i, c in enumerate(feat_cols):
        pos_vals = X_train[y_train == 1, i]
        neg_vals = X_train[y_train == 0, i]
        dist[c] = {
            "pos_mean": round(float(pos_vals.mean()), 4),
            "neg_mean": round(float(neg_vals.mean()), 4),
            "pos_median": round(float(np.median(pos_vals)), 4),
            "neg_median": round(float(np.median(neg_vals)), 4),
        }
    with open(os.path.join(REPORT_DIR, f"feature_distributions_{args.config}.json"), "w") as f:
        json.dump(dist, f, indent=2)
    log("Wrote feature distribution report (positive vs negative, train split)")

    # ground truth per S1 for entity-level eval, and the FULL original val sample
    # (including any S1 with zero candidates, absent from val_df entirely)
    log("Reconstructing full val sample (including zero-candidate S1s)...")
    val_sample_full = reconstruct_val_sample_ids(n_val=10000, seed=42)
    all_val_s1 = val_sample_full["entity_id"].tolist()
    s1_country = dict(zip(val_sample_full["entity_id"], val_sample_full["country"]))
    n_missing_from_pairs = len(set(all_val_s1) - set(val_df["s1_id"].unique()))
    log(f"Full val sample: {len(all_val_s1)} S1 ({n_missing_from_pairs} have zero candidates, "
        f"absent from the pairs file, now included via reconstruction)")

    # True positive labels must come from real ground truth for EVERY S1 in the
    # full sample, not just those derived from val_df's positive rows -- an S1
    # with zero candidates at all (blocking total miss) would otherwise be
    # indistinguishable from a genuine singleton, when in fact it's a
    # guaranteed-zero-recall entity that must count against macro F0.5.
    gt_path = os.path.join(HERE, "..", "..", "..", "student_resource", "dataset", "train", "train_ground_truth.tsv")
    gt_raw = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    gt_raw = gt_raw[gt_raw["source1_entity_id"].isin(set(all_val_s1))]
    gt_true_sets = {}
    for row in gt_raw.itertuples():
        if isinstance(row.matched_entity_ids, str) and row.matched_entity_ids:
            gt_true_sets[row.source1_entity_id] = frozenset(row.matched_entity_ids.split(","))
    n_true_singleton = sum(1 for s1 in all_val_s1 if s1 not in gt_true_sets)
    log(f"Ground truth resolved for full sample: {len(gt_true_sets)} matched S1, "
        f"{n_true_singleton} true singleton S1 (out of {len(all_val_s1)})")

    models = {}

    log("Training Logistic Regression (class_weight=balanced, standardized features)...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    lr.fit(X_train_s, y_train)
    lr_train_time = time.time() - t0
    t0 = time.time()
    lr_probs = lr.predict_proba(X_val_s)[:, 1]
    lr_inference_time = time.time() - t0
    log(f"LR trained in {lr_train_time:.1f}s, inference in {lr_inference_time:.2f}s")
    models["LogisticRegression"] = {"probs": lr_probs, "train_time": lr_train_time, "inference_time": lr_inference_time,
                                     "coefficients": dict(zip(feat_cols, lr.coef_[0].round(4).tolist()))}

    log("Training LightGBM (is_unbalance=True)...")
    t0 = time.time()
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
    params = {"objective": "binary", "metric": "binary_logloss", "is_unbalance": True,
              "verbosity": -1, "num_leaves": 31, "learning_rate": 0.05, "seed": 42}
    gbm = lgb.train(params, lgb_train, num_boost_round=200)
    lgb_train_time = time.time() - t0
    t0 = time.time()
    lgb_probs = gbm.predict(X_val)
    lgb_inference_time = time.time() - t0
    log(f"LightGBM trained in {lgb_train_time:.1f}s, inference in {lgb_inference_time:.2f}s")
    importance = dict(zip(feat_cols, gbm.feature_importance(importance_type="gain").round(1).tolist()))
    models["LightGBM"] = {"probs": lgb_probs, "train_time": lgb_train_time, "inference_time": lgb_inference_time,
                           "feature_importance_gain": importance}

    report = {"config": args.config, "n_train_pairs": len(train_df), "n_val_pairs": len(val_df),
              "train_positive_rate": float(train_df["label"].mean()),
              "val_positive_rate": float(val_df["label"].mean()),
              "n_val_s1": len(all_val_s1), "models": {}}

    for model_name, m in models.items():
        log(f"Threshold sweep for {model_name}...")
        sweep = []
        for th in THRESHOLDS:
            per_entity_df, n_dup, n_distinct = entity_level_eval(val_df, m["probs"], th, all_val_s1, gt_true_sets)
            summ = summarize_threshold(per_entity_df, s1_country, n_dup, n_distinct)
            summ["threshold"] = th
            sweep.append(summ)
            log(f"  {model_name} th={th}: macro_f05={summ['macro_f05']} "
                f"macro_precision={summ['macro_precision']} macro_recall={summ['macro_recall']} "
                f"avg_pred/S1={summ['avg_predicted_matches_per_s1']}")
        best = max(sweep, key=lambda r: r["macro_f05"])
        report["models"][model_name] = {
            "train_time_sec": round(m["train_time"], 2),
            "inference_time_sec": round(m["inference_time"], 3),
            "threshold_sweep": sweep,
            "best_threshold": best["threshold"],
            "best_result": best,
        }
        if "coefficients" in m:
            report["models"][model_name]["coefficients"] = m["coefficients"]
        if "feature_importance_gain" in m:
            report["models"][model_name]["feature_importance_gain"] = m["feature_importance_gain"]

    out_path = os.path.join(REPORT_DIR, f"matcher_results_{args.config}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {out_path}")
    log("Matcher training + evaluation complete.")


if __name__ == "__main__":
    main()
