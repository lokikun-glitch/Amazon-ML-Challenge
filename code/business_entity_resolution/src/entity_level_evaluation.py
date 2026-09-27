"""
Stage 6, entity-level evaluation of D1 / D3 / C.

The competition metric is macro F0.5 computed PER S1 entity, then averaged --
not pooled edge recall. A config can have good edge recall while still
leaving many individual S1 entities with at least one missed match (which
caps that entity's achievable recall term in F0.5) or, worse, leaving some
entities with NO surviving true match at all (recall=0 for that entity
regardless of the matcher). This script measures exactly that distinction
for the three configs still on the table after the asymmetric grid: D1, D3
(which dominates B), and C.

No candidate_pairs.tsv, no features, no model -- per the explicit hard stop.
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from blocking import ExactIndex, TokenIndex, build_ngram_series, char_ngram_string, rss_mb

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "normalized")
SPLIT_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "splits")
GT_PATH = os.path.join(HERE, "..", "..", "..", "student_resource", "dataset", "train", "train_ground_truth.tsv")
REPORT_DIR = os.path.join(HERE, "..", "..", "..", "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "entity_level_evaluation.json")

SAMPLE_SEED = 42
SAMPLE_N = 50000

CONFIGS = {
    "D1_lean_name_mid_addr": {"b3_maxdf": 500, "b5_maxdf": 1000, "b3_k": 4, "b5_k": 6,
                                "b7b_maxdf": None, "b8_4g_maxdf": None},
    "D3_lean_name_high_addr": {"b3_maxdf": 500, "b5_maxdf": 2000, "b3_k": 4, "b5_k": 6,
                                 "b7b_maxdf": None, "b8_4g_maxdf": None},
    "C_wider": {"b3_maxdf": 3000, "b5_maxdf": 3000, "b3_k": 4, "b5_k": 6,
                "b7b_maxdf": 5000, "b8_4g_maxdf": 500},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr)


def load_pool_column(columns):
    frames = []
    for label in ("train_source2", "train_source3"):
        p = os.path.join(CACHE_DIR, f"{label}.parquet")
        frames.append(pd.read_parquet(p, columns=columns))
    return pd.concat(frames, ignore_index=True)


def build_indexes(cfg):
    idx = {}
    col = load_pool_column(["country", "name_norm"])
    idx["b1"] = ExactIndex("B1").fit(col["country"], col["name_norm"])
    del col; gc.collect()
    col = load_pool_column(["country", "name_no_suffix"])
    idx["b2"] = ExactIndex("B2").fit(col["country"], col["name_no_suffix"])
    del col; gc.collect()
    col = load_pool_column(["country", "name_norm"])
    idx["b3"] = TokenIndex("B3").fit(col["country"], col["name_norm"], min_df=1, max_df=cfg["b3_maxdf"])
    idx["b3"].release_raw()
    del col; gc.collect()
    col = load_pool_column(["country", "address_norm"])
    idx["b5"] = TokenIndex("B5").fit(col["country"], col["address_norm"], min_df=1, max_df=cfg["b5_maxdf"])
    idx["b5"].release_raw()
    del col; gc.collect()
    col = load_pool_column(["country", "postal_code"])
    idx["b7a"] = ExactIndex("B7a").fit(col["country"], col["postal_code"])
    del col; gc.collect()
    if cfg["b7b_maxdf"] is not None:
        col = load_pool_column(["country", "street_number"])
        idx["b7b"] = ExactIndex("B7b").fit(col["country"], col["street_number"], max_df=cfg["b7b_maxdf"])
        del col; gc.collect()
    if cfg["b8_4g_maxdf"] is not None:
        col = load_pool_column(["country", "name_norm"])
        ngram_col = build_ngram_series(col["name_norm"], n=4)
        idx["b8"] = TokenIndex("B8").fit(col["country"], ngram_col, min_df=1, max_df=cfg["b8_4g_maxdf"])
        idx["b8"].release_raw()
        del col, ngram_col; gc.collect()
    return idx


def make_cands_fn(idx, cfg):
    has_b7b = "b7b" in idx
    has_b8 = "b8" in idx

    def cands_fn(row):
        c = (set(idx["b1"].query(row.country, row.name_norm).tolist())
             | set(idx["b2"].query(row.country, row.name_no_suffix).tolist())
             | set(idx["b3"].query(row.country, row.name_norm, max_query_tokens=cfg["b3_k"]).tolist())
             | set(idx["b5"].query(row.country, row.address_norm, max_query_tokens=cfg["b5_k"]).tolist())
             | set(idx["b7a"].query(row.country, row.postal_code).tolist()))
        if has_b7b:
            c |= set(idx["b7b"].query(row.country, row.street_number).tolist())
        if has_b8:
            qs = char_ngram_string(row.name_norm, 4) if row.name_norm else ""
            c |= set(idx["b8"].query(row.country, qs, max_query_tokens=12).tolist())
        return c

    return cands_fn


def entity_level_stats(cands_fn, s1_sample, gt_positions):
    rows = []
    for row in s1_sample.itertuples():
        true_pos = gt_positions.get(row.entity_id, frozenset())
        cands = cands_fn(row)
        n_true = len(true_pos)
        n_retained = len(true_pos & cands)
        rows.append({
            "country": row.country,
            "n_true": n_true,
            "n_retained": n_retained,
            "n_missed": n_true - n_retained,
            "n_candidates": len(cands),
            "is_singleton": n_true == 0,
        })
    return pd.DataFrame(rows)


def summarize(df):
    matched = df[df["n_true"] > 0].copy()
    singleton = df[df["n_true"] == 0].copy()
    affected = matched[matched["n_missed"] > 0]
    clean = matched[matched["n_missed"] == 0]

    def bucket(n):
        if n == 1:
            return "1"
        if n <= 3:
            return "2-3"
        if n <= 5:
            return "4-5"
        return "6+"

    matched["bucket"] = matched["n_true"].map(bucket)
    bucket_recall = {}
    for b, g in matched.groupby("bucket"):
        bucket_recall[b] = {
            "n_s1": int(len(g)),
            "edge_recall": round(g["n_retained"].sum() / g["n_true"].sum(), 4),
            "pct_all_retained": round(100 * (g["n_missed"] == 0).mean(), 2),
        }

    country_recall = {}
    for c, g in matched.groupby("country"):
        country_recall[c] = {
            "n_s1": int(len(g)),
            "edge_recall": round(g["n_retained"].sum() / g["n_true"].sum(), 4),
            "pct_all_retained": round(100 * (g["n_missed"] == 0).mean(), 2),
        }

    result = {
        "n_s1_total": int(len(df)),
        "n_s1_singleton": int(len(singleton)),
        "n_s1_matched": int(len(matched)),
        "overall_edge_recall": round(matched["n_retained"].sum() / matched["n_true"].sum(), 4),
        "pct_s1_all_matches_retained": round(100 * (matched["n_missed"] == 0).mean(), 3),
        "n_s1_all_matches_retained": int((matched["n_missed"] == 0).sum()),
        "pct_s1_at_least_one_retained": round(100 * (matched["n_retained"] > 0).mean(), 3),
        "n_s1_at_least_one_retained": int((matched["n_retained"] > 0).sum()),
        "n_s1_zero_retained": int((matched["n_retained"] == 0).sum()),
        "n_positive_edges_total": int(matched["n_true"].sum()),
        "n_positive_edges_missed": int(matched["n_missed"].sum()),
        "n_s1_affected_by_a_miss": int(len(affected)),
        "missed_per_affected_s1": {
            "mean": round(float(affected["n_missed"].mean()), 3) if len(affected) else None,
            "median": float(affected["n_missed"].median()) if len(affected) else None,
            "p95": float(affected["n_missed"].quantile(0.95)) if len(affected) else None,
        },
        "recall_by_match_count_bucket": bucket_recall,
        "recall_by_country": country_recall,
        "candidate_count_for_s1_with_missed_matches": {
            "mean": round(float(affected["n_candidates"].mean()), 1) if len(affected) else None,
            "median": float(affected["n_candidates"].median()) if len(affected) else None,
            "p95": float(affected["n_candidates"].quantile(0.95)) if len(affected) else None,
        },
        "candidate_count_for_s1_no_missed_matches": {
            "mean": round(float(clean["n_candidates"].mean()), 1) if len(clean) else None,
            "median": float(clean["n_candidates"].median()) if len(clean) else None,
            "p95": float(clean["n_candidates"].quantile(0.95)) if len(clean) else None,
        },
        "singleton_candidate_count": {
            "mean": round(float(singleton["n_candidates"].mean()), 2) if len(singleton) else None,
            "median": float(singleton["n_candidates"].median()) if len(singleton) else None,
            "p95": float(singleton["n_candidates"].quantile(0.95)) if len(singleton) else None,
            "pct_with_zero_candidates": round(100 * (singleton["n_candidates"] == 0).mean(), 2) if len(singleton) else None,
        },
    }
    return result


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    log("Prep (seed=42, same sample as all prior Stage 6 runs)")
    pool_base = load_pool_column(["entity_id", "country"])
    n_pool = len(pool_base)
    id_to_pos = pd.Series(np.arange(n_pool, dtype=np.int64), index=pool_base["entity_id"].values)
    del pool_base
    gc.collect()

    val_ids = set(pd.read_csv(os.path.join(SPLIT_DIR, "val_s1_ids.txt"), header=None)[0].astype(str))
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    gt = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    gt["match_list"] = gt["matched_entity_ids"].map(lambda x: x.split(",") if isinstance(x, str) and x else [])
    gt_positions = {}
    for row in gt.itertuples():
        if not row.match_list:
            continue
        positions = [int(id_to_pos[mid]) for mid in row.match_list if mid in id_to_pos.index]
        if positions:
            gt_positions[row.source1_entity_id] = frozenset(positions)
    del id_to_pos
    gc.collect()

    s1_full = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"))
    s1_val = s1_full[s1_full["entity_id"].isin(val_ids)].reset_index(drop=True)
    del s1_full
    gc.collect()
    rng = np.random.default_rng(SAMPLE_SEED)
    sample_idx = rng.choice(len(s1_val), size=min(SAMPLE_N, len(s1_val)), replace=False)
    s1_sample = s1_val.iloc[sample_idx].reset_index(drop=True)
    del s1_val
    gc.collect()
    log(f"Prep done. sample={len(s1_sample)} rss={rss_mb():.0f}MB")

    report = {}
    for name, cfg in CONFIGS.items():
        log(f"Building indexes for {name}: {cfg}")
        idx = build_indexes(cfg)
        log(f"  built. rss={rss_mb():.0f}MB")
        cands_fn = make_cands_fn(idx, cfg)
        t0 = time.time()
        df = entity_level_stats(cands_fn, s1_sample, gt_positions)
        log(f"  entity-level stats computed in {time.time()-t0:.1f}s")
        summary = summarize(df)
        report[name] = summary
        log(f"  {name}: edge_recall={summary['overall_edge_recall']} "
            f"pct_all_retained={summary['pct_s1_all_matches_retained']}% "
            f"pct_at_least_one={summary['pct_s1_at_least_one_retained']}% "
            f"n_zero_retained={summary['n_s1_zero_retained']}")
        del idx, cands_fn, df
        gc.collect()
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        log(f"  released. rss={rss_mb():.0f}MB")

    log("Entity-level evaluation complete.")


if __name__ == "__main__":
    main()
