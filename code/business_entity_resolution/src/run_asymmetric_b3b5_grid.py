"""
Stage 6 follow-up: small, targeted B3/B5 asymmetric grid.

A/B/C used symmetric B3==B5 max_df, but address channels already showed more
*exclusive* recall contribution than name channels (10.38% address-only vs
5.02% name-only in the DBA/alias 2x2). This tests whether shifting budget
from name to address (or vice versa) gets more recall per candidate than the
symmetric configs -- 4 targeted points, not the full 4x4=16 grid, per the
"use A/B/C to pick the most informative combinations" instruction.

Channel set kept identical to config A (B1, B2, B3, B5, B7a -- no B7b/B8) so
the ONLY variable is the B3/B5 max_df split, isolating that comparison.
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from blocking import ExactIndex, TokenIndex, rss_mb

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "normalized")
SPLIT_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "splits")
GT_PATH = os.path.join(HERE, "..", "..", "..", "student_resource", "dataset", "train", "train_ground_truth.tsv")
REPORT_DIR = os.path.join(HERE, "..", "..", "..", "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "asymmetric_b3b5_grid.json")

SAMPLE_SEED = 42
SAMPLE_N = 50000
TEST_S1_COUNT = 1_732_544
AVG_ID_BYTES = 13

GRID = {
    "D1_lean_name_mid_addr": {"b3_maxdf": 500, "b5_maxdf": 1000, "b3_k": 4, "b5_k": 6},
    "D2_mid_name_lean_addr": {"b3_maxdf": 1000, "b5_maxdf": 500, "b3_k": 4, "b5_k": 4},
    "D3_lean_name_high_addr": {"b3_maxdf": 500, "b5_maxdf": 2000, "b3_k": 4, "b5_k": 6},
    "D4_high_name_lean_addr": {"b3_maxdf": 2000, "b5_maxdf": 500, "b3_k": 4, "b5_k": 4},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr)


def load_pool_column(columns):
    frames = []
    for label in ("train_source2", "train_source3"):
        p = os.path.join(CACHE_DIR, f"{label}.parquet")
        frames.append(pd.read_parquet(p, columns=columns))
    return pd.concat(frames, ignore_index=True)


def evaluate_query_fn(query_fn, sample_df, gt_positions, n_pool):
    t0 = time.time()
    cand_counts = []
    tp = 0
    total_pos = 0
    for row in sample_df.itertuples():
        cands = query_fn(row)
        cand_counts.append(len(cands))
        true_pos = gt_positions.get(row.entity_id)
        if true_pos:
            total_pos += len(true_pos)
            tp += len(true_pos & cands)
    elapsed = time.time() - t0
    arr = np.array(cand_counts) if cand_counts else np.array([0])
    mean_cand = float(arr.mean())
    projected_test_pairs = int(round(mean_cand * TEST_S1_COUNT))
    return {
        "n_true_positive_edges": total_pos,
        "n_true_positive_edges_retained": tp,
        "blocking_recall": round(tp / total_pos, 4) if total_pos else None,
        "avg_candidates": round(mean_cand, 2),
        "median_candidates": float(np.median(arr)),
        "p95_candidates": float(np.percentile(arr, 95)),
        "p99_candidates": float(np.percentile(arr, 99)),
        "max_candidates": int(arr.max()),
        "total_candidate_pairs_in_sample": int(arr.sum()),
        "reduction_ratio": round(1 - mean_cand / n_pool, 6),
        "runtime_sec": round(elapsed, 2),
        "rss_mb_at_measurement": round(rss_mb(), 0),
        "projected_test_pairs_at_1_732_544_test_s1": projected_test_pairs,
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    log("Prep (same seed=42 sample as prior Stage 6 runs)")
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
    log(f"Prep done. rss={rss_mb():.0f}MB")

    # channels shared across all 4 grid points -- build once
    log("Building shared B1/B2/B7a...")
    col = load_pool_column(["country", "name_norm"])
    b1 = ExactIndex("B1").fit(col["country"], col["name_norm"])
    del col; gc.collect()
    col = load_pool_column(["country", "name_no_suffix"])
    b2 = ExactIndex("B2").fit(col["country"], col["name_no_suffix"])
    del col; gc.collect()
    col = load_pool_column(["country", "postal_code"])
    b7a = ExactIndex("B7a").fit(col["country"], col["postal_code"])
    del col; gc.collect()

    log("Building shared B3/B5 df_counts (postings rebuilt per grid point, cheap)...")
    col_name = load_pool_column(["country", "name_norm"])
    b3 = TokenIndex("B3")
    b3.fit_df_counts(col_name["country"], col_name["name_norm"])
    col_addr = load_pool_column(["country", "address_norm"])
    b5 = TokenIndex("B5")
    b5.fit_df_counts(col_addr["country"], col_addr["address_norm"])
    log(f"df_counts done. rss={rss_mb():.0f}MB")

    results = {}
    for name, cfg in GRID.items():
        log(f"Config {name}: {cfg}")
        b3.build_postings(min_df=1, max_df=cfg["b3_maxdf"])
        b5.build_postings(min_df=1, max_df=cfg["b5_maxdf"])

        def cands_fn(row, _b3k=cfg["b3_k"], _b5k=cfg["b5_k"]):
            return (set(b1.query(row.country, row.name_norm).tolist())
                    | set(b2.query(row.country, row.name_no_suffix).tolist())
                    | set(b3.query(row.country, row.name_norm, max_query_tokens=_b3k).tolist())
                    | set(b5.query(row.country, row.address_norm, max_query_tokens=_b5k).tolist())
                    | set(b7a.query(row.country, row.postal_code).tolist()))

        r = evaluate_query_fn(cands_fn, s1_sample, gt_positions, n_pool)
        r["config"] = cfg
        results[name] = r
        log(f"  {name}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} "
            f"median={r['median_candidates']} p95={r['p95_candidates']} p99={r['p99_candidates']} "
            f"projected_test_pairs={r['projected_test_pairs_at_1_732_544_test_s1']:,} rss={rss_mb():.0f}MB")
        with open(REPORT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    log("Asymmetric B3/B5 grid complete.")


if __name__ == "__main__":
    main()
