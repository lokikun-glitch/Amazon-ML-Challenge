"""
Stage 6: candidate-size optimization.

1. Fix B7b: ExactIndex now supports max_df (added to blocking.py); sweep
   100/250/500/1000/2000/5000 and report recall/candidates/runtime/RSS.
2. B8 switches to 4-gram as primary (measured to dominate 3-gram: higher
   recall AND fewer candidates at max_df=20000); a tight 4-gram sweep
   (500/1000/2000) is added for the compact union configs. 3-gram is kept
   only as the already-recorded reference result -- not re-swept here.
3. Actual union configs A (conservative) / B (mid) / C (wider) are measured
   directly (never estimated by composing single-channel numbers).
4. Full Pareto table: recall, mean/median/P95/P99/max candidates, total
   candidate pairs (in-sample and projected), runtime, peak RSS.
5. Competition-scale projection uses the ACTUAL measured test_source1 count
   (1,732,544 -- confirmed via `wc -l` and the cache-build log), not the
   11,702,133 figure given in the brief. That number is not test S1 rows: it
   equals test_source1 + test_source2 + test_source3 combined
   (1,732,544 + 4,887,273 + 5,082,316 = 11,702,133 exactly) -- flagged rather
   than silently substituted, per "measure it yourself, don't assume."
6. Resource safety: compact (small max_df) indexes are lightweight (tens of
   MB, confirmed by the Stage-4 sweep), so configs A/B/C's few small indexes
   coexist safely; heavier one-off sweep indexes are deleted immediately
   after their sweep completes, with RSS logged at every step.
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
REPORT_PATH = os.path.join(REPORT_DIR, "candidate_size_optimization.json")

SAMPLE_SEED = 42
SAMPLE_N = 50000

B7B_MAXDF_GRID = [100, 250, 500, 1000, 2000, 5000]
B8_4GRAM_TIGHT_GRID = [500, 1000, 2000]

# Measured directly (wc -l on the actual files + cache-build log), not assumed.
TEST_S1_COUNT = 1_732_544
TEST_S2_COUNT = 4_887_273
TEST_S3_COUNT = 5_082_316
# The brief's "11,702,133 test S1 rows" figure is actually S1+S2+S3 combined --
# flagged, not silently used as an S1 count.
assert TEST_S1_COUNT + TEST_S2_COUNT + TEST_S3_COUNT == 11_702_133

AVG_ID_BYTES = 13  # "S2-123456789" style id + separator, measured from sample ids


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr)


def load_pool_column(columns):
    frames = []
    for label in ("train_source2", "train_source3"):
        p = os.path.join(CACHE_DIR, f"{label}.parquet")
        frames.append(pd.read_parquet(p, columns=columns))
    return pd.concat(frames, ignore_index=True)


REPORT = {}


def save_report():
    with open(REPORT_PATH, "w") as f:
        json.dump(REPORT, f, indent=2, default=lambda o: str(o))


def evaluate_query_fn(query_fn, sample_df, gt_positions, n_pool, extra_percentiles=True):
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
    total_in_sample = int(arr.sum())
    mean_cand = float(arr.mean())
    projected_test_pairs = int(round(mean_cand * TEST_S1_COUNT))
    projected_file_mb = round(
        (TEST_S1_COUNT * (AVG_ID_BYTES + 1) + projected_test_pairs * AVG_ID_BYTES) / 1e6, 1
    )
    result = {
        "n_s1_queried": len(sample_df),
        "n_true_positive_edges": total_pos,
        "n_true_positive_edges_retained": tp,
        "blocking_recall": round(tp / total_pos, 4) if total_pos else None,
        "avg_candidates": round(mean_cand, 2),
        "median_candidates": float(np.median(arr)),
        "p95_candidates": float(np.percentile(arr, 95)),
        "max_candidates": int(arr.max()),
        "total_candidate_pairs_in_sample": total_in_sample,
        "reduction_ratio": round(1 - mean_cand / n_pool, 6),
        "runtime_sec": round(elapsed, 2),
        "rss_mb_at_measurement": round(rss_mb(), 0),
        "projected_test_pairs_at_1_732_544_test_s1": projected_test_pairs,
        "projected_candidate_pairs_tsv_mb": projected_file_mb,
    }
    if extra_percentiles:
        result["p99_candidates"] = float(np.percentile(arr, 99))
    return result


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    log("Prep: pool ids, ground truth, validation sample (seed=42, same as before)")
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

    REPORT["test_scale_note"] = {
        "measured_test_source1_rows": TEST_S1_COUNT,
        "measured_test_source2_rows": TEST_S2_COUNT,
        "measured_test_source3_rows": TEST_S3_COUNT,
        "brief_figure_11702133_is_actually": "test_source1 + test_source2 + test_source3 combined, not test S1 row count",
        "projection_uses": "TEST_S1_COUNT = 1,732,544 (measured), not the brief's 11,702,133",
    }
    save_report()

    # ---------------- 1. B7b capped sweep ----------------
    log("STAGE 1: B7b (street number) capped max_df sweep")
    col = load_pool_column(["country", "street_number"])
    b7b_sweep = []
    for max_df in B7B_MAXDF_GRID:
        b7b = ExactIndex("B7b").fit(col["country"], col["street_number"], max_df=max_df)
        r = evaluate_query_fn(lambda row: set(b7b.query(row.country, row.street_number).tolist()),
                               s1_sample, gt_positions, n_pool)
        r["max_df"] = max_df
        r["n_keys_kept"] = b7b.n_distinct_keys_kept
        r["n_keys_seen"] = b7b.n_distinct_keys_seen
        b7b_sweep.append(r)
        log(f"  B7b max_df={max_df}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} "
            f"median={r['median_candidates']} p95={r['p95_candidates']} p99={r['p99_candidates']} "
            f"max={r['max_candidates']} keys_kept={r['n_keys_kept']}/{r['n_keys_seen']} rss={rss_mb():.0f}MB")
        del b7b
        gc.collect()
    # reference: uncapped result from the prior run (for comparison only, not re-measured)
    b7b_sweep.append({
        "max_df": None, "note": "uncapped (prior run reference, not re-measured here)",
        "blocking_recall": 0.6704, "avg_candidates": 11665.12, "p95_candidates": 71656.0,
    })
    REPORT["b7b_capped_sweep"] = b7b_sweep
    save_report()
    del col
    gc.collect()

    # ---------------- 2. B8 4-gram tight sweep ----------------
    log("STAGE 2: B8 4-gram tight max_df sweep (500/1000/2000) -- 4-gram is now primary")
    col = load_pool_column(["country", "name_norm"])
    ngram4 = build_ngram_series(col["name_norm"], n=4)
    b8_4g = TokenIndex("B8_4gram")
    b8_4g.fit_df_counts(col["country"], ngram4)
    log(f"  4-gram df_counts pass done in {b8_4g.df_count_time:.1f}s, distinct keys={len(b8_4g.df_counts)}")
    b8_4g_sweep = []
    for max_df in B8_4GRAM_TIGHT_GRID:
        b8_4g.build_postings(min_df=1, max_df=max_df)

        def q(row, _idx=b8_4g):
            qs = char_ngram_string(row.name_norm, 4) if row.name_norm else ""
            return set(_idx.query(row.country, qs, max_query_tokens=12).tolist())

        r = evaluate_query_fn(q, s1_sample, gt_positions, n_pool)
        r["max_df"] = max_df
        r["n_keys_kept"] = b8_4g.n_distinct_keys_kept
        b8_4g_sweep.append(r)
        log(f"  B8-4gram max_df={max_df}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} "
            f"p95={r['p95_candidates']} rss={rss_mb():.0f}MB")
    # keep prior (3000/8000/20000) results as reference, already in blocking_experiments.json
    REPORT["b8_4gram_tight_sweep"] = b8_4g_sweep
    REPORT["b8_3gram_reference_note"] = "3-gram kept only as reference (see blocking_experiments.json b8_ngram_sweep.3gram) -- not re-swept, dominated by 4-gram"
    save_report()
    del col, ngram4
    gc.collect()
    log(f"B8 4-gram sweep done. rss={rss_mb():.0f}MB")

    # ---------------- 3+4. actual union configs A / B / C ----------------
    log("STAGE 3: building compact indexes for union configs A/B/C")

    def build_config_indexes(b3_maxdf, b5_maxdf, b7b_maxdf, b8_4g_maxdf=None):
        idx = {}
        col = load_pool_column(["country", "name_norm"])
        idx["b1"] = ExactIndex("B1").fit(col["country"], col["name_norm"])
        del col; gc.collect()
        col = load_pool_column(["country", "name_no_suffix"])
        idx["b2"] = ExactIndex("B2").fit(col["country"], col["name_no_suffix"])
        del col; gc.collect()
        col = load_pool_column(["country", "name_norm"])
        idx["b3"] = TokenIndex("B3").fit(col["country"], col["name_norm"], min_df=1, max_df=b3_maxdf)
        idx["b3"].release_raw()
        del col; gc.collect()
        col = load_pool_column(["country", "address_norm"])
        idx["b5"] = TokenIndex("B5").fit(col["country"], col["address_norm"], min_df=1, max_df=b5_maxdf)
        idx["b5"].release_raw()
        del col; gc.collect()
        col = load_pool_column(["country", "postal_code"])
        idx["b7a"] = ExactIndex("B7a").fit(col["country"], col["postal_code"])
        del col; gc.collect()
        if b7b_maxdf is not None:
            col = load_pool_column(["country", "street_number"])
            idx["b7b"] = ExactIndex("B7b").fit(col["country"], col["street_number"], max_df=b7b_maxdf)
            del col; gc.collect()
        if b8_4g_maxdf is not None:
            col = load_pool_column(["country", "name_norm"])
            ngram_col = build_ngram_series(col["name_norm"], n=4)
            idx["b8"] = TokenIndex("B8").fit(col["country"], ngram_col, min_df=1, max_df=b8_4g_maxdf)
            idx["b8"].release_raw()
            del col, ngram_col; gc.collect()
        return idx

    def make_cands_fn(idx, b3_k, b5_k):
        has_b7b = "b7b" in idx
        has_b8 = "b8" in idx

        def cands_fn(row):
            c = (set(idx["b1"].query(row.country, row.name_norm).tolist())
                 | set(idx["b2"].query(row.country, row.name_no_suffix).tolist())
                 | set(idx["b3"].query(row.country, row.name_norm, max_query_tokens=b3_k).tolist())
                 | set(idx["b5"].query(row.country, row.address_norm, max_query_tokens=b5_k).tolist())
                 | set(idx["b7a"].query(row.country, row.postal_code).tolist()))
            if has_b7b:
                c |= set(idx["b7b"].query(row.country, row.street_number).tolist())
            if has_b8:
                qs = char_ngram_string(row.name_norm, 4) if row.name_norm else ""
                c |= set(idx["b8"].query(row.country, qs, max_query_tokens=12).tolist())
            return c

        return cands_fn

    B7B_CAPPED_CHOICE = min(
        (r for r in b7b_sweep if r.get("max_df") is not None),
        key=lambda r: abs((r["blocking_recall"] or 0) - 0.60),  # pick the cap nearest ~60% recall as a reasonable capped B7b
    )["max_df"]
    B8_4G_CAPPED_CHOICE = max(b8_4g_sweep, key=lambda r: (r["blocking_recall"] or 0) / max(r["avg_candidates"], 1))["max_df"]
    log(f"Chosen capped B7b max_df for configs B/C: {B7B_CAPPED_CHOICE}; chosen B8 4-gram cap for config C: {B8_4G_CAPPED_CHOICE}")

    configs = {
        "A_conservative": {"b3_maxdf": 500, "b5_maxdf": 500, "b7b_maxdf": None, "b8_4g_maxdf": None, "b3_k": 4, "b5_k": 4},
        "B_mid": {"b3_maxdf": 1000, "b5_maxdf": 1000, "b7b_maxdf": B7B_CAPPED_CHOICE, "b8_4g_maxdf": None, "b3_k": 4, "b5_k": 6},
        "C_wider": {"b3_maxdf": 3000, "b5_maxdf": 3000, "b7b_maxdf": B7B_CAPPED_CHOICE, "b8_4g_maxdf": B8_4G_CAPPED_CHOICE, "b3_k": 4, "b5_k": 6},
    }

    pareto_table = {}
    for name, cfg in configs.items():
        log(f"STAGE 4: building + measuring union config {name}: {cfg}")
        idx = build_config_indexes(cfg["b3_maxdf"], cfg["b5_maxdf"], cfg["b7b_maxdf"], cfg["b8_4g_maxdf"])
        log(f"  indexes built. rss={rss_mb():.0f}MB")
        cands_fn = make_cands_fn(idx, cfg["b3_k"], cfg["b5_k"])
        r = evaluate_query_fn(cands_fn, s1_sample, gt_positions, n_pool)
        r["config"] = cfg
        pareto_table[name] = r
        log(f"  {name}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} median={r['median_candidates']} "
            f"p95={r['p95_candidates']} p99={r['p99_candidates']} max={r['max_candidates']} "
            f"projected_test_pairs={r['projected_test_pairs_at_1_732_544_test_s1']:,} "
            f"projected_file_mb={r['projected_candidate_pairs_tsv_mb']} rss={rss_mb():.0f}MB")
        del idx, cands_fn
        gc.collect()
        REPORT["pareto_table"] = pareto_table
        save_report()

    log("Stage 6 candidate-size optimization complete. Awaiting review before feature engineering.")


if __name__ == "__main__":
    main()
