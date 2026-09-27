"""
Phase 4: Blocking channel benchmark, incorporating technical-lead review feedback:
  - max_df swept (500/1000/3000/5000/10000/25000), not fixed a priori
  - top-K query-token count swept (1/2/3/4/6/8)
  - OR vs minimum-overlap(>=2) retrieval compared
  - transliteration measured in isolation (TP recovered ONLY by translit)
  - DBA/alias measured as a full 2x2 (name / address / both / neither)
  - RSS (actual process memory), not just posting-array nbytes, reported
  - development sample (50K) vs full validation (441,364) explicitly separated
  - sample representativeness (singleton rate / country dist / match-count
    dist) checked against the full validation population

Writes reports/blocking_experiments.json incrementally (after every stage) so
partial results survive a crash/interruption; each stage also prints a
one-line summary to stderr.
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
REPORT_PATH = os.path.join(REPORT_DIR, "blocking_experiments.json")

SAMPLE_SEED = 42
SAMPLE_N = 50000
MAX_DF_GRID = [500, 1000, 3000, 5000, 10000, 25000]
TOPK_GRID = [1, 2, 3, 4, 6, 8]


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
    return {
        "n_s1_queried": len(sample_df),
        "n_true_positive_edges": total_pos,
        "n_true_positive_edges_retained": tp,
        "blocking_recall": round(tp / total_pos, 4) if total_pos else None,
        "avg_candidates": round(float(arr.mean()), 2),
        "median_candidates": float(np.median(arr)),
        "p95_candidates": float(np.percentile(arr, 95)),
        "max_candidates": int(arr.max()),
        "reduction_ratio": round(1 - arr.mean() / n_pool, 6),
        "runtime_sec": round(elapsed, 2),
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ---------------- Stage 0: prep ----------------
    log("STAGE 0: loading pool ids + ground truth + validation split")
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
    n_missing_lookup = 0
    for row in gt.itertuples():
        if not row.match_list:
            continue
        positions = []
        for mid in row.match_list:
            pos = id_to_pos.get(mid)
            if pos is None:
                n_missing_lookup += 1
            else:
                positions.append(int(pos))
        if positions:
            gt_positions[row.source1_entity_id] = frozenset(positions)
    del id_to_pos
    gc.collect()
    log(f"pool={n_pool} val_s1={len(val_ids)} positive_val_s1={len(gt_positions)} unresolved_match_ids={n_missing_lookup}")

    s1_full = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"))
    s1_val = s1_full[s1_full["entity_id"].isin(val_ids)].reset_index(drop=True)
    del s1_full
    gc.collect()

    rng = np.random.default_rng(SAMPLE_SEED)
    sample_idx = rng.choice(len(s1_val), size=min(SAMPLE_N, len(s1_val)), replace=False)
    s1_sample = s1_val.iloc[sample_idx].reset_index(drop=True)

    # sample representativeness check
    gt_by_s1 = gt.set_index("source1_entity_id")["match_list"]
    full_val_match_counts = gt_by_s1.reindex(s1_val["entity_id"]).map(lambda x: len(x) if isinstance(x, list) else 0)
    sample_match_counts = gt_by_s1.reindex(s1_sample["entity_id"]).map(lambda x: len(x) if isinstance(x, list) else 0)

    representativeness = {
        "full_validation": {
            "n": len(s1_val),
            "singleton_pct": round(100 * (full_val_match_counts == 0).mean(), 3),
            "country_dist_pct": (s1_val["country"].value_counts(normalize=True) * 100).round(2).to_dict(),
            "match_count_mean": round(float(full_val_match_counts.mean()), 3),
            "match_count_median": float(full_val_match_counts.median()),
        },
        "dev_sample_50k": {
            "n": len(s1_sample),
            "singleton_pct": round(100 * (sample_match_counts == 0).mean(), 3),
            "country_dist_pct": (s1_sample["country"].value_counts(normalize=True) * 100).round(2).to_dict(),
            "match_count_mean": round(float(sample_match_counts.mean()), 3),
            "match_count_median": float(sample_match_counts.median()),
        },
    }
    log(f"representativeness: full singleton%={representativeness['full_validation']['singleton_pct']} "
        f"sample singleton%={representativeness['dev_sample_50k']['singleton_pct']}")

    REPORT["pool_size"] = n_pool
    REPORT["sample_seed"] = SAMPLE_SEED
    REPORT["representativeness"] = representativeness
    save_report()

    # ---------------- Stage A: exact-match channels (B1, B2, B7a, B7b) ----------------
    log("STAGE A: exact-match channels B1/B2/B7a/B7b")
    exact_results = {}

    col = load_pool_column(["country", "name_norm"])
    b1 = ExactIndex("B1_exact_name").fit(col["country"], col["name_norm"])
    del col; gc.collect()
    r = evaluate_query_fn(lambda row: set(b1.query(row.country, row.name_norm).tolist()), s1_sample, gt_positions, n_pool)
    r["postings_array_mb"] = round(b1.postings_array_mb(), 1)
    r["fit_time_sec"] = round(b1.fit_time, 1)
    r["fit_rss_delta_mb"] = round(b1.fit_rss_delta_mb, 1) if b1.fit_rss_delta_mb is not None else None
    exact_results["B1_exact_name"] = r
    log(f"B1_exact_name: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} rss={rss_mb():.0f}MB")

    col = load_pool_column(["country", "name_no_suffix"])
    b2 = ExactIndex("B2_exact_name_no_suffix").fit(col["country"], col["name_no_suffix"])
    del col; gc.collect()
    r = evaluate_query_fn(lambda row: set(b2.query(row.country, row.name_no_suffix).tolist()), s1_sample, gt_positions, n_pool)
    r["postings_array_mb"] = round(b2.postings_array_mb(), 1)
    r["fit_time_sec"] = round(b2.fit_time, 1)
    exact_results["B2_exact_name_no_suffix"] = r
    log(f"B2_exact_name_no_suffix: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} rss={rss_mb():.0f}MB")

    col = load_pool_column(["country", "postal_code"])
    b7a = ExactIndex("B7a_postal").fit(col["country"], col["postal_code"])
    del col; gc.collect()
    r = evaluate_query_fn(lambda row: set(b7a.query(row.country, row.postal_code).tolist()), s1_sample, gt_positions, n_pool)
    r["postings_array_mb"] = round(b7a.postings_array_mb(), 1)
    r["fit_time_sec"] = round(b7a.fit_time, 1)
    exact_results["B7a_postal"] = r
    log(f"B7a_postal: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} rss={rss_mb():.0f}MB")

    col = load_pool_column(["country", "street_number"])
    b7b = ExactIndex("B7b_street_number").fit(col["country"], col["street_number"])
    del col; gc.collect()
    r = evaluate_query_fn(lambda row: set(b7b.query(row.country, row.street_number).tolist()), s1_sample, gt_positions, n_pool)
    r["postings_array_mb"] = round(b7b.postings_array_mb(), 1)
    r["fit_time_sec"] = round(b7b.fit_time, 1)
    exact_results["B7b_street_number"] = r
    log(f"B7b_street_number: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} rss={rss_mb():.0f}MB")

    REPORT["exact_channels"] = exact_results
    save_report()

    # ---------------- Stage B: max_df sweep, name token (B3) ----------------
    log("STAGE B: max_df sweep -- name token channel (B3)")
    col = load_pool_column(["country", "name_norm"])
    name_ti = TokenIndex("B3_name_token")
    name_ti.fit_df_counts(col["country"], col["name_norm"])
    log(f"B3 df_counts pass done in {name_ti.df_count_time:.1f}s, distinct (country,token) keys={len(name_ti.df_counts)}")

    b3_maxdf_sweep = []
    for max_df in MAX_DF_GRID:
        name_ti.build_postings(min_df=1, max_df=max_df)
        r = evaluate_query_fn(
            lambda row: set(name_ti.query(row.country, row.name_norm, max_query_tokens=6).tolist()),
            s1_sample, gt_positions, n_pool,
        )
        r["max_df"] = max_df
        r["n_keys_kept"] = name_ti.n_distinct_keys_kept
        r["build_postings_time_sec"] = round(name_ti.build_postings_time, 1)
        r["postings_array_mb"] = round(name_ti.postings_array_mb(), 1)
        b3_maxdf_sweep.append(r)
        log(f"  B3 max_df={max_df}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} "
            f"median={r['median_candidates']} p95={r['p95_candidates']} keys_kept={r['n_keys_kept']} "
            f"postings_mb={r['postings_array_mb']} rss={rss_mb():.0f}MB")
        REPORT["b3_maxdf_sweep"] = b3_maxdf_sweep
        save_report()

    # pick elbow: smallest max_df achieving >=99% of the best-observed recall
    best_recall = max(r["blocking_recall"] for r in b3_maxdf_sweep if r["blocking_recall"] is not None)
    b3_chosen_maxdf = next(r["max_df"] for r in b3_maxdf_sweep if r["blocking_recall"] is not None and r["blocking_recall"] >= 0.99 * best_recall)
    log(f"B3 chosen max_df (elbow, >=99% of best recall {best_recall}): {b3_chosen_maxdf}")

    # ---------------- Stage C: top-K sweep at chosen max_df, name token ----------------
    log(f"STAGE C: top-K sweep -- name token channel at max_df={b3_chosen_maxdf}")
    name_ti.build_postings(min_df=1, max_df=b3_chosen_maxdf)
    b3_topk_sweep = []
    for k in TOPK_GRID:
        r = evaluate_query_fn(
            lambda row: set(name_ti.query(row.country, row.name_norm, max_query_tokens=k).tolist()),
            s1_sample, gt_positions, n_pool,
        )
        r["top_k"] = k
        b3_topk_sweep.append(r)
        log(f"  B3 K={k}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} p95={r['p95_candidates']}")
    REPORT["b3_maxdf_chosen"] = b3_chosen_maxdf
    REPORT["b3_topk_sweep"] = b3_topk_sweep
    save_report()

    # min-overlap comparison at chosen max_df / elbow K (recall is monotonic non-decreasing in K
    # since query() only unions more postings as K grows, so picking argmax(recall) would trivially
    # always select the largest K in the grid -- instead pick the smallest K reaching >=99% of the
    # recall achieved at the largest K, mirroring the max_df elbow logic above).
    best_k_recall = max(r["blocking_recall"] or 0 for r in b3_topk_sweep)
    best_k = next(r["top_k"] for r in b3_topk_sweep if (r["blocking_recall"] or 0) >= 0.99 * best_k_recall)
    r_or = evaluate_query_fn(
        lambda row: set(name_ti.query(row.country, row.name_norm, max_query_tokens=best_k, min_overlap=1).tolist()),
        s1_sample, gt_positions, n_pool)
    r_ov2 = evaluate_query_fn(
        lambda row: set(name_ti.query(row.country, row.name_norm, max_query_tokens=best_k, min_overlap=2).tolist()),
        s1_sample, gt_positions, n_pool)
    REPORT["b3_retrieval_strategy_comparison"] = {"OR_k" + str(best_k): r_or, "min_overlap2_k" + str(best_k): r_ov2}
    log(f"B3 retrieval strategy @K={best_k}: OR recall={r_or['blocking_recall']} avg_cand={r_or['avg_candidates']} | "
        f"min_overlap2 recall={r_ov2['blocking_recall']} avg_cand={r_ov2['avg_candidates']}")
    save_report()

    B3_FINAL_MAXDF, B3_FINAL_K = b3_chosen_maxdf, best_k
    name_ti.build_postings(min_df=1, max_df=B3_FINAL_MAXDF)  # leave built at chosen config for later stages
    name_ti.release_raw()  # no more max_df rebuilds needed -- drop the resident raw column
    del col
    gc.collect()

    # ---------------- Stage D: max_df sweep, address token (B5) ----------------
    log("STAGE D: max_df sweep -- address token channel (B5)")
    col = load_pool_column(["country", "address_norm"])
    addr_ti = TokenIndex("B5_address_token")
    addr_ti.fit_df_counts(col["country"], col["address_norm"])
    log(f"B5 df_counts pass done in {addr_ti.df_count_time:.1f}s, distinct keys={len(addr_ti.df_counts)}")

    b5_maxdf_sweep = []
    for max_df in MAX_DF_GRID:
        addr_ti.build_postings(min_df=1, max_df=max_df)
        r = evaluate_query_fn(
            lambda row: set(addr_ti.query(row.country, row.address_norm, max_query_tokens=8).tolist()),
            s1_sample, gt_positions, n_pool,
        )
        r["max_df"] = max_df
        r["n_keys_kept"] = addr_ti.n_distinct_keys_kept
        r["postings_array_mb"] = round(addr_ti.postings_array_mb(), 1)
        b5_maxdf_sweep.append(r)
        log(f"  B5 max_df={max_df}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} "
            f"median={r['median_candidates']} p95={r['p95_candidates']} rss={rss_mb():.0f}MB")
        REPORT["b5_maxdf_sweep"] = b5_maxdf_sweep
        save_report()

    best_recall_addr = max(r["blocking_recall"] for r in b5_maxdf_sweep if r["blocking_recall"] is not None)
    b5_chosen_maxdf = next(r["max_df"] for r in b5_maxdf_sweep if r["blocking_recall"] is not None and r["blocking_recall"] >= 0.99 * best_recall_addr)
    log(f"B5 chosen max_df: {b5_chosen_maxdf}")

    addr_ti.build_postings(min_df=1, max_df=b5_chosen_maxdf)
    b5_topk_sweep = []
    for k in TOPK_GRID:
        r = evaluate_query_fn(
            lambda row: set(addr_ti.query(row.country, row.address_norm, max_query_tokens=k).tolist()),
            s1_sample, gt_positions, n_pool,
        )
        r["top_k"] = k
        b5_topk_sweep.append(r)
        log(f"  B5 K={k}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} p95={r['p95_candidates']}")
    REPORT["b5_maxdf_chosen"] = b5_chosen_maxdf
    REPORT["b5_topk_sweep"] = b5_topk_sweep
    save_report()

    best_k_addr_recall = max(r["blocking_recall"] or 0 for r in b5_topk_sweep)
    best_k_addr = next(r["top_k"] for r in b5_topk_sweep if (r["blocking_recall"] or 0) >= 0.99 * best_k_addr_recall)
    r_or = evaluate_query_fn(
        lambda row: set(addr_ti.query(row.country, row.address_norm, max_query_tokens=best_k_addr, min_overlap=1).tolist()),
        s1_sample, gt_positions, n_pool)
    r_ov2 = evaluate_query_fn(
        lambda row: set(addr_ti.query(row.country, row.address_norm, max_query_tokens=best_k_addr, min_overlap=2).tolist()),
        s1_sample, gt_positions, n_pool)
    REPORT["b5_retrieval_strategy_comparison"] = {"OR_k" + str(best_k_addr): r_or, "min_overlap2_k" + str(best_k_addr): r_ov2}
    log(f"B5 retrieval strategy @K={best_k_addr}: OR recall={r_or['blocking_recall']} | min_overlap2 recall={r_ov2['blocking_recall']}")
    save_report()

    B5_FINAL_MAXDF, B5_FINAL_K = b5_chosen_maxdf, best_k_addr
    # B6 = same index, distinctive/rare-only query (small K, no min_overlap)
    r_b6 = evaluate_query_fn(
        lambda row: set(addr_ti.query(row.country, row.address_norm, max_query_tokens=2).tolist()),
        s1_sample, gt_positions, n_pool)
    REPORT["b6_address_token_rare_k2"] = r_b6
    log(f"B6 (rare address token, K=2): recall={r_b6['blocking_recall']} avg_cand={r_b6['avg_candidates']}")
    save_report()

    addr_ti.build_postings(min_df=1, max_df=B5_FINAL_MAXDF)
    addr_ti.release_raw()
    del col
    gc.collect()

    # ---------------- Stage E: transliteration isolation (B4) ----------------
    log("STAGE E: transliteration channel isolation (B4)")
    col = load_pool_column(["country", "name_translit"])
    translit_ti = TokenIndex("B4_name_translit_token").fit(col["country"], col["name_translit"], min_df=1, max_df=B3_FINAL_MAXDF)
    translit_ti.release_raw()
    del col
    gc.collect()

    def name_orig_cands(row):
        return set(name_ti.query(row.country, row.name_norm, max_query_tokens=B3_FINAL_K).tolist())

    def name_translit_cands(row):
        return set(translit_ti.query(row.country, row.name_translit, max_query_tokens=B3_FINAL_K).tolist())

    r_orig_only = evaluate_query_fn(name_orig_cands, s1_sample, gt_positions, n_pool)
    r_translit_only = evaluate_query_fn(name_translit_cands, s1_sample, gt_positions, n_pool)
    r_both = evaluate_query_fn(lambda row: name_orig_cands(row) | name_translit_cands(row), s1_sample, gt_positions, n_pool)

    # TP recovered ONLY by translit (present in translit retrieval, absent from original-token retrieval)
    tp_only_translit = 0
    tp_total_sample = 0
    for row in s1_sample.itertuples():
        true_pos = gt_positions.get(row.entity_id)
        if not true_pos:
            continue
        tp_total_sample += len(true_pos)
        orig_hit = true_pos & name_orig_cands(row)
        both_hit = true_pos & (name_orig_cands(row) | name_translit_cands(row))
        tp_only_translit += len(both_hit - orig_hit)

    REPORT["b4_translit_isolation"] = {
        "original_name_token_only": r_orig_only,
        "translit_name_token_only": r_translit_only,
        "union_orig_plus_translit": r_both,
        "tp_recovered_only_by_translit": tp_only_translit,
        "tp_recovered_only_by_translit_pct_of_sample_positives": round(100 * tp_only_translit / tp_total_sample, 3) if tp_total_sample else None,
    }
    log(f"B4 translit isolation: orig_recall={r_orig_only['blocking_recall']} translit_recall={r_translit_only['blocking_recall']} "
        f"union_recall={r_both['blocking_recall']} TP_only_via_translit={tp_only_translit} "
        f"({REPORT['b4_translit_isolation']['tp_recovered_only_by_translit_pct_of_sample_positives']}% of sample positives)")
    save_report()

    # ---------------- Stage F: character n-gram channel (B8) ----------------
    log("STAGE F: character n-gram channel (B8) -- n=3 vs n=4, max_df sweep")
    col = load_pool_column(["country", "name_norm"])
    b8_results = {}
    for n in (3, 4):
        ngram_col = build_ngram_series(col["name_norm"], n=n)
        ng_ti = TokenIndex(f"B8_name_{n}gram")
        ng_ti.fit_df_counts(col["country"], ngram_col)
        sweep = []
        for max_df in [3000, 8000, 20000]:
            ng_ti.build_postings(min_df=1, max_df=max_df)

            def q(row, _n=n, _idx=ng_ti):
                qs = char_ngram_string(row.name_norm, _n) if row.name_norm else ""
                return set(_idx.query(row.country, qs, max_query_tokens=12).tolist())

            r = evaluate_query_fn(q, s1_sample, gt_positions, n_pool)
            r["max_df"] = max_df
            r["n_keys_kept"] = ng_ti.n_distinct_keys_kept
            r["postings_array_mb"] = round(ng_ti.postings_array_mb(), 1)
            sweep.append(r)
            log(f"  B8 {n}-gram max_df={max_df}: recall={r['blocking_recall']} avg_cand={r['avg_candidates']} "
                f"p95={r['p95_candidates']} postings_mb={r['postings_array_mb']} rss={rss_mb():.0f}MB")
        b8_results[f"{n}gram"] = sweep
        del ng_ti, ngram_col
        gc.collect()
        REPORT["b8_ngram_sweep"] = b8_results
        save_report()
    del col
    gc.collect()

    # pick B8 final config: best recall among 3-gram sweep (typically best tolerance to typos)
    b8_best = max(b8_results["3gram"], key=lambda r: (r["blocking_recall"] or 0))
    log(f"B8 chosen: 3-gram max_df={b8_best['max_df']} recall={b8_best['blocking_recall']} avg_cand={b8_best['avg_candidates']}")

    col = load_pool_column(["country", "name_norm"])
    ngram_col = build_ngram_series(col["name_norm"], n=3)
    b8_ti = TokenIndex("B8_final").fit(col["country"], ngram_col, min_df=1, max_df=b8_best["max_df"])
    b8_ti.release_raw()
    del col, ngram_col
    gc.collect()

    def b8_cands(row):
        qs = char_ngram_string(row.name_norm, 3) if row.name_norm else ""
        return set(b8_ti.query(row.country, qs, max_query_tokens=12).tolist())

    # ---------------- Stage G: unions + DBA/alias 2x2 ----------------
    log("STAGE G: name-union / address-union / full-union + DBA/alias 2x2 breakdown")

    def name_union_cands(row):
        return (set(b1.query(row.country, row.name_norm).tolist())
                | set(b2.query(row.country, row.name_no_suffix).tolist())
                | set(name_ti.query(row.country, row.name_norm, max_query_tokens=B3_FINAL_K).tolist())
                | set(translit_ti.query(row.country, row.name_translit, max_query_tokens=B3_FINAL_K).tolist())
                | b8_cands(row))

    def address_union_cands(row):
        return (set(addr_ti.query(row.country, row.address_norm, max_query_tokens=B5_FINAL_K).tolist())
                | set(b7a.query(row.country, row.postal_code).tolist())
                | set(b7b.query(row.country, row.street_number).tolist()))

    def full_union_cands(row):
        return name_union_cands(row) | address_union_cands(row)

    r_name_union = evaluate_query_fn(name_union_cands, s1_sample, gt_positions, n_pool)
    r_addr_union = evaluate_query_fn(address_union_cands, s1_sample, gt_positions, n_pool)
    r_full_union = evaluate_query_fn(full_union_cands, s1_sample, gt_positions, n_pool)
    log(f"name_union: recall={r_name_union['blocking_recall']} avg_cand={r_name_union['avg_candidates']} p95={r_name_union['p95_candidates']}")
    log(f"address_union: recall={r_addr_union['blocking_recall']} avg_cand={r_addr_union['avg_candidates']} p95={r_addr_union['p95_candidates']}")
    log(f"full_union: recall={r_full_union['blocking_recall']} avg_cand={r_full_union['avg_candidates']} p95={r_full_union['p95_candidates']}")

    # DBA/alias full 2x2
    name_only_edges = addr_only_edges = both_edges = neither_edges = 0
    total_edges_sample = 0
    neither_examples = []
    for row in s1_sample.itertuples():
        true_pos = gt_positions.get(row.entity_id)
        if not true_pos:
            continue
        total_edges_sample += len(true_pos)
        nset = name_union_cands(row)
        aset = address_union_cands(row)
        name_hit = true_pos & nset
        addr_hit = true_pos & aset
        both = name_hit & addr_hit
        only_name = name_hit - addr_hit
        only_addr = addr_hit - name_hit
        neither = true_pos - nset - aset
        both_edges += len(both)
        name_only_edges += len(only_name)
        addr_only_edges += len(only_addr)
        neither_edges += len(neither)
        if neither and len(neither_examples) < 25:
            neither_examples.append({"s1_id": row.entity_id, "n_missed": len(neither)})

    dba_2x2 = {
        "total_positive_edges_in_sample": total_edges_sample,
        "name_only": {"edges": name_only_edges, "pct": round(100 * name_only_edges / total_edges_sample, 3)},
        "address_only": {"edges": addr_only_edges, "pct": round(100 * addr_only_edges / total_edges_sample, 3)},
        "both": {"edges": both_edges, "pct": round(100 * both_edges / total_edges_sample, 3)},
        "neither": {"edges": neither_edges, "pct": round(100 * neither_edges / total_edges_sample, 3)},
        "neither_examples_s1_ids": neither_examples,
    }
    log(f"DBA/alias 2x2: name_only={dba_2x2['name_only']} address_only={dba_2x2['address_only']} "
        f"both={dba_2x2['both']} neither={dba_2x2['neither']}")

    REPORT["union_results_sample"] = {
        "name_union": r_name_union,
        "address_union": r_addr_union,
        "full_union": r_full_union,
    }
    REPORT["dba_alias_2x2"] = dba_2x2
    REPORT["final_channel_configs"] = {
        "B3_name_token": {"max_df": B3_FINAL_MAXDF, "top_k": B3_FINAL_K},
        "B5_address_token": {"max_df": B5_FINAL_MAXDF, "top_k": B5_FINAL_K},
        "B8_name_trigram": {"max_df": b8_best["max_df"]},
    }
    save_report()

    log("Blocking experiments (development-sample stage) complete. "
        "Run the full-validation confirmation + miss analysis next.")


if __name__ == "__main__":
    main()
