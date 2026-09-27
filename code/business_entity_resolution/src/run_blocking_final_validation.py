"""
Stage I + J: full-validation confirmation and miss analysis.

Re-measures the chosen full-union blocking configuration (read from
reports/blocking_experiments.json, produced by run_blocking_experiments.py)
on ALL 441,364 validation S1 entities -- not the 50K development sample --
and samples true positives missed by every channel ("neither" class) for
manual failure-mode inspection.
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
DEV_REPORT_PATH = os.path.join(REPORT_DIR, "blocking_experiments.json")
OUT_PATH = os.path.join(REPORT_DIR, "blocking_full_validation.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr)


def load_pool_column(columns):
    frames = []
    for label in ("train_source2", "train_source3"):
        p = os.path.join(CACHE_DIR, f"{label}.parquet")
        frames.append(pd.read_parquet(p, columns=columns))
    return pd.concat(frames, ignore_index=True)


def main():
    with open(DEV_REPORT_PATH) as f:
        dev_report = json.load(f)
    cfg = dev_report["final_channel_configs"]
    b3_maxdf, b3_k = cfg["B3_name_token"]["max_df"], cfg["B3_name_token"]["top_k"]
    b5_maxdf, b5_k = cfg["B5_address_token"]["max_df"], cfg["B5_address_token"]["top_k"]
    b8_maxdf = cfg["B8_name_trigram"]["max_df"]
    log(f"Using chosen configs: B3 max_df={b3_maxdf} k={b3_k}; B5 max_df={b5_maxdf} k={b5_k}; B8 max_df={b8_maxdf}")

    pool_base = load_pool_column(["entity_id", "country"])
    n_pool = len(pool_base)
    id_to_pos = pd.Series(np.arange(n_pool, dtype=np.int64), index=pool_base["entity_id"].values)
    del pool_base
    gc.collect()

    val_ids = set(pd.read_csv(os.path.join(SPLIT_DIR, "val_s1_ids.txt"), header=None)[0].astype(str))
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    gt = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    gt["match_list"] = gt["matched_entity_ids"].map(lambda x: x.split(",") if isinstance(x, str) and x else [])

    gt_pairs = {}  # s1_id -> list of (matched_entity_id, pool_position)
    gt_positions = {}  # s1_id -> frozenset(pool_position), for fast set ops
    for row in gt.itertuples():
        if not row.match_list:
            continue
        pairs = []
        for mid in row.match_list:
            pos = id_to_pos.get(mid)
            if pos is not None:
                pairs.append((mid, int(pos)))
        if pairs:
            gt_pairs[row.source1_entity_id] = pairs
            gt_positions[row.source1_entity_id] = frozenset(p for _, p in pairs)
    del id_to_pos
    gc.collect()
    log(f"Full validation S1: {len(val_ids)}; with resolvable positives: {len(gt_positions)}")

    s1_full = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"))
    s1_val = s1_full[s1_full["entity_id"].isin(val_ids)].reset_index(drop=True)
    del s1_full
    gc.collect()

    # --- build all channel indexes at chosen configs ---
    log("Building B1/B2 exact indexes...")
    col = load_pool_column(["country", "name_norm"])
    b1 = ExactIndex("B1").fit(col["country"], col["name_norm"])
    del col; gc.collect()
    col = load_pool_column(["country", "name_no_suffix"])
    b2 = ExactIndex("B2").fit(col["country"], col["name_no_suffix"])
    del col; gc.collect()

    log("Building B3 name token index...")
    col = load_pool_column(["country", "name_norm"])
    name_ti = TokenIndex("B3").fit(col["country"], col["name_norm"], min_df=1, max_df=b3_maxdf)
    name_ti.release_raw()
    del col; gc.collect()

    log("Building B4 translit token index...")
    col = load_pool_column(["country", "name_translit"])
    translit_ti = TokenIndex("B4").fit(col["country"], col["name_translit"], min_df=1, max_df=b3_maxdf)
    translit_ti.release_raw()
    del col; gc.collect()

    log("Building B5 address token index...")
    col = load_pool_column(["country", "address_norm"])
    addr_ti = TokenIndex("B5").fit(col["country"], col["address_norm"], min_df=1, max_df=b5_maxdf)
    addr_ti.release_raw()
    del col; gc.collect()

    log("Building B7a/B7b exact indexes...")
    col = load_pool_column(["country", "postal_code"])
    b7a = ExactIndex("B7a").fit(col["country"], col["postal_code"])
    del col; gc.collect()
    col = load_pool_column(["country", "street_number"])
    b7b = ExactIndex("B7b").fit(col["country"], col["street_number"])
    del col; gc.collect()

    log("Building B8 trigram index...")
    col = load_pool_column(["country", "name_norm"])
    ngram_col = build_ngram_series(col["name_norm"], n=3)
    b8_ti = TokenIndex("B8").fit(col["country"], ngram_col, min_df=1, max_df=b8_maxdf)
    b8_ti.release_raw()
    del col, ngram_col; gc.collect()

    def b8_cands(row):
        qs = char_ngram_string(row.name_norm, 3) if row.name_norm else ""
        return set(b8_ti.query(row.country, qs, max_query_tokens=12).tolist())

    def name_union_cands(row):
        return (set(b1.query(row.country, row.name_norm).tolist())
                | set(b2.query(row.country, row.name_no_suffix).tolist())
                | set(name_ti.query(row.country, row.name_norm, max_query_tokens=b3_k).tolist())
                | set(translit_ti.query(row.country, row.name_translit, max_query_tokens=b3_k).tolist())
                | b8_cands(row))

    def address_union_cands(row):
        return (set(addr_ti.query(row.country, row.address_norm, max_query_tokens=b5_k).tolist())
                | set(b7a.query(row.country, row.postal_code).tolist())
                | set(b7b.query(row.country, row.street_number).tolist()))

    def full_union_cands(row):
        return name_union_cands(row) | address_union_cands(row)

    log(f"Querying full union over ALL {len(s1_val)} validation S1 entities...")
    t0 = time.time()
    cand_counts = []
    tp = 0
    total_pos = 0
    neither_examples = []
    dba_name_only = dba_addr_only = dba_both = dba_neither = 0
    for row in s1_val.itertuples():
        nset = name_union_cands(row)
        aset = address_union_cands(row)
        cands = nset | aset
        cand_counts.append(len(cands))
        true_pos = gt_positions.get(row.entity_id)
        if true_pos:
            total_pos += len(true_pos)
            tp += len(true_pos & cands)
            name_hit = true_pos & nset
            addr_hit = true_pos & aset
            dba_both += len(name_hit & addr_hit)
            dba_name_only += len(name_hit - addr_hit)
            dba_addr_only += len(addr_hit - name_hit)
            missed = true_pos - cands
            dba_neither += len(missed)
            if missed and len(neither_examples) < 200:
                missed_ids = [mid for mid, pos in gt_pairs[row.entity_id] if pos in missed]
                neither_examples.append({"s1_id": row.entity_id, "n_missed": len(missed),
                                          "missed_match_ids": missed_ids})
    elapsed = time.time() - t0

    arr = np.array(cand_counts)
    full_val_result = {
        "n_s1_queried": len(s1_val),
        "n_true_positive_edges": total_pos,
        "n_true_positive_edges_retained": tp,
        "blocking_recall": round(tp / total_pos, 4) if total_pos else None,
        "avg_candidates": round(float(arr.mean()), 2),
        "median_candidates": float(np.median(arr)),
        "p95_candidates": float(np.percentile(arr, 95)),
        "max_candidates": int(arr.max()),
        "reduction_ratio": round(1 - arr.mean() / n_pool, 6),
        "runtime_sec": round(elapsed, 2),
        "dba_alias_2x2": {
            "name_only": {"edges": dba_name_only, "pct": round(100 * dba_name_only / total_pos, 3)},
            "address_only": {"edges": dba_addr_only, "pct": round(100 * dba_addr_only / total_pos, 3)},
            "both": {"edges": dba_both, "pct": round(100 * dba_both / total_pos, 3)},
            "neither": {"edges": dba_neither, "pct": round(100 * dba_neither / total_pos, 3)},
        },
    }
    log(f"FULL VALIDATION full-union: recall={full_val_result['blocking_recall']} "
        f"avg_cand={full_val_result['avg_candidates']} p95={full_val_result['p95_candidates']} "
        f"max={full_val_result['max_candidates']} runtime={elapsed:.1f}s")
    log(f"DBA/alias 2x2 (full val): {full_val_result['dba_alias_2x2']}")

    with open(OUT_PATH, "w") as f:
        json.dump({"full_union_full_validation": full_val_result,
                    "neither_examples_s1_ids": neither_examples[:50],
                    "configs_used": cfg}, f, indent=2)
    log(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
