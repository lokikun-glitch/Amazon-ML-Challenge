"""
Stage 6, item 6: concrete miss analysis on positives unrecoverable by the
*maximal*-recall channel union (the same configuration that produced the
510-edge "neither" figure in the prior blocking-experiments run: B3 token
max_df=25000/K=2, B4 translit max_df=25000/K=2, B5 token max_df=25000/K=6,
B8 3-gram max_df=20000/K=12, B1/B2 exact, B7a exact, B7b exact UNCAPPED).

Reproducing this exact config (rather than a new one) is deliberate: these
510-ish edges are unrecoverable at the *recall ceiling* -- no smaller,
cheaper configuration explored in Stage 6 will do any better on them, so
they represent the current channel set's structural blind spot.

For every captured miss, resolves S1 and true-match records' raw + normalized
fields, records which channels were queried and confirms none hit, and prints
a classification aid (heuristic tags) for manual review.
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
OUT_PATH = os.path.join(REPORT_DIR, "miss_analysis.json")

SAMPLE_SEED = 42
SAMPLE_N = 50000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr)


def load_pool_column(columns):
    frames = []
    for label in ("train_source2", "train_source3"):
        p = os.path.join(CACHE_DIR, f"{label}.parquet")
        frames.append(pd.read_parquet(p, columns=columns))
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    log("Prep: pool ids, ground truth, validation sample (same seed=42 as before)")
    pool_base = load_pool_column(["entity_id", "country"])
    n_pool = len(pool_base)
    id_to_pos = pd.Series(np.arange(n_pool, dtype=np.int64), index=pool_base["entity_id"].values)
    del pool_base
    gc.collect()

    val_ids = set(pd.read_csv(os.path.join(SPLIT_DIR, "val_s1_ids.txt"), header=None)[0].astype(str))
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    gt = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    gt["match_list"] = gt["matched_entity_ids"].map(lambda x: x.split(",") if isinstance(x, str) and x else [])

    gt_pairs = {}
    gt_positions = {}
    for row in gt.itertuples():
        if not row.match_list:
            continue
        pairs = [(mid, int(id_to_pos[mid])) for mid in row.match_list if mid in id_to_pos.index]
        if pairs:
            gt_pairs[row.source1_entity_id] = pairs
            gt_positions[row.source1_entity_id] = frozenset(p for _, p in pairs)
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
    log(f"Sample ready: {len(s1_sample)} S1 rows")

    # ---- rebuild the exact maximal-recall config from the prior run ----
    log("Building B1/B2/B7a/B7b (uncapped) exact indexes...")
    col = load_pool_column(["country", "name_norm"])
    b1 = ExactIndex("B1").fit(col["country"], col["name_norm"])
    del col; gc.collect()
    col = load_pool_column(["country", "name_no_suffix"])
    b2 = ExactIndex("B2").fit(col["country"], col["name_no_suffix"])
    del col; gc.collect()
    col = load_pool_column(["country", "postal_code"])
    b7a = ExactIndex("B7a").fit(col["country"], col["postal_code"])
    del col; gc.collect()
    col = load_pool_column(["country", "street_number"])
    b7b = ExactIndex("B7b").fit(col["country"], col["street_number"])  # uncapped, matches original run
    del col; gc.collect()
    log(f"Exact indexes built. rss={rss_mb():.0f}MB")

    log("Building B3 name token (max_df=25000)...")
    col = load_pool_column(["country", "name_norm"])
    name_ti = TokenIndex("B3").fit(col["country"], col["name_norm"], min_df=1, max_df=25000)
    name_ti.release_raw()
    del col; gc.collect()

    log("Building B4 translit token (max_df=25000)...")
    col = load_pool_column(["country", "name_translit"])
    translit_ti = TokenIndex("B4").fit(col["country"], col["name_translit"], min_df=1, max_df=25000)
    translit_ti.release_raw()
    del col; gc.collect()

    log("Building B5 address token (max_df=25000)...")
    col = load_pool_column(["country", "address_norm"])
    addr_ti = TokenIndex("B5").fit(col["country"], col["address_norm"], min_df=1, max_df=25000)
    addr_ti.release_raw()
    del col; gc.collect()

    log("Building B8 3-gram (max_df=20000, matching the original 510-edge baseline)...")
    col = load_pool_column(["country", "name_norm"])
    ngram_col = build_ngram_series(col["name_norm"], n=3)
    b8_ti = TokenIndex("B8").fit(col["country"], ngram_col, min_df=1, max_df=20000)
    b8_ti.release_raw()
    del col, ngram_col; gc.collect()
    log(f"All maximal-config indexes built. rss={rss_mb():.0f}MB")

    def b8_cands(row):
        qs = char_ngram_string(row.name_norm, 3) if row.name_norm else ""
        return set(b8_ti.query(row.country, qs, max_query_tokens=12).tolist())

    def name_union_cands(row):
        return (set(b1.query(row.country, row.name_norm).tolist())
                | set(b2.query(row.country, row.name_no_suffix).tolist())
                | set(name_ti.query(row.country, row.name_norm, max_query_tokens=2).tolist())
                | set(translit_ti.query(row.country, row.name_translit, max_query_tokens=2).tolist())
                | b8_cands(row))

    def address_union_cands(row):
        return (set(addr_ti.query(row.country, row.address_norm, max_query_tokens=6).tolist())
                | set(b7a.query(row.country, row.postal_code).tolist())
                | set(b7b.query(row.country, row.street_number).tolist()))

    log("Querying full union over sample, capturing every missed positive edge...")
    t0 = time.time()
    misses = []  # (s1_id, missed_match_id)
    tp = 0
    total_pos = 0
    for row in s1_sample.itertuples():
        true_pos = gt_positions.get(row.entity_id)
        if not true_pos:
            continue
        total_pos += len(true_pos)
        cands = name_union_cands(row) | address_union_cands(row)
        tp += len(true_pos & cands)
        missed_positions = true_pos - cands
        if missed_positions:
            for mid, pos in gt_pairs[row.entity_id]:
                if pos in missed_positions:
                    misses.append((row.entity_id, mid))
    elapsed = time.time() - t0
    log(f"Done in {elapsed:.1f}s. recall={tp/total_pos:.4f} total_pos={total_pos} "
        f"n_missed_edges={len(misses)} (prior run reported 510 on this same sample/config)")

    # release heavy indexes before the (comparatively light) lookup/classification pass
    del b1, b2, b7a, b7b, name_ti, translit_ti, addr_ti, b8_ti
    gc.collect()
    log(f"Indexes released. rss={rss_mb():.0f}MB")

    # ---- resolve full record details for every missed edge ----
    log("Resolving raw + normalized fields for missed S1/S2/S3 records...")
    s1_ids_needed = {s1 for s1, _ in misses}
    match_ids_needed = {m for _, m in misses}

    s1_lookup = {}
    full_cols = ["entity_id", "country", "name_norm", "name_no_suffix", "name_translit",
                 "address_norm", "address_translit", "postal_code", "street_number"]
    s1_cache = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"), columns=full_cols)
    hit = s1_cache[s1_cache["entity_id"].isin(s1_ids_needed)]
    for r in hit.itertuples(index=False):
        s1_lookup[r.entity_id] = r._asdict()
    del s1_cache, hit
    gc.collect()

    match_lookup = {}
    for label in ("train_source2", "train_source3"):
        p = os.path.join(CACHE_DIR, f"{label}.parquet")
        chunk = pd.read_parquet(p, columns=full_cols)
        hit = chunk[chunk["entity_id"].isin(match_ids_needed)]
        for r in hit.itertuples(index=False):
            match_lookup[r.entity_id] = r._asdict()
        del chunk, hit
    gc.collect()

    # also need RAW (pre-normalization) name/address for human review -- read from source TSVs
    raw_lookup = {}
    needed_all = s1_ids_needed | match_ids_needed
    raw_files = [
        os.path.join(HERE, "..", "..", "..", "student_resource", "dataset", "train", "train_source1.tsv"),
        os.path.join(HERE, "..", "..", "..", "student_resource", "dataset", "train", "train_source2.tsv"),
        os.path.join(HERE, "..", "..", "..", "student_resource", "dataset", "train", "train_source3.tsv"),
    ]
    for p in raw_files:
        for chunk in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, na_values=[""], chunksize=500000):
            hit = chunk[chunk["entity_id"].isin(needed_all)]
            if len(hit):
                for r in hit.itertuples(index=False):
                    raw_lookup[r.entity_id] = {"business_name": r.business_name, "business_address": r.business_address}
            if len(raw_lookup) >= len(needed_all):
                break

    def classify(s1_rec, s1_raw, m_rec, m_raw):
        tags = []
        n1, n2 = s1_rec["name_norm"], m_rec["name_norm"]
        a1, a2 = s1_rec["address_norm"], m_rec["address_norm"]
        t1 = set(n1.split())
        t2 = set(n2.split())
        jac = len(t1 & t2) / len(t1 | t2) if (t1 or t2) else 1.0
        if jac < 0.15:
            tags.append("DBA/alias (name essentially unrelated)")
        if s1_rec["name_translit"] != n1 or m_rec["name_translit"] != n2:
            if s1_rec["name_translit"].split() and m_rec["name_translit"].split():
                tt1, tt2 = set(s1_rec["name_translit"].split()), set(m_rec["name_translit"].split())
                if len(tt1 & tt2) / len(tt1 | tt2 | {"_"}) > jac + 0.1:
                    tags.append("transliteration (translit tokens overlap more than original)")
        if not a1 or not a2:
            tags.append("missing address information")
        if a1 and a2 and set(a1.split()) & set(a2.split()) == set() and a1 != a2:
            tags.append("address corruption / no shared tokens")
        if n1 and n2 and n1 != n2 and jac >= 0.5:
            tags.append("minor name variation (typo/abbreviation/word-order)")
        if not tags:
            tags.append("other / needs manual review")
        return tags

    examples = []
    for s1_id, m_id in misses[:60]:
        s1_rec = s1_lookup.get(s1_id)
        m_rec = match_lookup.get(m_id)
        if not s1_rec or not m_rec:
            continue
        s1_raw = raw_lookup.get(s1_id, {})
        m_raw = raw_lookup.get(m_id, {})
        tags = classify(s1_rec, s1_raw, m_rec, m_raw)
        examples.append({
            "s1_id": s1_id, "match_id": m_id,
            "s1_raw_name": s1_raw.get("business_name"), "s1_raw_address": s1_raw.get("business_address"),
            "s1_country": s1_rec["country"],
            "match_raw_name": m_raw.get("business_name"), "match_raw_address": m_raw.get("business_address"),
            "match_country": m_rec["country"],
            "s1_name_norm": s1_rec["name_norm"], "match_name_norm": m_rec["name_norm"],
            "s1_address_norm": s1_rec["address_norm"], "match_address_norm": m_rec["address_norm"],
            "s1_name_translit": s1_rec["name_translit"], "match_name_translit": m_rec["name_translit"],
            "classification_tags": tags,
        })

    from collections import Counter
    tag_counts = Counter()
    for ex in examples:
        for t in ex["classification_tags"]:
            tag_counts[t] += 1

    report = {
        "sample_recall_at_maximal_config": round(tp / total_pos, 4),
        "n_positive_edges_in_sample": total_pos,
        "n_missed_edges": len(misses),
        "n_examples_classified": len(examples),
        "classification_tag_counts": dict(tag_counts),
        "examples": examples,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f"Wrote {OUT_PATH}: {len(misses)} missed edges, {len(examples)} classified examples")
    log(f"Tag counts: {dict(tag_counts)}")


if __name__ == "__main__":
    main()
