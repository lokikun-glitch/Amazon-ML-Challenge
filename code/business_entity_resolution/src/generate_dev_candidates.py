"""
Matcher-development candidate generation.

Generates labeled candidate pairs WITH per-channel provenance for a given
blocking config (D3 or C), on a fixed-size S1 sample -- separately for the
TRAIN split (train_s1_ids.txt, used to fit the classifier) and the VAL split
(val_s1_ids.txt, used for entity-level macro F0.5 evaluation). This is a
development dataset, not the full competition-scale candidate_pairs.tsv
(D3 alone would be ~2.09B pairs at full test scale) -- that stays
unmaterialized per the explicit instruction not to generate it yet.

Sample size is reduced from the 50K blocking-development sample to 10,000 S1
per side (train and val) so that total pairs stay tractable for the first
feature-engineering + model pass: D3 (~1,207 avg candidates) -> ~12M pairs
per side; C (~3,180 avg) -> ~32M pairs per side. Flagged as a scale choice,
not a silent shortcut -- rerun at 50K once the approach is validated.

Candidate identity is stored as an integer position into the concatenated
train_source2+train_source3 pool (int64), not the raw entity_id string, to
keep the pairs table compact; a small entity_id lookup is saved separately
for only the distinct positions actually touched.
"""
import argparse
import gc
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
OUT_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "dev_pairs")

CONFIGS = {
    "D3": {"b3_maxdf": 500, "b5_maxdf": 2000, "b3_k": 4, "b5_k": 6, "b7b_maxdf": None, "b8_4g_maxdf": None,
           "channels": ["b1", "b2", "b3", "b5", "b7a"]},
    "C": {"b3_maxdf": 3000, "b5_maxdf": 3000, "b3_k": 4, "b5_k": 6, "b7b_maxdf": 5000, "b8_4g_maxdf": 500,
          "channels": ["b1", "b2", "b3", "b5", "b7a", "b7b", "b8"]},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr)


def load_pool_column(columns, prefix="train", country=None):
    """S2+S3 pool columns in fixed (source2, source3, file-row) order. With `country`, only that
    country's rows (positions then index the country-filtered pool). Every D3 index key already
    contains the country, and max_df is counted per (country, key), so a per-country index returns
    exactly the same candidate records as the all-country index."""
    frames = []
    for label in (f"{prefix}_source2", f"{prefix}_source3"):
        p = os.path.join(CACHE_DIR, f"{label}.parquet")
        cols = columns if (country is None or "country" in columns) else columns + ["country"]
        d = pd.read_parquet(p, columns=cols)
        if country is not None:
            d = d[d["country"] == country]
            if "country" not in columns:
                d = d.drop(columns="country")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def build_indexes(cfg, prefix="train", country=None):
    idx = {}

    def pool_col(columns):
        return load_pool_column(columns, prefix=prefix, country=country)

    col = pool_col(["country", "name_norm"])
    idx["b1"] = ExactIndex("B1").fit(col["country"], col["name_norm"])
    del col; gc.collect()
    col = pool_col(["country", "name_no_suffix"])
    idx["b2"] = ExactIndex("B2").fit(col["country"], col["name_no_suffix"])
    del col; gc.collect()
    col = pool_col(["country", "name_norm"])
    idx["b3"] = TokenIndex("B3").fit(col["country"], col["name_norm"], min_df=1, max_df=cfg["b3_maxdf"])
    idx["b3"].release_raw()
    del col; gc.collect()
    col = pool_col(["country", "address_norm"])
    idx["b5"] = TokenIndex("B5").fit(col["country"], col["address_norm"], min_df=1, max_df=cfg["b5_maxdf"])
    idx["b5"].release_raw()
    del col; gc.collect()
    col = pool_col(["country", "postal_code"])
    idx["b7a"] = ExactIndex("B7a").fit(col["country"], col["postal_code"])
    del col; gc.collect()
    if cfg["b7b_maxdf"] is not None:
        col = pool_col(["country", "street_number"])
        idx["b7b"] = ExactIndex("B7b").fit(col["country"], col["street_number"], max_df=cfg["b7b_maxdf"])
        del col; gc.collect()
    if cfg["b8_4g_maxdf"] is not None:
        col = pool_col(["country", "name_norm"])
        ngram_col = build_ngram_series(col["name_norm"], n=4)
        idx["b8"] = TokenIndex("B8").fit(col["country"], ngram_col, min_df=1, max_df=cfg["b8_4g_maxdf"])
        idx["b8"].release_raw()
        del col, ngram_col; gc.collect()
    return idx


def per_channel_candidates(idx, cfg, row):
    out = {
        "b1": set(idx["b1"].query(row.country, row.name_norm).tolist()),
        "b2": set(idx["b2"].query(row.country, row.name_no_suffix).tolist()),
        "b3": set(idx["b3"].query(row.country, row.name_norm, max_query_tokens=cfg["b3_k"]).tolist()),
        "b5": set(idx["b5"].query(row.country, row.address_norm, max_query_tokens=cfg["b5_k"]).tolist()),
        "b7a": set(idx["b7a"].query(row.country, row.postal_code).tolist()),
    }
    if "b7b" in idx:
        out["b7b"] = set(idx["b7b"].query(row.country, row.street_number).tolist())
    if "b8" in idx:
        qs = char_ngram_string(row.name_norm, 4) if row.name_norm else ""
        out["b8"] = set(idx["b8"].query(row.country, qs, max_query_tokens=12).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["D3", "C"], required=True)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val2-seed", type=int, default=None,
                    help="generate ONLY a second untouched val sample: n_val S1 drawn with this seed from "
                         "val_s1_ids minus the first val sample (seed=--seed)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = CONFIGS[args.config]
    channels = cfg["channels"]

    log("Loading pool ids + ground truth...")
    pool_base = load_pool_column(["entity_id"])
    n_pool = len(pool_base)
    id_to_pos = pd.Series(np.arange(n_pool, dtype=np.int64), index=pool_base["entity_id"].values)
    pos_to_id = pool_base["entity_id"].values
    del pool_base
    gc.collect()

    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
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

    train_ids_all = set(pd.read_csv(os.path.join(SPLIT_DIR, "train_s1_ids.txt"), header=None)[0].astype(str))
    val_ids_all = set(pd.read_csv(os.path.join(SPLIT_DIR, "val_s1_ids.txt"), header=None)[0].astype(str))
    s1_full = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"))

    def sample_split(ids_all, n, seed):
        s1 = s1_full[s1_full["entity_id"].isin(ids_all)].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(s1), size=min(n, len(s1)), replace=False)
        return s1.iloc[idx].reset_index(drop=True)

    s1_train = sample_split(train_ids_all, args.n_train, args.seed)
    s1_val = sample_split(val_ids_all, args.n_val, args.seed)
    if args.val2_seed is not None:
        s1_val = sample_split(val_ids_all - set(s1_val["entity_id"]), args.n_val, args.val2_seed)
        s1_train = None
    del s1_full
    gc.collect()
    if s1_train is not None:
        log(f"train sample={len(s1_train)} (from train_s1_ids), val sample={len(s1_val)} (from val_s1_ids)")
    else:
        log(f"val2 sample={len(s1_val)} (from val_s1_ids minus first val sample, seed={args.val2_seed})")

    log(f"Building indexes for config {args.config}: {cfg}")
    idx = build_indexes(cfg)
    log(f"indexes built. rss={rss_mb():.0f}MB")

    def generate(s1_df, split_name):
        s1_ids_out, cand_pos_out, labels_out = [], [], []
        flag_cols = {c: [] for c in channels}
        n_pos_total = 0
        t0 = time.time()
        for i, row in enumerate(s1_df.itertuples()):
            per_chan = per_channel_candidates(idx, cfg, row)
            union = set()
            for c in channels:
                union |= per_chan[c]
            true_pos = gt_positions.get(row.entity_id, frozenset())
            n_pos_total += len(true_pos)
            for pos in union:
                s1_ids_out.append(row.entity_id)
                cand_pos_out.append(pos)
                labels_out.append(1 if pos in true_pos else 0)
                for c in channels:
                    flag_cols[c].append(1 if pos in per_chan[c] else 0)
            if (i + 1) % 2000 == 0:
                log(f"  {split_name}: {i+1}/{len(s1_df)} S1 processed, {len(s1_ids_out)} pairs so far")
        df = pd.DataFrame({
            "s1_id": s1_ids_out,
            "cand_pos": np.array(cand_pos_out, dtype=np.int64),
            "label": np.array(labels_out, dtype=np.int8),
        })
        for c in channels:
            df[f"from_{c}"] = np.array(flag_cols[c], dtype=np.int8)
        out_path = os.path.join(OUT_DIR, f"{args.config}_{split_name}.parquet")
        df.to_parquet(out_path)
        log(f"{split_name}: {len(df)} pairs, {int(df['label'].sum())} positive "
            f"({100*df['label'].mean():.4f}% positive rate), {n_pos_total} true edges available "
            f"({time.time()-t0:.1f}s) -> wrote {out_path}")
        return df

    if args.val2_seed is not None:
        df_val = generate(s1_val, "val2")
        touched = sorted(set(df_val["cand_pos"].tolist()))
        lookup_path = os.path.join(OUT_DIR, f"{args.config}_val2_cand_id_lookup.parquet")
    else:
        df_train = generate(s1_train, "train")
        df_val = generate(s1_val, "val")
        touched = sorted(set(df_train["cand_pos"].tolist()) | set(df_val["cand_pos"].tolist()))
        lookup_path = os.path.join(OUT_DIR, f"{args.config}_cand_id_lookup.parquet")
    touched_arr = np.array(touched, dtype=np.int64)
    touched_ids = pos_to_id[touched_arr]
    pd.DataFrame({"cand_pos": touched_arr, "entity_id": touched_ids}).to_parquet(lookup_path)
    log(f"Saved cand_pos->entity_id lookup for {len(touched)} distinct positions -> {lookup_path}")
    log("Candidate generation complete.")


if __name__ == "__main__":
    main()
