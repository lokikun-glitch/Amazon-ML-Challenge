"""
Pairwise feature engineering for the matcher-development candidate pairs
produced by generate_dev_candidates.py.

Cheap, interpretable first feature set only (per instruction: do not throw
every possible feature at the first model). Token-set computations are
memoized per distinct entity (S1 side and candidate side each reused across
many pairs), so the expensive part -- tokenizing/hashing -- happens once per
distinct record, not once per pair.
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "normalized")
PAIRS_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "dev_pairs")
FEAT_DIR = os.path.join(HERE, "..", "..", "..", "data_cache", "dev_features")

FULL_COLS = ["entity_id", "country", "name_norm", "name_no_suffix", "name_translit",
             "address_norm", "address_translit", "postal_code", "street_number"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}]  {msg}", file=sys.stderr)


def digit_tokens(s):
    return frozenset(t for t in s.split() if t.isdigit()) if s else frozenset()


def tokset(s):
    return frozenset(s.split()) if s else frozenset()


def _pairwise(scorer, a, b):
    """Element-wise rapidfuzz score / 100 as float32. cpdist (C++, multithreaded) computes the
    same scorer in double precision as the per-pair fuzz.* calls it replaces; equality with the
    original per-pair loop is asserted by verify_production_equivalence.py."""
    if len(a) == 0:
        return np.zeros(0, dtype=np.float32)
    return (process.cpdist(list(a), list(b), scorer=scorer, dtype=np.float64, workers=-1) / 100.0).astype(np.float32)


def compute_pair_features(s1_fields, cand_fields, channels, log=lambda m: None):
    """The 28 D3 pair features, in stored column order. s1_fields / cand_fields: per-pair
    frames with FULL_COLS fields (row-aligned); channels: per-pair from_* flag columns."""
    t0 = time.time()
    feat = pd.DataFrame(index=channels.index)

    n1 = s1_fields["name_norm"].values
    n2 = cand_fields["name_norm"].values
    ns1 = s1_fields["name_no_suffix"].values
    ns2 = cand_fields["name_no_suffix"].values
    nt1 = s1_fields["name_translit"].values
    nt2 = cand_fields["name_translit"].values
    a1 = s1_fields["address_norm"].values
    a2 = cand_fields["address_norm"].values
    pc1 = s1_fields["postal_code"].values
    pc2 = cand_fields["postal_code"].values
    sn1 = s1_fields["street_number"].values
    sn2 = cand_fields["street_number"].values
    c1 = s1_fields["country"].values
    c2 = cand_fields["country"].values

    # ---- NAME features ----
    feat["name_exact"] = (n1 == n2).astype(np.int8)
    feat["name_nosuffix_exact"] = (ns1 == ns2).astype(np.int8)

    # memoized token sets: distinct strings only, not per-pair
    uniq_names = set(n1.tolist()) | set(n2.tolist())
    tok_cache = {s: tokset(s) for s in uniq_names}

    def jaccard_overlap_containment(strs1, strs2, cache):
        jac = np.zeros(len(strs1), dtype=np.float32)
        overlap = np.zeros(len(strs1), dtype=np.int16)
        contain = np.zeros(len(strs1), dtype=np.float32)
        for i in range(len(strs1)):
            t1, t2 = cache[strs1[i]], cache[strs2[i]]
            inter = len(t1 & t2)
            union = len(t1 | t2)
            jac[i] = inter / union if union else 1.0
            overlap[i] = inter
            m = min(len(t1), len(t2))
            contain[i] = inter / m if m else (1.0 if union == 0 else 0.0)
        return jac, overlap, contain

    jac, ov, contain = jaccard_overlap_containment(n1, n2, tok_cache)
    feat["name_token_jaccard"] = jac
    feat["name_token_overlap_count"] = ov
    feat["name_token_containment"] = contain

    log(f"  name token features done ({time.time()-t0:.1f}s)")
    t0 = time.time()

    feat["name_fuzz_ratio"] = _pairwise(fuzz.ratio, n1, n2)
    feat["name_fuzz_token_sort_ratio"] = _pairwise(fuzz.token_sort_ratio, n1, n2)
    feat["name_fuzz_token_set_ratio"] = _pairwise(fuzz.token_set_ratio, n1, n2)
    feat["name_translit_fuzz_ratio"] = _pairwise(fuzz.ratio, nt1, nt2)
    feat["name_len_diff"] = np.abs(np.array([len(x) for x in n1]) - np.array([len(x) for x in n2])).astype(np.int16)
    log(f"  name rapidfuzz features done ({time.time()-t0:.1f}s)")
    t0 = time.time()

    # ---- ADDRESS features ----
    both_addr_present = (a1 != "") & (a2 != "")
    feat["address_exact"] = ((a1 == a2) & both_addr_present).astype(np.int8)
    feat["address_missing_s1"] = (a1 == "").astype(np.int8)
    feat["address_missing_cand"] = (a2 == "").astype(np.int8)

    uniq_addrs = set(a1.tolist()) | set(a2.tolist())
    addr_tok_cache = {s: tokset(s) for s in uniq_addrs}
    ajac, aov, acontain = jaccard_overlap_containment(a1, a2, addr_tok_cache)
    feat["address_token_jaccard"] = ajac
    feat["address_token_overlap_count"] = aov
    feat["address_token_containment"] = acontain

    digit_cache = {s: digit_tokens(s) for s in uniq_addrs}
    num_overlap = np.zeros(len(a1), dtype=np.int16)
    for i in range(len(a1)):
        num_overlap[i] = len(digit_cache[a1[i]] & digit_cache[a2[i]])
    feat["address_numeric_token_overlap"] = num_overlap

    feat["street_number_equal"] = ((sn1 == sn2) & (sn1 != "") & (sn2 != "")).astype(np.int8)
    feat["postal_equal"] = ((pc1 == pc2) & (pc1 != "") & (pc2 != "")).astype(np.int8)

    feat["address_fuzz_ratio"] = _pairwise(fuzz.ratio, a1, a2)
    log(f"  address features done ({time.time()-t0:.1f}s)")

    # ---- CROSS features ----
    feat["country_equal"] = (c1 == c2).astype(np.int8)  # invariant check: should be ~100% by construction
    feat["name_addr_product"] = feat["name_fuzz_ratio"] * feat["address_fuzz_ratio"]
    for c in channels.columns:
        feat[c] = channels[c].values
    feat["n_channels_hit"] = channels.sum(axis=1).astype(np.int8)
    return feat



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["D3", "C"], required=True)
    ap.add_argument("--split", choices=["train", "val", "val2"], required=True)
    args = ap.parse_args()

    os.makedirs(FEAT_DIR, exist_ok=True)
    pairs_path = os.path.join(PAIRS_DIR, f"{args.config}_{args.split}.parquet")
    pairs = pd.read_parquet(pairs_path)
    log(f"Loaded {len(pairs)} pairs from {pairs_path}")

    channel_cols = [c for c in pairs.columns if c.startswith("from_")]

    # ---- resolve S1-side fields ----
    s1_ids_needed = set(pairs["s1_id"].unique().tolist())
    s1_cache = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"), columns=FULL_COLS)
    s1_lookup_df = s1_cache[s1_cache["entity_id"].isin(s1_ids_needed)].set_index("entity_id")
    del s1_cache
    gc.collect()
    log(f"Resolved {len(s1_lookup_df)} distinct S1 records")

    # ---- resolve candidate-side fields via the saved cand_pos->entity_id lookup, then pool ----
    lookup_name = f"{args.config}_val2_cand_id_lookup.parquet" if args.split == "val2" else f"{args.config}_cand_id_lookup.parquet"
    lookup_path = os.path.join(PAIRS_DIR, lookup_name)
    cand_id_lookup = pd.read_parquet(lookup_path).set_index("cand_pos")["entity_id"]
    cand_ids_needed = set(cand_id_lookup.loc[pairs["cand_pos"].unique()].tolist())

    cand_lookup_frames = []
    for label in ("train_source2", "train_source3"):
        p = os.path.join(CACHE_DIR, f"{label}.parquet")
        chunk = pd.read_parquet(p, columns=FULL_COLS)
        hit = chunk[chunk["entity_id"].isin(cand_ids_needed)]
        if len(hit):
            cand_lookup_frames.append(hit)
        del chunk
        gc.collect()
    cand_lookup_df = pd.concat(cand_lookup_frames, ignore_index=True).set_index("entity_id")
    del cand_lookup_frames
    gc.collect()
    log(f"Resolved {len(cand_lookup_df)} distinct candidate records")

    # ---- build the merged pair table (s1_* and cand_* columns) ----
    pairs = pairs.copy()
    pairs["cand_id"] = cand_id_lookup.loc[pairs["cand_pos"].values].values

    s1_fields = s1_lookup_df.loc[pairs["s1_id"].values].reset_index(drop=True)
    cand_fields = cand_lookup_df.loc[pairs["cand_id"].values].reset_index(drop=True)
    del s1_lookup_df, cand_lookup_df
    gc.collect()

    log("Computing features...")
    feat = compute_pair_features(s1_fields, cand_fields, pairs[channel_cols], log=log)

    feat["s1_id"] = pairs["s1_id"].values
    feat["cand_id"] = pairs["cand_id"].values
    feat["label"] = pairs["label"].values
    feat["country"] = c1

    out_path = os.path.join(FEAT_DIR, f"{args.config}_{args.split}_features.parquet")
    feat.to_parquet(out_path)
    log(f"Wrote {len(feat)} rows x {feat.shape[1]} cols -> {out_path}")
    log(f"country_equal invariant check: {feat['country_equal'].mean():.6f} (should be 1.0)")


if __name__ == "__main__":
    main()
