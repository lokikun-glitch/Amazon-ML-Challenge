"""
Pairwise address-number agreement / contradiction features for the EXISTING
D3 candidate pairs (no blocking or candidate change).

Reads the row order of data_cache/dev_features/{config}_{split}_features.parquet
and writes data_cache/dev_features/{config}_{split}_numfeat.parquet with the new
columns in exactly that order (plus s1_id/cand_id for an alignment check), so
the existing 28 features are reused byte-for-byte and the baseline is untouched.

Per-entity numbers come from data_cache/address_numbers (address_numbers.py).

AGREEMENT features (variant C and B)
  num_hn_head_equal       both have a house number and the leading numeric atom is equal
  num_hn_full_equal       both have a house number and the canonical expression is equal
  num_compound_equal      both have a compound house number (>=2 atoms) and they are equal
  num_unit_equal          both have a unit and it is equal
  num_postal_equal        both have a postal code (PIN) and it is equal
  num_token_jaccard       Jaccard of numeric-atom sets (0 if either side has none)
  num_token_overlap       |shared numeric atoms|, clipped at 5
CONTRADICTION features (variant B only)
  num_hn_head_mismatch    both have a house number and the leading atoms differ
  num_compound_mismatch   both have a compound house number and they differ (26/244 vs 26/248)
  num_unit_mismatch       both have a unit and it differs
  num_postal_mismatch     both have a postal code and it differs
  num_token_contradiction both have numeric atoms and each side has one the other lacks
                          (adding a unit/PMB/range on one side is NOT a contradiction)
  num_token_symdiff       |symmetric difference| of atom sets (0 if either empty), clipped at 5
  num_strongname_contra   name_fuzz_token_set_ratio >= 0.90 AND (head mismatch OR compound
                          mismatch OR token contradiction) -- the forensic failure mode.
                          0.90 is a fixed prior, not tuned.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from blocking import rss_mb

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
FEAT_DIR = os.path.join(ROOT, "data_cache", "dev_features")
NUM_DIR = os.path.join(ROOT, "data_cache", "address_numbers")
REPORT_DIR = os.path.join(ROOT, "reports")

AGREEMENT = ["num_hn_head_equal", "num_hn_full_equal", "num_compound_equal", "num_unit_equal",
             "num_postal_equal", "num_token_jaccard", "num_token_overlap"]
CONTRADICTION = ["num_hn_head_mismatch", "num_compound_mismatch", "num_unit_mismatch", "num_postal_mismatch",
                 "num_token_contradiction", "num_token_symdiff", "num_strongname_contra"]
STRONG_NAME = 0.90


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr, flush=True)


def load_numbers(ids, files):
    parts = []
    for f in files:
        d = pd.read_parquet(os.path.join(NUM_DIR, f))
        parts.append(d[d["entity_id"].isin(ids)])
    return pd.concat(parts).set_index("entity_id")


def compute_number_features(ns, nc, s1_idx, c_idx, name_ts, log=lambda m: None):
    """The 14 number features. ns / nc: per-entity address_numbers frames (FIELDS columns,
    "" for missing) for the S1 side and candidate side; s1_idx / c_idx: per-pair row
    positions into ns / nc; name_ts: per-pair name_fuzz_token_set_ratio.
    Returns {feature: array} for AGREEMENT + CONTRADICTION."""
    n = len(s1_idx)
    # shared integer codes per field so equality is a vectorized int compare ("" -> -1)
    def codes(field):
        u = pd.Index(pd.unique(np.concatenate([ns[field].values, nc[field].values])))
        a = u.get_indexer(ns[field].values)
        b = u.get_indexer(nc[field].values)
        a[ns[field].values == ""] = -1
        b[nc[field].values == ""] = -1
        return a[s1_idx], b[c_idx]

    out = {}
    for field, eq, mm in (("house_head", "num_hn_head_equal", "num_hn_head_mismatch"),
                          ("primary_house_number", "num_hn_full_equal", None),
                          ("compound_number", "num_compound_equal", "num_compound_mismatch"),
                          ("unit_number", "num_unit_equal", "num_unit_mismatch"),
                          ("postal_code", "num_postal_equal", "num_postal_mismatch")):
        a, b = codes(field)
        both = (a >= 0) & (b >= 0)
        out[eq] = (both & (a == b)).astype(np.int8)
        if mm:
            out[mm] = (both & (a != b)).astype(np.int8)
    log("exact-field features done")

    tok_s = [frozenset(x.split()) for x in ns["number_tokens"].values]
    tok_c = [frozenset(x.split()) for x in nc["number_tokens"].values]
    jac = np.zeros(n, dtype=np.float32)
    ov = np.zeros(n, dtype=np.int8)
    contra = np.zeros(n, dtype=np.int8)
    sd = np.zeros(n, dtype=np.int8)
    for i in range(n):
        A = tok_s[s1_idx[i]]
        B = tok_c[c_idx[i]]
        if A and B:
            k = len(A & B)
            u = len(A) + len(B) - k
            jac[i] = k / u
            ov[i] = min(k, 5)
            sd[i] = min(u - k, 5)
            contra[i] = (k < len(A)) and (k < len(B))
    out["num_token_jaccard"] = jac
    out["num_token_overlap"] = ov
    out["num_token_contradiction"] = contra
    out["num_token_symdiff"] = sd
    strong = name_ts >= STRONG_NAME
    out["num_strongname_contra"] = (strong & ((out["num_hn_head_mismatch"] == 1) | (out["num_compound_mismatch"] == 1)
                                              | (contra == 1))).astype(np.int8)
    log("token features done")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="D3", choices=["D3"])
    ap.add_argument("--split", required=True, choices=["train", "val", "val2"])
    args = ap.parse_args()
    t0 = time.time()
    src = os.path.join(FEAT_DIR, f"{args.config}_{args.split}_features.parquet")
    t = pq.read_table(src, columns=["s1_id", "cand_id", "name_fuzz_token_set_ratio"],
                      read_dictionary=["s1_id", "cand_id"])
    s1_col = t.column("s1_id").combine_chunks()
    c_col = t.column("cand_id").combine_chunks()
    s1_vals = s1_col.dictionary.to_pylist()
    c_vals = c_col.dictionary.to_pylist()
    s1_idx = s1_col.indices.to_numpy()
    c_idx = c_col.indices.to_numpy()
    name_ts = t.column("name_fuzz_token_set_ratio").to_numpy()
    n = len(s1_idx)
    log(f"{n} pairs, {len(s1_vals)} S1, {len(c_vals)} candidates")

    ns = load_numbers(set(s1_vals), ["train_source1.parquet"]).reindex(s1_vals).fillna("")
    nc = load_numbers(set(c_vals), ["train_source2.parquet", "train_source3.parquet"]).reindex(c_vals).fillna("")
    assert (ns.index == s1_vals).all() and (nc.index == c_vals).all()

    out = compute_number_features(ns, nc, s1_idx, c_idx, name_ts, log=log)
    df = pd.DataFrame({c: out[c] for c in AGREEMENT + CONTRADICTION})
    df["s1_id"] = pd.Categorical.from_codes(s1_idx, categories=s1_vals)
    df["cand_id"] = pd.Categorical.from_codes(c_idx, categories=c_vals)
    path = os.path.join(FEAT_DIR, f"{args.config}_{args.split}_numfeat.parquet")
    df.to_parquet(path, index=False)
    stats = {"rows": n, "seconds": round(time.time() - t0, 1), "peak_rss_mb_parent": round(rss_mb()),
             "file_mb": round(os.path.getsize(path) / 1e6, 1),
             "means": {c: round(float(df[c].mean()), 4) for c in AGREEMENT + CONTRADICTION}}
    with open(os.path.join(REPORT_DIR, f"number_features_{args.config}_{args.split}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log(f"wrote {path}: {stats}")


if __name__ == "__main__":
    main()
