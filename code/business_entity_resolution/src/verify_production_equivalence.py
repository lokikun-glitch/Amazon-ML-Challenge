"""
Proves the production components reproduce the frozen dev artifacts (no labels are used to
choose anything; this only compares values). Run with PYTHONHASHSEED=0.

V1  compute_pair_features (refactored, rapidfuzz cpdist)  == stored D3_val_features (first N rows)
V2  compute_number_features (refactored, chunk-local)     == stored D3_val_numfeat  (first N rows)
V3  production load_model/score from B_frozen.npz          == stored D3B_num_val2_z  (all val2 rows)
V4  production decide()                                     == tune_decision_rules_B H2 acceptance on val2
V5  per-country D3 indexes == all-country D3 indexes (same hash seed) on training-side S1s,
    plus agreement with the stored dev candidate pairs (generated without a fixed hash seed)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_number_features import AGREEMENT, CONTRADICTION, compute_number_features
from build_pairwise_features import FULL_COLS, compute_pair_features
from production_inference import decide, h2_ok, load_model, log, score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
CACHE = os.path.join(ROOT, "data_cache")
REPORT = os.path.join(ROOT, "reports", "production_equivalence.json")
N = 300_000
N_S1_V5 = 1000


def head_rows(path, n, columns=None):
    pf = pq.ParquetFile(path)
    out, got = [], 0
    for b in pf.iter_batches(batch_size=100_000, columns=columns):
        out.append(b)
        got += b.num_rows
        if got >= n:
            break
    return pa.Table.from_batches(out).slice(0, n).to_pandas()


def v1_v2():
    pairs = head_rows(os.path.join(CACHE, "dev_pairs", "D3_val.parquet"), N)
    lookup = pd.read_parquet(os.path.join(CACHE, "dev_pairs", "D3_cand_id_lookup.parquet")).set_index("cand_pos")["entity_id"]
    pairs["cand_id"] = lookup.loc[pairs["cand_pos"].values].values
    s1 = pd.read_parquet(os.path.join(CACHE, "normalized", "train_source1.parquet"), columns=FULL_COLS)
    s1 = s1[s1["entity_id"].isin(set(pairs["s1_id"]))].set_index("entity_id")
    cand = pd.concat([pd.read_parquet(os.path.join(CACHE, "normalized", f"train_source{k}.parquet"), columns=FULL_COLS)
                      for k in (2, 3)])
    cand = cand[cand["entity_id"].isin(set(pairs["cand_id"]))].set_index("entity_id")
    s1_fields = s1.loc[pairs["s1_id"].values].reset_index(drop=True)
    cand_fields = cand.loc[pairs["cand_id"].values].reset_index(drop=True)
    channel_cols = [c for c in pairs.columns if c.startswith("from_")]
    t0 = time.time()
    feat = compute_pair_features(s1_fields, cand_fields, pairs[channel_cols].reset_index(drop=True))
    t28 = time.time() - t0
    stored = head_rows(os.path.join(CACHE, "dev_features", "D3_val_features.parquet"), N)
    assert (stored["cand_id"].astype(str).values == pairs["cand_id"].values).all()
    cols28 = [c for c in stored.columns if c not in ("s1_id", "cand_id", "label", "country")]
    assert list(feat.columns) == cols28, "column order differs"
    mism28 = {c: int((feat[c].values != stored[c].values).sum()) for c in cols28}

    # number features, chunk-local entity frames exactly as production uses them
    n1 = pd.read_parquet(os.path.join(CACHE, "address_numbers", "train_source1.parquet")).set_index("entity_id")
    nn = pd.concat([pd.read_parquet(os.path.join(CACHE, "address_numbers", f"train_source{k}.parquet")) for k in (2, 3)])
    nn = nn[nn["entity_id"].isin(set(pairs["cand_id"]))].set_index("entity_id")
    us, s_inv = np.unique(pairs["s1_id"].values, return_inverse=True)
    uc, c_inv = np.unique(pairs["cand_id"].values, return_inverse=True)
    t0 = time.time()
    num = compute_number_features(n1.reindex(us).fillna("").reset_index(drop=True),
                                  nn.reindex(uc).fillna("").reset_index(drop=True),
                                  s_inv, c_inv, feat["name_fuzz_token_set_ratio"].values)
    t14 = time.time() - t0
    stored_n = head_rows(os.path.join(CACHE, "dev_features", "D3_val_numfeat.parquet"), N)
    mism14 = {c: int((num[c] != stored_n[c].values).sum()) for c in AGREEMENT + CONTRADICTION}
    return {"rows": N, "mismatches_28": mism28, "mismatches_14": mism14, "seconds_28": round(t28, 1),
            "seconds_14": round(t14, 1), "pass": sum(mism28.values()) == 0 and sum(mism14.values()) == 0}


def v3_v4():
    from number_feature_experiment import load_split
    from tune_decision_rules_B import PART_COLS, RulePart
    from evaluate_val2 import val2_s1_table
    model = load_model()
    va2 = load_split("val2")
    z = score(model, va2[model["cols"]].values.astype(np.float32))
    z_stored = pd.read_parquet(os.path.join(CACHE, "dev_scores", "D3B_num_val2_z.parquet"))["z_val2"].values
    d = np.abs(z - z_stored)
    near = sum(int((np.abs(z_stored - t) <= 1e-9).sum()) for t in (model["t"], model["t_low"]))
    v3 = {"rows": len(z), "max_abs_diff": float(d.max()), "pairs_differing": int((d > 0).sum()),
          "pairs_within_1e-9_of_a_threshold": near,
          "criterion": "float summation-order noise only: max diff < 1e-9 and no pair near 8.3 / 3.0"}
    v3["pass"] = v3["max_abs_diff"] < 1e-9 and near == 0

    # V4: production decide vs RulePart acceptance (the frozen val2 evaluation logic)
    ok = h2_ok(va2).values
    rows = va2[["s1_id", "cand_id", "label"] + PART_COLS].copy()
    rows["z"] = z
    rows["num_ok"] = ok
    v2 = val2_s1_table()
    rp = RulePart(v2.reset_index(drop=True), rows)
    acc = rp.accepted("H2", model["t_low"])
    ref = {}
    for s, c in zip(rp.p.r_s[acc], rp.p.r_cand[acc]):
        ref.setdefault(rp.p.s1_ids[s], set()).add(c)
    order = np.argsort(va2["s1_id"].values, kind="stable")
    s_sorted = va2["s1_id"].values[order]
    starts = np.r_[0, np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1, len(s_sorted)]
    preds = decide(starts, z[order], va2["cand_id"].values[order], ok[order], model["t"], model["t_low"])
    prod = {s_sorted[starts[i]]: set(p) for i, p in enumerate(preds) if p}
    all_s1 = set(ref) | set(prod)
    diff = [s for s in all_s1 if ref.get(s, set()) != prod.get(s, set())]
    # a difference is acceptable only if it is an exact top-z tie between H2-eligible candidates
    zc = pd.DataFrame({"s1": va2["s1_id"].values, "z": z, "ok": ok})
    tie_only = []
    for s in diff:
        g = zc[zc["s1"] == s]
        top = g[g["z"] == g["z"].max()]
        tie_only.append(len(top) >= 2 and top["ok"].all() and len(ref.get(s, ())) == len(prod.get(s, ())) == 1)
    v4 = {"s1_compared": len(set(s_sorted)), "s1_with_predictions": len(prod), "s1_differing": len(diff),
          "differing_all_exact_top_z_ties": all(tie_only), "examples": diff[:5],
          "criterion": "identical except exact top-z ties (production tie-break: smallest entity_id)",
          "pass": all(tie_only)}
    return v3, v4


def v5():
    from generate_dev_candidates import CONFIGS, build_indexes, load_pool_column, per_channel_candidates
    cfg = CONFIGS["D3"]
    s1tab = pd.read_parquet(os.path.join(CACHE, "dev_scores", "D3_s1_table.parquet"))
    ids = s1tab.loc[s1tab["part"] == "val", "entity_id"].values[:N_S1_V5]
    s1 = pd.read_parquet(os.path.join(CACHE, "normalized", "train_source1.parquet"))
    s1 = s1.set_index("entity_id").loc[ids].reset_index()

    def run(idx, pool_ids):
        out = {}
        for row in s1.itertuples(index=False):
            per = per_channel_candidates(idx, cfg, row)
            u = set()
            for c in cfg["channels"]:
                u |= per[c]
            out[row.entity_id] = {pool_ids[p] for p in u}
        return out

    t0 = time.time()
    idx = build_indexes(cfg, prefix="train")
    glob = run(idx, load_pool_column(["entity_id"], prefix="train")["entity_id"].values)
    del idx
    log(f"V5 global done ({time.time() - t0:.0f}s)")
    per_c = {}
    for c in sorted(s1["country"].unique()):
        idx = build_indexes(cfg, prefix="train", country=c)
        pids = load_pool_column(["entity_id"], prefix="train", country=c)["entity_id"].values
        sub = s1[s1["country"] == c]
        per_c.update({k: v for k, v in run(idx, pids).items() if k in set(sub["entity_id"])})
        del idx
    diff_pc = sum(glob[s] != per_c[s] for s in ids)
    # stored dev candidates (generated without fixed hash seed)
    dev = pd.read_parquet(os.path.join(CACHE, "dev_pairs", "D3_val.parquet"), columns=["s1_id", "cand_pos"],
                          filters=[("s1_id", "in", list(ids))])
    lookup = pd.read_parquet(os.path.join(CACHE, "dev_pairs", "D3_cand_id_lookup.parquet")).set_index("cand_pos")["entity_id"]
    dev["cand_id"] = lookup.loc[dev["cand_pos"].values].values
    devset = dev.groupby("s1_id")["cand_id"].agg(set).to_dict()
    same_dev = sum(glob[s] == devset.get(s, set()) for s in ids)
    sym = sum(len(glob[s] ^ devset.get(s, set())) for s in ids)
    tot = sum(len(glob[s]) for s in ids)
    return {"n_s1": len(ids), "per_country_vs_global_s1_differing": int(diff_pc),
            "stored_dev_identical_s1": int(same_dev), "stored_dev_symdiff_pairs": int(sym),
            "global_pairs": int(tot), "pass": diff_pc == 0}


def main():
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
    rep = {}
    rep["V1_V2_features"] = v1_v2()
    log(f"V1/V2: {rep['V1_V2_features']}")
    rep["V3_scores"], rep["V4_decision"] = v3_v4()
    log(f"V3: {rep['V3_scores']}  V4: {rep['V4_decision']}")
    rep["V5_candidates"] = v5()
    log(f"V5: {rep['V5_candidates']}")
    rep["all_pass"] = all(v["pass"] for v in rep.values() if isinstance(v, dict))
    with open(REPORT, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
