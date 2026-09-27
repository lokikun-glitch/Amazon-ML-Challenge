"""
FINAL-B-H2 production inference (streaming, deterministic). See reports/FINAL_CONFIG.md.

  test S1 (normalized cache)  ->  D3 candidates (per country)  ->  candidate lines written
  -> 28 + 14 features for exactly those candidates -> frozen B scores -> B + H2 decision
  -> matching lines written

Reused, unchanged logic:
  normalization          data_cache/normalized/test_source*.parquet (build_cache.py / normalization.py)
  D3 blocking            generate_dev_candidates.CONFIGS["D3"], build_indexes, per_channel_candidates
  28 base features       build_pairwise_features.compute_pair_features
  address numbers        data_cache/address_numbers/test_source*.parquet (address_numbers.extract)
  14 number features     build_number_features.compute_number_features
  model                  data_cache/models/B_frozen.npz (frozen scaler + LR, threshold, H2 t_low)

Streaming / memory: countries are processed one at a time (every D3 key contains the country,
so a per-country index yields exactly the all-country candidates). Inside a country, S1 are
processed in fixed-size chunks; every S1's candidates are generated, written, featurized,
scored and decided inside one chunk, so the H2 "no candidate >= 8.3 / top candidate" decision
always sees the S1's complete candidate set. The same in-memory candidate arrays are written to
candidate_pairs and fed to the model, so the two can never diverge. Nothing larger than one
chunk of pairs is ever materialized.

Determinism: TokenIndex.query breaks document-frequency ties in set() order, which depends on
string hashing; the pipeline therefore refuses to run unless PYTHONHASHSEED=0 (it re-launches
itself with it). Candidate and match lists are sorted by entity_id; tie-break for the H2 top
candidate is (highest z, then smallest entity_id). Output rows follow test_source1 file order.
"""
import argparse
import heapq
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
CACHE_DIR = os.path.join(ROOT, "data_cache", "normalized")
NUM_DIR = os.path.join(ROOT, "data_cache", "address_numbers")
MODEL_PATH = os.path.join(ROOT, "data_cache", "models", "B_frozen.npz")
FULL_COLS = ["entity_id", "country", "name_norm", "name_no_suffix", "name_translit",
             "address_norm", "address_translit", "postal_code", "street_number"]
CHANNELS = ["b1", "b2", "b3", "b5", "b7a"]


def _ensure_hashseed():
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = dict(os.environ, PYTHONHASHSEED="0")
        sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))


try:
    import psutil
    _P = psutil.Process(os.getpid())
except ImportError:  # pragma: no cover
    _P = None


def rss_mb():
    return _P.memory_info().rss / 1e6 if _P else float("nan")


def peak_mb():
    mi = _P.memory_info() if _P else None
    return getattr(mi, "peak_wset", getattr(mi, "rss", 0)) / 1e6 if mi else float("nan")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB peak={peak_mb():.0f}MB  {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- frozen model + decision

def load_model(path=MODEL_PATH):
    m = np.load(path, allow_pickle=False)
    cols = [str(c) for c in m["feature_order"]]
    sc = StandardScaler()
    sc.mean_ = m["scaler_mean"]
    sc.scale_ = m["scaler_scale"]
    sc.var_ = m["scaler_scale"] ** 2
    sc.n_features_in_ = len(cols)
    sc.n_samples_seen_ = 1
    lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
    lr.coef_ = m["coef"].reshape(1, -1)
    lr.intercept_ = m["intercept"]
    lr.classes_ = np.array([0, 1])
    lr.n_features_in_ = len(cols)
    return {"cols": cols, "scaler": sc, "lr": lr, "t": float(m["threshold_z"]), "t_low": float(m["h2_t_low"])}


def score(model, X32):
    return model["lr"].decision_function(model["scaler"].transform(X32))


def h2_ok(num):
    return ((num["num_hn_head_equal"] == 1) & (num["num_hn_head_mismatch"] == 0)
            & (num["num_compound_mismatch"] == 0) & (num["num_token_contradiction"] == 0))


def decide(starts, z, cand_ids, ok, t, t_low):
    """FINAL-B-H2 for S1 segments [starts[i], starts[i+1]) of the pair arrays.
    Returns a list (one per S1) of sorted accepted candidate IDs."""
    out = []
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i + 1]
        if a == b:
            out.append([])
            continue
        zs = z[a:b]
        main = zs >= t
        if main.any():
            out.append(sorted(cand_ids[a:b][main].tolist()))
            continue
        # H2: top candidate = highest z, ties -> smallest entity_id
        zmax = zs.max()
        top_rel = min(np.flatnonzero(zs == zmax), key=lambda j: cand_ids[a + j])
        out.append([cand_ids[a + top_rel]] if (zmax >= t_low and ok[a + top_rel]) else [])
    return out


# ----------------------------------------------------------------------------- per-country processing

def load_country(country, s1_all):
    from generate_dev_candidates import load_pool_column
    pool = load_pool_column(FULL_COLS, prefix="test", country=country)
    nums = pd.concat([pd.read_parquet(os.path.join(NUM_DIR, f"test_source{k}.parquet")) for k in (2, 3)])
    nums = nums[nums["entity_id"].isin(set(pool["entity_id"]))].set_index("entity_id")
    pool_num = nums.reindex(pool["entity_id"].values).fillna("")
    del nums
    s1 = s1_all[s1_all["country"] == country]
    n1 = pd.read_parquet(os.path.join(NUM_DIR, "test_source1.parquet"))
    s1_num = n1[n1["entity_id"].isin(set(s1["entity_id"]))].set_index("entity_id").reindex(s1["entity_id"].values).fillna("")
    return pool, pool_num, s1, s1_num


def process_country(country, s1_all, model, cand_fh, match_fh, sel_rows, chunk, stats):
    from build_number_features import compute_number_features
    from build_pairwise_features import compute_pair_features
    from generate_dev_candidates import CONFIGS, build_indexes, per_channel_candidates

    cfg = CONFIGS["D3"]
    t0 = time.time()
    idx = build_indexes(cfg, prefix="test", country=country)
    t_index = time.time() - t0
    log(f"{country}: D3 indexes built in {t_index:.0f}s")
    pool, pool_num, s1, s1_num = load_country(country, s1_all)
    pool_ids = pool["entity_id"].values
    pool_country = pool["country"].values
    log(f"{country}: pool={len(pool)} S1={len(s1)} loaded")
    if sel_rows is not None:
        keep = np.isin(s1["row"].values, sel_rows)
        s1, s1_num = s1[keep], s1_num[keep]
    s1 = s1.reset_index(drop=True)
    s1_num = s1_num.reset_index(drop=True)
    st = {"n_s1": len(s1), "pool": len(pool), "index_build_s": round(t_index, 1), "cands_per_s1": [],
          "time": {"gen": 0.0, "write_cand": 0.0, "feat28": 0.0, "feat14": 0.0, "score": 0.0, "decide_write": 0.0},
          "pairs": 0, "country_leak": 0, "self_or_bad_prefix": 0, "dup_within_s1": 0,
          "predicted_pairs": 0, "s1_zero_pred": 0, "s1_one_pred": 0, "s1_multi_pred": 0, "h2_recovered": 0}
    for c0 in range(0, len(s1), chunk):
        part = s1.iloc[c0:c0 + chunk]
        t1 = time.time()
        s_idx, c_pos, flags, starts = [], [], {c: [] for c in CHANNELS}, [0]
        for li, row in zip(range(c0, c0 + len(part)), part.itertuples(index=False)):
            per = per_channel_candidates(idx, cfg, row)
            union = set()
            for c in CHANNELS:
                union |= per[c]
            order = sorted(union, key=lambda p: pool_ids[p])
            s_idx.extend([li] * len(order))
            c_pos.extend(order)
            for c in CHANNELS:
                sc_ = per[c]
                flags[c].extend(1 if p in sc_ else 0 for p in order)
            starts.append(starts[-1] + len(order))
        s_idx = np.asarray(s_idx, dtype=np.int64)
        c_pos = np.asarray(c_pos, dtype=np.int64)
        cand_ids = pool_ids[c_pos] if len(c_pos) else np.empty(0, dtype=object)
        st["time"]["gen"] += time.time() - t1

        # exact candidate set -> candidate_pairs (same arrays feed the model below)
        t1 = time.time()
        for i, (row, sid) in enumerate(zip(part["row"].values, part["entity_id"].values)):
            ids = cand_ids[starts[i]:starts[i + 1]]
            cand_fh.write(f"{row}\t{sid}\t{','.join(ids)}\n")
            n = len(ids)
            st["cands_per_s1"].append(n)
            if n != len(set(ids)):
                st["dup_within_s1"] += 1
        st["pairs"] += len(c_pos)
        st["country_leak"] += int((pool_country[c_pos] != country).sum())
        st["self_or_bad_prefix"] += int(sum(1 for x in cand_ids if not (x.startswith("S2-") or x.startswith("S3-"))))
        st["time"]["write_cand"] += time.time() - t1

        if len(c_pos):
            t1 = time.time()
            s1_fields = s1.iloc[s_idx][FULL_COLS[1:]].reset_index(drop=True)
            cand_fields = pool.iloc[c_pos][FULL_COLS[1:]].reset_index(drop=True)
            channels = pd.DataFrame({f"from_{c}": np.asarray(flags[c], dtype=np.int8) for c in CHANNELS})
            feat = compute_pair_features(s1_fields, cand_fields, channels)
            st["time"]["feat28"] += time.time() - t1
            t1 = time.time()
            # only the entities present in this chunk (features depend on value equality only)
            uniq_c, c_inv = np.unique(c_pos, return_inverse=True)
            num = compute_number_features(s1_num.iloc[c0:c0 + len(part)].reset_index(drop=True),
                                          pool_num.iloc[uniq_c].reset_index(drop=True),
                                          s_idx - c0, c_inv, feat["name_fuzz_token_set_ratio"].values)
            for k, v in num.items():
                feat[k] = v
            st["time"]["feat14"] += time.time() - t1
            t1 = time.time()
            z = score(model, feat[model["cols"]].values.astype(np.float32))
            ok = h2_ok(feat).values
            st["time"]["score"] += time.time() - t1
            del s1_fields, cand_fields, channels, feat
        else:
            z = np.empty(0)
            ok = np.empty(0, dtype=bool)
        t1 = time.time()
        preds = decide(np.asarray(starts), z, cand_ids, ok, model["t"], model["t_low"])
        for i, (row, sid, pr) in enumerate(zip(part["row"].values, part["entity_id"].values, preds)):
            match_fh.write(f"{row}\t{sid}\t{','.join(pr)}\n")
            a, b = starts[i], starts[i + 1]
            n = len(pr)
            st["predicted_pairs"] += n
            st["s1_zero_pred"] += n == 0
            st["s1_one_pred"] += n == 1
            st["s1_multi_pred"] += n >= 2
            if n == 1 and b > a and not (z[a:b] >= model["t"]).any():
                st["h2_recovered"] += 1
            assert set(pr) <= set(cand_ids[a:b]), "prediction outside candidate set"
        st["time"]["decide_write"] += time.time() - t1
        done_s1 = min(c0 + chunk, len(s1))
        n_zero_c = sum(1 for n in st["cands_per_s1"] if n == 0)
        log(f"{country}: {done_s1}/{len(s1)} S1, pairs so far {st['pairs']} "
            f"({st['pairs'] / max(done_s1, 1):.0f}/S1), zero-cand {n_zero_c} ({100 * n_zero_c / max(done_s1, 1):.2f}%), "
            f"pred {st['predicted_pairs']} [0:{st['s1_zero_pred']} 1:{st['s1_one_pred']} 2+:{st['s1_multi_pred']}], "
            f"H2 {st['h2_recovered']}")
    del idx, pool, pool_num
    cps = np.asarray(st.pop("cands_per_s1"))
    st["cand_per_s1"] = {"mean": round(float(cps.mean()), 1) if len(cps) else 0,
                         "p50": float(np.percentile(cps, 50)) if len(cps) else 0,
                         "p90": float(np.percentile(cps, 90)) if len(cps) else 0,
                         "p99": float(np.percentile(cps, 99)) if len(cps) else 0,
                         "max": int(cps.max()) if len(cps) else 0,
                         "zero_candidate_s1": int((cps == 0).sum()),
                         "zero_candidate_rate_pct": round(100 * float((cps == 0).mean()), 2) if len(cps) else 0}
    st["time"] = {k: round(v, 1) for k, v in st["time"].items()}
    stats[country] = st


def smoke_rows(s1_all, countries, n):
    """Deterministic smoke sample: n evenly spaced test_source1 rows per country."""
    sel = []
    for c in countries:
        rows = s1_all.loc[s1_all["country"] == c, "row"].values
        pick = np.linspace(0, len(rows) - 1, min(n, len(rows))).round().astype(int)
        sel.extend(rows[np.unique(pick)].tolist())
    return np.asarray(sorted(sel))


def merge_parts(parts, out_path, header):
    """k-way merge of per-country part files (each sorted by test_source1 row) into one TSV."""
    fhs = [open(p, encoding="utf-8") for p in parts]
    try:
        streams = [((int(line.split("\t", 1)[0]), line) for line in fh) for fh in fhs]
        with open(out_path, "w", encoding="utf-8", newline="\n") as out:
            out.write(header + "\n")
            for _, line in heapq.merge(*streams, key=lambda x: x[0]):
                out.write(line.split("\t", 1)[1])
    finally:
        for fh in fhs:
            fh.close()


def main():
    _ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--smoke-per-country", type=int, default=0,
                    help="if >0: only this many evenly spaced S1 per country (smoke test, not a submission)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--countries", default=None, help="comma-separated subset (smoke/debug only)")
    args = ap.parse_args()
    t_all = time.time()
    model = load_model()
    s1_all = pd.read_parquet(os.path.join(CACHE_DIR, "test_source1.parquet"), columns=FULL_COLS)
    s1_all["row"] = np.arange(len(s1_all), dtype=np.int64)
    countries = sorted(s1_all["country"].unique())
    if args.countries:
        countries = [c for c in countries if c in args.countries.split(",")]
    sel = smoke_rows(s1_all, countries, args.smoke_per_country) if args.smoke_per_country else None
    part_dir = os.path.join(args.out_dir, "_parts")
    os.makedirs(part_dir, exist_ok=True)
    stats = {"hashseed": os.environ.get("PYTHONHASHSEED"), "n_test_s1": len(s1_all),
             "s1_by_country": s1_all["country"].value_counts().to_dict(), "countries": {}}
    for c in countries:
        with open(os.path.join(part_dir, f"{c}_cand.tsv"), "w", encoding="utf-8", newline="\n") as cf, \
                open(os.path.join(part_dir, f"{c}_match.tsv"), "w", encoding="utf-8", newline="\n") as mf:
            process_country(c, s1_all, model, cf, mf, sel, args.chunk, stats["countries"])
        log(f"{c} done: {json.dumps(stats['countries'][c])}")
    merge_parts([os.path.join(part_dir, f"{c}_cand.tsv") for c in countries],
                os.path.join(args.out_dir, "candidate_pairs.tsv"), "source1_entity_id\tcandidate_entity_ids")
    merge_parts([os.path.join(part_dir, f"{c}_match.tsv") for c in countries],
                os.path.join(args.out_dir, "matching_results.tsv"), "source1_entity_id\tmatched_entity_ids")
    stats["total_s"] = round(time.time() - t_all, 1)
    stats["peak_rss_mb"] = round(peak_mb())
    with open(os.path.join(args.out_dir, "inference_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=int)
    log(f"done in {stats['total_s']}s")


if __name__ == "__main__":
    main()
