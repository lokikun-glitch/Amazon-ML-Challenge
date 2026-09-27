"""
Forensics of high-scoring singleton false positives (D3 + LR, training side).

Analysis only -- the matcher is NOT changed and nothing here feeds inference.
Ground truth is used for exactly two forensic purposes:
  (1) to identify TRUE singleton S1s (n_true == 0) and pair labels (group membership);
  (2) to look up whether a singleton's top candidate is claimed by some OTHER S1
      in the full training GT (for manual categorization of GT-omission cases).
It is never used to construct a pair/S1 feature.

Data: D3 training-side scored pairs (FIT 8K + TUNE 2K S1; LR fit on FIT only,
so FIT rows are in-sample for the LR -- results are reported per part too).

Groups compared
  A      true-positive pairs (label=1), all
  A_acc  true-positive pairs with z >= 7 (the accepted TPs C must be separated from)
  B      ordinary negatives: uniform random sample of label=0 pairs
  C      top candidate of a TRUE singleton S1 with z >= 7 (high-scoring singleton FP)
  D      top candidate of a MATCHED S1 that is a false positive with z >= 7 (reference)

Rarity features (observable, computed from the S2+S3 pool / S1 population, per country):
  name / address / postal / street-number exact-value frequency, token DF / IDF.

Outputs
  reports/singleton_forensics_cases.tsv   one row per singleton with top z >= 7
  reports/singleton_forensics_pairs.parquet  all analysed pairs with features (groups A/B/C/D)
  reports/singleton_forensics.json        counts + distribution comparison
"""
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
SCORE_DIR = os.path.join(ROOT, "data_cache", "dev_scores")
FEAT_DIR = os.path.join(ROOT, "data_cache", "dev_features")
CACHE_DIR = os.path.join(ROOT, "data_cache", "normalized")
RAW_DIR = os.path.join(ROOT, "student_resource", "dataset", "train")
REPORT_DIR = os.path.join(ROOT, "reports")
CONFIG = "D3"
Z_LEVELS = (7.0, 9.0, 11.0)
Z_HI = 7.0
N_NEG_SAMPLE = 20000
SEED = 42
NON_FEATURE_COLS = {"s1_id", "cand_id", "label", "country"}
REC_COLS = ["entity_id", "country", "name_norm", "name_no_suffix", "name_translit",
            "address_norm", "postal_code", "street_number"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}]  {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- selection

def select_rows():
    s1 = pd.read_parquet(os.path.join(SCORE_DIR, f"{CONFIG}_s1_table.parquet"))
    s1 = s1[s1["part"].isin(["fit", "tune"])]
    n_true = dict(zip(s1["entity_id"], s1["n_true"]))
    sc = pd.read_parquet(os.path.join(SCORE_DIR, f"{CONFIG}_train_scores.parquet"),
                         columns=["s1_id", "cand_id", "label", "z", "part"])
    sc["s1_id"] = sc["s1_id"].astype(str)
    sc["row"] = np.arange(len(sc))
    sc["n_true"] = sc["s1_id"].map(n_true).astype(int)
    log(f"scores: {len(sc)} rows, {sc['s1_id'].nunique()} S1 with candidates")

    top_idx = sc.groupby("s1_id", sort=False)["z"].idxmax().values
    top = sc.loc[top_idx]
    # rank of each S1's second-best z (margin) -- observable
    sc_sorted = sc[["s1_id", "z"]].sort_values(["s1_id", "z"], ascending=[True, False])
    z2 = sc_sorted.groupby("s1_id", sort=False)["z"].nth(1)
    z2 = pd.Series(z2.values, index=sc_sorted.loc[z2.index, "s1_id"].values)
    n_cand = sc.groupby("s1_id", sort=False).size()
    top = top.assign(z2=top["s1_id"].map(z2).fillna(-np.inf).values,
                     n_cand=top["s1_id"].map(n_cand).values)

    single_s1 = s1[s1["n_true"] == 0]
    counts = {"n_true_singletons": int(len(single_s1)),
              "singletons_with_zero_candidates": int((~single_s1["entity_id"].isin(set(top["s1_id"]))).sum())}
    ts = top[top["n_true"] == 0]
    for zl in Z_LEVELS:
        m = ts["z"] >= zl
        counts[f"singleton_top_z_ge_{zl:g}"] = {
            "n": int(m.sum()), "pct_of_singletons": round(100 * m.sum() / len(single_s1), 2),
            "fit": int((m & (ts["part"] == "fit")).sum()), "tune": int((m & (ts["part"] == "tune")).sum())}
    # operational threshold from decision_rules_D3 (A: t=7.3)
    counts["singleton_top_z_ge_7.3_operational"] = int((ts["z"] >= 7.3).sum())

    rng = np.random.default_rng(SEED)
    groups = {}
    groups["C"] = ts.loc[ts["z"] >= Z_HI, "row"].values
    tm = top[(top["n_true"] > 0) & (top["label"] == 0) & (top["z"] >= Z_HI)]
    groups["D"] = tm["row"].values
    pos = sc.loc[sc["label"] == 1, ["row", "z"]]
    groups["A"] = pos["row"].values
    groups["A_acc"] = pos.loc[pos["z"] >= Z_HI, "row"].values
    neg_rows = sc.loc[sc["label"] == 0, "row"].values
    groups["B"] = rng.choice(neg_rows, size=N_NEG_SAMPLE, replace=False)
    log("group sizes: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    fit_mask = (sc["part"] == "fit").values
    top_info = top.set_index("row")[["z2", "n_cand"]]
    return sc, groups, counts, fit_mask, top_info


# ----------------------------------------------------------------------------- features + LR contributions

def fetch_features(rows_needed, fit_mask):
    """Stream the feature parquet; collect the selected rows and FIT-part scaler stats."""
    pf = pq.ParquetFile(os.path.join(FEAT_DIR, f"{CONFIG}_train_features.parquet"))
    names = pf.schema_arrow.names
    feat_cols = [c for c in names if c not in NON_FEATURE_COLS]
    need = np.zeros(len(fit_mask), dtype=bool)
    need[rows_needed] = True
    s = np.zeros(len(feat_cols))
    ss = np.zeros(len(feat_cols))
    n_fit = 0
    parts = []
    off = 0
    for b in pf.iter_batches(batch_size=1_000_000, columns=feat_cols + ["cand_id"]):
        n = b.num_rows
        X = np.column_stack([b.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in feat_cols])
        fm = fit_mask[off:off + n]
        Xf = X[fm].astype(np.float64)
        s += Xf.sum(0)
        ss += (Xf ** 2).sum(0)
        n_fit += len(Xf)
        sel = np.where(need[off:off + n])[0]
        if len(sel):
            df = pd.DataFrame(X[sel], columns=feat_cols)
            df["cand_id_chk"] = b.column("cand_id").take(pa.array(sel)).to_pylist()
            df["row"] = sel + off
            parts.append(df)
        off += n
    mu = s / n_fit
    sd = np.sqrt(np.maximum(ss / n_fit - mu ** 2, 0))
    sd[sd == 0] = 1.0   # StandardScaler convention
    return pd.concat(parts, ignore_index=True).set_index("row"), feat_cols, mu, sd


# ----------------------------------------------------------------------------- records

def load_records(ids, path):
    t = pq.read_table(path, columns=REC_COLS)
    t = t.filter(pc.is_in(t["entity_id"], value_set=pa.array(sorted(ids))))
    return t.to_pandas().set_index("entity_id")


def load_raw(ids, fname):
    out = []
    for ch in pd.read_csv(os.path.join(RAW_DIR, fname), sep="\t", dtype=str, keep_default_na=False,
                          na_values=[""], chunksize=500_000):
        out.append(ch[ch["entity_id"].isin(ids)])
    return pd.concat(out).set_index("entity_id")


# ----------------------------------------------------------------------------- rarity

def rarity_tables(target):
    """target: dict col -> set of values. Returns per-(country,value) counts in the S2+S3 pool,
    token DF (per country) for name/address tokens in target, and per-country pool size."""
    exact = {c: Counter() for c in ("name_norm", "address_norm", "postal_code", "street_number")}
    tok_df = {"name": Counter(), "addr": Counter()}
    n_country = Counter()
    vs = {c: pa.array(sorted(v)) for c, v in target.items() if c in exact}
    for lab in ("train_source2", "train_source3"):
        pf = pq.ParquetFile(os.path.join(CACHE_DIR, f"{lab}.parquet"))
        for b in pf.iter_batches(batch_size=500_000, columns=["country"] + list(exact)):
            ctry = b.column("country")
            n_country.update(ctry.to_pylist())
            for c in exact:
                col = b.column(c)
                m = pc.is_in(col, value_set=vs[c])
                exact[c].update(zip(pc.filter(ctry, m).to_pylist(), pc.filter(col, m).to_pylist()))
            for key, c in (("name", "name_norm"), ("addr", "address_norm")):
                tset = target[f"{key}_tokens"]
                cnt = tok_df[key]
                for co, s in zip(ctry.to_pylist(), b.column(c).to_pylist()):
                    if s:
                        for tk in set(s.split()):
                            if (co, tk) in tset:
                                cnt[(co, tk)] += 1
        log(f"  rarity scan done: {lab}")
    return exact, tok_df, n_country


def s1_population_counts(target_names, target_name_addr):
    """How many S1 records (full train_source1, 2.2M) share the S1's name / name+address."""
    t = pq.read_table(os.path.join(CACHE_DIR, "train_source1.parquet"),
                      columns=["entity_id", "country", "name_norm", "address_norm"])
    m = pc.is_in(t["name_norm"], value_set=pa.array(sorted(target_names)))
    sub = t.filter(m).to_pandas()
    name_cnt = Counter(zip(sub["country"], sub["name_norm"]))
    na_cnt = Counter(zip(sub["country"], sub["name_norm"], sub["address_norm"]))
    by_name = defaultdict(list)
    for r in sub.itertuples():
        by_name[(r.country, r.name_norm)].append(r.entity_id)
    return name_cnt, na_cnt, by_name


def gt_claims(cand_ids, s1_ids):
    """Forensic only: which S1 claim each candidate in the full training GT, and each S1's matches."""
    claims = defaultdict(list)
    s1_matches = {}
    for ch in pd.read_csv(os.path.join(RAW_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str,
                          keep_default_na=False, na_values=[""], chunksize=500_000):
        for sid, ms in zip(ch["source1_entity_id"], ch["matched_entity_ids"]):
            if not isinstance(ms, str) or not ms:
                continue
            lst = ms.split(",")
            if sid in s1_ids:
                s1_matches[sid] = lst
            for mid in lst:
                if mid in cand_ids:
                    claims[mid].append(sid)
    return claims, s1_matches


def idf(df, n):
    return math.log((n + 1) / (df + 1)) + 1


def pair_rarity(p, rec_s1, rec_c, exact, tok_df, n_country):
    out = defaultdict(list)
    for s1_id, cid in zip(p["s1_id"], p["cand_id"]):
        a, b = rec_s1.loc[s1_id], rec_c.loc[cid]
        co = a["country"]
        N = n_country[co]
        out["cand_name_freq"].append(exact["name_norm"][(co, b["name_norm"])])
        out["s1_name_freq_pool"].append(exact["name_norm"][(co, a["name_norm"])])
        out["cand_addr_freq"].append(exact["address_norm"][(co, b["address_norm"])] if b["address_norm"] else np.nan)
        out["s1_addr_freq_pool"].append(exact["address_norm"][(co, a["address_norm"])] if a["address_norm"] else np.nan)
        out["cand_postal_freq"].append(exact["postal_code"][(co, b["postal_code"])] if b["postal_code"] else np.nan)
        out["cand_streetnum_freq"].append(exact["street_number"][(co, b["street_number"])] if b["street_number"] else np.nan)
        for key, col in (("name", "name_norm"), ("addr", "address_norm")):
            t1 = set(a[col].split()) if a[col] else set()
            t2 = set(b[col].split()) if b[col] else set()
            w = {tk: idf(tok_df[key][(co, tk)], N) for tk in t1 | t2}
            sh, un = t1 & t2, t1 | t2
            diff = un - sh
            out[f"{key}_shared_idf_sum"].append(sum(w[t] for t in sh))
            out[f"{key}_shared_idf_max"].append(max((w[t] for t in sh), default=0.0))
            out[f"{key}_shared_min_df"].append(min((tok_df[key][(co, t)] for t in sh), default=np.nan))
            out[f"{key}_idf_jaccard"].append(sum(w[t] for t in sh) / sum(w.values()) if un else np.nan)
            out[f"{key}_diff_idf_max"].append(max((w[t] for t in diff), default=0.0))
            out[f"{key}_diff_idf_sum"].append(sum(w[t] for t in diff))
            out[f"{key}_mean_idf_s1"].append(np.mean([w[t] for t in t1]) if t1 else np.nan)
    df = pd.DataFrame(out, index=p.index)
    for c in ("cand_name_freq", "s1_name_freq_pool", "cand_addr_freq", "s1_addr_freq_pool",
              "cand_postal_freq", "cand_streetnum_freq"):
        df[f"log_{c}"] = np.log1p(df[c])
    return df


# ----------------------------------------------------------------------------- comparison

def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 ties, NaN dropped."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def zmatched_weights(z_ref, z_other, bins=(7, 8, 9, 10, 11, 12, 14, 99)):
    """Importance weights so `other` has the same z-bin distribution as `ref`."""
    br = pd.cut(z_ref, bins).value_counts(normalize=True)
    bo = pd.cut(z_other, bins)
    fo = bo.value_counts(normalize=True)
    w = bo.map(lambda b: br.get(b, 0) / fo.get(b, 1) if fo.get(b, 0) > 0 else 0).astype(float).values
    return w


def compare(P, cols):
    g = {k: P[P[f"g_{k}"]] for k in ("A", "A_acc", "B", "C", "D")}
    res = {}
    wA = zmatched_weights(g["C"]["z"], g["A_acc"]["z"])
    for c in cols:
        r = {}
        for k, d in g.items():
            v = d[c].astype(float)
            r[k] = {"mean": round(float(v.mean()), 4), "median": round(float(v.median()), 4),
                    "p10": round(float(v.quantile(.1)), 4), "p90": round(float(v.quantile(.9)), 4),
                    "missing_pct": round(100 * float(v.isna().mean()), 1)}
        a = auc(g["C"][c], g["A_acc"][c])
        r["auc_C_vs_A_acc"] = None if a is None else round(a, 4)
        # z-matched mean of A_acc (reweighted to C's z distribution)
        v = g["A_acc"][c].astype(float).values
        ok = ~np.isnan(v)
        r["A_acc_zmatched_mean"] = round(float(np.average(v[ok], weights=wA[ok])), 4) if wA[ok].sum() > 0 else None
        res[c] = r
    return res


# ----------------------------------------------------------------------------- main

def main():
    sc, groups, counts, fit_mask, top_info = select_rows()
    all_rows = np.unique(np.concatenate(list(groups.values())))
    meta = sc.loc[all_rows, ["s1_id", "cand_id", "label", "z", "part", "n_true"]].copy()
    del sc

    log(f"Fetching feature rows ({len(all_rows)}) + FIT scaler stats...")
    F, feat_cols, mu, sd = fetch_features(all_rows, fit_mask)
    F = F.loc[all_rows]
    assert (F["cand_id_chk"].values == meta["cand_id"].values).all(), "score/feature row misalignment"
    coef = pd.read_json(os.path.join(SCORE_DIR, f"{CONFIG}_lr_coef.json"), typ="series")[feat_cols].values
    contrib = (F[feat_cols].values - mu) / sd * coef
    icpt = meta["z"].values - contrib.sum(1)
    log(f"reconstructed intercept: mean={icpt.mean():.5f} sd={icpt.std():.2e} (sd~0 validates contributions)")
    P = pd.concat([meta, F[feat_cols]], axis=1)
    P = P.join(pd.DataFrame(contrib, index=P.index, columns=[f"ctb_{c}" for c in feat_cols]))
    P["prob"] = 1 / (1 + np.exp(-P["z"]))
    for k, v in groups.items():
        P[f"g_{k}"] = P.index.isin(v)
    P = P.join(top_info, how="left")

    log("Loading normalized records...")
    rec_s1 = load_records(set(P["s1_id"]), os.path.join(CACHE_DIR, "train_source1.parquet"))
    cids = set(P["cand_id"])
    rec_c = pd.concat([load_records(cids, os.path.join(CACHE_DIR, f"train_source{k}.parquet")) for k in (2, 3)])

    log("Computing rarity tables over S2+S3 pool...")
    target = {"name_norm": set(rec_s1["name_norm"]) | set(rec_c["name_norm"]),
              "address_norm": set(rec_s1["address_norm"]) | set(rec_c["address_norm"]),
              "postal_code": set(rec_c["postal_code"]) | set(rec_s1["postal_code"]),
              "street_number": set(rec_c["street_number"]) | set(rec_s1["street_number"])}
    for key, col in (("name", "name_norm"), ("addr", "address_norm")):
        ts = set()
        for rec in (rec_s1, rec_c):
            for co, s in zip(rec["country"], rec[col]):
                if s:
                    ts.update((co, t) for t in s.split())
        target[f"{key}_tokens"] = ts
    exact, tok_df, n_country = rarity_tables(target)
    R = pair_rarity(P, rec_s1, rec_c, exact, tok_df, n_country)
    P = P.join(R)
    rar_cols = list(R.columns)

    log("S1-population duplicate counts + GT claims (forensic only)...")
    name_cnt, na_cnt, s1_by_name = s1_population_counts(set(rec_s1["name_norm"]), None)
    P["s1_name_freq_s1pop"] = [name_cnt[(rec_s1.at[s, "country"], rec_s1.at[s, "name_norm"])] for s in P["s1_id"]]
    P["s1_name_addr_freq_s1pop"] = [na_cnt[(rec_s1.at[s, "country"], rec_s1.at[s, "name_norm"], rec_s1.at[s, "address_norm"])] for s in P["s1_id"]]
    rar_cols += ["s1_name_freq_s1pop", "s1_name_addr_freq_s1pop"]

    CD = P[P["g_C"] | P["g_D"]]
    claims, _ = gt_claims(set(CD["cand_id"]), set())
    other_s1 = set(s for c in CD["cand_id"] for s in claims.get(c, []))
    claims_s1_rec = load_records(other_s1, os.path.join(CACHE_DIR, "train_source1.parquet")) if other_s1 else pd.DataFrame()
    raw_s1 = load_raw(set(CD["s1_id"]) | other_s1, "train_source1.tsv")
    raw_c = pd.concat([load_raw(set(CD["cand_id"]), f"train_source{k}.tsv") for k in (2, 3)])

    # ---------------- case table (singletons, z >= 7)
    C = P[P["g_C"]].sort_values("z", ascending=False)
    rows = []
    for idx, r in C.iterrows():
        a, b = rec_s1.loc[r.s1_id], rec_c.loc[r.cand_id]
        cl = [s for s in claims.get(r.cand_id, []) if s != r.s1_id]
        cl_same_name = [s for s in cl if s in claims_s1_rec.index and claims_s1_rec.at[s, "name_norm"] == a["name_norm"]]
        cl_same_na = [s for s in cl_same_name if claims_s1_rec.at[s, "address_norm"] == a["address_norm"]]
        chans = ",".join(c[5:] for c in feat_cols if c.startswith("from_") and r[c] > 0)
        top_ctb = sorted(((r[f"ctb_{c}"], c) for c in feat_cols), reverse=True)[:4]
        rows.append({
            "s1_id": r.s1_id, "part": r.part, "country": a["country"],
            "s1_name": raw_s1.at[r.s1_id, "business_name"], "s1_address": raw_s1.at[r.s1_id, "business_address"],
            "cand_id": r.cand_id, "cand_source": r.cand_id[:2],
            "cand_name": raw_c.at[r.cand_id, "business_name"], "cand_address": raw_c.at[r.cand_id, "business_address"],
            "s1_name_norm": a["name_norm"], "cand_name_norm": b["name_norm"],
            "s1_addr_norm": a["address_norm"], "cand_addr_norm": b["address_norm"],
            "s1_postal": a["postal_code"], "cand_postal": b["postal_code"],
            "s1_streetnum": a["street_number"], "cand_streetnum": b["street_number"],
            "z": round(r.z, 3), "prob": r.prob, "z2": round(r.z2, 3), "n_cand": int(r.n_cand),
            **{c: round(float(r[c]), 3) for c in feat_cols if not c.startswith("from_")},
            "channels": chans,
            "exact_name_and_address": int(r.name_exact == 1 and r.address_exact == 1),
            **{c: r[c] for c in rar_cols},
            "gt_claimed_by_other_s1": len(cl), "gt_other_s1_same_name": len(cl_same_name),
            "gt_other_s1_same_name_addr": len(cl_same_na),
            "gt_other_s1_ids": ",".join(cl[:5]),
            "gt_other_s1_name": raw_s1.at[cl[0], "business_name"] if cl and cl[0] in raw_s1.index else "",
            "gt_other_s1_address": raw_s1.at[cl[0], "business_address"] if cl and cl[0] in raw_s1.index else "",
            "top_lr_contributions": "; ".join(f"{c}={v:+.2f}" for v, c in top_ctb),
        })
    cases = pd.DataFrame(rows)
    cases.to_csv(os.path.join(REPORT_DIR, "singleton_forensics_cases.tsv"), sep="\t", index=False)

    # ---------------- distribution comparison
    cmp_cols = ["z"] + [c for c in feat_cols if c != "country_equal"] + rar_cols
    report = {"config": CONFIG, "counts": counts,
              "group_sizes": {k: int(P[f"g_{k}"].sum()) for k in groups},
              "group_defs": {"A": "label=1 pairs", "A_acc": "label=1 pairs with z>=7",
                             "B": f"random {N_NEG_SAMPLE} label=0 pairs",
                             "C": "top candidate of TRUE singleton S1, z>=7",
                             "D": "top candidate of MATCHED S1 that is a FP, z>=7"},
              "C_part_breakdown": C["part"].value_counts().to_dict(),
              "C_country_breakdown": cases["country"].value_counts().to_dict(),
              "C_source_breakdown": cases["cand_source"].value_counts().to_dict(),
              "C_exact_name_and_address": int(cases["exact_name_and_address"].sum()),
              "C_gt_claimed_by_other_s1": int((cases["gt_claimed_by_other_s1"] > 0).sum()),
              "C_gt_other_s1_same_name": int((cases["gt_other_s1_same_name"] > 0).sum()),
              "C_gt_other_s1_same_name_addr": int((cases["gt_other_s1_same_name_addr"] > 0).sum()),
              "D_gt_claimed_by_other_s1": int(sum(bool([s for s in claims.get(c, []) if s != s1])
                                                  for s1, c in zip(P.loc[P["g_D"], "s1_id"], P.loc[P["g_D"], "cand_id"]))),
              "lr_contrib_mean": {k: {c: round(float(P.loc[P[f"g_{k}"], f"ctb_{c}"].mean()), 3) for c in feat_cols}
                                  for k in ("A_acc", "B", "C", "D")},
              "comparison": compare(P, cmp_cols)}
    ranked = sorted(((abs(v["auc_C_vs_A_acc"] - 0.5), c, v["auc_C_vs_A_acc"]) for c, v in report["comparison"].items()
                     if v["auc_C_vs_A_acc"] is not None and c != "z"), reverse=True)
    report["top_separating_features_C_vs_A_acc"] = [{"feature": c, "auc": a} for _, c, a in ranked[:25]]
    with open(os.path.join(REPORT_DIR, "singleton_forensics.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    keep = ["s1_id", "cand_id", "label", "z", "part", "n_true", "z2", "n_cand"] + feat_cols + rar_cols + \
           [f"g_{k}" for k in groups]
    P[keep].to_parquet(os.path.join(REPORT_DIR, "singleton_forensics_pairs.parquet"))
    log(f"Wrote {len(cases)} singleton cases + report.")


if __name__ == "__main__":
    main()
