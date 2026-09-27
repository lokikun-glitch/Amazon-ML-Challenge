"""
Run address_numbers.extract over the raw training sources once and cache the
result per entity (data_cache/address_numbers/train_source{1,2,3}.parquet).
Feature building joins these by entity_id, so extraction happens once per
record, not once per pair.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

from address_numbers import FIELDS, extract
from blocking import rss_mb

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
DATASET_DIR = os.path.join(ROOT, "student_resource", "dataset")
OUT_DIR = os.path.join(ROOT, "data_cache", "address_numbers")
REPORT_DIR = os.path.join(ROOT, "reports")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_mb():.0f}MB  {msg}", file=sys.stderr, flush=True)


def _work(chunk):
    rows = [extract(a, c) for a, c in zip(chunk["business_address"], chunk["country"])]
    out = pd.DataFrame(rows, columns=FIELDS)
    out.insert(0, "entity_id", chunk["entity_id"].values)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="train", choices=["train", "test"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {}
    t_all = time.time()
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for k in (1, 2, 3):
            t0 = time.time()
            reader = pd.read_csv(os.path.join(DATASET_DIR, args.prefix, f"{args.prefix}_source{k}.tsv"), sep="\t", dtype=str,
                                 keep_default_na=False, usecols=["entity_id", "business_address", "country"],
                                 chunksize=100_000)
            parts = list(pool.imap(_work, reader))
            df = pd.concat(parts, ignore_index=True)
            path = os.path.join(OUT_DIR, f"{args.prefix}_source{k}.parquet")
            df.to_parquet(path, index=False)
            ctry = pd.read_parquet(os.path.join(ROOT, "data_cache", "normalized", f"{args.prefix}_source{k}.parquet"),
                                   columns=["entity_id", "country"])
            df = df.merge(ctry, on="entity_id", how="left")
            cov = {c: {f: round(100 * float((g[f] != "").mean()), 2) for f in FIELDS}
                   for c, g in df.groupby("country")}
            stats[f"{args.prefix}_source{k}"] = {"rows": len(df), "seconds": round(time.time() - t0, 1),
                                         "file_mb": round(os.path.getsize(path) / 1e6, 1), "coverage_pct": cov}
            log(f"source{k}: {len(df)} rows in {time.time() - t0:.0f}s; coverage {cov}")
    stats["total_seconds"] = round(time.time() - t_all, 1)
    stats["parent_peak_rss_mb_note"] = "workers are separate processes; see log for parent RSS"
    with open(os.path.join(REPORT_DIR, "address_numbers_extraction.json" if args.prefix == "train"
                           else f"address_numbers_extraction_{args.prefix}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
