"""
Verify country agreement between S1 and its ground-truth matches over the
*complete* training ground truth (not a sample), memory-safely.

Design: only entity_id + country columns are ever loaded (usecols) -- name and
address are skipped entirely, so peak memory is two small id->country dicts
(~12.5M entries total) rather than the full ~1.2GB source files. Country
strings are interned so the dict value is a shared pointer, not a fresh string
per row.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter

import pandas as pd


def build_country_dict(path, usecols=("entity_id", "country")):
    d = {}
    t0 = time.time()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        eid_i, ctry_i = idx[usecols[0]], idx[usecols[1]]
        for row in reader:
            if not row:
                continue
            d[row[eid_i]] = sys.intern(row[ctry_i])
    print(f"  loaded {len(d)} ids from {path} in {time.time()-t0:.1f}s", file=sys.stderr)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="../../student_resource/dataset")
    ap.add_argument("--out-dir", default="../../reports")
    args = ap.parse_args()

    import os
    train_dir = os.path.join(args.dataset_dir, "train")

    print("Building id->country dicts (entity_id/country columns only)...", file=sys.stderr)
    s1_country = build_country_dict(os.path.join(train_dir, "train_source1.tsv"))
    s2_country = build_country_dict(os.path.join(train_dir, "train_source2.tsv"))
    s3_country = build_country_dict(os.path.join(train_dir, "train_source3.tsv"))

    print("Streaming ground truth and comparing country on every edge...", file=sys.stderr)
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    total_edges = 0
    agree = 0
    disagree = 0
    disagree_examples = []
    missing_s1 = 0
    missing_match = 0
    mismatch_country_pairs = Counter()

    t0 = time.time()
    with open(gt_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row or len(row) < 1:
                continue
            s1_id = row[0]
            matched = row[1] if len(row) > 1 else ""
            if not matched:
                continue
            c1 = s1_country.get(s1_id)
            if c1 is None:
                missing_s1 += 1
                continue
            for mid in matched.split(","):
                total_edges += 1
                if mid.startswith("S2-"):
                    c2 = s2_country.get(mid)
                elif mid.startswith("S3-"):
                    c2 = s3_country.get(mid)
                else:
                    c2 = None
                if c2 is None:
                    missing_match += 1
                    continue
                if c1 == c2:
                    agree += 1
                else:
                    disagree += 1
                    mismatch_country_pairs[(c1, c2)] += 1
                    if len(disagree_examples) < 20:
                        disagree_examples.append({"s1_id": s1_id, "s1_country": c1,
                                                    "match_id": mid, "match_country": c2})

    elapsed = time.time() - t0
    print(f"  processed {total_edges} edges in {elapsed:.1f}s", file=sys.stderr)

    result = {
        "total_edges": total_edges,
        "agree": agree,
        "disagree": disagree,
        "agree_pct": round(100 * agree / total_edges, 6) if total_edges else None,
        "missing_s1_lookup": missing_s1,
        "missing_match_lookup": missing_match,
        "mismatch_country_pairs": {f"{a}->{b}": c for (a, b), c in mismatch_country_pairs.most_common(20)},
        "disagree_examples": disagree_examples,
    }
    print(json.dumps(result, indent=2))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "country_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
