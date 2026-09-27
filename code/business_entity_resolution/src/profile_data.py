"""
Phase 1: Dataset Forensics.

Profiles every source/ground-truth file: size, missingness, country distribution,
name/address length & token statistics, and ground-truth match distribution.
Writes a machine-readable JSON report plus a human-readable text summary.

Usage:
    python src/profile_data.py --dataset-dir ../../student_resource/dataset --out-dir ../../reports
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_source(path):
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[""]
    )


def tokenize(s):
    if not isinstance(s, str):
        return []
    return TOKEN_RE.findall(s.lower())


def field_stats(series, name):
    n = len(series)
    missing = series.isna().sum()
    non_missing = series.dropna()
    lengths = non_missing.str.len()
    token_counts = non_missing.map(lambda s: len(tokenize(s)))
    dup = non_missing.duplicated().sum()
    unique = non_missing.nunique()

    stats = {
        "field": name,
        "n": int(n),
        "missing": int(missing),
        "missing_pct": round(100 * missing / n, 3) if n else 0.0,
        "unique": int(unique),
        "unique_pct_of_nonmissing": round(100 * unique / len(non_missing), 3) if len(non_missing) else 0.0,
        "duplicate_nonmissing": int(dup),
        "char_len_mean": round(float(lengths.mean()), 2) if len(lengths) else None,
        "char_len_median": float(lengths.median()) if len(lengths) else None,
        "char_len_p95": float(lengths.quantile(0.95)) if len(lengths) else None,
        "char_len_max": int(lengths.max()) if len(lengths) else None,
        "token_count_mean": round(float(token_counts.mean()), 2) if len(token_counts) else None,
        "token_count_median": float(token_counts.median()) if len(token_counts) else None,
    }
    return stats


def common_tokens(series, top_n=30):
    counter = Counter()
    for s in series.dropna():
        counter.update(tokenize(s))
    return counter.most_common(top_n)


def profile_source_file(path, label, sample_n=500000):
    t0 = time.time()
    df = load_source(path)
    n = len(df)

    # Sample for expensive token/pattern analysis if huge
    sample = df if n <= sample_n else df.sample(sample_n, random_state=42)

    report = {
        "label": label,
        "path": path,
        "n_records": int(n),
        "n_unique_entity_id": int(df["entity_id"].nunique()),
        "load_time_sec": round(time.time() - t0, 2),
        "fields": {},
    }

    for col in ["business_name", "business_address", "country"]:
        report["fields"][col] = field_stats(sample[col], col)

    country_counts = df["country"].value_counts(dropna=False)
    report["country_distribution"] = {
        (k if pd.notna(k) else "<MISSING>"): int(v) for k, v in country_counts.items()
    }
    report["country_pct"] = {
        k: round(100 * v / n, 3) for k, v in report["country_distribution"].items()
    }

    report["top_name_tokens"] = common_tokens(sample["business_name"])
    report["top_address_tokens"] = common_tokens(sample["business_address"])

    # postal / numeric token frequency in address (sampled)
    addr_nonmissing = sample["business_address"].dropna()
    has_digit_token = addr_nonmissing.map(lambda s: any(t.isdigit() for t in tokenize(s)))
    has_5digit_postal = addr_nonmissing.str.contains(r"\b\d{5,6}\b", regex=True)
    report["address_has_any_numeric_token_pct"] = round(100 * has_digit_token.mean(), 2)
    report["address_has_5_6_digit_code_pct"] = round(100 * has_5digit_postal.mean(), 2)

    return report, df


def profile_ground_truth(path, s1_ids, s2_ids, s3_ids):
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    n_s1 = len(df)

    def split_ids(x):
        if not isinstance(x, str) or x == "":
            return []
        return x.split(",")

    match_lists = df["matched_entity_ids"].map(split_ids)
    n_matches = match_lists.map(len)

    n_singleton = int((n_matches == 0).sum())
    n_matched_s1 = int((n_matches > 0).sum())

    all_matched_ids = [mid for lst in match_lists for mid in lst]
    n_s2_matches = sum(1 for m in all_matched_ids if m.startswith("S2-"))
    n_s3_matches = sum(1 for m in all_matched_ids if m.startswith("S3-"))

    unknown_s2 = sum(1 for m in all_matched_ids if m.startswith("S2-") and m not in s2_ids)
    unknown_s3 = sum(1 for m in all_matched_ids if m.startswith("S3-") and m not in s3_ids)
    self_ref = sum(1 for m in all_matched_ids if m.startswith("S1-"))

    s1_in_gt = set(df["source1_entity_id"])
    s1_missing_from_gt = s1_ids - s1_in_gt

    report = {
        "n_s1_rows": int(n_s1),
        "n_s1_ids_missing_from_gt_file": len(s1_missing_from_gt),
        "n_singleton_s1": n_singleton,
        "n_matched_s1": n_matched_s1,
        "singleton_pct": round(100 * n_singleton / n_s1, 3),
        "matches_per_matched_s1_mean": round(float(n_matches[n_matches > 0].mean()), 3) if n_matched_s1 else None,
        "matches_per_matched_s1_median": float(n_matches[n_matches > 0].median()) if n_matched_s1 else None,
        "matches_per_matched_s1_p95": float(n_matches[n_matches > 0].quantile(0.95)) if n_matched_s1 else None,
        "matches_per_matched_s1_max": int(n_matches.max()),
        "total_match_edges": len(all_matched_ids),
        "s2_match_edges": n_s2_matches,
        "s3_match_edges": n_s3_matches,
        "s2_edge_pct": round(100 * n_s2_matches / len(all_matched_ids), 2) if all_matched_ids else None,
        "s3_edge_pct": round(100 * n_s3_matches / len(all_matched_ids), 2) if all_matched_ids else None,
        "unknown_s2_ids_referenced": unknown_s2,
        "unknown_s3_ids_referenced": unknown_s3,
        "self_referential_s1_in_matches": self_ref,
        "match_count_histogram": {int(k): int(v) for k, v in n_matches.value_counts().sort_index().items()},
        "load_time_sec": round(time.time() - t0, 2),
    }
    return report, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="student_resource/dataset")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--sample-n", type=int, default=500000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    full_report = {"generated_by": "profile_data.py", "sample_n_for_token_stats": args.sample_n}

    files = {
        "train_source1": os.path.join(args.dataset_dir, "train", "train_source1.tsv"),
        "train_source2": os.path.join(args.dataset_dir, "train", "train_source2.tsv"),
        "train_source3": os.path.join(args.dataset_dir, "train", "train_source3.tsv"),
        "test_source1": os.path.join(args.dataset_dir, "test", "test_source1.tsv"),
        "test_source2": os.path.join(args.dataset_dir, "test", "test_source2.tsv"),
        "test_source3": os.path.join(args.dataset_dir, "test", "test_source3.tsv"),
    }

    ids = {}
    for label, path in files.items():
        print(f"[profile] {label} ...", file=sys.stderr)
        report, df = profile_source_file(path, label, sample_n=args.sample_n)
        full_report[label] = report
        ids[label] = set(df["entity_id"])
        del df

    print("[profile] train_ground_truth ...", file=sys.stderr)
    gt_path = os.path.join(args.dataset_dir, "train", "train_ground_truth.tsv")
    gt_report, gt_df = profile_ground_truth(
        gt_path, ids["train_source1"], ids["train_source2"], ids["train_source3"]
    )
    full_report["train_ground_truth"] = gt_report

    out_json = os.path.join(args.out_dir, "data_profile.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_json}")

    # Human-readable summary
    out_txt = os.path.join(args.out_dir, "data_profile_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for label in files:
            r = full_report[label]
            f.write(f"=== {label} ===\n")
            f.write(f"records: {r['n_records']}  unique_entity_id: {r['n_unique_entity_id']}\n")
            for col, s in r["fields"].items():
                f.write(
                    f"  {col}: missing={s['missing_pct']}%  unique%={s['unique_pct_of_nonmissing']}%  "
                    f"len_mean={s['char_len_mean']}  len_p95={s['char_len_p95']}  "
                    f"tok_mean={s['token_count_mean']}\n"
                )
            f.write(f"  country_dist: {r['country_distribution']}\n")
            f.write(
                f"  address_numeric_token_pct={r['address_has_any_numeric_token_pct']}%  "
                f"address_5-6digit_code_pct={r['address_has_5_6_digit_code_pct']}%\n"
            )
            f.write("\n")
        f.write("=== train_ground_truth ===\n")
        for k, v in gt_report.items():
            if k != "match_count_histogram":
                f.write(f"  {k}: {v}\n")
        f.write(f"  match_count_histogram (first 20): {dict(list(gt_report['match_count_histogram'].items())[:20])}\n")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
