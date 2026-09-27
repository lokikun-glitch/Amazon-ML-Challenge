"""
Deep-dive on ground truth: multi-mapping check + sampled true-positive pairs
with normalized-name / normalized-address agreement rates.
"""
import argparse
import os
import re
import json
from collections import Counter

import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9]+")


def norm_name(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def tokset(s):
    return set(TOKEN_RE.findall(s.lower())) if isinstance(s, str) else set()


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="student_resource/dataset")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--sample-pairs", type=int, default=30)
    ap.add_argument("--sample-for-rates", type=int, default=200000)
    args = ap.parse_args()

    train_dir = os.path.join(args.dataset_dir, "train")
    gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t",
                      dtype=str, keep_default_na=False, na_values=[""])

    def split_ids(x):
        return x.split(",") if isinstance(x, str) and x else []

    gt["match_list"] = gt["matched_entity_ids"].map(split_ids)

    # multi-mapping check: does any S2/S3 id appear under >1 S1?
    counter = Counter()
    for lst in gt["match_list"]:
        for mid in lst:
            counter[mid] += 1
    multi = {k: v for k, v in counter.items() if v > 1}
    print(f"S2/S3 IDs matched to >1 S1 entity: {len(multi)} (out of {len(counter)} distinct matched ids)")
    if multi:
        sample_multi = list(multi.items())[:10]
        print("sample:", sample_multi)

    # Build flat list of positive pairs (s1_id, matched_id)
    pos_pairs = [(row.source1_entity_id, mid) for row in gt.itertuples() for mid in row.match_list]
    print(f"Total positive pairs: {len(pos_pairs)}")

    import random
    random.seed(42)
    rates_sample = random.sample(pos_pairs, min(args.sample_for_rates, len(pos_pairs)))
    extra_examples_sample = random.sample(pos_pairs, min(args.sample_pairs, len(pos_pairs)))

    needed_ids = set()
    for s1_id, m_id in rates_sample:
        needed_ids.add(s1_id)
        needed_ids.add(m_id)
    for s1_id, m_id in extra_examples_sample:
        needed_ids.add(s1_id)
        needed_ids.add(m_id)
    print(f"Distinct IDs needed for lookup: {len(needed_ids)}")

    # Stream through each source file in chunks, keeping only needed rows.
    lookup_table = {}

    def stream_filter(path, chunksize=500000):
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                  na_values=[""], chunksize=chunksize):
            hit = chunk[chunk["entity_id"].isin(needed_ids)]
            if len(hit):
                for row in hit.itertuples(index=False):
                    lookup_table[row.entity_id] = (row.business_name, row.business_address, row.country)

    stream_filter(os.path.join(train_dir, "train_source1.tsv"))
    stream_filter(os.path.join(train_dir, "train_source2.tsv"))
    stream_filter(os.path.join(train_dir, "train_source3.tsv"))
    print(f"Resolved {len(lookup_table)} / {len(needed_ids)} needed IDs")

    def lookup(entity_id):
        return lookup_table[entity_id]

    exact_name_match = 0
    name_jaccard_sum = 0.0
    exact_addr_match = 0
    country_match = 0
    both_empty_addr = 0
    name_jaccard_ge_08 = 0

    for s1_id, m_id in rates_sample:
        n1, a1, c1 = lookup(s1_id)
        n2, a2, c2 = lookup(m_id)
        nn1, nn2 = norm_name(n1), norm_name(n2)
        if nn1 == nn2 and nn1 != "":
            exact_name_match += 1
        jac = jaccard(tokset(n1), tokset(n2))
        name_jaccard_sum += jac
        if jac >= 0.8:
            name_jaccard_ge_08 += 1
        na1 = a1.lower().strip() if isinstance(a1, str) else ""
        na2 = a2.lower().strip() if isinstance(a2, str) else ""
        if na1 == na2 and na1 != "":
            exact_addr_match += 1
        if c1 == c2:
            country_match += 1

    n = len(rates_sample)
    summary = {
        "n_sampled_positive_pairs": n,
        "exact_normalized_name_match_pct": round(100 * exact_name_match / n, 2),
        "name_token_jaccard_mean": round(name_jaccard_sum / n, 4),
        "name_token_jaccard_ge_0.8_pct": round(100 * name_jaccard_ge_08 / n, 2),
        "exact_raw_address_match_pct": round(100 * exact_addr_match / n, 2),
        "country_match_pct": round(100 * country_match / n, 2),
        "multi_mapped_s2s3_ids": len(multi),
        "distinct_matched_ids": len(counter),
    }
    print(json.dumps(summary, indent=2))

    # sample readable examples
    examples = []
    for s1_id, m_id in extra_examples_sample:
        n1, a1, c1 = lookup(s1_id)
        n2, a2, c2 = lookup(m_id)
        examples.append({
            "s1_id": s1_id, "s1_name": n1, "s1_addr": a1, "s1_country": c1,
            "match_id": m_id, "match_name": n2, "match_addr": a2, "match_country": c2,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "gt_deep_dive.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": examples}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {os.path.join(args.out_dir, 'gt_deep_dive.json')}")


if __name__ == "__main__":
    main()
