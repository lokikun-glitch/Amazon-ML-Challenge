"""
Consolidate country shards into the final submission files. No candidates are regenerated,
nothing is rescored, no prediction is altered: shard lines are copied verbatim (minus the
leading test_source1_row column) in exact test_source1 order.

Pre-merge checks (fail -> nothing is written):
  * every expected shard has manifest status == success
  * all shards used identical code, input-data, model hashes and package versions
  * shard files are unchanged since their manifest (SHA-256)
  * S1 coverage: shard S1 counts equal the per-country counts derived from test_source1, no
    country missing, total == test S1 count (skipped in --smoke)
During the merge:
  * test_source1_row sequence == 0..N-1 exactly (smoke: strictly increasing) and the S1 ID at each
    row equals test_source1's ID at that row -> exact original order, no S1 missing or duplicated
  * candidate and matching streams carry the identical S1 sequence
  * per-shard copied row / pair / prediction counts equal the shard manifests
Writes <out>/candidate_pairs.tsv, <out>/matching_results.tsv, <out>/consolidation_manifest.json.
"""
import argparse
import heapq
import json
import os
import sys

import pandas as pd

from run_country_shard import sha256

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")


def stream(path, code):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            row, rest = line.split("\t", 1)
            yield int(row), code, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-root", default=os.path.join(ROOT, "output", "shards"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--shards", default="FR,IN,US")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    codes = args.shards.split(",")
    man = {c: json.load(open(os.path.join(args.shard_root, c, f"manifest_{c}.json"))) for c in codes}
    errors = []
    for c, m in man.items():
        if m.get("status") != "success":
            errors.append(f"shard {c}: status {m.get('status')}")
        for fname, h in m.get("output_sha256", {}).items():
            if sha256(os.path.join(args.shard_root, c, fname)) != h:
                errors.append(f"shard {c}: {fname} changed since its manifest")
        if bool(m.get("smoke_per_country")) != args.smoke:
            errors.append(f"shard {c}: smoke setting {m.get('smoke_per_country')} does not match --smoke={args.smoke}")
    env_keys = ("code_sha256", "input_sha256", "packages", "PYTHONHASHSEED", "python")
    ref = man[codes[0]]["environment"]
    for c in codes[1:]:
        for k in env_keys:
            if man[c]["environment"][k] != ref[k]:
                errors.append(f"shard {c}: environment {k} differs from shard {codes[0]}")

    s1_order = pd.read_parquet(os.path.join(ROOT, "data_cache", "normalized", "test_source1.parquet"),
                               columns=["entity_id", "country"])
    by_country = s1_order["country"].value_counts().to_dict()
    countries = {man[c]["country"] for c in codes}
    coverage = {}
    if not args.smoke:
        missing = set(by_country) - countries
        if missing:
            errors.append(f"countries without a shard: {sorted(missing)}")
        for c in codes:
            n = man[c]["file_stats"]["s1_rows"]
            coverage[c] = {"shard_s1": n, "expected_from_test_source1": by_country[man[c]["country"]]}
            if n != by_country[man[c]["country"]]:
                errors.append(f"shard {c}: {n} S1 rows, test_source1 has {by_country[man[c]['country']]}")
        total = sum(man[c]["file_stats"]["s1_rows"] for c in codes)
        coverage["total"] = {"shard_s1": total, "test_s1": len(s1_order)}
        if total != len(s1_order):
            errors.append(f"total shard S1 {total} != test S1 {len(s1_order)}")
    if errors:
        print("CONSOLIDATION ABORTED:\n  " + "\n  ".join(errors))
        sys.exit(1)

    ids = s1_order["entity_id"].values
    out_c = os.path.join(args.out_dir, "candidate_pairs.tsv")
    out_m = os.path.join(args.out_dir, "matching_results.tsv")
    copied = {c: {"rows": 0, "pairs": 0, "preds": 0} for c in codes}
    prev = -1
    n_rows = 0
    order_errors = seq_errors = 0
    cs = heapq.merge(*[stream(os.path.join(args.shard_root, c, f"candidate_pairs_{c}.tsv"), c) for c in codes])
    ms = heapq.merge(*[stream(os.path.join(args.shard_root, c, f"matching_results_{c}.tsv"), c) for c in codes])
    with open(out_c + ".tmp", "w", encoding="utf-8", newline="\n") as fc, \
            open(out_m + ".tmp", "w", encoding="utf-8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for (rc, code_c, lc), (rm, code_m, lm) in zip(cs, ms):
            s1c, cand = lc.split("\t", 1)
            s1m, pred = lm.split("\t", 1)
            if rc != rm or s1c != s1m or code_c != code_m:
                seq_errors += 1
            expected_row = n_rows if not args.smoke else None
            if (expected_row is not None and rc != expected_row) or rc <= prev or ids[rc] != s1c:
                order_errors += 1
            prev = rc
            fc.write(f"{s1c}\t{cand}")
            fm.write(f"{s1m}\t{pred}")
            copied[code_c]["rows"] += 1
            copied[code_c]["pairs"] += cand.count(",") + 1 if cand.strip() else 0
            copied[code_m]["preds"] += pred.count(",") + 1 if pred.strip() else 0
            n_rows += 1
    leftovers = next(cs, None) is not None or next(ms, None) is not None
    count_errors = [c for c in codes if (copied[c]["rows"], copied[c]["pairs"], copied[c]["preds"]) !=
                    (man[c]["file_stats"]["s1_rows"], man[c]["file_stats"]["candidate_pairs"],
                     man[c]["file_stats"]["predicted_pairs"])]
    ok = order_errors == 0 and seq_errors == 0 and not leftovers and not count_errors and \
        (args.smoke or n_rows == len(s1_order))
    if ok:
        os.replace(out_c + ".tmp", out_c)
        os.replace(out_m + ".tmp", out_m)
    res = {"status": "success" if ok else "failed", "rows_written": n_rows, "test_s1": len(s1_order),
           "order_errors": order_errors, "cand_match_sequence_errors": seq_errors, "unmatched_trailing_lines": leftovers,
           "shard_count_mismatches": count_errors, "copied_per_shard": copied, "coverage": coverage,
           "shard_manifest_sha256": {c: sha256(os.path.join(args.shard_root, c, f"manifest_{c}.json")) for c in codes},
           "model_sha256": ref["input_sha256"].get("data_cache/models/B_frozen.npz"),
           "code_sha256": ref["code_sha256"], "smoke": args.smoke}
    if ok:
        res["output_sha256"] = {"candidate_pairs.tsv": sha256(out_c), "matching_results.tsv": sha256(out_m)}
    with open(os.path.join(args.out_dir, "consolidation_manifest.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("code_sha256", "shard_manifest_sha256")}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
