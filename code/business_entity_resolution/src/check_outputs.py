"""
Streaming correctness checks for candidate_pairs.tsv + matching_results.tsv (both files are
read line by line in lockstep; nothing proportional to the candidate count is held in memory
except the set of valid S2/S3 IDs).

  1/2  row counts (== test S1 count unless --smoke)
  3    identical S1 sequence in both files; (full run) equals test_source1.tsv order
  4    candidate IDs are S2-/S3- and exist in test S2/S3
  5    no duplicate candidate IDs within an S1
  6    no S1 self matches / S1 IDs anywhere
  7    every predicted ID is in that S1's candidate list
  8    no duplicate predicted IDs
  9    empty lists are an empty second field
  + SHA-256 of both files, candidate / prediction totals
"""
import argparse
import hashlib
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
TEST_DIR = os.path.join(ROOT, "student_resource", "dataset", "test")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="subset run: skip the all-test-S1 coverage checks")
    args = ap.parse_args()
    cand_p = os.path.join(args.dir, "candidate_pairs.tsv")
    match_p = os.path.join(args.dir, "matching_results.tsv")
    valid = set()
    for k in (2, 3):
        valid |= set(pd.read_csv(os.path.join(TEST_DIR, f"test_source{k}.tsv"), sep="\t", usecols=["entity_id"],
                                 dtype=str)["entity_id"])
    s1_order = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", usecols=["entity_id"],
                           dtype=str)["entity_id"].tolist()
    bad = {k: 0 for k in ("s1_mismatch", "cand_bad_prefix", "cand_unknown_id", "cand_dup", "self_match",
                          "pred_not_in_cand", "pred_dup", "pred_bad_prefix", "malformed", "empty_not_blank")}
    n_rows = n_cand = n_pred = n_empty_cand = n_empty_pred = 0
    seq = []
    with open(cand_p, encoding="utf-8") as fc, open(match_p, encoding="utf-8") as fm:
        hc, hm = fc.readline().rstrip("\n"), fm.readline().rstrip("\n")
        header_ok = hc == "source1_entity_id\tcandidate_entity_ids" and hm == "source1_entity_id\tmatched_entity_ids"
        for lc, lm in zip(fc, fm):
            pc, pm = lc.rstrip("\n").split("\t"), lm.rstrip("\n").split("\t")
            if len(pc) != 2 or len(pm) != 2:
                bad["malformed"] += 1
                continue
            n_rows += 1
            s1 = pc[0]
            seq.append(s1)
            if pm[0] != s1:
                bad["s1_mismatch"] += 1
            cands = pc[1].split(",") if pc[1] else []
            preds = pm[1].split(",") if pm[1] else []
            if (pc[1] and pc[1].strip() == "") or (pm[1] and pm[1].strip() == ""):
                bad["empty_not_blank"] += 1
            n_cand += len(cands)
            n_pred += len(preds)
            n_empty_cand += not cands
            n_empty_pred += not preds
            cs = set(cands)
            bad["cand_dup"] += len(cs) != len(cands)
            bad["pred_dup"] += len(set(preds)) != len(preds)
            bad["cand_bad_prefix"] += sum(not (c.startswith("S2-") or c.startswith("S3-")) for c in cands)
            bad["pred_bad_prefix"] += sum(not (c.startswith("S2-") or c.startswith("S3-")) for c in preds)
            bad["self_match"] += sum(c == s1 or c.startswith("S1-") for c in cands + preds)
            bad["cand_unknown_id"] += sum(c not in valid for c in cands)
            bad["pred_not_in_cand"] += sum(p not in cs for p in preds)
        extra = (fc.readline() != "") or (fm.readline() != "")
    res = {"header_ok": header_ok, "rows": n_rows, "files_same_length": not extra,
           "candidate_pairs": n_cand, "predicted_pairs": n_pred, "s1_empty_candidates": n_empty_cand,
           "s1_empty_predictions": n_empty_pred, "violations": bad,
           "sha256": {"candidate_pairs.tsv": sha256(cand_p), "matching_results.tsv": sha256(match_p)},
           "bytes": {"candidate_pairs.tsv": os.path.getsize(cand_p), "matching_results.tsv": os.path.getsize(match_p)}}
    if not args.smoke:
        res["rows_equal_test_s1"] = n_rows == len(s1_order)
        res["order_equals_test_source1"] = seq == s1_order
    else:
        pos = {s: i for i, s in enumerate(s1_order)}
        idx = [pos.get(s, -1) for s in seq]
        res["all_s1_exist_in_test"] = min(idx) >= 0
        res["order_follows_test_source1"] = idx == sorted(idx)
    res["pass"] = header_ok and not extra and sum(bad.values()) == 0 and all(
        v for k, v in res.items() if k in ("rows_equal_test_s1", "order_equals_test_source1",
                                           "all_s1_exist_in_test", "order_follows_test_source1"))
    with open(os.path.join(args.dir, "output_checks.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
