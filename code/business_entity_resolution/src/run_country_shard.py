"""
Run FINAL-B-H2 production inference for exactly ONE country (one shard per machine).

Thin wrapper around production_inference.process_country -- the same function the tested
sequential pipeline calls -- so candidates, features, scores and decisions are produced by
the identical code path. Only orchestration differs: one country, its own output directory.

    python run_country_shard.py --country FR     # also: IN, US, or a literal country value

Outputs (output/shards/<CODE>/), written to *.tmp and renamed only on success:
  candidate_pairs_<CODE>.tsv   test_source1_row \t source1_entity_id \t candidate_entity_ids
  matching_results_<CODE>.tsv  test_source1_row \t source1_entity_id \t matched_entity_ids
  manifest_<CODE>.json         counts, distributions, timings, peak RSS, environment and input
                               hashes, status (a failure manifest is written on exceptions)
test_source1_row is the 0-based row of the S1 in test_source1 (the normalized cache preserves
file order); consolidate_shards.py uses it to rebuild the exact original order.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd

import production_inference as pi

ALIASES = {"FR": "France", "IN": "India", "US": "US"}   # CLI convenience only; no country logic
ROOT = pi.ROOT
CODE_FILES = ["production_inference.py", "run_country_shard.py", "generate_dev_candidates.py", "blocking.py",
              "build_pairwise_features.py", "build_number_features.py", "address_numbers.py", "normalization.py"]
INPUT_FILES = ([os.path.join("data_cache", "normalized", f"test_source{k}.parquet") for k in (1, 2, 3)]
               + [os.path.join("data_cache", "address_numbers", f"test_source{k}.parquet") for k in (1, 2, 3)]
               + [os.path.join("data_cache", "models", "B_frozen.npz")])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def environment():
    import numpy, pandas, pyarrow, rapidfuzz, sklearn
    src = os.path.dirname(os.path.abspath(__file__))
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": {m.__name__: m.__version__ for m in (numpy, pandas, pyarrow, rapidfuzz, sklearn)},
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "code_sha256": {f: sha256(os.path.join(src, f)) for f in CODE_FILES},
            "input_sha256": {f.replace(os.sep, "/"): sha256(os.path.join(ROOT, f)) for f in INPUT_FILES}}


def shard_file_stats(cand_path, match_path):
    """Streaming stats straight from the written shard files (what consolidation will read)."""
    n_c, n_m = [], []
    for path, acc in ((cand_path, n_c), (match_path, n_m)):
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                ids = line.rstrip("\n").split("\t")[2]
                acc.append(ids.count(",") + 1 if ids else 0)
    c, m = np.asarray(n_c), np.asarray(n_m)
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else 0.0
    return {"s1_rows": len(c), "candidate_pairs": int(c.sum()),
            "cand_per_s1": {"mean": round(float(c.mean()), 2) if len(c) else 0, "median": q(c, 50), "p95": q(c, 95),
                            "p99": q(c, 99), "max": int(c.max()) if len(c) else 0},
            "zero_candidate_s1": int((c == 0).sum()),
            "zero_candidate_rate_pct": round(100 * float((c == 0).mean()), 3) if len(c) else 0,
            "predicted_pairs": int(m.sum()), "s1_zero_pred": int((m == 0).sum()), "s1_one_pred": int((m == 1).sum()),
            "s1_multi_pred": int((m >= 2).sum())}


def main():
    pi._ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, help="FR, IN, US (aliases) or a literal test country value")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "output", "shards"))
    ap.add_argument("--smoke-per-country", type=int, default=0, help="subset smoke test (same sample as production_inference)")
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()
    country = ALIASES.get(args.country, args.country)
    code = next((k for k, v in ALIASES.items() if v == country), country)
    out_dir = os.path.join(args.out_root, code)
    os.makedirs(out_dir, exist_ok=True)
    cand_path = os.path.join(out_dir, f"candidate_pairs_{code}.tsv")
    match_path = os.path.join(out_dir, f"matching_results_{code}.tsv")
    man_path = os.path.join(out_dir, f"manifest_{code}.json")
    manifest = {"country": country, "shard_code": code, "status": "running", "smoke_per_country": args.smoke_per_country,
                "chunk": args.chunk, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "environment": environment()}
    t0 = time.time()
    try:
        s1_all = pd.read_parquet(os.path.join(pi.CACHE_DIR, "test_source1.parquet"), columns=pi.FULL_COLS)
        s1_all["row"] = np.arange(len(s1_all), dtype=np.int64)
        counts = s1_all["country"].value_counts().to_dict()
        if country not in counts:
            raise ValueError(f"country {country!r} not in test S1 (have {sorted(counts)})")
        manifest["test_s1_total"] = len(s1_all)
        manifest["test_s1_by_country"] = counts
        manifest["expected_shard_s1"] = int(counts[country]) if not args.smoke_per_country else None
        sel = pi.smoke_rows(s1_all, [country], args.smoke_per_country) if args.smoke_per_country else None
        model = pi.load_model()
        stats = {}
        with open(cand_path + ".tmp", "w", encoding="utf-8", newline="\n") as cf, \
                open(match_path + ".tmp", "w", encoding="utf-8", newline="\n") as mf:
            cf.write("test_source1_row\tsource1_entity_id\tcandidate_entity_ids\n")
            mf.write("test_source1_row\tsource1_entity_id\tmatched_entity_ids\n")
            pi.process_country(country, s1_all, model, cf, mf, sel, args.chunk, stats)
        os.replace(cand_path + ".tmp", cand_path)
        os.replace(match_path + ".tmp", match_path)
        st = stats[country]
        manifest["inference_stats"] = st
        manifest["file_stats"] = shard_file_stats(cand_path, match_path)
        fs = manifest["file_stats"]
        checks = {"file_rows_eq_processed": fs["s1_rows"] == st["n_s1"],
                  "file_pairs_eq_processed": fs["candidate_pairs"] == st["pairs"],
                  "file_preds_eq_processed": fs["predicted_pairs"] == st["predicted_pairs"],
                  "no_country_leak": st["country_leak"] == 0, "no_bad_prefix": st["self_or_bad_prefix"] == 0,
                  "no_dup_candidates": st["dup_within_s1"] == 0}
        if not args.smoke_per_country:
            checks["all_country_s1_processed"] = fs["s1_rows"] == counts[country]
        manifest["checks"] = checks
        manifest["h2_recovered"] = st["h2_recovered"]
        manifest["output_sha256"] = {os.path.basename(cand_path): sha256(cand_path),
                                     os.path.basename(match_path): sha256(match_path)}
        manifest["status"] = "success" if all(checks.values()) else "failed_checks"
    except BaseException as e:  # noqa: BLE001 -- record any failure (incl. KeyboardInterrupt) in the manifest
        manifest["status"] = "failure"
        manifest["error"] = repr(e)
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        manifest["elapsed_s"] = round(time.time() - t0, 1)
        manifest["peak_rss_mb"] = round(pi.peak_mb())
        manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2, default=int)
        pi.log(f"shard {code}: status={manifest['status']} elapsed={manifest['elapsed_s']}s")
    sys.exit(0 if manifest["status"] == "success" else 1)


if __name__ == "__main__":
    main()
