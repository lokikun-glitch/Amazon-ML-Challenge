"""
Phase: S1-level train/validation split.

80/20 split at the Source-1 entity level, fixed seed=42. No S1 entity appears
in both sides. Per spec section 16 of the technical-lead directive, the S2/S3
reference pools stay available to both sides (blocking must retrieve against
the full population) -- what must never leak is validation S1 *labels* into
training.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEED = 42
VAL_FRACTION = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="../../student_resource/dataset")
    ap.add_argument("--out-dir", default="../../data_cache/splits")
    args = ap.parse_args()

    s1_path = os.path.join(args.dataset_dir, "train", "train_source1.tsv")
    s1_ids = pd.read_csv(s1_path, sep="\t", usecols=["entity_id"], dtype=str)["entity_id"].values

    rng = np.random.default_rng(SEED)
    shuffled = s1_ids.copy()
    rng.shuffle(shuffled)

    n_val = int(len(shuffled) * VAL_FRACTION)
    val_ids = shuffled[:n_val]
    train_ids = shuffled[n_val:]

    assert set(val_ids).isdisjoint(set(train_ids))
    assert len(train_ids) + len(val_ids) == len(s1_ids)

    os.makedirs(args.out_dir, exist_ok=True)
    pd.Series(sorted(train_ids)).to_csv(os.path.join(args.out_dir, "train_s1_ids.txt"), index=False, header=False)
    pd.Series(sorted(val_ids)).to_csv(os.path.join(args.out_dir, "val_s1_ids.txt"), index=False, header=False)

    meta = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "n_total_s1": len(s1_ids),
        "n_train_s1": len(train_ids),
        "n_val_s1": len(val_ids),
    }
    with open(os.path.join(args.out_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
