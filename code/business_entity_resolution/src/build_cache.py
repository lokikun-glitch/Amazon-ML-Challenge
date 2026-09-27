"""
Precompute normalized representations for every source file and cache them as
Parquet. Streams the raw TSV in chunks and writes Parquet incrementally
(pyarrow.parquet.ParquetWriter) so peak memory stays bounded by one chunk,
not the whole file -- required given the ~5GB free-RAM constraint on this
machine for 5M+ row source files.

Usage:
    python src/build_cache.py --dataset-dir ../../student_resource/dataset --out-dir ../../data_cache/normalized
"""
import argparse
import os
import sys
import time
import tracemalloc

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from normalization import build_normalized_frame

FILES = {
    "train_source1": ("train", "train_source1.tsv"),
    "train_source2": ("train", "train_source2.tsv"),
    "train_source3": ("train", "train_source3.tsv"),
    "test_source1": ("test", "test_source1.tsv"),
    "test_source2": ("test", "test_source2.tsv"),
    "test_source3": ("test", "test_source3.tsv"),
}


def process_file(path, out_path, chunksize):
    writer = None
    total = 0
    t0 = time.time()
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                              na_values=[""], chunksize=chunksize):
        normed = build_normalized_frame(chunk)
        table = pa.Table.from_pandas(normed, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(chunk)
        print(f"    ... {total} rows ({time.time()-t0:.1f}s elapsed)", file=sys.stderr)
    if writer:
        writer.close()
    return total, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="../../student_resource/dataset")
    ap.add_argument("--out-dir", default="../../data_cache/normalized")
    ap.add_argument("--chunksize", type=int, default=300000)
    ap.add_argument("--only", default=None, help="comma-separated subset of FILES keys")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = FILES if not args.only else {k: FILES[k] for k in args.only.split(",")}

    tracemalloc.start()
    for label, (split, fname) in targets.items():
        in_path = os.path.join(args.dataset_dir, split, fname)
        out_path = os.path.join(args.out_dir, f"{label}.parquet")
        print(f"[cache] {label}: {in_path} -> {out_path}", file=sys.stderr)
        n, secs = process_file(in_path, out_path, args.chunksize)
        cur, peak = tracemalloc.get_traced_memory()
        print(f"[cache] {label}: {n} rows in {secs:.1f}s "
              f"(python-tracked mem: cur={cur/1e6:.0f}MB peak={peak/1e6:.0f}MB)", file=sys.stderr)
        tracemalloc.reset_peak()
        out_size = os.path.getsize(out_path) / 1e6
        print(f"[cache] {label}: parquet size = {out_size:.1f}MB", file=sys.stderr)


if __name__ == "__main__":
    main()
