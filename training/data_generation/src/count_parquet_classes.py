#!/usr/bin/env python3
"""Count rows per class (Label) in a parquet file.

Usage:
    python3 count_parquet_classes.py [parquet_path]

Defaults to the hybrid test set. Reads only the 'Label' column (memory-safe).
Prints the per-class counts and also writes them to
<basename>_class_counts.csv next to this script.
"""
import sys
import os
import collections
import csv
import pyarrow.parquet as pq

DEFAULT = "./training/data_generation/src/output/train/hybrid_data/train_3_pkts.parquet"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    vc = collections.Counter(
        pq.ParquetFile(path).read(columns=["Label"]).column("Label").to_pylist()
    )
    total = sum(vc.values())

    rows = sorted(vc.items(), key=lambda x: -x[1])
    print(f"{'class':<12}{'count':>14}")
    for c, n in rows:
        print(f"{c:<12}{n:>14,}")
    print(f"{'TOTAL':<12}{total:>14,}")

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.splitext(os.path.basename(path))[0] + "_class_counts.csv",
    )
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for c, n in rows:
            w.writerow([c, n])
        w.writerow(["TOTAL", total])
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
