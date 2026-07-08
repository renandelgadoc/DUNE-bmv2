#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
from pathlib import Path


PROTO_MAP = {
    "icmp": "1",
    "tcp": "6",
    "udp": "17",
    "icmpv6": "58",
}

def normalize_transport_proto(df: pd.DataFrame) -> pd.Series:
    if "transport_proto" in df.columns:
        return df["transport_proto"].astype(str).str.strip()

    if "proto" not in df.columns:
        raise ValueError(
            "CSV must contain either 'transport_proto' or 'proto' column."
        )

    proto = df["proto"].astype(str).str.strip().str.lower()
    return proto.map(lambda x: PROTO_MAP.get(x, x))


def main(input_dir: str, output_csv: str, flow_column: str = "Flow ID") -> int:
    input_path = Path(input_dir)

    if not input_path.is_dir():
        print(f"Error: Directory '{input_path}' does not exist or is not a directory.", file=sys.stderr)
        return 1

    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{input_path}'.", file=sys.stderr)
        return 1

    dfs = []
    print(f"Found {len(csv_files)} CSV files. Reading and concatenating...")
    for file in csv_files:
        try:
            df_part = pd.read_csv(file, dtype={"src_bytes": str})
            df_part['src_bytes'] = pd.to_numeric(df_part['src_bytes'], errors='coerce')
            df_part = df_part.dropna(subset=['src_bytes'])
            df_part['src_bytes'] = df_part['src_bytes'].astype(int)
            dfs.append(df_part)
        except Exception as e:
            print(f"Warning: Could not read {file}. Error: {e}", file=sys.stderr)

    if not dfs:
        print("Error: No valid data could be read from the CSV files.", file=sys.stderr)
        return 1

    df = pd.concat(dfs, ignore_index=True)
    print(f"Successfully concatenated into a DataFrame with {len(df)} rows. Processing...")

    required = ["src_ip", "dst_ip", "src_port", "dst_port"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in the merged dataset: {missing}")

    transport_proto = normalize_transport_proto(df)

    df[flow_column] = (
        df["src_ip"].astype(str).str.strip() + " " +
        df["dst_ip"].astype(str).str.strip() + " " +
        df["src_port"].astype(str).str.strip() + " " +
        df["dst_port"].astype(str).str.strip() + " " +
        transport_proto.astype(str).str.strip()
    )
    
    df[flow_column + " Inv"] = (
        df["dst_ip"].astype(str).str.strip() + " " +
        df["src_ip"].astype(str).str.strip() + " " +
        df["dst_port"].astype(str).str.strip() + " " +
        df["src_port"].astype(str).str.strip() + " " +
        transport_proto.astype(str).str.strip()
    )

    if "type" not in df.columns:
        raise ValueError("CSV must contain a 'type' column to create 'Label'.")
    df["Label"] = df["type"]

    # 6. Save the final output
    df.to_csv(output_csv, index=False)
    print(f"Success! Wrote {len(df)} rows to {output_csv}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())