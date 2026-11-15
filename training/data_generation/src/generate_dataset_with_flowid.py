#!/usr/bin/env python3
import argparse
import sys
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path


PROTO_MAP = {
    "icmp": "1",
    "tcp": "6",
    "udp": "17",
    "icmpv6": "58",
}

# Classes kept in the labels: all classes of the ground truth. The training chooses
# which ones it uses in the classes_filter of each params.ini.
USED_CLASSES = ["normal", "scanning", "ddos", "dos", "injection", "password", "xss", "backdoor",
                "ransomware", "mitm"]
# Fraction of the flows of each class that goes to train (the rest goes to test).
TRAIN_FRACTION = 0.75
# Seed of the shuffle, so the same split comes out on every run.
RANDOM_SEED = 42
# Maximum number of flows (bidirectional 5-tuples) of each class, applied before the
# train/test split: the flows whose first connection starts first in time are kept.
# A class with fewer flows keeps all of them. None keeps every flow (no limit).
MAX_FLOWS_PER_CLASS = 10000

# Columns shared by the sample (train_test_network.csv) and the full ground truth
# (Network_dataset_*.csv) that identify one connection, used to recover its ts.
# Text columns, compared as lowercase text.
JOIN_TEXT_COLUMNS = ["src_ip", "dst_ip", "proto", "conn_state", "type"]
# Integer columns, compared as numbers (so "4444" and "4444.0" are equal).
JOIN_INT_COLUMNS = ["src_port", "dst_port", "src_bytes", "dst_bytes", "missed_bytes",
                    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
# The 15 columns of the join: 5 text, 9 integer and duration.
JOIN_COLUMNS = JOIN_TEXT_COLUMNS + JOIN_INT_COLUMNS + ["duration"]
# Slack around [ts, ts + duration], in seconds. Used as 1 s when ts has no fraction.
TIME_TOLERANCE = 0.001
# Rows read at a time from each Network_dataset_*.csv, to keep memory low.
FULL_GT_CHUNK_ROWS = 1_000_000


def normalize_join_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Join columns in one comparable form, whatever their text formatting."""
    # Empty table with the same row numbers (index) as df, to be filled column by column.
    out = pd.DataFrame(index=df.index)
    for column in JOIN_TEXT_COLUMNS:
        # astype(str): every value as text; .str.strip(): removes spaces at the ends;
        # .str.lower(): lowercase, so "TCP" and "tcp" are equal.
        out[column] = df[column].astype(str).str.strip().str.lower()
    for column in JOIN_INT_COLUMNS:
        # pd.to_numeric(..., errors="coerce"): text to number; a value that is not a
        # number ("-") becomes NaN (missing). .round(): no decimals.
        # .astype("Int64"): integer type that also accepts a missing value.
        out[column] = pd.to_numeric(df[column], errors="coerce").round().astype("Int64")
    # duration: number with up to 6 decimals; a missing duration ("-") becomes 0.
    # .fillna(0): replaces NaN with 0. .round(6): same rounding on both files.
    out["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).round(6)
    return out


def flow_id_from_normalized(norm: pd.DataFrame) -> pd.Series:
    """Flow ID in the format built from tshark: 'src dst sport dport proto'."""
    # .map(function): applies the function to every value; turns "tcp" into "6".
    proto = norm["proto"].map(lambda x: PROTO_MAP.get(x, x))
    # Adding text columns with + joins them row by row: "1.1.1.1 2.2.2.2 1000 80 6".
    return (norm["src_ip"] + " " + norm["dst_ip"] + " " + norm["src_port"].astype(str) + " "
            + norm["dst_port"].astype(str) + " " + proto)


def build_connections(sample: pd.DataFrame, flows: pd.DataFrame, full_dir: str,
                      flow_column: str) -> pd.DataFrame:
    """
    Recover the start time (ts) of each sample connection from the full ground truth
    and return one time window per connection: key (undirected Flow ID), start, end,
    Label, Split and the packet count the ground truth lists for it.
    """
    from datageneration import undirected_flow_id

    # The CSVs can be in the folder or inside the zip as downloaded (read in place,
    # so the zip does not need to be extracted). Each entry is (name, function that
    # opens the file); the file is only opened when it is read below.
    files = [(path.name, lambda path=path: open(path, "rb")) for path in sorted(Path(full_dir).glob("*.csv"))]
    for archive in sorted(Path(full_dir).glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            members = sorted(name for name in zf.namelist() if name.endswith(".csv"))
        files += [(f"{archive.name}:{name}", lambda archive=archive, name=name: zipfile.ZipFile(archive).open(name))
                  for name in members]
    if not files:
        raise FileNotFoundError(f"No CSV files (or zip with CSV files) of the full ground truth in '{full_dir}'.")

    # sample["type"].isin(list): True for the rows whose class is in the list.
    # sample[mask]: keeps only those rows. reset_index(drop=True): renumbers the rows
    # 0, 1, 2..., so the row number can be used as the id of the connection.
    sample = sample[sample["type"].isin(USED_CLASSES)].reset_index(drop=True)
    sample_norm = normalize_join_columns(sample)
    # "row": the id of the sample connection, kept through the join to know which
    # sample row each full ground truth row matched.
    sample_norm["row"] = sample_norm.index
    # Undirected 5-tuple of each sample connection, and the set of all of them.
    sample_keys = undirected_flow_id(sample[flow_column])
    wanted = set(sample_keys)

    parts = []
    print(f"Reading {len(files)} CSV files of the full ground truth...")
    for name, opener in files:
        print(f"  {name}", flush=True)
        with opener() as handle:
            # pd.read_csv(..., chunksize=N) does not read the whole file: it returns
            # blocks of N rows, one at a time. dtype=str: every column as text.
            # usecols: only the 15 join columns and ts. encoding="utf-8-sig": ignores
            # the invisible mark (BOM) at the start of the file.
            for chunk in pd.read_csv(handle, dtype=str, usecols=JOIN_COLUMNS + ["ts"], encoding="utf-8-sig",
                                     chunksize=FULL_GT_CHUNK_ROWS):
                norm = normalize_join_columns(chunk)
                # True for the rows whose 5-tuple is one of the sample 5-tuples.
                # .to_numpy(): the True/False list as a plain array.
                keep = undirected_flow_id(flow_id_from_normalized(norm)).isin(wanted).to_numpy()
                if not keep.any():
                    continue
                # norm[keep]: only the rows marked True.
                norm = norm[keep]
                # ts as a number (seconds since 1970), for the same rows.
                norm["ts"] = pd.to_numeric(chunk["ts"][keep], errors="coerce")
                # .dropna(subset=["ts"]): drops the rows whose ts is missing.
                parts.append(norm.dropna(subset=["ts"]))
    # pd.concat: stacks the blocks into one table. ignore_index=True: renumbers rows.
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=JOIN_COLUMNS + ["ts"])
    print(f"Full ground truth rows on the sample 5-tuples: {len(full)}")

    # .merge(other, on=columns, how="inner"): for each sample row, finds the full rows
    # with the same values in all 15 columns and makes one output row per pair, with
    # the sample "row" and the full "ts". A sample row with no equal full row is left
    # out; one with 3 equal full rows (same connection logged 3 times at different
    # ts) gives 3 output rows.
    matched = sample_norm.merge(full, on=JOIN_COLUMNS, how="inner")
    # .groupby("row").size(): how many full rows each sample row matched.
    # .reindex(..., fill_value=0): adds the sample rows that matched nothing, with 0.
    per_row = matched.groupby("row").size().reindex(sample_norm["row"], fill_value=0)
    print(f"Sample connections: {len(sample_norm)}; matched once: {(per_row == 1).sum()}, "
          f"more than once: {(per_row > 1).sum()}, not matched: {(per_row == 0).sum()}")

    # ts % 1: the fraction of a second. If no ts has one, ts is in whole seconds and
    # the window gets 1 s of slack on each side; otherwise 1 ms.
    tolerance = TIME_TOLERANCE if (matched["ts"] % 1 != 0).any() else 1.0
    print(f"Time tolerance: {tolerance} s")
    # One row per matched pair. .iloc[positions]: takes the rows at those positions,
    # here the sample row of each pair. .to_numpy(): plain array, so the values are
    # placed in order, without pandas aligning them by row number.
    connections = pd.DataFrame({
        "key": sample_keys.iloc[matched["row"]].to_numpy(),
        "start": matched["ts"].to_numpy() - tolerance,
        "end": matched["ts"].to_numpy() + matched["duration"].to_numpy() + tolerance,
        "Label": sample["type"].iloc[matched["row"]].to_numpy(),
        "expected_pkts": (sample_norm["src_pkts"].iloc[matched["row"]].fillna(0).to_numpy()
                          + sample_norm["dst_pkts"].iloc[matched["row"]].fillna(0).to_numpy()),
    # .drop_duplicates(columns): keeps one row among those equal in these columns
    # (two identical sample rows that matched the same full row give the same window).
    }).drop_duplicates(["key", "start", "end", "Label"])

    # Dictionary 5-tuple -> "train"/"test", from labels.csv (built by build_flow_split).
    # zip(a, b): pairs the values of a and b one by one.
    flow_split = dict(zip(undirected_flow_id(flows[flow_column]), flows["Split"]))
    # .map(dictionary): replaces each key by its value; each connection gets the
    # Split of its 5-tuple. A 5-tuple not in labels.csv gets NaN.
    connections["Split"] = connections["key"].map(flow_split)
    # Drops connections without Split, sorts by 5-tuple and start, renumbers rows.
    connections = connections.dropna(subset=["Split"]).sort_values(["key", "start"]).reset_index(drop=True)
    # Count of connections and sum of expected packets per (class, split), to print.
    # .agg(name=(column, function)): one output column per pair.
    summary = connections.groupby(["Label", "Split"]).agg(connections=("key", "size"),
                                                          expected_pkts=("expected_pkts", "sum"))
    # .unstack(): turns the Split level into columns (train, test side by side).
    print(summary.unstack(fill_value=0).to_string())
    return connections[["key", "start", "end", "Label", "Split", "expected_pkts"]]


def build_flow_split(df: pd.DataFrame, flow_column: str, break_ties: bool = False) -> pd.DataFrame:
    """
    Return one row per bidirectional 5-tuple, with its majority class and a
    stratified train/test split. Tuples tied between classes are dropped, unless
    break_ties is set: then the tie goes to the class with more packets, then to
    the first class by name. Use break_ties when packets are labeled per
    connection, where the class of the tuple only stratifies the split.
    """
    # Only the used classes. .copy(): independent table, so new columns can be added.
    df = df[df["type"].isin(USED_CLASSES)].copy()
    inverse_column = flow_column + " Inv"
    # .min(axis=1): the smaller of the two texts on each row (Flow ID or its inverse),
    # so both directions of a flow get the same key.
    df["flow_key"] = df[[flow_column, inverse_column]].min(axis=1)
    # Packets of each connection: src_pkts + dst_pkts, with missing values as 0.
    df["packets"] = (pd.to_numeric(df.get("src_pkts", 0), errors="coerce").fillna(0)
                     + pd.to_numeric(df.get("dst_pkts", 0), errors="coerce").fillna(0))

    # One row per (5-tuple, class): how many connections of that class the 5-tuple
    # has (votes) and how many packets they add up to.
    # .groupby([a, b]): groups the rows with the same a and b. .agg(...): one value
    # per group. .reset_index(): turns the group columns back into normal columns.
    votes = (df.groupby(["flow_key", "Label"])
               .agg(votes=("Label", "size"), packets=("packets", "sum")).reset_index())
    # .transform("max"): the largest vote count of each 5-tuple, repeated on every
    # row of that 5-tuple.
    votes["max_votes"] = votes.groupby("flow_key")["votes"].transform("max")
    # The classes with the most votes in their 5-tuple.
    winners = votes[votes["votes"] == votes["max_votes"]]
    # .duplicated(keep=False): True for every 5-tuple that appears more than once,
    # that is, more than one class with the most votes (a tie).
    tied = winners["flow_key"].duplicated(keep=False)
    # .nunique(): number of distinct 5-tuples tied.
    n_tied = winners.loc[tied, "flow_key"].nunique()
    if break_ties:
        # Sorts each 5-tuple's candidates by packets (most first), then by class name,
        # and keeps the first row of each 5-tuple (.drop_duplicates("flow_key")).
        winners = (winners.sort_values(["flow_key", "packets", "Label"], ascending=[True, False, True])
                          .drop_duplicates("flow_key"))
        print(f"Flows with more than one class: {(votes.groupby('flow_key').size() > 1).sum()}; "
              f"ties broken by packets: {n_tied}")
    else:
        # ~tied: not tied. Keeps only the 5-tuples with a single winner.
        winners = winners.loc[~tied]
        print(f"Flows with more than one class: {(votes.groupby('flow_key').size() > 1).sum()}; "
              f"dropped as ties: {n_tied}")
    winners = winners[["flow_key", "Label"]]

    # One row per 5-tuple, with its Flow ID and inverse (from its first connection).
    flows = df.drop_duplicates("flow_key")[["flow_key", flow_column, inverse_column]]
    # Adds the winning class of each 5-tuple; 5-tuples without a winner are left out.
    # Sorted by 5-tuple so the shuffle below gives the same result on every run.
    flows = flows.merge(winners, on="flow_key").sort_values("flow_key").reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_SEED)
    # Every flow starts as test; 75% of each class are then set to train.
    flows["Split"] = "test"
    # .groupby("Label").groups: dictionary class -> row numbers of its flows.
    for label, index in flows.groupby("Label").groups.items():
        # rng.permutation: the row numbers of the class in random order.
        shuffled = rng.permutation(index.to_numpy())
        train_size = int(round(len(shuffled) * TRAIN_FRACTION))
        # .loc[rows, column] = value: sets the column on those rows.
        flows.loc[shuffled[:train_size], "Split"] = "train"
    # Number of flows per class and split, to print.
    print(flows.groupby(["Label", "Split"]).size().unstack(fill_value=0).to_string())
    return flows[[flow_column, inverse_column, "Label", "Split"]]


def split_train_test(flows: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified train/test split of the flows: TRAIN_FRACTION of the flows of each class
    go to train and the rest to test, with the same seeded shuffle of build_flow_split.
    The flows must come sorted, so the same split comes out on every run.
    """
    flows = flows.reset_index(drop=True).copy()
    rng = np.random.default_rng(RANDOM_SEED)
    # Every flow starts as test; TRAIN_FRACTION of each class are then set to train.
    flows["Split"] = "test"
    # .groupby("Label").groups: dictionary class -> row numbers of its flows.
    for label, index in flows.groupby("Label").groups.items():
        # rng.permutation: the row numbers of the class in random order.
        shuffled = rng.permutation(index.to_numpy())
        train_size = int(round(len(shuffled) * TRAIN_FRACTION))
        flows.loc[shuffled[:train_size], "Split"] = "train"
    return flows


def limit_flows_per_class(flows: pd.DataFrame, connections: pd.DataFrame, flow_column: str,
                          max_flows: int) -> pd.DataFrame:
    """
    Keep at most max_flows flows of each class, the ones whose first connection starts
    first, and split them again into train and test (split_train_test).
    flows: one row per 5-tuple (labels.csv). connections: one row per connection, with
    the undirected 5-tuple in "key" and its start time (connections.csv).
    """
    from datageneration import undirected_flow_id

    # Start of the first connection of each 5-tuple. .groupby("key")["start"].min():
    # the smallest start among the connections of the same 5-tuple.
    first_start = connections.groupby("key")["start"].min()
    flows = flows.copy()
    # Undirected 5-tuple of each flow, the same key used in connections.csv.
    flows["key"] = undirected_flow_id(flows[flow_column]).to_numpy()
    # .map(first_start): the start of each flow. A flow without any connection gets NaN:
    # it has no time window, so none of its packets is labeled, and it is dropped.
    flows["first_start"] = flows["key"].map(first_start)
    flows = flows.dropna(subset=["first_start"])
    # Sorted by class, then by start (ties by 5-tuple), and .groupby("Label").head(n):
    # the first n flows of each class.
    kept = (flows.sort_values(["Label", "first_start", "key"])
                 .groupby("Label").head(max_flows)
                 .sort_values("key"))
    kept = split_train_test(kept)
    print(f"Flows kept per class (at most {max_flows}, the first in time):")
    print(kept.groupby(["Label", "Split"]).size().unstack(fill_value=0).to_string())
    return kept.drop(columns=["key", "first_start"])


def apply_flow_limit(flows: pd.DataFrame, connections: pd.DataFrame, flow_column: str,
                     max_flows: int):
    """
    Limit the flows of each class (limit_flows_per_class) and give each connection the
    new split of its 5-tuple; the connections of the dropped flows are dropped.
    Returns the new flows and connections.
    """
    flows = limit_flows_per_class(flows, connections, flow_column, max_flows)
    from datageneration import undirected_flow_id
    # Dictionary 5-tuple -> "train"/"test" with the new split.
    flow_split = dict(zip(undirected_flow_id(flows[flow_column]), flows["Split"]))
    connections = connections.copy()
    connections["Split"] = connections["key"].map(flow_split)
    connections = connections.dropna(subset=["Split"]).reset_index(drop=True)
    return flows, connections


def limit_existing_files(labels_csv: str, connections_csv: str, max_flows: int = MAX_FLOWS_PER_CLASS,
                         flow_column: str = "Flow ID"):
    """
    Apply the flow limit to labels.csv and connections.csv already built, without
    reading the ground truth again. The original files are kept with the suffix
    _all_flows (e.g. labels_all_flows.csv), and the limited ones take their place.
    """
    labels_path, connections_path = Path(labels_csv), Path(connections_csv)
    labels_backup = labels_path.with_name(labels_path.stem + "_all_flows.csv")
    connections_backup = connections_path.with_name(connections_path.stem + "_all_flows.csv")
    # The originals are read from the backup when it exists, so running this twice
    # limits the full files again instead of the already limited ones.
    if not labels_backup.exists():
        labels_path.rename(labels_backup)
    if not connections_backup.exists():
        connections_path.rename(connections_backup)
    flows = pd.read_csv(labels_backup, dtype=str)
    connections = pd.read_csv(connections_backup, dtype={"key": str, "Label": str, "Split": str})
    flows, connections = apply_flow_limit(flows, connections, flow_column, max_flows)
    flows.to_csv(labels_path, index=False)
    connections.to_csv(connections_path, index=False)
    print(f"Wrote {len(flows)} flows to {labels_path} and {len(connections)} connections to {connections_path}")


def normalize_transport_proto(df: pd.DataFrame) -> pd.Series:
    if "transport_proto" in df.columns:
        return df["transport_proto"].astype(str).str.strip()

    if "proto" not in df.columns:
        raise ValueError(
            "CSV must contain either 'transport_proto' or 'proto' column."
        )

    proto = df["proto"].astype(str).str.strip().str.lower()
    return proto.map(lambda x: PROTO_MAP.get(x, x))


def main(input_dir: str, output_csv: str, flow_column: str = "Flow ID",
         full_ground_truth_dir: str = None, connections_csv: str = None) -> int:
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
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # One row per 5-tuple with its class and train/test split. With connections,
    # ties are broken instead of dropped (packets get the class of their connection).
    flows = build_flow_split(df, flow_column, break_ties=bool(connections_csv))
    flows.to_csv(output_csv, index=False)
    print(f"Success! Wrote {len(flows)} flows to {output_csv}")

    # Time window of each connection, used to label packets by 5-tuple and time.
    if connections_csv:
        if not full_ground_truth_dir:
            raise ValueError("connections_csv needs full_ground_truth_dir (the Network_dataset_*.csv files).")
        connections = build_connections(df, flows, full_ground_truth_dir, flow_column)
        # The full files are kept with the suffix _all_flows, and the limited ones
        # (at most MAX_FLOWS_PER_CLASS flows per class) take the usual names.
        if MAX_FLOWS_PER_CLASS:
            flows.to_csv(Path(output_csv).with_name(Path(output_csv).stem + "_all_flows.csv"), index=False)
            Path(connections_csv).parent.mkdir(parents=True, exist_ok=True)
            connections.to_csv(Path(connections_csv).with_name(Path(connections_csv).stem + "_all_flows.csv"),
                               index=False)
            flows, connections = apply_flow_limit(flows, connections, flow_column, MAX_FLOWS_PER_CLASS)
            flows.to_csv(output_csv, index=False)
            print(f"Success! Wrote {len(flows)} limited flows to {output_csv}")
        Path(connections_csv).parent.mkdir(parents=True, exist_ok=True)
        connections.to_csv(connections_csv, index=False)
        print(f"Success! Wrote {len(connections)} connections to {connections_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
