from collections import defaultdict

import os

import numpy as np
import pandas as pd
import subprocess
import glob
from dataclasses import dataclass
from typing import List

flow_pkt_feature_map = {'Min Packet Length': 'ip.len', 'Max Packet Length': 'ip.len',
                           'Packet Length Mean': 'ip.len', 'Packet Length Total': 'ip.len',
                           'UDP Len Min': 'udp.length', 'UDP Len Max': 'udp.length',
                           'Flow IAT Min': 'flow_iat', 'Flow IAT Max': 'flow_iat',
                           'Flow IAT Mean': 'flow_iat', 'Time to Inference': 'frame.time_relative',
                           'SYN Flag Count': 'tcp.flags.syn', 'ACK Flag Count': 'tcp.flags.ack',
                           'PSH Flag Count': 'tcp.flags.push', 'FIN Flag Count': 'tcp.flags.fin',
                           'RST Flag Count': 'tcp.flags.reset', 'ECE Flag Count': 'tcp.flags.ece'}
flow_feature_names = list(flow_pkt_feature_map.keys())
flow_time_feature_names = ['Flow IAT Min', 'Flow IAT Max', 'Flow IAT Mean', 'Time to Inference']
flow_op_feature_map = {'Min Packet Length': 'min', 'Max Packet Length': 'max',
                           'Packet Length Mean': 'mean', 'Packet Length Total': 'sum',
                           'UDP Len Min': 'min', 'UDP Len Max': 'max',
                           'Flow IAT Min': 'min', 'Flow IAT Max': 'min',
                           'Flow IAT Mean': 'mean', 'Time to Inference': 'ptp',
                           'SYN Flag Count': 'sum', 'ACK Flag Count': 'sum',
                           'PSH Flag Count': 'sum', 'FIN Flag Count': 'sum',
                           'RST Flag Count': 'sum', 'ECE Flag Count': 'sum'}




@dataclass
class DataGenerationConfig:
    use_case: str
    labels_path: str  # The path to the file containing the labels map
    output_path: str  # The path to the data location
    inference_points_list: List[int]  # A list of inference points
    ground_truth_path: str  # The path to the ground truth data (pcap files)
    split: str  # 'train' or 'test'
    # Connections with their time window (see generate_dataset_with_flowid.py).
    # When set, packets are labeled by 5-tuple and time instead of 5-tuple only.
    connections_path: str = None




def parse_tshark_csv_to_dataframe(filename_in):
    """
    Parses the output from tshark into a pandas DataFrame, allowing analysis of packet
    data and simplifying further processing.

    This function reads the input file containing packet capture data, processes it
    by selecting specified columns, applies appropriate data types, and calculates
    new features needed for flow identification. The resulting dataframe
    offers a structured representation for further processing or analysis.

    Parameters:
    filename_in: str
        Path to the input file containing tshark tool output in CSV format.

    Returns:
    pd.DataFrame
        A processed DataFrame where columns represent selected packet information.
        This DataFrame includes derived features such as source/destination ports
        and flow identifiers.

    Raises:
    FileNotFoundError
        If the input file specified by filename_in does not exist.
    ValueError
        If the input file is not formatted as expected to match selected columns or
        data types.
    """

    dtypes=defaultdict(lambda: 'Int64')
    dtypes["ip.src"] = "str"
    dtypes["ip.dst"] = "str"
    dtypes["eth.src"] = "str"
    dtypes["eth.dst"] = "str"
    dtypes["frame.time_relative"] = "float"
    dtypes["frame.time_epoch"] = "float64"

    for col in [
        "tcp.flags.syn",
        "tcp.flags.ack",
        "tcp.flags.push",
        "tcp.flags.fin",
        "tcp.flags.reset",
        "tcp.flags.ece",
    ]:
        dtypes[col] = "boolean"

    # ToDo: this hardcodes the pcap columns. Make this configurable
    packet_data_cols = ["frame.number","frame.time_relative","frame.time_epoch","ip.src","ip.dst","tcp.srcport","tcp.dstport","ip.len",
                    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.push", "tcp.flags.fin",
                    "tcp.flags.reset", "tcp.flags.ece","ip.proto", "udp.srcport", "udp.dstport",
                    "eth.src","eth.dst", "ip.hdr_len", "ip.tos", "ip.ttl", "tcp.window_size_value",
                    "tcp.hdr_len", "udp.length"]

    packet_data = pd.read_csv(filename_in, usecols=packet_data_cols, dtype=dtypes)

    defaults = {
        "Int64": 0,
        "float64": 0.0,
        "object": "",
        "boolean": False
    }
    fill_values = {
        col: defaults[str(packet_data[col].dtype)]
        for col in packet_data.columns
        if str(packet_data[col].dtype) in defaults
    }

    packet_data = packet_data.fillna(fill_values)

    # divide by 4, since the tcp_offset counts 32-bit words
    packet_data["tcp.hdr_len"] = packet_data["tcp.hdr_len"]/4

    packet_data["srcport"] = np.where(((packet_data["ip.proto"] == 6)), packet_data["tcp.srcport"], packet_data["udp.srcport"])
    packet_data["dstport"] = np.where(((packet_data["ip.proto"] == 6)), packet_data["tcp.dstport"], packet_data["udp.dstport"])

    packet_data = packet_data.drop(["tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"],axis=1)
    packet_data = packet_data.reset_index(drop=True)

    packet_data["Flow ID"] = packet_data[["ip.src", "ip.dst", "srcport", "dstport", "ip.proto"]].astype(str).agg(
        " ".join,
        axis=1)

    return packet_data

def label_packet_data(packet_data: pd.DataFrame, config: DataGenerationConfig):
    """
    Labels the packet data based on the specified configuration and use case. The function assigns
    labels derived from the given label data to the packet data and filters the data accordingly
    based on the use case configuration.

    Args:
        packet_data (pd.DataFrame): The DataFrame containing the packet data that needs to be labeled.
        config (DataGenerationConfig): Configuration object containing paths and parameters for
            data labeling, including `labels_path` and `use_case`.

    Returns:
        pd.DataFrame: The labeled and filtered packet data.
    """
    label_data = pd.read_csv(config.labels_path)

    if config.use_case == "UNSW":
        packet_data["Label"] = [0] * len(packet_data)
        for i in range(len(label_data)):
            packet_data["Label"] = np.where((packet_data["eth.src"]==label_data["MAC ADDRESS"][i]),
                                                label_data["List of Devices"][i], packet_data["Label"])
        for i in range(len(label_data)):
            packet_data["Label"] = np.where((packet_data["eth.dst"] ==label_data["MAC ADDRESS"][i]) &
                                            (packet_data["eth.src"]=="14:cc:20:51:33:ea"),
                                            label_data["List of Devices"][i], packet_data["Label"])

        packet_data = packet_data[packet_data['Label']!="TPLink Router Bridge LAN (Gateway)"]
        packet_data = packet_data[packet_data['Label']!="0"]
        packet_data = packet_data[packet_data['Label']!="Nest Dropcam"]
        packet_data = packet_data[packet_data['Label']!="MacBook/Iphone"]
        packet_data = packet_data.reset_index(drop=True)

    # With the connection windows (connections.csv), a packet is labeled by its 5-tuple
    # AND its time, so packets of connections outside the ground truth are dropped.
    if config.use_case == "TON-IOT" and config.connections_path:
        return label_packets_by_time(packet_data, config.connections_path)

    # Without them, a packet is labeled by its 5-tuple only.
    if config.use_case == "TON-IOT":
        # .set_index("Flow ID")["Label"].to_dict(): dictionary Flow ID -> class.
        # The second dictionary does the same for the inverse direction, and | joins
        # both, so packets of either direction find their flow.
        flow_id_label_dict = label_data.set_index("Flow ID")["Label"].to_dict()
        flow_id_label_dict = flow_id_label_dict | label_data.set_index("Flow ID Inv")["Label"].to_dict()
        packet_data.loc[packet_data['Flow ID'].isin(flow_id_label_dict.keys()), "Label"] = packet_data["Flow ID"].map(flow_id_label_dict)
        if "Split" in label_data.columns:
            # Same for the train/test split of the flow: Flow ID -> "train"/"test".
            # .map(dictionary): each packet gets the value of its Flow ID, or NaN.
            flow_id_split_dict = label_data.set_index("Flow ID")["Split"].to_dict()
            flow_id_split_dict = flow_id_split_dict | label_data.set_index("Flow ID Inv")["Split"].to_dict()
            packet_data["Split"] = packet_data["Flow ID"].map(flow_id_split_dict)

    return packet_data

def undirected_flow_id(flow_ids: pd.Series) -> pd.Series:
    """Same key for both directions: the smaller of the Flow ID and its inverse."""
    # .str.split(" ", expand=True): splits "src dst sport dport proto" into 5 columns
    # (0 = src, 1 = dst, 2 = sport, 3 = dport, 4 = proto).
    parts = flow_ids.str.split(" ", expand=True)
    # The same flow seen from the other side: "dst src dport sport proto".
    inverse = parts[1] + " " + parts[0] + " " + parts[3] + " " + parts[2] + " " + parts[4]
    # np.where(condition, a, b): a where the condition is True, b elsewhere. Keeps the
    # alphabetically smaller text, so both directions give the same key.
    # index=flow_ids.index: keeps the row numbers of the input.
    return pd.Series(np.where(flow_ids < inverse, flow_ids, inverse), index=flow_ids.index)


def match_packets_to_connections(packets: pd.DataFrame, connections: pd.DataFrame) -> pd.DataFrame:
    """
    Match each packet to the ground truth connection whose window [start, end] holds its
    absolute time (frame.time_epoch), among the connections of its undirected 5-tuple
    (column '_key'). The connection that started last is used; when it already ended,
    the earlier connection of the same 5-tuple whose window ends last is used, so a long
    connection still gets its packets after a shorter one started. Returns the matched
    packets with the Label, Split and row ('conn') of their connection; packets outside
    every window are left out.
    """
    # Connections sorted by 5-tuple and start time, numbered 0, 1, 2... ("conn").
    # kind="stable": rows with equal values keep their original order.
    connections = connections.sort_values(["key", "start"], kind="stable").reset_index(drop=True)
    connections["conn"] = connections.index
    # .groupby("key")["end"].cummax(): for each connection, the latest end among it and
    # the connections of the same 5-tuple that started before it. If a packet comes
    # after the end of the latest connection but before max_end, an earlier connection
    # of the same 5-tuple is still open.
    connections["max_end"] = connections.groupby("key")["end"].cummax()
    # True on the connections that hold that latest end themselves.
    holds_max = connections["end"] >= connections["max_end"]
    # Which connection holds max_end: .where(holds_max) keeps "conn" only on those rows
    # (NaN elsewhere); .groupby("key").ffill() fills each NaN with the last value above
    # it within the same 5-tuple. .astype("int64"): back to integers.
    connections["max_conn"] = (connections["conn"].where(holds_max)
                               .groupby(connections["key"]).ffill().astype("int64"))

    # pd.merge_asof needs both sides sorted by the time columns it compares.
    packets = packets.sort_values("frame.time_epoch", kind="stable")
    right = connections.sort_values("start", kind="stable")[["key", "start", "end", "max_end", "conn", "max_conn"]]
    # pd.merge_asof: for each packet, the connection with the same 5-tuple
    # (left_by="_key" equals right_by="key") whose start is the closest one at or
    # before the packet time (direction="backward"), that is, the connection of that
    # 5-tuple that started last before the packet. A packet with no such connection
    # gets NaN in the connection columns.
    matched = pd.merge_asof(packets, right, left_on="frame.time_epoch", right_on="start",
                            left_by="_key", right_by="key", direction="backward")
    time = matched["frame.time_epoch"]
    # own: that connection is still open at the packet time (time <= end).
    own = time <= matched["end"]
    # earlier: it already ended, but an earlier connection of the same 5-tuple is
    # still open (time <= max_end). NaN compares as False, so unmatched packets are
    # neither own nor earlier.
    earlier = ~own & (time <= matched["max_end"])
    # Keeps only the packets inside some window; the rest are dropped.
    matched = matched[own | earlier].copy()
    # The connection of each packet: its own, or the earlier one still open.
    matched["conn"] = np.where(own[own | earlier], matched["conn"], matched["max_conn"]).astype("int64")
    # array[positions]: the Label and Split of the chosen connection of each packet.
    matched["Label"] = connections["Label"].to_numpy()[matched["conn"].to_numpy()]
    matched["Split"] = connections["Split"].to_numpy()[matched["conn"].to_numpy()]
    # Removes the helper columns; keeps the packet columns, conn, Label and Split.
    return matched.drop(columns=["key", "start", "end", "max_end", "max_conn"]).reset_index(drop=True)


def label_packets_by_time(packet_data: pd.DataFrame, connections_path: str) -> pd.DataFrame:
    """
    Give each packet the Label and Split of the ground truth connection it belongs to:
    same 5-tuple in either direction and an absolute time (frame.time_epoch) inside the
    connection window [start, end]. Packets outside every window are dropped.
    """
    # One row per connection: key, start, end, Label, Split, expected_pkts.
    # dtype: reads the 5-tuple, class and split as text and the times as decimals.
    connections = pd.read_csv(connections_path, dtype={"key": str, "Label": str, "Split": str,
                                                       "start": "float64", "end": "float64"})
    # Undirected 5-tuple of each packet, comparable to connections["key"].
    packet_data["_key"] = undirected_flow_id(packet_data["Flow ID"])
    # Keeps only packets whose 5-tuple has a connection in the ground truth; most
    # packets of the pcap are dropped here, which keeps the matching below small.
    packets = packet_data[packet_data["_key"].isin(set(connections["key"]))]
    if packets.empty:
        # No packet of this pcap is in the ground truth: empty table with the columns
        # the caller expects (.assign adds the empty Label and Split columns).
        return packets.drop(columns="_key").assign(Label=pd.Series(dtype=str), Split=pd.Series(dtype=str))
    labeled = match_packets_to_connections(packets, connections)
    # Removes the helper columns.
    return labeled.drop(columns=["_key", "conn"])


def process_pcap_to_csv(pcap_file, output_file):
    """
    Processes a PCAP file and exports relevant extracted fields into a CSV-like formatted text file.

    Parameters
    ----------
    pcap_file_name : str
        The name of the PCAP file to be processed.

    Raises
    ------
    subprocess.CalledProcessError
        If the external `tshark` command fails or encounters an error during execution.

    Notes
    -----
    The function extracts specific fields from the given PCAP file using the `tshark` tool. Filtered data, excluding
    ICMP traffic but including TCP and UDP traffic, is written to an output file located in a subdirectory of `self.data_path`.
    Configuration of the fields and their format is managed by the `tshark` command options provided.
    """
    command = [
        "tshark", "-r", pcap_file, "-Y", "not icmp and (tcp or udp)", "-T", "fields",
        "-e", "frame.number", "-e", "frame.time_relative", "-e", "frame.time_epoch",
        "-e", "ip.src", "-e", "ip.dst",
        "-e", "tcp.srcport", "-e", "tcp.dstport", "-e", "ip.len",
        "-e", "tcp.flags.syn", "-e", "tcp.flags.ack", "-e", "tcp.flags.push",
        "-e", "tcp.flags.fin", "-e", "tcp.flags.reset", "-e", "tcp.flags.ece",
        "-e", "ip.proto", "-e", "udp.srcport", "-e", "udp.dstport",
        "-e", "eth.src", "-e", "eth.dst", "-e", "ip.hdr_len", "-e", "ip.tos",
        "-e", "ip.ttl", "-e", "tcp.window_size_value", "-e", "tcp.hdr_len", "-e", "udp.length",
        "-E", "separator=,", "-E", "quote=n", "-E", "header=y"
    ]

    with open(output_file, "w") as out_file:
        subprocess.run(command, stdout=out_file, check=True)

def frames_to_ranges(frame_numbers):
    """Compact sorted frame numbers into editcap selections such as '3-90'."""
    ranges = []
    start = previous = None
    for frame in frame_numbers:
        if start is None:
            # First frame: opens a range.
            start = previous = frame
        elif frame == previous + 1:
            # Next frame in sequence: extends the open range.
            previous = frame
        else:
            # Gap: closes the open range ("3-90", or "3" for a single frame) and opens
            # a new one.
            ranges.append(f"{start}-{previous}" if previous != start else f"{start}")
            start = previous = frame
    # Closes the last range.
    if start is not None:
        ranges.append(f"{start}-{previous}" if previous != start else f"{start}")
    return ranges


def extract_frames(pcap_file, frame_numbers, output_file, block_size=500):
    """
    Copy the selected frames of pcap_file to output_file with editcap, in
    blocks because editcap silently ignores selections beyond 512, joining the
    blocks with mergecap.
    """
    ranges = frames_to_ranges(sorted(int(frame) for frame in frame_numbers))
    parts = []
    try:
        # One editcap per block of up to block_size ranges. "editcap -r in out 3-90 95"
        # keeps (-r) only those frames of the input pcap.
        for i in range(0, len(ranges), block_size):
            part = f"{output_file}.part{len(parts)}"
            subprocess.run(["editcap", "-r", pcap_file, part] + ranges[i:i + block_size], check=True)
            parts.append(part)
        if len(parts) == 1:
            os.replace(parts[0], output_file)
        else:
            # Joins the blocks into one pcap, in time order.
            subprocess.run(["mergecap", "-w", output_file] + parts, check=True)
    finally:
        # Removes the block files, even if editcap or mergecap failed.
        for part in parts:
            if os.path.exists(part):
                os.remove(part)


def get_flow_length(data_config: DataGenerationConfig, split: str):
    """
    Fetch and process the packet data (CSV) files to calculate each flow's length.

    This function extracts the columns ('Flow ID' and 'ip.len'),
    computes the count of occurrences for each 'Flow ID', and adds additional metadata such as the file source.
    It then combines the processed data from all files into a single DataFrame.

    Parameters:
        data_config (DataGenerationConfig): Configuration object
        split (str): 'train' or 'test' to filter which files to process

    Returns:
        pd.DataFrame: A DataFrame consolidating information on each unique 'Flow ID'.
            The DataFrame contains:
            - 'Flow ID': Identifier of network flows.
            - 'ip.len': Length of the IP packets in each flow.
            - 'count': Number of occurrences of each 'Flow ID' in the source file.
            - 'File': Filename from which the data is extracted.

    Raises:
        FileNotFoundError: If the provided directory in `data_config.data_path` does
            not exist or contains no CSV files.
        ValueError: If an error occurs during data concatenation due to mismatched schema
            across the files.
    """
    # Create a list to store the data frames
    flow_length_dfs = []
    filenames = glob.glob(f"{data_config.output_path}/{split}/csv_files/*_{split}.csv")
    for filename in filenames:
        df = pd.read_csv(filename, usecols=['Flow ID', 'ip.len', 'Label'])
        # Count occurrences of each category and add as a new column
        df['count'] = df.groupby('Flow ID')['Flow ID'].transform('count')
        df['File'] = filename.split('/')[-1].split('.')[0]
        df = df.drop_duplicates(subset=['Flow ID'])
        df['type'] = df['Label']
        flow_length_dfs.append(df)

    if not flow_length_dfs:
        return pd.DataFrame()

    flow_length_df = pd.concat(flow_length_dfs)
    flow_length_df = flow_length_df.dropna()

    # Each file was de-duplicated on its own, but the SAME Flow ID (broadcast /
    # multicast 5-tuples such as NetBIOS/mDNS) shows up in several scenario
    # files, so the concat above still holds one row per (Flow ID, File). The
    # simulation classifies a flow by its 5-tuple, so all those instances are a
    # single flow -> collapse to ONE row per Flow ID. Sum the packet counts
    # across files; the label is consistent per Flow ID so keep the first.
    if flow_length_df['Flow ID'].duplicated().any():
        flow_length_df = (
            flow_length_df
            .groupby('Flow ID', as_index=False)
            .agg({'ip.len': 'first', 'Label': 'first', 'count': 'sum',
                  'File': 'first', 'type': 'first'})
        )

    return flow_length_df

def save_merged_data(data_config: DataGenerationConfig):
    """
    Save merged data from multiple CSV files into separate train and test CSV files.

    The function processes data files matching a specific pattern, combines them into a single
    dataframe for each inference point value and split (train/test), and then saves the merged data into new files.

    Parameters:
        data_config (DataGenerationConfig): Configuration object containing parameters such as
            the data path, inference points list, and use case.

    Raises:
        None
    """
    # Iterate over the values of N
    for n in data_config.inference_points_list:
        for split in ['train', 'test']:
            filenames = sorted(glob.glob(f"{data_config.output_path}/{split}/hybrid_data/{data_config.use_case}_{split}_*_N_{n}.csv"))
            if not filenames:
                continue

            output_parquet = f"{data_config.output_path}/{split}/hybrid_data/{split}_{n}_pkts.parquet"
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq_writer = None
            for filename in filenames:
                df = pd.read_csv(filename)
                df = df.dropna()
                if not df.empty:
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if pq_writer is None:
                        pq_writer = pq.ParquetWriter(output_parquet, table.schema, compression='snappy')
                    pq_writer.write_table(table)
            if pq_writer is not None:
                pq_writer.close()

def generate_hybrid_data(packet_data, csv_file_name, n):
    """
    Generate hybrid features for packet data by grouping and processing specified columns.

    The function processes input packet data by computing flow features over the n first packets for each flow. Then,
    it assigns these features to consecutive packets. Additionally, it handles conversions of time-related features.

    Parameters:
        packet_data: pandas.DataFrame
            The input DataFrame containing packet-related data with columns necessary for
            feature computations (e.g., 'Flow ID', 'frame.time_relative').
        csv_file_name: str
            The name of the CSV file associated with the processed packet data.
        n: int
            The inference point over which to compute flow features.

    Returns:
        pandas.DataFrame: The updated DataFrame with hybrid features and the associated file name.
    """

    # Sort by group key and time
    packet_data = packet_data.sort_values(by=['Flow ID', 'frame.time_relative'])

    # Compute the packet number
    packet_data['pkt_number'] = (
            packet_data
            .groupby('Flow ID')  # Group by the 'key' column
            .cumcount() + 1  # Create a cumulative count starting at 1
    )

    # Compute flow_iat
    packet_data['flow_iat'] = packet_data.groupby('Flow ID')['frame.time_relative'].diff().fillna(0)

    packet_data.loc[packet_data['pkt_number'] < n, flow_feature_names] = -1

    # count_feature_names = (feature for feature in flow_feature_names if 'Count' in feature)
    for feature in flow_feature_names:
        op = flow_op_feature_map[feature]
        # handle the ptp case
        if op == 'ptp':
            op = np.ptp

        # Step 1: Compute the group-wise sum of 'tcp.flags.syn' for rows where 'pkt_number' < n
        group_aggregations = (
            packet_data[packet_data['pkt_number'] <= n]
            .groupby('Flow ID')[flow_pkt_feature_map[feature]]
            .agg(op)
        )

        # Step 2: Map the group sums back to rows where 'pkt_number' >= n
        packet_data.loc[
            packet_data['pkt_number'] >= n,
            feature
        ] = packet_data.loc[
            packet_data['pkt_number'] >= n, 'Flow ID'
        ].map(group_aggregations)

        # Step 3: convert time features to ns
        if feature in flow_time_feature_names:
            packet_data.loc[packet_data['pkt_number'] >= n, feature] = (
                    packet_data.loc[packet_data['pkt_number'] >= n, feature] * 10e9).round(9)

    packet_data = packet_data.assign(File=csv_file_name)

    return packet_data
