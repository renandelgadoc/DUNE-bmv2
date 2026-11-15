import configparser
import logging
import os
import multiprocessing as mp
import subprocess
import sys
import glob
import time
from datageneration import *
import generate_dataset_with_flowid
from os import path
from ast import literal_eval
from itertools import product
                
OUTPUT_COLUMNS = ['Flow ID', 'ip.len', 'ip.ttl', 'tcp.flags.syn', 'tcp.flags.ack',
                  'tcp.flags.push', 'tcp.flags.fin', 'tcp.flags.reset', 'tcp.flags.ece',
                  'ip.proto', 'srcport', 'dstport', 'ip.hdr_len', 'ip.tos',
                  'tcp.window_size_value', 'tcp.hdr_len', 'udp.length',
                  'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                  'Packet Length Total', 'UDP Len Min', 'UDP Len Max',
                  'Flow IAT Min', 'Flow IAT Max', 'Flow IAT Mean',
                  'Time to Inference', 'SYN Flag Count', 'ACK Flag Count',
                  'PSH Flag Count', 'FIN Flag Count', 'RST Flag Count',
                  'ECE Flag Count', 'Label', 'File']

def run_data_generation(input_data):
    filename = input_data[0]
    data_generator_obj = input_data[1]
    try:
        return __run_data_generation(filename, data_generator_obj)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []

def __run_data_generation(file: str, config: DataGenerationConfig):
    logger.info(f"Starting the data generation for File: {file}")

    pcap_file_path = os.path.join(file['path'], file['name'])
    pcap_file_name = file['name'].split('.')[0]
    csv_file_name = f"{pcap_file_name}.csv"

    tshark_file_path = os.path.join(config.output_path, config.split, 'tshark_files', f"{pcap_file_name}.csv")
    hybrid_results_dir = f"{config.output_path}/{config.split}/hybrid_data"

    all_outputs_exist = all(
        os.path.exists(f"{hybrid_results_dir}/{config.use_case}_{config.split}_{pcap_file_name}_N_{n}.csv")
        for n in config.inference_points_list
    )
    if all_outputs_exist:
        logger.info(f"Skipping (already processed): {file['name']}")
        return []

    process_pcap_to_csv(pcap_file_path, tshark_file_path)
    packet_data = parse_tshark_csv_to_dataframe(tshark_file_path)

    labeled_packet_data = label_packet_data(packet_data, config)

    # Drop packets that could not be labeled (NaN Label) to prevent NaN errors downstream
    labeled_packet_data = labeled_packet_data.dropna(subset=['Label'])

    output_columns = ['Flow ID', 'ip.len', 'ip.ttl', 'tcp.flags.syn', 'tcp.flags.ack',
                    'tcp.flags.push', 'tcp.flags.fin', 'tcp.flags.reset', 'tcp.flags.ece',
                    'ip.proto', 'srcport', 'dstport', 'ip.hdr_len', 'ip.tos',
                    'tcp.window_size_value', 'tcp.hdr_len', 'udp.length',
                    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                    'Packet Length Total', 'UDP Len Min', 'UDP Len Max',
                    'Flow IAT Min', 'Flow IAT Max', 'Flow IAT Mean',
                    'Time to Inference', 'SYN Flag Count', 'ACK Flag Count',
                    'PSH Flag Count', 'FIN Flag Count', 'RST Flag Count',
                    'ECE Flag Count', 'Label', 'File']

    packet_data_split = labeled_packet_data
    if not packet_data_split.empty:
        packet_data_split.to_csv(f"{config.output_path}/{config.split}/csv_files/{pcap_file_name}_{config.split}.csv", index=False)

    for n in config.inference_points_list:
        if not packet_data_split.empty:
            hybrid_split = generate_hybrid_data(packet_data_split.copy(), csv_file_name, n)
            # Drop rows with any NaN (malformed attack packets with switched fields)
            hybrid_split = hybrid_split.dropna()
            hybrid_split.to_csv(f"{hybrid_results_dir}/{config.use_case}_{config.split}_{pcap_file_name}_N_{n}.csv", columns=output_columns, index=False)

def __run_flow_split_generation(file: dict, config: DataGenerationConfig):
    """
    Process one pcap ONCE, without the train/test split: tshark, the class of each
    packet (by its connection) and the hybrid features of all its labeled packets.
    The result is saved in output/packets/ and output/hybrid_all/; the split is done
    later by split_processed_pcaps, so changing the split does not reprocess the pcap.
    Only packets of ground truth connections reach generate_hybrid_data.
    """
    logger.info(f"Starting the data generation for File: {file['name']}")
    began = time.time()

    pcap_file_path = os.path.join(file['path'], file['name'])
    pcap_file_name = file['name'].split('.')[0]
    csv_file_name = f"{pcap_file_name}.csv"
    # A finished pcap leaves an empty marker file; a restarted run skips it.
    marker = f"{config.output_path}/processed_markers/{pcap_file_name}.done"
    if os.path.exists(marker):
        # The pcap is not processed again, but the inference points of the list that have
        # no hybrid file yet are built from its labeled packets (output/packets/), with
        # the same function and columns as below.
        missing = [n for n in config.inference_points_list
                   if not os.path.exists(f"{config.output_path}/hybrid_all/{pcap_file_name}_N_{n}.csv")]
        if missing:
            labeled_packet_data = pd.read_csv(f"{config.output_path}/packets/{pcap_file_name}.csv")
            for n in missing:
                hybrid = generate_hybrid_data(labeled_packet_data.copy(), csv_file_name, n)
                hybrid = hybrid.dropna()
                hybrid.to_csv(f"{config.output_path}/hybrid_all/{pcap_file_name}_N_{n}.csv",
                              columns=OUTPUT_COLUMNS, index=False)
            logger.info(f"Already processed: {file['name']}; built the missing inference points {missing}")
        else:
            logger.info(f"Skipping (already processed): {file['name']}")
        return []

    # tshark: one CSV row per packet of the pcap, then read into a table.
    tshark_file_path = os.path.join(config.output_path, 'tshark_files', f"{pcap_file_name}.csv")
    process_pcap_to_csv(pcap_file_path, tshark_file_path)
    packet_data = parse_tshark_csv_to_dataframe(tshark_file_path)
    # Adds Label and Split to each packet, from its flow or connection.
    labeled_packet_data = label_packet_data(packet_data, config)
    # Frees the memory of the unlabeled table.
    del packet_data
    # .dropna(subset=['Label']): drops the packets without Label, that is, the packets
    # that are not in any ground truth connection. The Split column is dropped: the
    # split is applied later, from labels.csv (split_processed_pcaps).
    labeled_packet_data = labeled_packet_data.dropna(subset=['Label']).drop(columns='Split')

    # The labeled packets of this pcap, with their frame numbers (used to copy the
    # test frames and to count the packets of each flow).
    labeled_packet_data.to_csv(f"{config.output_path}/packets/{pcap_file_name}.csv", index=False)
    for n in config.inference_points_list:
        # Packet and flow features at inference point n, for all packets of the pcap.
        # A flow is entirely in train or in test, and its features only use its own
        # packets, so computing them before the split gives the same values. Rows with
        # any missing value are dropped.
        hybrid = generate_hybrid_data(labeled_packet_data.copy(), csv_file_name, n)
        hybrid = hybrid.dropna()
        hybrid.to_csv(f"{config.output_path}/hybrid_all/{pcap_file_name}_N_{n}.csv",
                      columns=OUTPUT_COLUMNS, index=False)

    # The tshark CSV is large and no longer needed.
    os.remove(tshark_file_path)
    open(marker, 'w').close()
    logger.info(f"Finished File: {file['name']} in {time.time() - began:.0f} s "
                f"({len(labeled_packet_data)} labeled packets)")


def split_processed_pcaps(data_config: DataGenerationConfig, pcaps_path: str, test_pcap_classes=None):
    """
    Split the processed pcaps (output/packets/ and output/hybrid_all/) into train and
    test by the split of each flow in labels.csv, and write the per-pcap files read by
    save_merged_data, get_flow_length and merge_test_pcaps. It only reads the saved
    tables and copies the test frames with editcap, so it is fast and runs on every run.

    test_pcap_classes: classes whose test packets go to the test pcap of the simulation
    (None = all). Only the pcap is filtered; the csv_files and hybrid_data (parquets)
    keep all the classes.
    """
    labels = pd.read_csv(data_config.labels_path)
    # Split of each 5-tuple, the same for both directions (the same rule as the split
    # of the connections in connections.csv). undirected_flow_id: one key for both
    # directions of a 5-tuple.
    flow_split = dict(zip(undirected_flow_id(labels['Flow ID']), labels['Split']))

    # Removes the files of a previous split, so a pcap that no longer has packets in
    # one split leaves no old file behind.
    old_files = (glob.glob(f"{data_config.output_path}/*/csv_files/*.csv")
                 + glob.glob(f"{data_config.output_path}/*/hybrid_data/*")
                 + glob.glob(f"{data_config.output_path}/test/filtered_pcaps/*.pcap"))
    for old_file in old_files:
        os.remove(old_file)

    for packets_file in sorted(glob.glob(f"{data_config.output_path}/packets/*.csv")):
        pcap_file_name = os.path.basename(packets_file)[:-len('.csv')]
        packets = pd.read_csv(packets_file)
        # "train", "test", or NaN for a 5-tuple that is not in labels.csv.
        packets_split = undirected_flow_id(packets['Flow ID']).map(flow_split)
        # Hybrid features of each inference point, with the split of each row.
        hybrids = {}
        for n in data_config.inference_points_list:
            hybrid = pd.read_csv(f"{data_config.output_path}/hybrid_all/{pcap_file_name}_N_{n}.csv")
            hybrids[n] = (hybrid, undirected_flow_id(hybrid['Flow ID']).map(flow_split))

        for split in ['train', 'test']:
            packets_of_split = packets[packets_split == split]
            if packets_of_split.empty:
                continue
            # The labeled packets of this pcap and split (used by get_flow_length).
            packets_of_split.to_csv(f"{data_config.output_path}/{split}/csv_files/{pcap_file_name}_{split}.csv", index=False)
            if split == 'test':
                # Copies only the test packets of this pcap, of the classes in
                # test_pcap_classes, into the test pcap of the simulation, by their
                # frame numbers.
                if test_pcap_classes is None:
                    pcap_packets = packets_of_split
                else:
                    pcap_packets = packets_of_split[packets_of_split['Label'].isin(test_pcap_classes)]
                if not pcap_packets.empty:
                    pcap_file = glob.glob(f"{pcaps_path}/{pcap_file_name}.pcap*")[0]
                    extract_frames(pcap_file, pcap_packets['frame.number'],
                                   f"{data_config.output_path}/test/filtered_pcaps/{pcap_file_name}.pcap")
            for n, (hybrid, hybrid_split) in hybrids.items():
                hybrid[hybrid_split == split].to_csv(
                    f"{data_config.output_path}/{split}/hybrid_data/{data_config.use_case}_{split}_{pcap_file_name}_N_{n}.csv",
                    index=False)
        logger.info(f"Split {pcap_file_name}: train {(packets_split == 'train').sum()} packets, "
                    f"test {(packets_split == 'test').sum()} packets")

def run_flow_split_generation(input_data):
    # A failed pcap is logged (output_path/failed_pcaps.log) and the run goes on with
    # the next one; it has no marker, so a rerun processes it again.
    try:
        return __run_flow_split_generation(input_data[0], input_data[1])
    except Exception as e:
        logger.exception(f"Failed on {input_data[0]['name']}: {e}")
        with open(f"{input_data[1].output_path}/failed_pcaps.log", 'a') as log:
            log.write(f"{input_data[0]['name']}\t{e}\n")
        return []

def merge_test_pcaps():
    # Joins the test packets of all pcaps into the single pcap replayed by the
    # simulation; mergecap orders the packets by time.
    filtered = sorted(glob.glob(f"{output_path}/test/filtered_pcaps/*.pcap"))
    if filtered:
        subprocess.run(["mergecap", "-w", f"{output_path}/test/ToN_IoT_test.pcap"] + filtered, check=True)

def main_flow_split():
    """Generate the data with the train/test split defined per flow in label_data."""
    # Builds labels.csv (flows and their split) and connections.csv (time windows)
    # only when one of them is missing; delete or rename them to rebuild.
    if not os.path.isfile(label_data) or (connections_data and not os.path.isfile(connections_data)):
        status = generate_dataset_with_flowid.main(f"{ground_truth_path}", f"{label_data}", flow_column="Flow ID",
                                                   full_ground_truth_dir=full_ground_truth_path,
                                                   connections_csv=connections_data)
        if status:
            raise SystemExit(f"Label generation failed with status {status}")

    os.makedirs(f"{output_path}/tshark_files", exist_ok=True)
    os.makedirs(f"{output_path}/processed_markers", exist_ok=True)
    os.makedirs(f"{output_path}/test/filtered_pcaps", exist_ok=True)
    # Per-pcap results before the split (kept between runs).
    os.makedirs(f"{output_path}/packets", exist_ok=True)
    os.makedirs(f"{output_path}/hybrid_all", exist_ok=True)
    for split in ['train', 'test']:
        os.makedirs(f"{output_path}/{split}/csv_files", exist_ok=True)
        os.makedirs(f"{output_path}/{split}/hybrid_data", exist_ok=True)

    data_config = DataGenerationConfig(
        use_case=use_case,
        labels_path=label_data,
        output_path=output_path,
        ground_truth_path=ground_truth_path,
        inference_points_list=inference_points_list,
        split='all',
        connections_path=connections_data
    )
    # Every processed pcap, each paired with the same configuration.
    pcap_files = [{'path': pcaps_path, 'name': f} for f in sorted(os.listdir(pcaps_path))
                  if f.endswith(('.pcap', '.pcapng'))]
    input_data = list(product(pcap_files, [data_config]))

    consumed_cores = min([max_usable_cores, len(input_data) if input_data else 1])
    logger.info(f'Will use {consumed_cores} cores for {len(input_data)} pcaps. Starting pool...')
    # maxtasksperchild returns the memory of each pcap to the system.
    with mp.get_context('fork').Pool(processes=consumed_cores, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(run_flow_split_generation, input_data, chunksize=1):
            pass

    # The train/test split of the processed pcaps, from labels.csv.
    split_processed_pcaps(data_config, pcaps_path, test_pcap_classes)
    # After all pcaps: one parquet per split and inference point, the packet count of
    # each flow (ground truth of the simulation) and the single test pcap.
    save_merged_data(data_config)
    for split in ['train', 'test']:
        flow_length_data = get_flow_length(data_config, split)
        if not flow_length_data.empty:
            flow_length_data.to_csv(f"{output_path}/{split}/{use_case}_{split}_flow_length.csv", index=False)
    merge_test_pcaps()
    logger.info(f"Finished data generation, Data at: {output_path}")

def main():
    
    if not os.path.isfile(label_data):
        generate_dataset_with_flowid.main(f"{ground_truth_path}", f"{label_data}", flow_column="Flow ID")

    for split in ['train', 'test']:
        os.makedirs(f"{output_path}/{split}/tshark_files", exist_ok=True)
        os.makedirs(f"{output_path}/{split}/csv_files", exist_ok=True)
        os.makedirs(f"{output_path}/{split}/hybrid_data", exist_ok=True)
    
    input_data = []
    
    if os.path.exists(train_pcaps_path):
        train_pcap_files = [{'path': train_pcaps_path, 'name': f} for f in os.listdir(train_pcaps_path) if ((f.endswith('.pcap') | (f.endswith('.pcapng'))))]
        train_config = DataGenerationConfig(
            use_case=use_case,
            labels_path=label_data,
            output_path=output_path,
            ground_truth_path=ground_truth_path,
            inference_points_list=inference_points_list,
            split='train'
        )
        input_data.extend(list(product(train_pcap_files, [train_config])))

    if os.path.exists(test_pcaps_path):
        test_pcap_files = [{'path': test_pcaps_path, 'name': f} for f in os.listdir(test_pcaps_path) if ((f.endswith('.pcap') | (f.endswith('.pcapng'))))]
        test_config = DataGenerationConfig(
            use_case=use_case,
            labels_path=label_data,
            output_path=output_path,
            ground_truth_path=ground_truth_path,
            inference_points_list=inference_points_list,
            split='test'
        )
        input_data.extend(list(product(test_pcap_files, [test_config])))

    consumed_cores = min([max_usable_cores, len(input_data) if input_data else 1])
    logger.info(f'Will use {consumed_cores} cores. Starting pool...')
    if input_data:
        with mp.get_context('fork').Pool(processes=consumed_cores) as pool:
            try:
                # issue tasks to the process pool
                pool.imap_unordered(run_data_generation, input_data, chunksize=chunksize)
                # shutdown the process pool
                pool.close()
            except KeyboardInterrupt:
                logger.error("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
            # wait for all issued task to complete
            pool.join()
        del pool

    # Once all files have been processed, we can merge them into individual files and save them
    if input_data:
        save_merged_data(input_data[0][1])

    for split in ['train', 'test']:
        if input_data:
            flow_length_data = get_flow_length(input_data[0][1], split)
            if not flow_length_data.empty:
                flow_length_data.to_csv(f"{output_path}/{split}/{use_case}_{split}_flow_length.csv", index=False)

    logger.info(f"Finished data generation, Data at: {output_path}")

if __name__ == '__main__':
    basepath = path.dirname(__file__)
    # Date and time on every log line, e.g. "2026-11-04 16:52:48 INFO:TON-IOT:...".
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    config = configparser.ConfigParser()
    config.read(path.abspath(path.join(basepath,'params.ini')))
    config['DEFAULT']['training_root'] = path.abspath(path.join(basepath, '..', '..'))
    use_case = config['DEFAULT']['use_case']
    log_level = config['DEFAULT']['log_level']
    max_usable_cores = int(config['DEFAULT']['max_usable_cores'])
    chunksize = int(config['DEFAULT']['chunksize'])
    level = logging.getLevelName(log_level)
    logger = logging.getLogger(use_case)
    logger.setLevel(level)

    # These are parameters for the data generation
    label_data = config[use_case]['label_data']
    train_pcaps_path = config[use_case]['train_pcaps_path']
    test_pcaps_path = config[use_case]['test_pcaps_path']
    output_path = config[use_case]['output_path']
    inference_points_list = literal_eval(config[use_case]['inference_point_list'])
    ground_truth_path = config[use_case]['ground_truth_path']
    # Optional: label packets by 5-tuple and time, using the ts of the full ground truth.
    connections_data = config[use_case].get('connections_data')
    full_ground_truth_path = config[use_case].get('full_ground_truth_path')
    if 'pcaps_path' in config[use_case]:
        pcaps_path = config[use_case]['pcaps_path']
        # Classes of the test pcap of the simulation: the classes_filter of the
        # params.ini given in test_pcap_classes_params (the cluster analysis one), so
        # the pcap only carries the classes the models know. Without it, all classes.
        test_pcap_classes = None
        if 'test_pcap_classes_params' in config[use_case]:
            classes_config = configparser.ConfigParser()
            classes_config.read(config[use_case]['test_pcap_classes_params'])
            test_pcap_classes = literal_eval(classes_config[use_case]['classes_filter'])
        raise SystemExit(main_flow_split())
    raise SystemExit(main())