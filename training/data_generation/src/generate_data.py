import configparser
import logging
import os
import multiprocessing as mp
import sys
from datageneration import *
import generate_dataset_with_flowid
from os import path
from ast import literal_eval
from itertools import product
                
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
    logging.basicConfig()
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
    raise SystemExit(main())