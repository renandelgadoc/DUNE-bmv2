# Data Generation
Data preparation for the ML model that targets hybrid packet- and flow-level classification.
For this you can use the python programs and packages under the `src` folder.

For the ToN-IoT pipeline used in this fork (train/test split per flow, packets labeled by
5-tuple and time), see [ToN-IoT: flow split labeled by time](#ton-iot-flow-split-labeled-by-time).

## Getting Started

Install the dependencies of the whole pipeline from the repository root (see the README there):
```bash
pip install -r requirements.txt
```

All configurable parameters are provided via the `src/params.ini` file.
If you want to experiment, this is the only file you should modify.

### Configurable Parameters
The default block has a single parameter controlling the use case. Two values are possible for this parameter: `UNSW` or
`TON-IOT`. The value of the use_case parameter will decide which block will be parsed for the main parameters of the use
case.
- `data_type`: the data type specifying if the data will be used for testing or training the model, e.g., `test`. Required for naming the files.
- `label_data`: the location of the file used to label data, e.g., `/nas_storage/shared/UNSW_PCAPS/iot_device_list.csv`
- `data_path`: the location of the input pcap traces
- `inference_point_list`: the list of inference points to be used in model analysis

### What do we generate?
**Requirements:** The dataset needs to include train and test splits, and each split must have its corresponding PCAP traces.

1) **Input:** We take the pcap traces for the given split as an input to dataGenerator.
2) **Raw Packet Data Extraction:** From the pcap trace, we generate TXT file containing raw packet data along with packet-level features.
3) **Labeled Packet Data Creation:** From the TXT file, we generate a CSV file containing labeled packet data. Each packet is assigned a flow ID.
4) **Experiment Data Generation:** Using the labeled packet data, we generate the experiment dataset for hybrid packet- and flow-level classification, considering each inference point separately.
5) **Parallel Processing:** If multiple PCAP traces are provided, steps 1-4 are executed in parallel for each trace.
6) **Merging Data from Multiple Traces:** We finally merge multiple experiment data generated from different pcap traces for the same inference point.
7) **Flow Length Calculation:** We generate a separate file containing the flow length (i.e., number of packets) for each flow.

**IMPORTANT NOTE:** Flows with the same flow ID are considered unique if they originate from different PCAP traces.

### Running the analysis
After configuring the `src/params.ini` file you can trigger the analysis by:
```bash
cd src
python3 generate_data.py
```
You will see logs with the progress of the execution.

## ToN-IoT: flow split labeled by time

The ToN-IoT files are available at https://research.unsw.edu.au/projects/toniot-datasets.
All paths below are relative to `training/data_generation/src/`.

### Files to download and where to put them

| Directory | What goes in it | ToN-IoT folder |
|---|---|---|
| `ground_truth/` | `train_test_network.csv` only. Every `.csv` in this folder is read, so keep nothing else here. | `Train_Test_datasets/Train_Test_Network_dataset/` |
| `ground_truth_full/` | `Network_dataset_1.csv` to `Network_dataset_23.csv`, with the `ts` column. They can stay inside the downloaded `Processed_Network_dataset.zip`, which is read in place. | `Processed_datasets/Processed_Network_dataset/` |
| `pcaps_not_processed/` | The raw network pcaps, as `.pcap` files (extract the downloaded zips). | `Raw_datasets/network_data/Network_datasets_pcaps/` |

Created by the pipeline: `processed_pcaps/`, `label/` and `output/`.

### Running

```bash
cd training/data_generation/src
# 1. Preprocess the raw pcaps of pcaps_not_processed/ into processed_pcaps/ (pcapfix,
#    Linux SLL to Ethernet, ICMP removed), skipping the ones already processed.
python3 prepare_input_pcap.py
# 2. Labels, train/test split and hybrid data, into label/ and output/.
python3 generate_data.py
```

Only `processed_pcaps/` is expensive to rebuild: `label/` and `output/` can be
regenerated from it and the ground truth.

### What `generate_data.py` does

1. **Labels** (`generate_dataset_with_flowid.py`, only when `label/labels.csv` or
   `label/connections.csv` is missing; rename them to regenerate):
   - `labels.csv`: one row per 5-tuple, both directions together, with its majority
     class and a train/test split stratified by class (`TRAIN_FRACTION = 0.75`,
     `RANDOM_SEED = 42`). The classes are `USED_CLASSES`: all 10 classes of the ground truth
     (normal, scanning, ddos, dos, injection, password, xss, backdoor, ransomware and
     mitm). The training picks the classes it uses in the `classes_filter` of each
     `params.ini`.
   - `connections.csv`: the time window of each connection of `train_test_network.csv`.
     Its `ts` comes from the `Network_dataset_*.csv` row with the same 5-tuple,
     duration, bytes, packets, connection state and class. The window is
     `[ts, ts + duration]` with a small tolerance (1 s when `ts` has no fraction of a
     second). The script prints how many connections matched once, more than once or
     not at all.
   - **Flow limit** (`MAX_FLOWS_PER_CLASS = 10000`): before the train/test split, each
     class keeps at most 10,000 flows, the ones whose first connection starts first in
     time, and these are split 75/25 again. A class with fewer flows keeps all of
     them (in the train-test subset: ransomware 1,415, dos 2,759, backdoor 816, mitm
     950). The full files are kept as `labels_all_flows.csv` and
     `connections_all_flows.csv`. `MAX_FLOWS_PER_CLASS = None` turns the limit off.
2. **Each processed pcap, once, without the split**: tshark extracts the packets; a
   packet gets the class of the connection with its 5-tuple (either direction) whose
   window holds its time (`frame.time_epoch`). Packets outside every window are
   dropped. The labeled packets go to `output/packets/<pcap>.csv` and the hybrid
   features of each inference point to `output/hybrid_all/<pcap>_N_<n>.csv`. This is
   the slow step (about 3 h for the 64 pcaps with 1 core).
3. **Split** (`split_processed_pcaps`, fast, runs on every run): each flow goes to
   train or test by the `Split` of its 5-tuple in `label/labels.csv`. It writes
   `output/{train,test}/csv_files/`, `output/{train,test}/hybrid_data/*_N_<n>.csv` and
   the test frames of each pcap in `output/test/filtered_pcaps/` (editcap). Only the
   test pcap is filtered by class: it gets the test packets of the classes in the
   `classes_filter` of the params.ini given in `test_pcap_classes_params` (the cluster
   analysis one), so the simulation only replays the classes the models know. The
   csv_files, hybrid_data and flow_length keep all the classes.
4. **Merge**: `output/{train,test}/hybrid_data/{train,test}_{n}_pkts.parquet`,
   `output/{train,test}/TON-IOT_{train,test}_flow_length.csv` and
   `output/test/ToN_IoT_test.pcap`.

A finished pcap leaves a marker in `output/processed_markers/`, so a stopped run can
be restarted and skips the pcaps already done (it only builds the inference points of the
list that are still missing for them).

The parquets keep all the classes of the ground truth; each training stage picks its
classes in the `classes_filter` of its `params.ini`. So:
- **To change the classes of the training**: only edit `classes_filter`, nothing here.
- **To change the train/test split** (`TRAIN_FRACTION`, `RANDOM_SEED`): rename
  `label/labels.csv` and `label/connections.csv` and run `generate_data.py` again. The
  pcaps are not reprocessed (they have their marker); only steps 1, 3 and 4 run.
- **To change the flow limit** (`MAX_FLOWS_PER_CLASS`) without reading the ground
  truth again: run, from `src/`,
  `python3 -c "import generate_dataset_with_flowid as g; g.limit_existing_files('label/labels.csv', 'label/connections.csv')"`
  It reads the `_all_flows` files, writes the limited `labels.csv` and
  `connections.csv`, and then `generate_data.py` only redoes steps 3 and 4.
- **To add inference points** to `inference_point_list`: run `generate_data.py` again. The
  pcaps already processed are not reprocessed; the missing points are built from their
  labeled packets in `output/packets/`.
- **To reprocess the pcaps** (e.g. new features): delete `output/`.

The simulation uses `output/test/ToN_IoT_test.pcap` and
`output/test/TON-IOT_test_flow_length.csv`, copied to `simulation/data/`.

### Parameters in `src/params.ini` (`[TON-IOT]`)

- `pcaps_path`: the processed pcaps. When set, `generate_data.py` uses this pipeline.
- `label_data`: the flow split (`label/labels.csv`).
- `connections_data`: the connection windows (`label/connections.csv`). Without it,
  packets are labeled by 5-tuple only, which also takes the packets of connections that
  are not in the ground truth.
- `full_ground_truth_path`: the folder with the `Network_dataset_*.csv` files.
- `output_path`: the output folder (`output`).
- `inference_point_list`: the inference points, e.g. `[2, 3]`.
- `test_pcap_classes_params`: the params.ini whose `classes_filter` sets the classes of
  the test pcap (`cluster_analysis/src/params.ini`). Without it, all classes.
