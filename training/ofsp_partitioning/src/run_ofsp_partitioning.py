#!/usr/bin/env python3
"""
Run OFSP (Optimal Feature Set Partitioning) on the training data and output
a cluster_info-compatible CSV that cluster_analysis can consume directly.

Output format (same as SPP):
    Cluster, Class List, Feature List
    0, ['normal', 'ddos', ...], ['ip.len', 'srcport', ...]
    1, ['normal', 'ddos', ...], ['ip.ttl', 'dstport', ...]

Usage:
    python3 run_ofsp_partitioning.py
"""

import ast
import configparser
import gc
import glob
import logging
import os
import sys
from os import path

import numpy as np
import pandas as pd

from ofsp import ofsp_partition

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_READ_DTYPES = {
    'tcp.flags.syn': np.int8, 'tcp.flags.ack': np.int8, 'tcp.flags.push': np.int8,
    'tcp.flags.fin': np.int8, 'tcp.flags.reset': np.int8, 'tcp.flags.ece': np.int8,
    'ip.proto': np.int8, 'ip.hdr_len': np.int8, 'ip.tos': np.int8, 'ip.ttl': np.uint8,
    'tcp.hdr_len': np.int8,
    'srcport': np.uint16, 'dstport': np.uint16,
    'ip.len': np.uint16, 'tcp.window_size_value': np.uint16, 'udp.length': np.uint16,
    'Min Packet Length': np.float32, 'Max Packet Length': np.float32,
    'Packet Length Mean': np.float32, 'Packet Length Total': np.float32,
    'UDP Len Min': np.float32, 'UDP Len Max': np.float32,
    'Flow IAT Min': np.float32, 'Flow IAT Max': np.float32,
    'Flow IAT Mean': np.float32, 'Time to Inference': np.float32,
    'SYN Flag Count': np.int8, 'ACK Flag Count': np.int8,
    'PSH Flag Count': np.int8, 'FIN Flag Count': np.int8,
    'RST Flag Count': np.int8, 'ECE Flag Count': np.int8,
}


def load_train_data(train_data_dir, flow_counts_path, inference_point, features, classes, logger, sample_target=None):
    """Load training data with flow-count-aware sample weights.

    Mirrors the sampling logic from modelAnalyzer._prepare_data:
    - 'flw' sample weight = (pkt_count - inference_point + 1) / pkt_count
    - 'pkt' sample weight = 1 / pkt_count
    """
    pq_path = os.path.join(train_data_dir, f"train_{inference_point}_pkts.parquet")
    csv_path = os.path.join(train_data_dir, f"train_{inference_point}_pkts.csv")

    if os.path.exists(pq_path):
        if sample_target:
            import pyarrow.parquet as _pq
            _pf = _pq.ParquetFile(pq_path)
            _total = _pf.metadata.num_rows
            _frac = min(1.0, (sample_target * 1.5) / _total)
            if _frac < 1.0:
                logger.info(f"Parquet is large ({_total:,} rows). Memory-safe sampling {_frac:.1%} from each row group...")
                _parts = []
                for _i in range(_pf.metadata.num_row_groups):
                    _rg = _pf.read_row_group(_i).to_pandas()
                    _parts.append(_rg.sample(frac=_frac, random_state=42))
                    del _rg
                data = pd.concat(_parts, ignore_index=True)
                del _parts
                gc.collect()
            else:
                data = pd.read_parquet(pq_path)
        else:
            # Full load with tight dtypes applied at the Arrow level. Parquet
            # stores wide int64/double columns, so cast to _READ_DTYPES before
            # to_pandas (mirrors the CSV branch and stays memory-safe).
            import pyarrow as _pa
            import pyarrow.parquet as _pq
            _arrow_t = {np.int8: _pa.int8(), np.uint8: _pa.uint8(),
                        np.uint16: _pa.uint16(), np.float32: _pa.float32()}
            _tbl = _pq.read_table(pq_path)
            for _i, _name in enumerate(_tbl.schema.names):
                if _name in _READ_DTYPES:
                    _tbl = _tbl.set_column(_i, _name, _tbl.column(_i).cast(_arrow_t[_READ_DTYPES[_name]]))
            data = _tbl.to_pandas(split_blocks=True, self_destruct=True)
            del _tbl
            gc.collect()
    elif os.path.exists(csv_path):
        data = pd.read_csv(csv_path, dtype=_READ_DTYPES)
    else:
        pattern = os.path.join(train_data_dir, f"*_N_{inference_point}.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No training files found for inference_point={inference_point}")
        data = pd.concat([pd.read_csv(f, dtype=_READ_DTYPES) for f in files], ignore_index=True)

    data = data[data['Label'].isin(classes)]
    data = data.dropna(subset=['srcport', 'dstport'])
    gc.collect()

    # Map flow packet counts
    flow_counts = pd.read_csv(flow_counts_path).set_index("Flow ID")["count"].to_dict()
    data["pkt_count"] = data["Flow ID"].map(flow_counts)
    data = data.dropna(subset=["pkt_count"])
    data["pkt_count"] = data["pkt_count"].astype(int)

    # Assign sample nature: 'pkt' when all flow-level stats are -1, else 'flw'
    _pkt_mask = (
        (data["Min Packet Length"] == -1) &
        (data["Max Packet Length"] == -1) &
        (data["Flow IAT Min"] == -1) &
        (data["Flow IAT Max"] == -1)
    )
    data["sample_nature"] = np.where(_pkt_mask, "pkt", "flw")

    # Per-sample weight (same formula as modelAnalyzer._prepare_data)
    data["weight"] = np.where(
        data["sample_nature"] == "flw",
        (data["pkt_count"] - inference_point + 1) / data["pkt_count"],
        1.0 / data["pkt_count"]
    )

    X = data[features].fillna(-1).values
    y = data["Label"].values
    weights = data["weight"].values
    return X, y, features, weights


def main():
    basepath = path.dirname(__file__)
    logging.basicConfig()
    config = configparser.ConfigParser()
    config.read(path.abspath(path.join(basepath, 'params.ini')))
    config['DEFAULT']['training_root'] = path.abspath(path.join(basepath, '..', '..'))

    use_case = config['DEFAULT']['use_case']
    log_level = config['DEFAULT']['log_level']
    n_views = int(config['DEFAULT']['n_views'])
    random_state = int(config['DEFAULT']['random_state'])
    n_estimators = int(config['DEFAULT']['n_estimators'])
    max_depth = int(config['DEFAULT']['max_depth']) if 'max_depth' in config['DEFAULT'] else None

    logger = logging.getLogger(use_case)
    logger.setLevel(logging.getLevelName(log_level))

    train_data_dir = config[use_case]['train_data_dir_path']
    flow_counts_path = config[use_case]['flow_counts_train_file_path']
    results_dir = config[use_case]['results_dir_path']
    # Optional env overrides for isolated experiments (do not touch params.ini).
    # OFSP_OBJECTIVE selects the greedy partitioning objective; OFSP_RESULTS_DIR
    # redirects where the solution CSV is written.
    ofsp_objective = os.environ.get('OFSP_OBJECTIVE', 'accuracy')
    results_dir = os.environ.get('OFSP_RESULTS_DIR', results_dir)
    classes = ast.literal_eval(config[use_case]['classes_filter'])
    inference_points = ast.literal_eval(config[use_case]['inference_point_list'])
    features_set = config[use_case]['features_set']
    features = ast.literal_eval(config['FEATURES'][features_set])

    os.makedirs(results_dir, exist_ok=True)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, n_jobs=12)

    # Full load was too slow for OFSP, so sample the training data down to 1M
    # rows: load_train_data does memory-safe per-row-group sampling to ~1.5M, then
    # the strict stratified subsample below trims to exactly this many rows.
    ofsp_sample_size = 1_000_000

    for n_point in inference_points:
        logger.info(f"Loading training data for inference point {n_point}...")
        X, y, feat_names, sample_weights = load_train_data(
            train_data_dir, flow_counts_path, n_point, features, classes, logger,
            sample_target=ofsp_sample_size
        )
        logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

        if ofsp_sample_size and X.shape[0] > ofsp_sample_size:
            logger.info(f"Applying strict stratified subsampling down to {ofsp_sample_size} rows...")
            rng = np.random.RandomState(random_state)
            idx = rng.choice(X.shape[0], size=ofsp_sample_size, replace=False)
            X, y, sample_weights = X[idx], y[idx], sample_weights[idx]
            logger.info(f"Final dataset ready: {X.shape[0]} rows")

        logger.info(f"Running OFSP with k={n_views} views (objective={ofsp_objective})...")
        views, view_accuracies, discarded = ofsp_partition(
            X, y, k=n_views, clf=clf, random_state=random_state,
            sample_weight=sample_weights, objective=ofsp_objective
        )

        # Build cluster_info-compatible CSV
        rows = []
        for i, (view_indices, acc) in enumerate(zip(views, view_accuracies)):
            view_features = [feat_names[idx] for idx in view_indices]
            rows.append({
                'Cluster': i,
                'Class List': classes,
                'Feature List': view_features,
                'OOB Accuracy': acc,
            })

        df = pd.DataFrame(rows)

        discarded_names = [feat_names[idx] for idx in discarded]
        outfile = os.path.join(results_dir, f"{use_case}_OFSP_solution_{n_point}pkts.csv")
        df.to_csv(outfile, index=False)

        logger.info(f"OFSP partitioning complete for N={n_point}")
        logger.info(f"Output: {outfile}")
        for i, row in df.iterrows():
            logger.info(f"  View {i}: {len(row['Feature List'])} features, "
                        f"OOB={row['OOB Accuracy']:.4f}, "
                        f"features={row['Feature List']}")
        if discarded_names:
            logger.info(f"  Discarded {len(discarded_names)} features: {discarded_names}")


if __name__ == "__main__":
    raise SystemExit(main())
